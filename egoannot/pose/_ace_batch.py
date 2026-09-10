#!/usr/bin/env python
"""
Batched, chunked ACE-Ego-Hand inference. Runs inside the estimator's env.

Three things the stock `infer_video.py` does not do, each of which cost us a
measured run:

* It VAE-encodes the WHOLE clip in one call and keeps the latent resident, so
  memory scales with clip length x frame area. At 1280x960 a 681-frame episode
  needs about 10 GB for the pixel tensor alone and dies on a 24 GB card --
  while the DiT itself only ever sees 81 frames at a time. Here the clip is
  encoded in chunks, so resolution is no longer traded against clip length.
* It reloads the 5 GB backbone per invocation. One process, one load.
* A failure takes the whole batch down. Each episode is caught and reported.

Chunking costs context, and context is the paper's whole argument: bidirectional
attention over the clip is what reconstructs occluded and out-of-sight hands.
So chunks OVERLAP by at least one 81-frame training window and each one's
margin is discarded, meaning every frame is decoded by a chunk that saw real
context on both sides of it -- except at the true clip ends, where none exists.

Resolution is chosen by trying the requested width and halving the chunk on
CUDA OOM, so the highest width that fits is used rather than the largest that
was guessed safe.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
GRID = 32          # VAE downsamples 16x; latent grids must be even
WINDOW = 81        # the training window, in pixel frames


def encode_size(w: int, h: int, target_w: int) -> tuple[int, int]:
    s = (target_w or w) / float(w)
    return (max(GRID, int(round(w * s / GRID)) * GRID),
            max(GRID, int(round(h * s / GRID)) * GRID))


def trim(n: int) -> int:
    """The causal VAE consumes 4k+1 frames."""
    return 4 * ((n - 1) // 4) + 1


def decode_range(path: str, start: int, count: int, size: tuple[int, int]):
    """(F,3,H,W) in [-1,1], RGB, for a frame range -- never the whole clip."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    while len(out) < count:
        ok, f = cap.read()
        if not ok:
            break
        if (f.shape[1], f.shape[0]) != size:
            f = cv2.resize(f, size, interpolation=cv2.INTER_AREA)
        out.append(f[:, :, ::-1].copy())
    cap.release()
    if not out:
        raise RuntimeError(f"{path}: no frames decoded at {start}")
    a = np.stack(out).astype(np.float32) / 255.0 * 2.0 - 1.0
    return torch.from_numpy(a).permute(0, 3, 1, 2).contiguous()


def plan_chunks(n: int, chunk: int, overlap: int) -> list[tuple[int, int]]:
    """(start, length) covering n frames, overlapping, each a valid 4k+1."""
    chunk = max(WINDOW, trim(chunk))
    if n <= chunk:
        return [(0, trim(n))]
    stride = max(WINDOW, chunk - overlap)
    spans, start = [], 0
    while start < n:
        length = trim(min(chunk, n - start))
        if length < WINDOW:                      # tail too short: pull it back
            start = max(0, n - chunk)
            spans.append((start, trim(min(chunk, n - start))))
            break
        spans.append((start, length))
        if start + length >= n:
            break
        start += stride
    return spans


def keep_window(spans, n: int) -> list[tuple[int, int]]:
    """
    Which frames each chunk owns: the midpoint of every overlap is the seam.

    Splitting at the midpoint means each frame is taken from the chunk that
    saw the most context around it, which is the whole reason for overlapping.
    """
    out = []
    for i, (s, ln) in enumerate(spans):
        lo = s if i == 0 else (spans[i - 1][0] + spans[i - 1][1] + s) // 2
        hi = s + ln if i == len(spans) - 1 else (
            s + ln + spans[i + 1][0]) // 2
        out.append((max(lo, s), min(hi, s + ln, n)))
    return out


def run_episode(item, model, vae, predict, dump_keys, args, device) -> dict:
    from ace_ego_hand.video_vae import encode

    video, cam_path = item["video"], item["camera"]
    cap = cv2.VideoCapture(video)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    n = trim(n_src)
    enc_w, enc_h = encode_size(src_w, src_h, args.encode_w)

    cam = json.load(open(cam_path))
    k = cam["frames"][0]["intrinsics"]
    cal_w = float(cam.get("image_width", src_w))
    cal_h = float(cam.get("image_height", src_h))
    sx, sy = enc_w / cal_w, enc_h / cal_h
    intr = {"fx": k["fx"] * sx, "fy": k["fy"] * sy,
            "cx": k["cx"] * sx, "cy": k["cy"] * sy,
            "image_width": enc_w, "image_height": enc_h}

    chunk, overlap = args.chunk, max(WINDOW, args.overlap)
    while True:
        spans = plan_chunks(n, chunk, overlap)
        keeps = keep_window(spans, n)
        acc = {key: [] for key in dump_keys}
        try:
            for (s, ln), (lo, hi) in zip(spans, keeps):
                px = decode_range(video, s, ln, (enc_w, enc_h))
                ctrl = encode(vae, px, device).float()
                del px
                pred = predict(model, args.tap, ctrl, intr, args.mode, device,
                               tile_w=args.tile_w)
                del ctrl
                torch.cuda.empty_cache()
                a, b = lo - s, hi - s
                for key in dump_keys:
                    acc[key].append(np.asarray(pred[key])[a:b])
            break
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if chunk <= WINDOW:
                raise
            chunk = max(WINDOW, trim(chunk // 2))
            print(f"    OOM -> retrying with chunk={chunk}", flush=True)

    out = {key: np.concatenate(v, 0) for key, v in acc.items()}
    f_use = len(out["global_orient"])
    rec = {key: out[key].astype(np.float32) for key in dump_keys}
    packed = {
        "go": rec["global_orient"].reshape(f_use, 2, 1, 3, 3),
        "hp": rec["hand_pose"], "betas": rec["betas"],
        "cam_trans": rec["cam_trans"],
        "is_right": np.tile(np.array([[0.0, 1.0]], np.float32), (f_use, 1)),
        "exists_2d": rec["exists_2d"], "exists_3d": rec["exists_3d"],
        "joints_2d": rec["direct_joints2d"],
        "joints_cam_direct": rec["direct_joints_cam"],
        "wrist_cam_direct": rec["direct_wrist_cam"],
        "intrinsics": intr, "pred_intrinsics": None,
        "chunks": [list(map(int, s)) for s in spans],
        "encode_size": [enc_w, enc_h],
    }
    out_dir = Path(item["out"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (Path(video).stem + ".pkl")
    with open(path, "wb") as fh:
        pickle.dump(packed, fh)
    return dict(episode=Path(video).stem, frames=f_use, pkl=str(path),
                encode=[enc_w, enc_h], chunks=len(spans), chunk_len=chunk)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--home", required=True)
    ap.add_argument("--opt", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--encode-w", type=int, default=1280)
    ap.add_argument("--chunk", type=int, default=241)
    ap.add_argument("--overlap", type=int, default=81)
    ap.add_argument("--tap", type=int, default=15)
    ap.add_argument("--mode", default="tiled")
    ap.add_argument("--tile-w", type=int, default=22)
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    home = Path(args.home)
    sys.path.insert(0, str(home))
    sys.path.insert(0, str(home / "third_party"))
    import os
    os.chdir(home)

    import yaml
    from ace_ego_hand.inference import DUMP_KEYS, predict_video
    from ace_ego_hand.models.geodit_model import GeoDitModel
    from ace_ego_hand.video_vae import load_vae

    device = torch.device("cuda")
    opt = yaml.safe_load(open(args.opt))
    model = GeoDitModel(opt, device)
    model.load_inference(str(Path(args.ckpt).resolve()))
    vae = load_vae(device)

    items = json.load(open(args.manifest))
    rows = []
    for i, item in enumerate(items, 1):
        name = Path(item["video"]).stem
        print(f"[{i}/{len(items)}] {name}", flush=True)
        try:
            r = run_episode(item, model, vae, predict_video, DUMP_KEYS, args,
                            device)
            r["ok"] = True
            print(f"    {r['frames']} frames  {r['encode'][0]}x{r['encode'][1]}"
                  f"  {r['chunks']} chunk(s) of {r['chunk_len']}", flush=True)
        except Exception as e:
            torch.cuda.empty_cache()
            r = dict(episode=name, ok=False,
                     error=f"{type(e).__name__}: {e}")
            print(f"    FAILED {r['error']}", flush=True)
            traceback.print_exc()
        rows.append(r)
    if args.report:
        json.dump(rows, open(args.report, "w"), indent=1)
    ok = sum(1 for r in rows if r.get("ok"))
    print(f"[batch] {ok}/{len(rows)} succeeded", flush=True)


if __name__ == "__main__":
    main()
