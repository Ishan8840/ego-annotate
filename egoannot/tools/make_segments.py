"""
Cut review segments from the corpus and precompute their pose overlays.

Each segment is cropped to the region the hands actually occupy (computed from
the shipped pose, padded, 4:3) instead of downscaling the whole 1920x1456
frame. That buys roughly 2x linear detail on the fingers at the same bitrate,
which is what contact judgement needs, and keeps eight minutes of video inside
one review page.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile

import numpy as np

from .. import config
from ..core.geometry import quat_to_R
from ..core.mcap_io import read_episode
from ..core.video import extract_h264

CFG = config.SEGMENT_RENDER


def project_all(ep, t0, t1, fps):
    """All 21 joints of both hands, in ORIGINAL full-frame pixel coords."""
    K, E = ep["K"], ep["extr"]
    times = np.arange(t0, t1, 1.0 / fps)
    frames = []
    for t in times:
        i = int(np.clip(np.searchsorted(E[:, 0], t), 0, len(E) - 1))
        R, tw = quat_to_R(*E[i, 4:8]), E[i, 1:4]
        row = {}
        for side in ("left", "right"):
            J = ep[f"/pose/{side}_hand_joints"]
            A = ep[f"/pose/{side}_hand"]
            if not len(J):
                continue
            k = int(np.clip(np.searchsorted(A[:, 0], t), 0, len(J) - 1))
            pc = (J[k] - tw) @ R
            z = pc[:, 2]
            points = []
            for n in range(len(pc)):
                if z[n] <= 1e-6:
                    points.append(None)
                    continue
                points.append([K["fx"] * pc[n, 0] / z[n] + K["cx"],
                               K["fy"] * pc[n, 1] / z[n] + K["cy"]])
            row[side] = dict(p=points,
                             ap=float(np.linalg.norm(J[k][4] - J[k][8])))
        frames.append(row)
    return times, frames


def crop_rect(frames, W, H, pad=None):
    """4:3 rect covering every projected joint, padded, clamped to the frame."""
    pad = CFG["crop_pad"] if pad is None else pad
    xs = [p[0] for f in frames for s in f.values() for p in s["p"] if p]
    ys = [p[1] for f in frames for s in f.values() for p in s["p"] if p]
    if not xs:
        return 0, 0, W, H
    x0, x1 = np.percentile(xs, 1), np.percentile(xs, 99)
    y0, y1 = np.percentile(ys, 1), np.percentile(ys, 99)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w = (x1 - x0) * (1 + 2 * pad)
    h = (y1 - y0) * (1 + 2 * pad)
    w = max(w, h * 4 / 3, W * 0.34)          # enforce 4:3 and a sane minimum
    h = w * 3 / 4
    x = int(round(max(0, min(W - w, cx - w / 2))))
    y = int(round(max(0, min(H - h, cy - h / 2))))
    w = int(round(min(w, W - x)))
    h = int(round(min(h, H - y)))
    return x, y, w - w % 2, h - h % 2


def render(only=None, outdir=None, cfg=CFG):
    outdir = str(outdir or config.SEGMENTS_DIR)
    os.makedirs(outdir, exist_ok=True)
    segs = json.load(open(config.SEGMENT_DEFS))
    if only:
        segs = [s for s in segs if s["id"] in set(only)]

    events_path = str(config.EVENTS_RECORDS)
    candidates = ([json.loads(l) for l in open(events_path)]
                  if os.path.exists(events_path) else [])

    out_w, out_h, ov_fps = cfg["out_w"], cfg["out_h"], cfg["overlay_fps"]
    h264_cache, manifest, total_bytes = {}, [], 0
    try:
        for seg in segs:
            name = os.path.basename(seg["source"]).rsplit(".", 1)[0]
            try:
                path = config.episode_path(name)
            except FileNotFoundError as e:
                print("MISSING", e)
                continue
            ep = read_episode(path, want_video=False)
            t0, t1 = seg["t0"], seg["t1"]
            W, H = ep["K"]["w"], ep["K"]["h"]
            _, frames = project_all(ep, t0, t1, ov_fps)
            cx, cy, cw, ch = crop_rect(frames, W, H)
            sx, sy = out_w / cw, out_h / ch

            # compact overlay: one flat int array per hand per frame
            # -> [aperture_mm, x0, y0, ..., x20, y20] in OUTPUT pixel coords
            lanes = {"L": [], "R": []}
            for f in frames:
                for side, key in (("left", "L"), ("right", "R")):
                    hand = f.get(side)
                    if hand is None:
                        lanes[key].append(None)
                        continue
                    flat = [int(round(1000 * hand["ap"]))]
                    for p in hand["p"]:
                        if p is None:
                            flat += [-999, -999]
                        else:
                            flat += [int(round((p[0] - cx) * sx)),
                                     int(round((p[1] - cy) * sy))]
                    lanes[key].append(flat)

            if path not in h264_cache:
                with tempfile.NamedTemporaryFile(suffix=".h264", delete=False) as fh:
                    fh.write(extract_h264(path))
                    h264_cache[path] = fh.name
            mp4 = os.path.join(outdir, seg["id"] + ".mp4")
            subprocess.run(
                ["ffmpeg", "-v", "error", "-f", "h264", "-r", str(ep["src_fps"]),
                 "-i", h264_cache[path], "-ss", str(t0), "-t", str(t1 - t0),
                 "-vf", f"crop={cw}:{ch}:{cx}:{cy},scale={out_w}:{out_h}",
                 "-c:v", "libx264", "-crf", str(cfg["crf"]), "-preset", "slow",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", mp4, "-y"],
                check=True)
            size = os.path.getsize(mp4)
            total_bytes += size

            overlay = os.path.join(outdir, seg["id"] + ".json")
            payload = dict(
                id=seg["id"], episode=name, cls=seg["cls"], t0=t0, t1=t1,
                fps=ov_fps, w=out_w, h=out_h, crop=[cx, cy, cw, ch],
                src_fps=ep["src_fps"], L=lanes["L"], R=lanes["R"],
                candidates=[{k: e[k] for k in
                             ("t", "hand", "type", "aperture_delta",
                              "aperture_held", "wrist_speed") if k in e}
                            for e in candidates
                            if e["episode"] == name and t0 <= e["t"] < t1])
            json.dump(payload, open(overlay, "w"), separators=(",", ":"))
            manifest.append(dict(
                id=seg["id"], episode=name, cls=seg["cls"], t0=t0, t1=t1,
                dur=round(t1 - t0, 2), mp4=size,
                ov=os.path.getsize(overlay), crop=[cx, cy, cw, ch],
                n_cand=len(payload["candidates"])))
            print(f"{seg['id']:<15s} {seg['cls']:<15s} {t1 - t0:5.1f}s  "
                  f"crop {cw}x{ch}@{cx},{cy}  mp4 {size / 1e6:5.2f} MB  "
                  f"ov {os.path.getsize(overlay) / 1e3:5.0f} KB  "
                  f"{manifest[-1]['n_cand']:3d} cand", flush=True)
            del ep
    finally:
        for tmp in h264_cache.values():
            os.unlink(tmp)

    # Merge into any existing manifest rather than replacing it: rendering a
    # subset with --only must not drop the entries for segments it did not
    # touch, or every downstream stage silently narrows to that subset.
    manifest_path = os.path.join(outdir, "manifest.json")
    merged = {}
    if os.path.exists(manifest_path):
        for entry in json.load(open(manifest_path)):
            merged[entry["id"]] = entry
    for entry in manifest:
        merged[entry["id"]] = entry
    order = [s["id"] for s in json.load(open(config.SEGMENT_DEFS))]
    manifest = [merged[i] for i in order if i in merged]
    json.dump(manifest, open(manifest_path, "w"), indent=1)
    duration = sum(m["dur"] for m in manifest)
    overlay_bytes = sum(m["ov"] for m in manifest)
    print(f"\n{len(manifest)} segments in manifest, {duration / 60:.1f} min "
          f"-> {outdir}")
    print(f"  video {total_bytes / 1e6:.2f} MB  overlays {overlay_bytes / 1e6:.2f} MB")
    print(f"  base64 page estimate: "
          f"{(total_bytes * 4 / 3 + overlay_bytes) / 1e6:.2f} MB (limit 16)")
    return manifest
