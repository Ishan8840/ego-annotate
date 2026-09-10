#!/usr/bin/env python
"""
ARCTIC ground truth -> 21 camera-frame joints. Runs in the estimator's env.

ARCTIC stores hands as MANO parameters in world coordinates (axis-angle
global orientation, 45-dim articulation, translation, 10 shape) plus a
per-frame egocentric camera pose. Turning that into something comparable with
a prediction means running MANO forward and moving the result into the camera
frame -- which needs the licensed model files, so this lives beside the
estimator rather than in the main package.

Joint order is made to match the estimator's exactly (OpenPose 21: wrist,
then thumb/index/middle/ring/pinky, four each) by reusing the estimator's own
remap constants. If those two disagreed, every number downstream would be
comparing different fingers.

`flat_hand_mean=False` matches both ARCTIC's own loader and the estimator's,
but a mismatch there would curl the fingers while leaving the wrist correct,
so `--check` reprojects the result onto the image for inspection instead of
taking the convention on trust.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True, help="ACE-Ego-Hand checkout")
    ap.add_argument("--raw", required=True, help="ARCTIC raw_seqs dir")
    ap.add_argument("--subject", default="s05")
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.home)
    import smplx
    from ace_ego_hand.mano_utils import (MANO_CHECKPOINT_DIR, _MANO_TIP_IDS,
                                         _MANO_TO_OP)

    models = {s: smplx.create(MANO_CHECKPOINT_DIR, model_type="mano",
                              is_rhand=(s == "right"), use_pca=False,
                              flat_hand_mean=False).eval()
              for s in ("left", "right")}
    os.makedirs(args.out, exist_ok=True)
    index = []

    for seq in args.seqs:
        base = Path(args.raw) / args.subject
        mano = np.load(base / f"{seq}.mano.npy", allow_pickle=True).item()
        ego = np.load(base / f"{seq}.egocam.dist.npy", allow_pickle=True).item()
        R = np.asarray(ego["R_k_cam_np"], np.float64)        # (N,3,3) world->cam
        T = np.asarray(ego["T_k_cam_np"], np.float64).reshape(-1, 3)
        K = np.asarray(ego["intrinsics"], np.float64)
        dist = np.asarray(ego["dist8"], np.float64)

        out = {"K_full": K, "dist8": dist, "R": R, "T": T}
        n = len(R)
        for side in ("left", "right"):
            p = mano[side]
            rot = torch.tensor(np.asarray(p["rot"], np.float32))
            pose = torch.tensor(np.asarray(p["pose"], np.float32))
            trans = torch.tensor(np.asarray(p["trans"], np.float32))
            betas = torch.tensor(np.tile(np.asarray(p["shape"], np.float32),
                                         (len(rot), 1)))
            with torch.no_grad():
                o = models[side](global_orient=rot, hand_pose=pose,
                                 betas=betas, transl=trans)
            j16 = o.joints.numpy()                       # (N,16,3) world
            tips = o.vertices.numpy()[:, _MANO_TIP_IDS]  # (N,5,3)
            j21 = np.concatenate([j16, tips], axis=1)[:, _MANO_TO_OP]
            m = min(n, len(j21))
            # world -> ego camera:  p_cam = R @ p_world + T
            cam = np.einsum("nij,nkj->nki", R[:m], j21[:m]) + T[:m, None, :]
            out[f"{side}_world"] = j21[:m]
            out[f"{side}_cam"] = cam
        np.savez_compressed(Path(args.out) / f"{seq}.npz", **out)
        index.append(dict(seq=seq, frames=int(min(n, len(j21))),
                          fx=float(K[0, 0]), fy=float(K[1, 1]),
                          cx=float(K[0, 2]), cy=float(K[1, 2]),
                          dist_max=float(np.abs(dist).max())))
        print(f"  {seq:<28} {min(n, len(j21)):5d} frames", flush=True)

    json.dump(index, open(Path(args.out) / "index.json", "w"), indent=1)
    print(f"[gt] {len(index)} sequences -> {args.out}")


if __name__ == "__main__":
    main()
