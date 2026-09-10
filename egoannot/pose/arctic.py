"""
ARCTIC as a validation corpus: the same pipeline, against real ground truth.

Everything measured on EgoStandard is *agreement* -- with an annotation whose
own accuracy is unknown, refereed by a stereo signal that the annotation
itself only satisfies about half the time. That answers "does this reproduce
the incumbent", not "how accurate is it".

ARCTIC answers the second question. It carries native mocap MANO ground
truth and an egocentric camera, and subject s05 is exactly the split the
estimator reports as its ARCTIC test set (protocol p2's val list, 34
sequences), so it is genuinely held out rather than memorised.

The adapter is deliberately thin. Ego frames become the same mp4 + cam.json
the EgoStandard path produces, so the identical runner, lift and PnP solve
execute here -- if they behaved differently the number would not transfer.

Two corpus facts worth recording:

* Camera 0 is the egocentric view. The distributed "cropped" ego images are
  840x600, a pure 0.3x resize of the 2800x2000 original with no cropping, so
  the intrinsics scale by 0.3 and nothing shifts.
* The ego camera is mildly distorted (an 8-parameter model, |coeff| <= 0.05).
  The estimator is pinhole-only, so frames are used as delivered and the
  distortion is a small unmodelled error, the same one the published ARCTIC
  numbers carry.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

from .. import config

ARCTIC = Path(os.environ.get("ARCTIC_ROOT", "/home/axibo/ego-data/arctic"))
EGO_CAM = 0
CROP_SCALE = 0.3        # 2800x2000 -> 840x600, verified as a pure resize
FPS = 30.0


def subject_meta(subject: str = "s05") -> dict:
    return json.load(open(ARCTIC / "meta" / "misc.json"))[subject]


def sequences(split: str = "val", protocol: str = "p2") -> list[str]:
    """Sequence ids for an ARCTIC split. p2 is the egocentric protocol."""
    p = json.load(open(ARCTIC / "splits_json" / f"protocol_{protocol}.json"))
    return p[split]


def build_clip(seq: str, subject: str = "s05", outdir=None) -> dict | None:
    """Ego frames -> mp4 + cam.json + frame sidecar, the pipeline's input."""
    src = ARCTIC / "seq" / seq / str(EGO_CAM)
    if not src.is_dir():
        return None
    outdir = Path(outdir or config.ARTIFACTS / "arctic" / "clips")
    outdir.mkdir(parents=True, exist_ok=True)
    gt = np.load(ARCTIC / "gt" / f"{seq}.npz")
    K = np.asarray(gt["K_full"], float) * CROP_SCALE

    mp4 = outdir / f"{seq}.mp4"
    if not mp4.exists():
        # Frame numbering is 1-based and offset from the mocap index by
        # `ioi_offset`; -start_number keeps ffmpeg's ordering aligned with it.
        ioi = int(subject_meta(subject)["ioi_offset"])
        cmd = ["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS),
               "-start_number", str(ioi), "-i", str(src / "%05d.jpg"),
               "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(mp4)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{seq}: ffmpeg failed: {r.stderr[:300]}")

    import cv2
    cap = cv2.VideoCapture(str(mp4))
    w, h = int(cap.get(3)), int(cap.get(4))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    json.dump({"image_width": w, "image_height": h, "model": "pinhole",
               "frames": [{"intrinsics": {"fx": K[0, 0], "fy": K[1, 1],
                                          "cx": K[0, 2], "cy": K[1, 2]}}]},
              open(outdir / f"{seq}.cam.json", "w"), indent=1)
    # the pipeline's lift expects a frame->time map and camera extrinsics;
    # here the prediction stays in the CAMERA frame, so identity extrinsics
    # keep `to_world` a no-op and the comparison happens where truth lives.
    ident = np.zeros((n, 8))
    ident[:, 0] = np.arange(n) / FPS
    ident[:, 4] = 1.0
    np.savez_compressed(outdir / f"{seq}.frames.npz",
                        vts=np.arange(n) / FPS, extr=ident, src_fps=FPS,
                        duration_s=n / FPS)
    return dict(seq=seq, frames=n, size=[w, h], mp4=str(mp4),
                fx=float(K[0, 0]), cx=float(K[0, 2]), cy=float(K[1, 2]))


def build_all(seqs=None, subject: str = "s05") -> list[dict]:
    avail = sorted(p.name for p in (ARCTIC / "seq").iterdir()) if (
        ARCTIC / "seq").is_dir() else []
    seqs = seqs or avail
    out = []
    for s in seqs:
        info = build_clip(s, subject)
        if info:
            out.append(info)
            print(f"  {s:<28} {info['frames']:4d} frames "
                  f"{info['size'][0]}x{info['size'][1]}", flush=True)
    return out


# ---------------------------------------------------------------- evaluate
def _procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Align A onto B with similarity (scale, rotation, translation)."""
    mu_a, mu_b = A.mean(0), B.mean(0)
    a, b = A - mu_a, B - mu_b
    U, S, Vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    var = (a ** 2).sum()
    s = (S * np.array([1, 1, d])).sum() / var if var > 1e-12 else 1.0
    return s * (a @ R) + mu_b


def evaluate(seqs=None, world_dir=None, gt_dir=None) -> dict:
    """
    Real accuracy: predicted joints against ARCTIC's mocap MANO ground truth.

    Reported in the camera frame, where the prediction is made and where the
    ground truth is placed, so nothing depends on an estimated world pose:

      MPJPE      wrist-aligned mean joint error -- articulation and
                 orientation, with placement divided out
      PA-MPJPE   after similarity alignment -- shape alone, scale removed
      wrist      absolute camera-frame wrist error -- placement, the term the
                 other two deliberately exclude
      EPE2D      reprojection error in pixels through the true intrinsics
    """
    world_dir = Path(world_dir or config.ARTIFACTS / "arctic" / "world")
    gt_dir = Path(gt_dir or ARCTIC / "gt")
    seqs = seqs or sorted(p.stem for p in world_dir.glob("*.npz"))
    rows = []
    for seq in seqs:
        wp, gp = world_dir / f"{seq}.npz", gt_dir / f"{seq}.npz"
        if not (wp.exists() and gp.exists()):
            continue
        z, g = np.load(wp), np.load(gp)
        K = np.asarray(g["K_full"], float) * CROP_SCALE
        iw, ih = 2800 * CROP_SCALE, 2000 * CROP_SCALE
        rec = {"seq": seq, "hands": {}}
        for side in ("left", "right"):
            P = z[f"{side}_hand_joints"]                 # predicted, cam frame
            G = g[f"{side}_cam"]
            n = min(len(P), len(G))
            P, G = P[:n], G[:n]
            ok = np.isfinite(P).all((1, 2)) & np.isfinite(G).all((1, 2))
            if ok.sum() < 8:
                continue
            P, G = P[ok], G[ok]
            # The paper's on-screen gate (Appendix A.1): a hand counts as
            # visible if at least one ground-truth joint projects inside the
            # image at a depth above 1 cm. Out-of-sight hands dominate the
            # error -- ~65% of the worst frames here have the hand entirely
            # outside the frame -- and are scored separately rather than
            # pooled, which is what makes these figures comparable with
            # published ARCTIC numbers instead of a stricter mixture.
            zg = np.clip(G[..., 2], 1e-6, None)
            ug = K[0, 0] * G[..., 0] / zg + K[0, 2]
            vg = K[1, 1] * G[..., 1] / zg + K[1, 2]
            onscreen = ((ug >= 0) & (ug < iw) & (vg >= 0) & (vg < ih)
                        & (G[..., 2] > 0.01)).any(axis=1)
            mpjpe = np.linalg.norm((P - P[:, :1]) - (G - G[:, :1]), axis=2).mean(1)
            pa = np.array([np.linalg.norm(_procrustes(P[i], G[i]) - G[i], axis=1).mean()
                           for i in range(len(P))])
            wrist = np.linalg.norm(P[:, 0] - G[:, 0], axis=1)
            zc = np.clip(P[..., 2], 1e-6, None)
            zg = np.clip(G[..., 2], 1e-6, None)
            du = K[0, 0] * (P[..., 0] / zc - G[..., 0] / zg)
            dv = K[1, 1] * (P[..., 1] / zc - G[..., 1] / zg)
            # Medians, not means. A handful of frames -- a hand leaving the
            # view, a failed solve -- carry errors two orders of magnitude
            # above the rest, and a mean over those describes the outliers
            # rather than the pose: on this corpus the mean wrist error is
            # 207 mm where the median is 28 mm. Means are kept alongside so
            # the tail stays visible instead of being quietly dropped.
            e2 = np.hypot(du, dv)
            oos = ~onscreen
            iv = onscreen
            rec["hands"][side] = dict(
                frames=int(ok.sum()),
                onscreen_frac=float(iv.mean()),
                mpjpe_oos_mm=(float(np.median(mpjpe[oos]) * 1000)
                              if oos.sum() >= 5 else None),
                wrist_oos_mm=(float(np.median(wrist[oos]) * 1000)
                              if oos.sum() >= 5 else None),
                mpjpe_mm=float(np.median(mpjpe[iv]) * 1000),
                mpjpe_mean_mm=float(np.mean(mpjpe[iv]) * 1000),
                pa_mpjpe_mm=float(np.median(pa[iv]) * 1000),
                wrist_mm=float(np.median(wrist[iv]) * 1000),
                wrist_p90_mm=float(np.percentile(wrist[iv], 90) * 1000),
                wrist_mean_mm=float(np.mean(wrist[iv]) * 1000),
                epe2d_px=float(np.median(e2[iv])),
                epe2d_mean_px=float(np.mean(e2[iv])),
                frac_wrist_over_100mm=float((wrist[iv] > 0.10).mean()),
                depth_bias_mm=float(np.median(P[iv, 0, 2] - G[iv, 0, 2]) * 1000))
        if rec["hands"]:
            rows.append(rec)

    if not rows:
        print("no sequences with both a prediction and ground truth")
        return {}
    print("=" * 92)
    print(f"ARCTIC s05 (held out): predicted vs MOCAP GROUND TRUTH, "
          f"{len(rows)} sequences")
    print("medians over ON-SCREEN frames (the paper's gate); 'vis' = share of "
          "frames on-screen, 'tail' = on-screen frames over 100 mm wrist error")
    print("%-26s %-5s %7s %9s %7s %7s %7s %7s" % (
        "sequence", "hand", "MPJPE", "PA-MPJPE", "wrist", "EPE2D", "z bias",
        "vis"))
    print("-" * 92)
    agg = {}
    for r in rows:
        for side, h in r["hands"].items():
            print("%-26s %-5s %7.1f %8.1f %7.1f %7.1f %+7.1f %6.0f%%" % (
                r["seq"][:26], side[0].upper(), h["mpjpe_mm"], h["pa_mpjpe_mm"],
                h["wrist_mm"], h["epe2d_px"], h["depth_bias_mm"],
                100 * h["onscreen_frac"]))
            for k in ("mpjpe_mm", "pa_mpjpe_mm", "wrist_mm", "epe2d_px",
                      "depth_bias_mm", "onscreen_frac"):
                agg.setdefault(k, []).append(h[k])
    print("-" * 92)
    summary = {k: float(np.median(v)) for k, v in agg.items()}
    print("median   MPJPE %.1f mm   PA-MPJPE %.1f mm   wrist %.1f mm   "
          "EPE2D %.1f px   z bias %+.1f mm   on-screen %.0f%%" % (
              summary["mpjpe_mm"], summary["pa_mpjpe_mm"], summary["wrist_mm"],
              summary["epe2d_px"], summary["depth_bias_mm"],
              100 * summary["onscreen_frac"]))
    out = str(config.artifact("arctic", "accuracy.json"))
    json.dump(dict(sequences=rows, median=summary), open(out, "w"), indent=1)
    print("wrote", out)
    return summary


def run(seqs=None, encode_w: int = 832, chunk: int = 241, overlap: int = 81):
    """
    Estimate, then lift, using the same batch runner the main path uses.

    Extrinsics are identity here, so `to_world` is a no-op and the arrays stay
    in the camera frame -- which is where ARCTIC's ground truth lives, so no
    estimated world pose enters the comparison.
    """
    from .ace import convert
    from .run import ACE_CKPT, ACE_ENV, ACE_HOME, ACE_OPT

    clips = Path(config.ARTIFACTS / "arctic" / "clips")
    raw = Path(config.ARTIFACTS / "arctic" / "raw")
    world = Path(config.ARTIFACTS / "arctic" / "world")
    for d in (raw, world):
        d.mkdir(parents=True, exist_ok=True)
    seqs = seqs or sorted(p.stem for p in clips.glob("*.mp4"))
    todo = [dict(name=s, video=str((clips / f"{s}.mp4").resolve()),
                 camera=str((clips / f"{s}.cam.json").resolve()),
                 out=str((raw / s).resolve())) for s in seqs]
    manifest = str(config.artifact("arctic", "manifest.json"))
    report = str(config.artifact("arctic", "report.json"))
    json.dump(todo, open(manifest, "w"), indent=1)
    entry = str(Path(__file__).with_name("_ace_batch.py"))
    cmd = ["conda", "run", "--no-capture-output", "-n", ACE_ENV, "python",
           entry, "--manifest", manifest, "--home", str(ACE_HOME),
           "--opt", str(ACE_HOME / ACE_OPT), "--ckpt", str(ACE_HOME / ACE_CKPT),
           "--encode-w", str(encode_w), "--chunk", str(chunk),
           "--overlap", str(overlap), "--mode", "tiled", "--report", report]
    r = subprocess.run(cmd, cwd=str(ACE_HOME))
    if r.returncode != 0:
        raise RuntimeError(f"batch runner exited {r.returncode}")
    done = []
    for row in json.load(open(report)):
        if not row.get("ok"):
            print(f"SKIP {row['episode']}: {row.get('error')}")
            continue
        s = row["episode"]
        convert(row["pkl"], str(clips / f"{s}.frames.npz"), axis=1.0,
                out=str(world / f"{s}.npz"), placement="pnp",
                orient_offset=False)     # ARCTIC GT is MANO's own frame
        done.append(s)
        print(f"  {s:<28} -> {world / (s + '.npz')}")
    return done
