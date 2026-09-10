"""
Placement arms, measured against each other on identical predictions.

The estimator's joints and its 2D anchors are good; the translation it hangs
them on is not. There are three ways to deal with that, and they are only
comparable if they run on the SAME predictions -- an earlier pass compared a
raw arm at one encode width against a corrected arm at another and the
resolution difference rode along inside the result.

    raw    the model's camera-space translation, untouched
    scale  one depth multiplier per episode, fitted against the model's own
           2D head (`ace.fit_depth_scale`)
    pnp    per-frame 6-DoF fit of the root-relative joints onto those same
           anchors (`ace.solve_placement`)

Each arm is scored two ways: agreement with the shipped annotation, and the
stereo referee that neither source owns. Agreement alone cannot say which is
right, and stereo alone is too noisy on this footage to resolve small
differences, so both are reported and neither is called the answer.
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np

from .. import config
from ..core.mcap_io import read_episode
from . import predictions_dir
from .ace import convert, measured_axis

MODES = ("raw", "scale", "pnp")


def _convert_all(paths, mode: str) -> list[str]:
    clips = str(config.ARTIFACTS / "pose" / "clips")
    out_dir = predictions_dir()
    os.makedirs(out_dir, exist_ok=True)
    names = []
    for path in paths:
        name = os.path.basename(str(path)).rsplit(".", 1)[0]
        pkl = os.path.join(str(config.ARTIFACTS / "pose" / "raw"), name,
                           name + ".pkl")
        frames = os.path.join(clips, name + ".frames.npz")
        if not (os.path.exists(pkl) and os.path.exists(frames)):
            continue
        ep = read_episode(str(path), want_video=False)
        convert(pkl, frames, ep=ep, axis=measured_axis(ep),
                out=os.path.join(out_dir, name + ".npz"),
                depth_scale=1.0 if mode == "raw" else None,
                placement="pnp" if mode == "pnp" else "raw")
        names.append(name)
        del ep
    return names


def run(paths, modes=MODES, with_stereo: bool = True, every: float = 2.0,
        scale: float = 0.5) -> dict:
    from . import evaluate, stereo_check

    results = {}
    for mode in modes:
        print(f"\n{'=' * 78}\n### placement = {mode}\n{'=' * 78}", flush=True)
        _convert_all(paths, mode)
        rows = evaluate.report(paths, "ace",
                               out=config.artifact("pose", f"agreement_{mode}.json"))
        hands = [h for r in rows for h in r["hands"].values()
                 if h.get("placement_mm")]
        med = lambda k: float(np.median([h[k] for h in hands
                                         if h.get(k) is not None
                                         and np.isfinite(h[k])]))
        rec = dict(placement_mm=med("placement_mm"), shape_mm=med("shape_mm"),
                   aperture_mae_mm=med("aperture_mae_mm"),
                   aperture_corr=med("aperture_corr"))
        wa = [h["wrist_axis_mm"] for h in hands if h.get("wrist_axis_mm")]
        if wa:
            for ax in ("x_right", "y_down", "z_depth"):
                rec[ax] = float(np.median([w[ax]["median_abs"] for w in wa]))
        if with_stereo:
            st = stereo_check.compare(
                paths, every, scale,
                out=config.artifact("pose", f"stereo_{mode}.json"))
            for label in ("ace", "shipped"):
                sel = [r for r in st if r["label"] == label]
                if sel:
                    rec[f"{label}_mae"] = float(np.median([r["mae"] for r in sel]))
                    rec[f"{label}_w5"] = float(np.median([r["within_5cm"] for r in sel]))
        results[mode] = rec
        shutil.copytree(predictions_dir(),
                        str(config.ARTIFACTS / "pose" / f"world_{mode}"),
                        dirs_exist_ok=True)

    print(f"\n{'=' * 86}")
    print("PLACEMENT ARMS on identical predictions "
          f"({len(paths)} episodes)")
    print("%-7s | %9s %8s %8s %6s | %6s %6s %6s | %8s %6s" % (
        "mode", "place mm", "shape", "ap MAE", "ap r",
        "|dx|", "|dy|", "|dz|", "stereo", "<5cm"))
    print("-" * 86)
    for mode, r in results.items():
        print("%-7s | %9.1f %8.1f %8.1f %6.2f | %6.1f %6.1f %6.1f | %8.3f %5.0f%%" % (
            mode, r["placement_mm"], r["shape_mm"], r["aperture_mae_mm"],
            r["aperture_corr"], r.get("x_right", float("nan")),
            r.get("y_down", float("nan")), r.get("z_depth", float("nan")),
            r.get("ace_mae", float("nan")), 100 * r.get("ace_w5", float("nan"))))
    if with_stereo and "ace_mae" in next(iter(results.values())):
        s = next(iter(results.values()))
        print("%-7s | %9s %8s %8s %6s | %6s %6s %6s | %8.3f %5.0f%%" % (
            "shipped", "-", "-", "-", "-", "-", "-", "-",
            s["shipped_mae"], 100 * s["shipped_w5"]))
    out = str(config.artifact("pose", "placement_arms.json"))
    json.dump(results, open(out, "w"), indent=1)
    print("wrote", out)
    return results
