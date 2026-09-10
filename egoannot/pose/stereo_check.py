"""
The referee: stereo depth, which neither pose source owns.

Agreement between two pose streams cannot say which is right. This corpus
does ship one independent measurement -- a calibrated 64 mm stereo pair -- and
`evaluation/stereo.py` already turns it into a test: a hand occludes whatever
is behind it, so the depth measured at the projected hand pixel must equal the
depth that hand's own pose predicts.

Run on the shipped pose that test failed (correlation ~ 0, MAE 75-251 mm),
which is why proximity-from-pose was abandoned. Running the SAME test on an
RGB-derived pose is the honest way to settle the 91 mm placement disagreement:
the estimator is anchored to pixels, so if the shipped pose is the one adrift,
this is where it shows.

Both sources go through one call into the same harness, sampling the same
frames at the same rectification, so the only thing that differs between the
two rows is which hand got projected.
"""
from __future__ import annotations

import json

import numpy as np

from .. import config
from ..evaluation import stereo
from . import _ace_path


def _hands_from_npz(path) -> dict:
    """World-frame joints in the (t, 21x3) shape `stereo.validate` expects."""
    z = np.load(path)
    out = {}
    for side in ("left", "right"):
        J, A = z[f"{side}_hand_joints"], z[f"{side}_hand"]
        out[side] = [(float(A[i, 0]), J[i].tolist()) for i in range(len(A))
                     if np.isfinite(J[i]).all()]
    return out


def compare(paths, every_s=2.0, scale=0.5, out=None) -> list[dict]:
    """Score shipped and estimated pose against stereo depth, side by side."""
    results = []
    for path in paths:
        path = str(path)
        name = path.split("/")[-1].rsplit(".", 1)[0]
        pred = _ace_path(name)
        arms = [("shipped", None)]
        try:
            arms.append(("ace", _hands_from_npz(pred)))
        except FileNotFoundError:
            print(f"{name}: no estimated pose, scoring shipped only")
        for label, hands in arms:
            try:
                r = stereo.validate(path, every_s, scale, hands=hands,
                                    label=label)
                if r and r.get("n"):
                    # Keep the per-sample rows: a summary cannot say whether a
                    # gap is a fixed offset or irreducible spread, and that is
                    # the difference between a correctable bias and a bad pose.
                    results.append(r)
            except Exception as e:
                print(f"FAIL {name}[{label}]: {type(e).__name__}: {e}")

    if results:
        print("=" * 88)
        print("STEREO REFEREE: predicted hand depth vs measured stereo depth "
              "at the same pixel")
        print("%-34s %-8s %6s %10s %8s %7s %9s" % (
            "episode", "source", "n", "median_err", "MAE", "corr", "<5cm"))
        for r in results:
            print("%-34s %-8s %6d %+10.3f %8.3f %+7.3f %8.0f%%" % (
                r["name"].split("[")[0][:34], r["label"], r["n"],
                r["median_err"], r["mae"], r["corr"], 100 * r["within_5cm"]))
        for label in ("shipped", "ace"):
            sel = [r for r in results if r["label"] == label]
            if sel:
                print(f"  {label:<8s} median MAE {np.median([r['mae'] for r in sel]):.3f} m"
                      f"   median corr {np.median([r['corr'] for r in sel]):+.3f}"
                      f"   median <5cm {100*np.median([r['within_5cm'] for r in sel]):.0f}%")
        bias_report(results)
        out = str(out or config.artifact("pose", "stereo_referee.json"))
        json.dump([{k: v for k, v in r.items() if k != "rows"} for r in results],
                  open(out, "w"), indent=1)
        print("wrote", out)
    return results


def bias_report(results: list[dict]) -> dict:
    """
    Is the gap a fixed offset, or irreducible spread?

    A systematic depth bias is a one-scalar fix; random error is not. This
    fits ONE global offset across every episode -- not one per episode, which
    would be fitting the answer -- and reports what removing it would buy.
    The offset is fitted on the same data it is scored on, so the corrected
    figures are an upper bound on what a real calibration could achieve, not
    a held-out result.
    """
    out = {}
    print("-" * 88)
    print("BIAS: one global depth offset, fitted and scored on the same samples "
          "(an upper bound)")
    for label in ("shipped", "ace"):
        rows = [x for r in results if r["label"] == label
                for x in r.get("rows", []) if x.get("z_stereo")]
        if not rows:
            continue
        zp = np.array([x["z_pose"] for x in rows])
        zs = np.array([x["z_stereo"] for x in rows])
        d = zs - zp
        off = float(np.median(d))
        raw_mae, cor_mae = float(np.abs(d).mean()), float(np.abs(d - off).mean())
        raw5, cor5 = float((np.abs(d) <= .05).mean()), float((np.abs(d - off) <= .05).mean())
        out[label] = dict(n=len(rows), offset_m=off, mae_m=raw_mae,
                          mae_corrected_m=cor_mae, within_5cm=raw5,
                          within_5cm_corrected=cor5)
        print(f"  {label:<8s} n={len(rows):4d}  offset {off:+.3f} m   "
              f"MAE {raw_mae:.3f} -> {cor_mae:.3f} m   "
              f"<5cm {100*raw5:.0f}% -> {100*cor5:.0f}%")
    return out
