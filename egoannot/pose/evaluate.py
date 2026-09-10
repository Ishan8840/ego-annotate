"""
Is the estimated pose as accurate as the shipped one?

Three readings, deliberately kept apart, because they answer different
questions and this corpus does not let one stand in for another.

AGREEMENT asks how close the two streams are to each other, in the units the
pipeline actually consumes. It is reported three ways, because a single MPJPE
would hide which half is wrong:

  placement  world-frame MPJPE with both hands where they claim to be. Wrong
             if either stream mislocates the hand in space.
  shape      MPJPE after both are canonicalised into the hand-local frame
             `features.canonical` already defines, so orientation and position
             divide out and only the finger configuration is left.
  aperture   thumb-tip to index-tip in millimetres -- the one scalar the
             events and spans stages read. Correlation and MAE.

COVERAGE asks how often there is an answer at all. A pose that is absent for
a third of the footage is not "as accurate" whatever its error on the frames
it does produce, and the two streams disagree about this by construction: the
shipped pose is dense by fiat, the estimator declares absence.

Neither of those is a truth test. Agreement with the shipped pose is only
meaningful where the shipped pose is right, and this repository has already
measured two places where it is not (head-coupling in `docs/quality.md`,
stereo disagreement in `evaluation/stereo.py`). So high agreement here should
be read as "reproduces the incumbent", and the stereo check -- run separately,
on both streams -- is the reading that neither source owns.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .. import config
from ..core.geometry import in_rect, project, quats_to_R
from ..core.mcap_io import read_episode
from ..stages.features import MCP, TIP, canonical
from . import _ace_path
from .ace import measured_axis

WRIST = 0


def _resample(t_src: np.ndarray, X: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """
    Nearest-in-time resample of a (N, ...) array onto a new timebase.

    Nearest, not linear: interpolating between two joint configurations
    invents a hand that was never measured, and at 25-30 fps the nearest
    sample is at most 20 ms away.
    """
    if not len(t_src):
        return np.full((len(t_dst),) + X.shape[1:], np.nan)
    idx = np.clip(np.searchsorted(t_src, t_dst), 0, len(t_src) - 1)
    prev = np.clip(idx - 1, 0, len(t_src) - 1)
    take_prev = np.abs(t_src[prev] - t_dst) < np.abs(t_src[idx] - t_dst)
    return X[np.where(take_prev, prev, idx)]


def aperture(J: np.ndarray) -> np.ndarray:
    """Thumb-tip to index-tip distance, the scalar the pipeline reads."""
    return np.linalg.norm(J[:, TIP["thumb"]] - J[:, TIP["index"]], axis=1)


def _shape_mpjpe(A: np.ndarray, B: np.ndarray):
    """
    Per-frame shape disagreement after canonicalising both hands.

    `features.canonical` divides through by hand scale, so its output is
    DIMENSIONLESS -- distances there are in units of the wrist-to-middle-MCP
    length, not metres. Returning that number scaled by 1000 and calling it
    "mm" overstates the error by roughly an order of magnitude, since a hand
    scale on this corpus is around 90 mm.

    So two values come back: the dimensionless figure, and the same error
    carried into millimetres against the reference hand's own measured scale.
    """
    ca, _ = canonical(A)
    cb, scale_b = canonical(B)
    frac = np.linalg.norm(ca - cb, axis=2).mean(axis=1)
    return frac, frac * scale_b * 1000.0


def _pck2d(u_a, v_a, u_b, v_b, K, thresh_frac=0.05) -> dict:
    """
    2D agreement as PCK, with the threshold tied to the image, not the hand.

    A bbox-relative threshold would move with whichever stream defines the
    box; a fixed fraction of the image diagonal is the same yardstick for
    both.
    """
    diag = float(np.hypot(K["w"], K["h"]))
    d = np.hypot(u_a - u_b, v_a - v_b)
    ok = np.isfinite(d)
    if not ok.any():
        return dict(n=0)
    d = d[ok]
    return dict(n=int(ok.sum()), px_p50=float(np.median(d)),
                px_p95=float(np.percentile(d, 95)),
                pck=float((d <= thresh_frac * diag).mean()),
                thresh_px=round(thresh_frac * diag, 1))


def _to_camera(P, R_cw, t_cw, axis):
    """World points -> standard camera coords (x right, y down, z forward)."""
    pc = np.einsum("nij,nkj->nki", np.transpose(R_cw, (0, 2, 1)), P - t_cw[:, None, :])
    pc[..., 2] *= axis
    return pc


def _mean_rotation(R):
    """
    The rotation closest to a set of rotations: SVD projection onto SO(3).

    Used to ask whether two orientation streams differ by a CONSTANT offset --
    two canonical hand frames -- rather than actually disagreeing. A constant
    offset is a convention mismatch and carries no information about accuracy.
    """
    U, _, Vt = np.linalg.svd(R.mean(axis=0))
    M = U @ Vt
    if np.linalg.det(M) < 0:
        U[:, -1] *= -1
        M = U @ Vt
    return M


def _geodesic(A, B):
    """Angle in degrees between two stacks of rotation matrices."""
    rel = np.einsum("nij,nkj->nik", A, B)          # A @ B^T
    tr = np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


def episode(path, source="ace", verbose=True) -> dict | None:
    """Agreement and coverage for one episode."""
    ep = read_episode(path, want_video=False)
    name = ep["name"]
    pred_path = _ace_path(name)
    if not os.path.exists(pred_path):
        if verbose:
            print(f"{name}: no {source} pose at {pred_path}")
        return None
    z = np.load(pred_path)
    K, extr = ep["K"], ep["extr"]
    axis = measured_axis(ep)
    row = dict(episode=name, axis=axis, src_fps=ep["src_fps"],
               duration_s=round(ep["duration_s"], 2), hands={})

    for side in ("left", "right"):
        Jm = z[f"{side}_hand_joints"]                 # mine, world frame
        tm = z[f"{side}_hand"][:, 0]
        Js = ep[f"/pose/{side}_hand_joints"]          # shipped, world frame
        ts = ep[f"/pose/{side}_hand"][:, 0]
        if not len(Js) or not len(Jm):
            continue
        Jr = _resample(ts, Js, tm)                    # shipped on my timebase

        have = np.isfinite(Jm).all(axis=(1, 2)) & np.isfinite(Jr).all(axis=(1, 2))
        h = dict(frames=int(len(tm)), coverage=float(have.mean()))
        if have.sum() >= 8:
            A, B = Jm[have], Jr[have]
            h["placement_mm"] = float(
                np.linalg.norm(A - B, axis=2).mean() * 1000)
            frac, mm = _shape_mpjpe(A, B)
            h["shape_frac_of_hand"] = float(frac.mean())
            h["shape_mm"] = float(mm.mean())
            h["hand_scale_mm"] = float(np.median(canonical(B)[1]) * 1000)
            ap_a, ap_b = aperture(A) * 1000, aperture(B) * 1000
            h["aperture_mae_mm"] = float(np.abs(ap_a - ap_b).mean())
            h["aperture_corr"] = (float(np.corrcoef(ap_a, ap_b)[0, 1])
                                  if ap_a.std() > 1e-9 and ap_b.std() > 1e-9
                                  else float("nan"))
            h["aperture_mine_p50_mm"] = float(np.median(ap_a))
            h["aperture_shipped_p50_mm"] = float(np.median(ap_b))
            # per-axis wrist error, in the camera's own frame: lateral and
            # vertical error mean something different from depth error, and
            # a single MPJPE hides which one is moving.
            if len(extr):
                ci = np.clip(np.searchsorted(extr[:, 0], tm[have]), 0,
                             len(extr) - 1)
                Rc, tc = quats_to_R(extr[ci, 4:8]), extr[ci, 1:4]
                ca = _to_camera(A[:, :1], Rc, tc, axis)[:, 0]
                cb = _to_camera(B[:, :1], Rc, tc, axis)[:, 0]
                d = ca - cb
                h["wrist_axis_mm"] = {
                    k: dict(median_signed=float(np.median(d[:, j]) * 1000),
                            median_abs=float(np.median(np.abs(d[:, j])) * 1000))
                    for j, k in enumerate(("x_right", "y_down", "z_depth"))}
                h["wrist_dist_mm"] = float(
                    np.median(np.linalg.norm(d, axis=1)) * 1000)
                h["depth_mine_mm"] = float(np.median(ca[:, 2]) * 1000)
                h["depth_shipped_mm"] = float(np.median(cb[:, 2]) * 1000)

            # orientation, raw and after removing a constant frame offset
            qa, qb = z[f"{side}_hand_quat"], ep.get(f"/pose/{side}_hand_quat")
            if qb is not None and len(qb):
                qb_r = _resample(qb[:, 0], qb[:, 1:5], tm)[have]
                qa_r = qa[have][:, 1:5]
                good = np.isfinite(qa_r).all(1) & np.isfinite(qb_r).all(1)
                if good.sum() >= 8:
                    Ra, Rb = quats_to_R(qa_r[good]), quats_to_R(qb_r[good])
                    raw = _geodesic(Ra, Rb)
                    off = _mean_rotation(np.einsum("nij,njk->nik",
                                                   np.transpose(Ra, (0, 2, 1)), Rb))
                    res = _geodesic(np.einsum("nij,jk->nik", Ra, off), Rb)
                    h["orientation_deg"] = dict(
                        raw_median=float(np.median(raw)),
                        after_constant_offset_median=float(np.median(res)),
                        after_constant_offset_p90=float(np.percentile(res, 90)),
                        offset_deg=float(_geodesic(off[None], np.eye(3)[None])[0]),
                        n=int(good.sum()))

            # 2D: both projected through the same shipped camera
            if len(extr):
                idx = np.clip(np.searchsorted(extr[:, 0], tm[have]), 0, len(extr) - 1)
                R = quats_to_R(extr[idx, 4:8])
                t = extr[idx, 1:4]
                ua, va, ub, vb = [], [], [], []
                for j in range(21):
                    for P, uu, vv in ((A[:, j], ua, va), (B[:, j], ub, vb)):
                        u, v, _, _ = _project_batch(P, R, t, K, axis)
                        uu.append(u); vv.append(v)
                h["pck2d"] = _pck2d(np.concatenate(ua), np.concatenate(va),
                                    np.concatenate(ub), np.concatenate(vb), K)
                inr = in_rect(np.concatenate(ua), np.concatenate(va), K)
                h["in_frame_mine"] = float(inr.mean())
                h["in_frame_shipped"] = float(in_rect(
                    np.concatenate(ub), np.concatenate(vb), K).mean())
        row["hands"][side] = h
    del ep
    return row


def _project_batch(P, R, t, K, axis):
    """Per-frame projection with a per-frame camera pose."""
    pc = np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), P - t)
    z = pc[:, 2] * axis
    ok = z > 1e-6
    u = np.full(len(P), np.nan)
    v = np.full(len(P), np.nan)
    u[ok] = K["fx"] * pc[ok, 0] / z[ok] + K["cx"]
    v[ok] = K["fy"] * pc[ok, 1] / z[ok] + K["cy"]
    return u, v, z, ok


def report(paths, source="ace", out=None) -> list[dict]:
    rows = [r for r in (episode(p, source) for p in paths) if r]
    if not rows:
        print("nothing to report - no predictions found")
        return rows
    print("=" * 96)
    print(f"POSE AGREEMENT: {source} vs shipped, {len(rows)} episodes")
    print("placement = world MPJPE (mm); shape = canonicalised MPJPE, carried "
          "back to mm by the reference hand's own scale; aperture = thumb-index tip")
    print("-" * 96)
    print("%-30s %-5s %6s %9s %8s %8s %9s %7s %7s" % (
        "episode", "hand", "cover", "placemm", "shapemm", "shape/h", "ap_mae",
        "ap_r", "pck2d"))
    agg = {}
    for r in rows:
        for side, h in r["hands"].items():
            print("%-30s %-5s %5.0f%% %9s %8s %8s %9s %7s %7s" % (
                r["episode"][:30], side[0].upper(), 100 * h["coverage"],
                _f(h.get("placement_mm")), _f(h.get("shape_mm")),
                _f(h.get("shape_frac_of_hand"), 3),
                _f(h.get("aperture_mae_mm")), _f(h.get("aperture_corr"), 2),
                _f(100 * h["pck2d"]["pck"], 0) if h.get("pck2d", {}).get("n") else "-"))
            for k in ("coverage", "placement_mm", "shape_mm",
                      "shape_frac_of_hand", "aperture_mae_mm", "aperture_corr",
                      "hand_scale_mm"):
                if h.get(k) is not None and np.isfinite(h.get(k, np.nan)):
                    agg.setdefault(k, []).append(h[k])
    print("-" * 96)
    print("  median over hands: " + "  ".join(
        f"{k}={np.median(v):.3g}" for k, v in agg.items()))
    out = str(out or config.artifact("pose", "agreement.json"))
    json.dump(rows, open(out, "w"), indent=1)
    print("wrote", out)
    return rows


def _f(x, nd=1):
    return "-" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"
