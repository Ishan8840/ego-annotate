"""
Per-frame confidence from cross-source agreement, and the validity gate.

The estimator claims a hand on every frame. On ARCTIC it claims 100% while
ground truth says 89% are actually on screen, so roughly one frame in nine
carries a confident hallucination with nothing marking it. Downstream that is
worse than a gap: `features` propagates validity but cannot detect a
fabrication, so a hallucinated hand enters the aperture signal as a real
measurement and a span boundary gets cut from it.

Nothing in a single estimator's output separates those frames -- its own
existence head is what claims 100%. What does separate them is a SECOND,
independent read of the same image. Where two models that share no weights
and no training data agree on where a hand is, both are probably right; where
they disagree wildly, at least one is wrong and the frame is not worth
trusting.

So confidence here is the 2D distance between this pipeline's reprojected
joints and an independent detector's, expressed as a fraction of the image
diagonal so it means the same thing at any resolution. Frames past a
threshold are written NaN, which is the one thing the downstream stages
already handle correctly.

This needs no ground truth and no calibration, which is the point: it is the
only quality control available on footage that has neither.
"""
from __future__ import annotations

import json

import numpy as np

# Chosen from the measured distribution, not guessed: across three corpora the
# estimator's agreement with an independent detector sits at 1.5-2.0% of the
# image diagonal on frames where the hand is genuinely present, so a frame
# past 5% is disagreeing by more than twice the worst honest case.
DEFAULT_THRESHOLD = 0.05


def agreement(proj: np.ndarray, detected: np.ndarray, diag: float) -> np.ndarray:
    """
    Per-frame 2D distance between two reads of the same hand, normalised.

    `proj` and `detected` are (N, 21, 2) in pixels. Returns (N,) as a fraction
    of the image diagonal; NaN where either read is missing.
    """
    d = np.linalg.norm(proj - detected, axis=-1)          # (N, 21)
    with np.errstate(invalid="ignore"):
        return np.nanmedian(d, axis=1) / float(diag)


def gate(arrays: dict, conf: dict, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Blank the hands whose confidence is worse than `threshold`.

    Writes NaN rather than dropping the frame, so the timebase stays intact
    and the features stage sees an honest gap instead of a fabrication.
    Returns the arrays plus a per-side report of what was removed.
    """
    out = dict(arrays)
    report = {}
    for side in ("left", "right"):
        c = conf.get(side)
        jk, wk, qk = (f"/pose/{side}_hand_joints", f"/pose/{side}_hand",
                      f"/pose/{side}_hand_quat")
        if c is None or jk not in out:
            continue
        n = min(len(c), len(out[jk]))
        # NaN means the referee never spoke -- not evidence, so not gated.
        # inf is set where it spoke and saw nothing here.
        bad = np.isfinite(c[:n]) & (c[:n] > threshold)
        bad |= np.isinf(c[:n])
        J = np.array(out[jk], float); J[:n][bad] = np.nan
        W = np.array(out[wk], float); W[:n][bad, 1:] = np.nan
        out[jk], out[wk] = J, W
        if qk in out:
            Q = np.array(out[qk], float); Q[:n][bad, 1:] = np.nan
            out[qk] = Q
        report[side] = dict(frames=int(n), gated=int(bad.sum()),
                            gated_frac=round(float(bad.mean()), 4),
                            median_conf=(None if not np.isfinite(c[:n]).any()
                                         else round(float(np.nanmedian(c[:n])), 4)))
    out["_gate"] = dict(threshold=threshold, **report)
    return out


def load_detections(path, episode: str) -> dict[int, list]:
    """Detections from the referee's dump: {frame index: [ {label, pts}, ... ]}."""
    blob = json.load(open(path))
    rec = blob.get(episode, {})
    return {int(k): v for k, v in rec.items()}


def confidence_from_detections(proj_by_side: dict, dets: dict[int, list],
                               diag: float, n_frames: int) -> tuple[dict, dict]:
    """
    Match each detected hand to the nearer projected hand, and score it.

    Matching is by wrist proximity rather than by the detector's own
    handedness label, which is unreliable in egocentric views -- a mislabelled
    hand would otherwise be scored against the wrong side and gated as a
    failure when the pose was fine.

    Two kinds of silence are separated, because conflating them costs far more
    than it saves. If the referee found NOTHING in a frame it may simply have
    failed, and that is no evidence against the pose -- measured on ARCTIC,
    treating it as evidence discarded 157 good frame-hands to catch 16 bad
    ones. If the referee found the OTHER hand but not this one, it was working
    and saw no hand here, which is evidence. Only the second case is gated;
    the first is returned as unknown.
    """
    conf = {s: np.full(n_frames, np.nan) for s in proj_by_side}
    checked = {s: np.zeros(n_frames, bool) for s in proj_by_side}
    for f, hands in dets.items():
        if f >= n_frames:
            continue
        for h in hands:
            M = np.asarray(h["pts"], float)
            best, score = None, np.inf
            for side, P in proj_by_side.items():
                if f >= len(P) or not np.isfinite(P[f]).all():
                    continue
                d = float(np.linalg.norm(P[f][0] - M[0]))
                if d < score:
                    best, score = side, d
            if best is None:
                continue
            v = float(np.nanmedian(
                np.linalg.norm(proj_by_side[best][f] - M, axis=-1))) / diag
            if not np.isfinite(conf[best][f]) or v < conf[best][f]:
                conf[best][f] = v
            checked[best][f] = True
    # a side the referee did not match, in a frame where it DID find a hand,
    # is a side the referee looked at and did not see
    for f, hands in dets.items():
        if f >= n_frames or not hands:
            continue
        for side in proj_by_side:
            if not checked[side][f]:
                conf[side][f] = np.inf
    return conf, checked
