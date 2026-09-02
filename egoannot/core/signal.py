"""
Shared 1-D signal helpers.

`smooth` is the one worth reading. The phase scripts used
`np.convolve(x, ones(n)/n, mode="same")`, which divides the first and last n/2
samples by the full window even though fewer samples contribute -- a constant
signal of 2.0 smoothed with n=5 comes back as [1.2, 1.6, 2.0, ...]. That is a
20-40% depression at both ends of every episode. Normalising by the actual
window coverage removes it at no cost.
"""
from __future__ import annotations

import numpy as np


def smooth(x: np.ndarray, n: int) -> np.ndarray:
    """Centred box filter with edge-correct normalisation. Handles 1-D and 2-D."""
    x = np.asarray(x, float)
    if n < 2 or len(x) < n:
        return x.astype(float, copy=True)
    k = np.ones(n)
    if x.ndim == 1:
        den = np.convolve(np.ones(len(x)), k, mode="same")
        return np.convolve(x, k, mode="same") / den
    den = np.convolve(np.ones(len(x)), k, mode="same")
    return np.stack([np.convolve(x[:, i], k, mode="same") / den
                     for i in range(x.shape[1])], axis=1)


def window(smooth_s: float, fps: float) -> int:
    """Smoothing window in samples, at least 2."""
    return max(2, int(round(smooth_s * fps)))


def speed(t: np.ndarray, P: np.ndarray, win: int = 0):
    """Speed along a 3-D path, on the midpoint timebase t[1:]."""
    if len(t) < 2:
        return np.zeros(0), np.zeros(0)
    dt = np.diff(t)
    ok = dt > 1e-9
    v = np.zeros(len(dt))
    v[ok] = np.linalg.norm(np.diff(P, axis=0)[ok], axis=1) / dt[ok]
    return t[1:], (smooth(v, win) if win else v)


def gradient(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """d x / d t, tolerant of short or degenerate inputs."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.zeros_like(x)
    return np.gradient(x, t)


def angular_speed(t: np.ndarray, Q: np.ndarray, win: int = 0):
    """Rotation rate from consecutive quaternions, on the timebase t[1:]."""
    if len(t) < 3:
        return np.zeros(0), np.zeros(0)
    dt = np.diff(t)
    dot = np.abs(np.sum(Q[1:] * Q[:-1], axis=1)).clip(0, 1)
    ang = 2 * np.arccos(dot)
    w = np.zeros(len(dt))
    ok = dt > 1e-9
    w[ok] = ang[ok] / dt[ok]
    return t[1:], (smooth(w, win) if win else w)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of each True run."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    d = np.diff(m.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0])
    if m[0]:
        starts.insert(0, 0)
    if m[-1]:
        ends.append(len(m) - 1)
    return list(zip(starts, ends))


def local_minima(v: np.ndarray, prominence: float, min_gap_samples: int = 0,
                 t: np.ndarray | None = None) -> np.ndarray:
    """
    Indices of local minima whose smaller flanking rise is >= prominence.

    Flanks are measured by walking outwards to the first sample below the
    minimum, so a shallow dip inside a deep valley does not qualify.
    """
    v = np.asarray(v, float)
    out = []
    for i in range(1, len(v) - 1):
        if not (v[i] <= v[i - 1] and v[i] <= v[i + 1]):
            continue
        j = i
        while j > 0 and v[j] >= v[i]:
            j -= 1
        left = v[max(0, j):i + 1].max()
        k = i
        while k < len(v) - 1 and v[k] >= v[i]:
            k += 1
        right = v[i:min(len(v), k + 1)].max()
        if min(left, right) - v[i] >= prominence:
            out.append(i)
    if not out:
        return np.zeros(0, int)
    if min_gap_samples <= 0 and t is None:
        return np.array(out, int)
    keep = [out[0]]
    for i in out[1:]:
        gap = (t[i] - t[keep[-1]]) if t is not None else (i - keep[-1])
        if gap >= min_gap_samples:
            keep.append(i)
    return np.array(keep, int)


def pct_rank(x: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """Map to [0, 1] by rank within `ref` (default: x itself). NaNs preserved."""
    x = np.asarray(x, float)
    r = x if ref is None else np.asarray(ref, float)
    good = np.isfinite(r)
    if good.sum() < 8:
        return np.full(len(x), np.nan)
    xs = np.sort(r[good])
    out = np.full(len(x), np.nan)
    m = np.isfinite(x)
    out[m] = np.searchsorted(xs, x[m], side="right") / len(xs)
    return out
