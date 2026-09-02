"""Rotations, projection, and the camera-convention test."""
from __future__ import annotations

import math

import numpy as np


def quat_to_R(w, x, y, z) -> np.ndarray:
    """Single quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quats_to_R(Q: np.ndarray) -> np.ndarray:
    """(N, 4) quaternions in (w, x, y, z) -> (N, 3, 3). Batched form of quat_to_R."""
    Q = np.asarray(Q, float)
    n = np.linalg.norm(Q, axis=1, keepdims=True)
    Q = np.divide(Q, n, out=np.tile(np.array([1.0, 0, 0, 0]), (len(Q), 1)), where=n > 0)
    w, x, y, z = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
    R = np.empty((len(Q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def project(points_world: np.ndarray, R: np.ndarray, t: np.ndarray, K: dict,
            axis: float = 1.0):
    """
    World points -> pixel coords in a camera at (R, t), where R is camera-to-world.

    world->camera is R.T @ (p - t). Returns (u, v, z, in_front), with u/v set to
    NaN where the point is behind the camera.
    """
    P = np.atleast_2d(np.asarray(points_world, float))
    pc = (P - t) @ R                      # == (R.T @ v) per row
    z = pc[:, 2] * axis
    ok = z > 1e-6
    u = np.full(len(P), np.nan)
    v = np.full(len(P), np.nan)
    u[ok] = K["fx"] * pc[ok, 0] / z[ok] + K["cx"]
    v[ok] = K["fy"] * pc[ok, 1] / z[ok] + K["cy"]
    return u, v, z, ok


def in_rect(u, v, K: dict, margin_frac: float = 0.0) -> np.ndarray:
    """Which projected points land inside the sensor rectangle, optionally inset."""
    mx, my = K["w"] * margin_frac, K["h"] * margin_frac
    with np.errstate(invalid="ignore"):
        return ((u >= mx) & (u < K["w"] - mx) & (v >= my) & (v < K["h"] - my)
                & np.isfinite(u) & np.isfinite(v))


def camera_convention(extr: np.ndarray, wrist: np.ndarray) -> dict | None:
    """
    Decide the optical-axis convention from geometry instead of assuming it.

    The head camera points where the person looks, and during manipulation the
    wrists are predominantly in front of it. For each candidate forward axis we
    measure the median angle between camera forward and the direction to the
    wrist: the correct convention puts the wrists in front (well under 90 deg).
    Wrist distance is convention-independent and validates that the extrinsic
    translation shares a frame with the pose stream.
    """
    if len(extr) == 0 or len(wrist) == 0:
        return None
    idx = np.clip(np.searchsorted(extr[:, 0], wrist[:, 0]), 0, len(extr) - 1)
    R = quats_to_R(extr[idx, 4:8])
    d = wrist[:, 1:4] - extr[idx, 1:4]
    nd = np.linalg.norm(d, axis=1)
    keep = nd > 1e-6
    if not keep.any():
        return None
    d_hat = d[keep] / nd[keep, None]
    cand = {}
    for name, sign in (("+z", 1.0), ("-z", -1.0)):
        f = R[keep][:, :, 2] * sign            # R @ [0, 0, +-1]
        cos = np.clip(np.sum(f * d_hat, axis=1), -1, 1)
        cand[name] = float(np.median(np.degrees(np.arccos(cos))))
    dists = nd[keep]
    return dict(axis=1.0 if cand["+z"] <= cand["-z"] else -1.0,
                median_angle_pz=round(cand["+z"], 1),
                median_angle_mz=round(cand["-z"], 1),
                wrist_dist_p50=round(float(np.median(dists)), 3),
                wrist_dist_p95=round(float(np.percentile(dists, 95)), 3))
