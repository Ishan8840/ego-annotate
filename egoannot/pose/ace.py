"""
ACE-Ego-Hand predictions -> the pose arrays every downstream stage reads.

The estimator predicts in the CAMERA frame; the pipeline's pose lives in the
WORLD frame. This module does that lift, and nothing else clever: no
smoothing, no gap filling, no re-association. A frame the model marks absent
stays absent, because the features stage propagates validity and never imputes
it, and a silently interpolated hand would enter the aperture signal as a real
measurement.

Three conventions have to line up, and only one of them is free:

* Joint order is already right. ACE emits 21 joints in OpenPose order
  (``mano_utils._MANO_TO_OP``): wrist, then thumb, index, middle, ring, pinky,
  four each. That is the MediaPipe layout ``stages/features.py`` assumes, so
  ``TIP[thumb]=4`` and ``TIP[index]=8`` address the same two points in both
  streams and the aperture is directly comparable. No permutation.
* Hand identity is fixed by construction. ACE writes slot 0 = left and slot
  1 = right with no detector in the loop, so the handedness flips that plague
  egocentric hand detection cannot occur here.
* The optical-axis convention is NOT free, and is not assumed. ACE predicts in
  a standard pinhole frame (+z forward). This corpus's own convention is
  measured per episode by ``geometry.camera_convention``, which decides the
  sign from where the wrists actually sit relative to the camera. The two are
  reconciled with that measured sign, and ``selfcheck`` verifies the result by
  reprojecting through the shipped K and comparing against the model's own 2D
  anchors -- a test that needs no ground truth and fails loudly if either
  convention is wrong.
"""
from __future__ import annotations

import os
import pickle

import numpy as np

from ..core.geometry import camera_convention, quats_to_R

SIDES = ("left", "right")          # ACE slot order: 0 = left, 1 = right
N_JOINTS = 21

# Canonical-frame offsets between the estimator's hand orientation and this
# corpus's. Raw geodesic error between the two streams is 100-167 deg, which
# is a frame convention and not disagreement: ONE constant per hand side,
# fitted once over all 8 held-out episodes and ~5000 frames each, brings it to
# ~15 deg -- within 2 deg of what 16 per-episode offsets achieve. The two
# sides need different axes, which is what MANO's mirrored left/right
# canonical frames predict.
#
# Unlike everything else in this module these ARE calibrated against the
# shipped annotation, because a convention has no ground truth to recover it
# from -- they express "the frame EgoStandard uses", not "the correct frame",
# and would have to be refitted for another corpus. Only the quaternion is
# affected; joints and aperture never pass through them.
ORIENT_OFFSET = {
    "left": np.array([[0.959509, -0.273754, -0.066334],
                      [0.027908, -0.141944, 0.989481],
                      [-0.28029, -0.951268, -0.128556]]),
    "right": np.array([[-0.951738, -0.306467, 0.01649],
                       [-0.006266, 0.073121, 0.997303],
                       [-0.306846, 0.949069, -0.071513]]),
}


def load_pkl(path) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def camera_joints(pred: dict, slot: int) -> np.ndarray:
    """
    (N, 21, 3) camera-frame joints for one hand slot, from the direct read.

    ``joints_cam_direct`` is ALREADY absolute: upstream composes it as
    ``direct_rootrel + direct_wrist_cam`` (memory_projector.py:818) and dumps
    the sum, while ``wrist_cam_direct`` is dumped alongside as the same wrist
    on its own. Adding the wrist here again would place every hand at twice
    its true depth -- roughly 0.9 m out on this corpus -- while still looking
    like a plausible hand, which is exactly the kind of error no downstream
    metric would flag.

    This path bypasses MANO and the mixed-PnP translation decode entirely, so
    it needs no licensed body model to reconstruct.
    """
    J = np.asarray(pred["joints_cam_direct"], float)
    return J[:, slot] if J.ndim == 4 else J


def rotmats(pred: dict, slot: int) -> np.ndarray:
    """(N, 3, 3) camera-frame global orientation for one slot."""
    go = np.asarray(pred["go"], float)          # (N, 2, 1, 3, 3)
    return go[:, slot, 0] if go.ndim == 5 else go[:, slot]


def presence(pred: dict, slot: int) -> np.ndarray:
    """(N,) bool: did the model claim a hand in this slot on this frame."""
    e3 = np.asarray(pred["exists_3d"], float)
    e = e3[:, slot] if e3.ndim == 2 else e3
    return e > 0.5


def _R_to_quat(R: np.ndarray) -> np.ndarray:
    """(N,3,3) -> (N,4) quaternions as (w, x, y, z), Shepperd's method."""
    N = len(R)
    q = np.zeros((N, 4))
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    big = tr > 0
    if big.any():
        s = np.sqrt(tr[big] + 1.0) * 2
        q[big, 0] = 0.25 * s
        q[big, 1] = (R[big, 2, 1] - R[big, 1, 2]) / s
        q[big, 2] = (R[big, 0, 2] - R[big, 2, 0]) / s
        q[big, 3] = (R[big, 1, 0] - R[big, 0, 1]) / s
    for i in np.flatnonzero(~big):
        d = np.array([R[i, 0, 0], R[i, 1, 1], R[i, 2, 2]])
        k = int(np.argmax(d))
        a, b = (k + 1) % 3, (k + 2) % 3
        s = np.sqrt(1.0 + R[i, k, k] - R[i, a, a] - R[i, b, b]) * 2
        q[i, 0] = (R[i, b, a] - R[i, a, b]) / s
        q[i, 1 + k] = 0.25 * s
        q[i, 1 + a] = (R[i, a, k] + R[i, k, a]) / s
        q[i, 1 + b] = (R[i, b, k] + R[i, k, b]) / s
    n = np.linalg.norm(q, axis=1, keepdims=True)
    return np.divide(q, n, out=np.tile([1.0, 0, 0, 0], (N, 1)), where=n > 0)


def _extr_at(extr: np.ndarray, t: np.ndarray):
    """Camera-to-world rotation and translation sampled at each time in t."""
    idx = np.clip(np.searchsorted(extr[:, 0], t), 0, len(extr) - 1)
    return quats_to_R(extr[idx, 4:8]), extr[idx, 1:4]


def fit_depth_scale(pred: dict, slots=(0, 1), lo=0.55, hi=1.45, n=181) -> dict:
    """
    Calibrate the depth head against the model's OWN 2D head.

    The estimator emits two independent reads of the same hand: 3D joints in
    the camera frame, and a soft-argmax 2D anchor per joint in the image. If
    the 3D read is placed at the right depth those agree under the camera the
    prediction was made with. Measured here they do not: the reprojected 3D
    hand comes out about 19% too small, which is what an over-estimated depth
    looks like, and matches the sign and size of the offset the stereo pair
    reports independently.

    Scaling depth alone -- z' = m*z, x and y untouched -- restores both the
    apparent size and the lateral shift, because the 3D hand is too far along
    the optical axis rather than too far along its own viewing ray. Scaling
    all three coordinates would be a no-op: fx*(x/m)/(z/m) is fx*x/z.

    This needs no ground truth. It compares the model against itself, so it
    calibrates on footage where nothing is annotated -- which is the case this
    whole stage exists for.
    """
    intr = pred.get("intrinsics") or {}
    if not intr:
        return dict(scale=1.0, note="no intrinsics in dump")
    iw, ih = float(intr["image_width"]), float(intr["image_height"])
    grid = np.linspace(lo, hi, n)
    tot = np.zeros(len(grid))
    kept = 0
    for slot in slots:
        J = camera_joints(pred, slot)
        p2 = np.asarray(pred["joints_2d"], float)
        p2 = (p2[:, slot] if p2.ndim == 4 else p2) * np.array([iw, ih])
        ok = presence(pred, slot) & np.isfinite(J).all(axis=(1, 2))
        if ok.sum() < 8:
            continue
        J, p2 = J[ok], p2[ok]
        kept += int(ok.sum())
        for i, m in enumerate(grid):
            z = np.clip(J[..., 2] * m, 1e-6, None)
            u = intr["fx"] * J[..., 0] / z + intr["cx"]
            v = intr["fy"] * J[..., 1] / z + intr["cy"]
            tot[i] += np.median(np.linalg.norm(
                np.stack([u, v], -1) - p2, axis=-1))
    if not kept:
        return dict(scale=1.0, note="no usable frames")
    j = int(np.argmin(tot))
    return dict(scale=float(grid[j]), frames=kept,
                px_before=float(tot[np.argmin(np.abs(grid - 1.0))] / len(slots)),
                px_after=float(tot[j] / len(slots)),
                at_bound=bool(j in (0, len(grid) - 1)))


def solve_placement(pred: dict, slot: int, min_joints: int = 8,
                    max_reproj_px: float = 60.0,
                    hand_scale: float = 1.0) -> tuple[np.ndarray, dict]:
    """
    Re-place each hand by fitting its own shape to its own 2D anchors.

    The estimator is good at two of the three things it outputs and bad at the
    third. Its root-relative joints agree with the shipped pose to 12-14 mm,
    and its 2D anchors are where the shipped pose reprojects to within 20 px --
    but the camera-space translation it hangs them on is systematically wrong,
    placing hands about 20% too far.

    So keep what works and recompute what does not: treat the root-relative
    joints as a rigid body and solve the pose that reprojects them onto the
    anchors, per frame, through the intrinsics the prediction was made with.
    This is the same idea as the paper's mixed-PnP decode, applied to the
    direct heads rather than the MANO ones, and it replaces a single scalar per
    episode with a full 6-DoF fit per frame.

    It uses only the model's own outputs, so it needs no annotation and no
    shipped pose -- the property that matters for footage where neither exists.

    Frames whose fit does not converge, or whose reprojection residual exceeds
    `max_reproj_px`, keep the original prediction rather than a bad solve; the
    returned stats say how often that happened.

    `hand_scale` rescales the rigid body before the fit. PnP places an object
    by matching its known metric size to its apparent size, so an error in
    predicted hand size becomes an error in depth of the same fraction, and
    inflates every metric quantity read off the hand -- aperture included.
    Measured on this corpus the estimator's hand is 14% larger than the
    shipped one (94-99 mm against 81-87 mm wrist-to-middle-MCP), which is a
    real disagreement about physical size and not something monocular vision
    can settle on its own.
    """
    import cv2

    intr = pred.get("intrinsics") or {}
    J = camera_joints(pred, slot)
    if not intr:
        return J, dict(solved=0, note="no intrinsics in dump")
    iw, ih = float(intr["image_width"]), float(intr["image_height"])
    Kmat = np.array([[intr["fx"], 0, intr["cx"]],
                     [0, intr["fy"], intr["cy"]], [0, 0, 1.0]])
    p2 = np.asarray(pred["joints_2d"], float)
    p2 = (p2[:, slot] if p2.ndim == 4 else p2) * np.array([iw, ih])
    ok = presence(pred, slot)

    out = J.copy()
    solved = 0
    resid = []
    for i in range(len(J)):
        if not ok[i] or not np.isfinite(J[i]).all() or not np.isfinite(p2[i]).all():
            continue
        rel = (J[i] - J[i, 0]).astype(np.float64) * hand_scale
        img = p2[i].astype(np.float64)
        inside = ((img[:, 0] > -iw) & (img[:, 0] < 2 * iw)
                  & (img[:, 1] > -ih) & (img[:, 1] < 2 * ih))
        if inside.sum() < min_joints:
            continue
        try:
            good, rvec, tvec = cv2.solvePnP(
                rel[inside], img[inside], Kmat, None,
                flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:
            continue
        if not good or tvec[2, 0] <= 1e-3:
            continue
        rvec, tvec = cv2.solvePnPRefineLM(rel[inside], img[inside], Kmat, None,
                                          rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        cand = (R @ rel.T).T + tvec.reshape(1, 3)
        z = np.clip(cand[:, 2], 1e-6, None)
        u = intr["fx"] * cand[:, 0] / z + intr["cx"]
        v = intr["fy"] * cand[:, 1] / z + intr["cy"]
        err = float(np.median(np.hypot(u - img[:, 0], v - img[:, 1])))
        if not np.isfinite(err) or err > max_reproj_px:
            continue
        out[i] = cand
        resid.append(err)
        solved += 1
    n_ok = int(ok.sum())
    return out, dict(solved=solved, candidates=n_ok,
                     rate=round(solved / n_ok, 3) if n_ok else 0.0,
                     reproj_px=round(float(np.median(resid)), 1) if resid else None)


def to_world(pred: dict, frames: dict, axis: float,
             depth_scale: float = 1.0, placement: str = "raw",
             stats: dict | None = None, orient_offset: bool = True,
             hand_scale: float = 1.0) -> dict:
    """
    Camera-frame predictions -> the six arrays a pose provider must supply.

    `axis` is the measured optical-axis sign for this corpus. ACE works in a
    +z-forward frame; where the episode's convention is -z forward the third
    camera axis is negated before the extrinsic rotation is applied, so both
    streams end up describing the same physical direction.
    """
    vts = np.asarray(frames["vts"], float)
    extr = np.asarray(frames["extr"], float)
    n = len(np.asarray(pred["cam_trans"]))
    if n > len(vts):
        raise ValueError(f"prediction has {n} frames but only {len(vts)} video "
                         "timestamps -- the mp4 and the mcap disagree")
    t = vts[:n]
    R_cw, t_cw = _extr_at(extr, t)
    flip = np.diag([1.0, 1.0, float(axis)])

    out = {}
    for slot, side in enumerate(SIDES):
        if placement == "pnp":
            Jfull, st = solve_placement(pred, slot, hand_scale=hand_scale)
            Jc = Jfull[:n].copy()
            if stats is not None:
                stats[side] = st
        else:
            Jc = camera_joints(pred, slot)[:n].copy()
            if depth_scale != 1.0:
                Jc[..., 2] *= depth_scale    # depth only; see fit_depth_scale
        Jc = Jc @ flip                                      # (n, 21, 3)
        ok = presence(pred, slot)[:n]
        # camera -> world: p_w = R_cw @ p_c + t_cw
        Jw = np.einsum("nij,nkj->nki", R_cw, Jc) + t_cw[:, None, :]
        Rh = flip @ rotmats(pred, slot)[:n]
        if orient_offset:
            # into this corpus's hand frame, so `spans` can read a twist off it
            Rh = np.einsum("nij,jk->nik", Rh, ORIENT_OFFSET[side])
        Rw = R_cw @ Rh
        Jw[~ok] = np.nan                                    # absent, not imputed

        wrist = np.column_stack([t, Jw[:, 0, :]])
        quat = np.column_stack([t, _R_to_quat(Rw)])
        quat[~ok, 1:] = np.nan
        out[f"/pose/{side}_hand"] = wrist
        out[f"/pose/{side}_hand_joints"] = Jw
        out[f"/pose/{side}_hand_quat"] = quat
        out[f"{side}_present"] = ok
    return out


def selfcheck(pred: dict, slot: int = 1) -> dict:
    """
    Reproject the camera-frame joints and compare with the model's own 2D read.

    This needs no ground truth: it asks only whether the 3D head and the 2D
    head of the SAME prediction agree under the SAME camera. Large error means
    a convention mismatch -- a flipped axis, or intrinsics at the wrong
    resolution -- not a bad hand.

    Two details the schema forces. The camera is the one recorded IN the pkl,
    already rescaled to the resolution the VAE encoded at, not the sensor's
    native K. And ``joints_2d`` is a sigmoid output, so it is normalised to
    [0, 1] and has to be taken back to pixels before anything is compared.
    """
    intr = pred.get("intrinsics") or {}
    if not intr:
        return dict(n=0, note="no intrinsics in dump")
    iw, ih = float(intr["image_width"]), float(intr["image_height"])
    J = camera_joints(pred, slot)
    p2 = np.asarray(pred["joints_2d"], float)
    p2 = p2[:, slot] if p2.ndim == 4 else p2
    p2 = p2 * np.array([iw, ih])                    # normalised -> pixels
    ok = presence(pred, slot) & np.isfinite(J).all(axis=(1, 2))
    if not ok.any():
        return dict(n=0)
    J, p2 = J[ok], p2[ok]
    z = np.clip(J[..., 2], 1e-6, None)
    u = intr["fx"] * J[..., 0] / z + intr["cx"]
    v = intr["fy"] * J[..., 1] / z + intr["cy"]
    d = np.linalg.norm(np.stack([u, v], -1) - p2, axis=-1)
    return dict(n=int(ok.sum()), px_p50=float(np.median(d)),
                px_p95=float(np.percentile(d, 95)),
                frac_of_width=round(float(np.median(d)) / iw, 4))


def measured_axis(ep: dict) -> float:
    """The optical-axis sign for this episode, measured rather than assumed."""
    conv = camera_convention(ep["extr"], ep["/pose/right_hand"])
    return 1.0 if conv is None else float(conv["axis"])


def convert(pkl_path, frames_path, ep=None, axis=None, out=None,
            depth_scale=None, placement="pnp", orient_offset=True,
            hand_scale=1.0) -> dict:
    """
    One episode: prediction pkl + frame sidecar -> world-frame pose npz.

    `depth_scale` None fits it from the model's own 2D head; pass 1.0 to keep
    the raw prediction.
    """
    pred = load_pkl(pkl_path)
    frames = dict(np.load(frames_path))
    if axis is None:
        axis = measured_axis(ep) if ep is not None else 1.0
    cal, st = None, {}
    if placement != "pnp" and depth_scale is None:
        cal = fit_depth_scale(pred)
        depth_scale = cal["scale"]
    arrays = to_world(pred, frames, axis, depth_scale or 1.0, placement, st,
                      orient_offset=orient_offset, hand_scale=hand_scale)
    arrays["_depth_cal"] = cal
    arrays["_placement"] = dict(mode=placement, **st)
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        np.savez_compressed(out, axis=axis,
                            depth_scale=float(depth_scale or 1.0), **{
            k.replace("/pose/", "").replace("/", "_"): v
            for k, v in arrays.items() if not k.startswith("_")})
    return arrays
