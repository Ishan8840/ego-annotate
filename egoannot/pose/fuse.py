"""
Place articulation from one source using image evidence from another.

The two things a hand pose needs -- what the fingers are doing, and where the
hand is -- are not best measured the same way, and on this corpus they are not
even measured by the same instrument well:

  articulation   EMG measures it through occlusion and out of frame, because
                 it reads the muscles rather than the picture. A vision model
                 loses it exactly when the hand is hidden, which is most of
                 what makes manipulation footage hard.
  placement      EMG cannot see it at all -- it has no idea where the arm is
                 in space. Only the image knows.

So fuse them at the point where they are each strong: take root-relative
joints from whichever source is better at shape, and solve the rigid pose that
puts them where the image says the hand is. That is the same PnP solve the
ACE path already uses to fix its own translation; nothing about it is specific
to where the joints came from, which is why it is factored out here.

`place` needs the articulation in metres, 2D anchors in pixels, and
intrinsics. It returns the joints in the camera frame and a residual, and
frames it cannot solve are returned untouched rather than guessed at.
"""
from __future__ import annotations

import numpy as np

MIN_JOINTS = 8
MAX_REPROJ_PX = 60.0


def place(joints_rel: np.ndarray, anchors_2d: np.ndarray, intr: dict,
          min_joints: int = MIN_JOINTS,
          max_reproj_px: float = MAX_REPROJ_PX) -> tuple[np.ndarray, dict]:
    """
    Rigidly place root-relative joints onto 2D anchors, per frame.

    joints_rel : (N, J, 3) metres, root-relative (joint 0 at the origin)
    anchors_2d : (N, J, 2) pixels in the same image the intrinsics describe
    intr       : fx, fy, cx, cy, image_width, image_height

    Returns (N, J, 3) camera-frame joints and a report. A frame whose solve
    fails, or whose reprojection residual exceeds `max_reproj_px`, keeps its
    input unplaced -- a bad solve is worse than an honest gap, because
    downstream cannot tell one from the other once it is written.
    """
    import cv2

    K = np.array([[intr["fx"], 0, intr["cx"]],
                  [0, intr["fy"], intr["cy"]], [0, 0, 1.0]])
    iw, ih = float(intr["image_width"]), float(intr["image_height"])
    out = np.array(joints_rel, float).copy()
    solved, resid = 0, []

    for i in range(len(joints_rel)):
        rel = np.asarray(joints_rel[i], np.float64)
        img = np.asarray(anchors_2d[i], np.float64)
        if not (np.isfinite(rel).all() and np.isfinite(img).all()):
            continue
        rel = rel - rel[0]                       # enforce root-relative
        inside = ((img[:, 0] > -iw) & (img[:, 0] < 2 * iw)
                  & (img[:, 1] > -ih) & (img[:, 1] < 2 * ih))
        if inside.sum() < min_joints:
            continue
        try:
            ok, rvec, tvec = cv2.solvePnP(rel[inside], img[inside], K, None,
                                          flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:
            continue
        if not ok or tvec[2, 0] <= 1e-3:
            continue
        rvec, tvec = cv2.solvePnPRefineLM(rel[inside], img[inside], K, None,
                                          rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        cand = (R @ rel.T).T + tvec.reshape(1, 3)
        z = np.clip(cand[:, 2], 1e-6, None)
        u = intr["fx"] * cand[:, 0] / z + intr["cx"]
        v = intr["fy"] * cand[:, 1] / z + intr["cy"]
        e = float(np.median(np.hypot(u - img[:, 0], v - img[:, 1])))
        if not np.isfinite(e) or e > max_reproj_px:
            continue
        out[i] = cand
        resid.append(e)
        solved += 1

    n = len(joints_rel)
    return out, dict(solved=solved, frames=n,
                     rate=round(solved / n, 3) if n else 0.0,
                     reproj_px=round(float(np.median(resid)), 2) if resid else None)


def articulation_from_angles(angles: np.ndarray, hand_model) -> np.ndarray:
    """
    Joint angles -> root-relative 3D joints, via whatever hand model produced
    them.

    EMG models emit angles, not positions: EgoEMG's is 20 per hand, HOT3D's
    UmeTrack 22. Those are only meaningful through the skeleton they were
    fitted with, so the model is passed in rather than assumed -- feeding
    angles from one convention through another's kinematics produces a
    plausible-looking hand that is wrong in a way no downstream metric
    catches.
    """
    if hand_model is None:
        raise ValueError(
            "joint angles need the hand model they were fitted with; pass the "
            "forward-kinematics callable from that model's own toolkit")
    J = np.asarray(hand_model(np.asarray(angles, float)), float)
    if J.ndim != 3 or J.shape[-1] != 3:
        raise ValueError(f"hand model returned {J.shape}, expected (N, J, 3)")
    return J - J[:, :1]
