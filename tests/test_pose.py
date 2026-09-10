"""The RGB pose path: conversion maths, conventions, and the provider swap.

None of these need the estimator, its weights or the corpus. They cover the
part that is easy to get silently wrong -- a frame that is off by one, an
axis that points the wrong way, an absent hand quietly filled in.
"""
import json
import os

import numpy as np
import pytest

from egoannot.core.geometry import quats_to_R
from egoannot.pose import POSE_KEYS, attach
from egoannot.pose.ace import _R_to_quat, to_world
from egoannot.pose.evaluate import _resample, aperture
from egoannot.pose.export import camera_json
from egoannot.stages.features import MCP, TIP

K = dict(fx=900.0, fy=900.0, cx=960.0, cy=728.0, w=1920, h=1456)


def _pred(n=6, present=True):
    """A minimal prediction in the schema infer_video.py dumps."""
    rng = np.random.default_rng(0)
    rel = rng.normal(0, 0.03, (n, 2, 21, 3))
    rel[:, :, 0] = 0.0                              # joint 0 is the root
    wrist = np.tile(np.array([0.05, -0.02, 0.45]), (n, 2, 1))
    # upstream dumps the SUM (memory_projector.py:818), not the root-relative part
    absolute = rel + wrist[:, :, None, :]
    e = np.full((n, 2), 1.0 if present else 0.0)
    return {
        "joints_cam_direct": absolute, "wrist_cam_direct": wrist,
        "go": np.tile(np.eye(3), (n, 2, 1, 1, 1)),
        "cam_trans": np.zeros((n, 2, 3)), "exists_3d": e, "exists_2d": e,
        "joints_2d": np.zeros((n, 2, 21, 2)),
    }


def _frames(n=6):
    """Identity extrinsics: the camera frame IS the world frame."""
    extr = np.zeros((n, 8))
    extr[:, 0] = np.arange(n) / 30.0
    extr[:, 4] = 1.0                                 # unit quaternion (w=1)
    return {"vts": np.arange(n) / 30.0, "extr": extr}


def test_quat_roundtrips_through_the_rotation_matrix():
    rng = np.random.default_rng(1)
    Q = rng.normal(size=(50, 4))
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    Q[Q[:, 0] < 0] *= -1                             # fix the sign convention
    back = _R_to_quat(quats_to_R(Q))
    back[back[:, 0] < 0] *= -1
    assert np.allclose(back, Q, atol=1e-6)


def test_quat_handles_the_degenerate_trace_branch():
    """A 180 degree rotation drives the trace negative -- Shepperd's other case."""
    R = np.diag([1.0, -1.0, -1.0])[None]             # pi about x
    q = _R_to_quat(R)
    assert np.allclose(np.abs(q[0]), [0, 1, 0, 0], atol=1e-6)


def test_identity_extrinsics_leave_the_camera_frame_alone():
    out = to_world(_pred(), _frames(), axis=1.0)
    J = out["/pose/right_hand_joints"]
    assert np.allclose(J[:, 0, 2], 0.45)             # wrist depth survives


def test_negative_axis_flips_the_optical_axis_and_nothing_else():
    p, f = _pred(), _frames()
    pos = to_world(p, f, axis=1.0)["/pose/right_hand_joints"]
    neg = to_world(p, f, axis=-1.0)["/pose/right_hand_joints"]
    assert np.allclose(pos[..., :2], neg[..., :2])
    assert np.allclose(pos[..., 2], -neg[..., 2])


def test_absent_hands_are_nan_and_never_imputed():
    """features.py propagates validity; a filled-in hand would enter as real."""
    out = to_world(_pred(present=False), _frames(), axis=1.0)
    assert np.isnan(out["/pose/left_hand_joints"]).all()
    assert np.isnan(out["/pose/left_hand_quat"][:, 1:]).all()
    assert not out["left_present"].any()


def test_wrist_array_carries_joint_zero_on_the_video_timebase():
    out = to_world(_pred(), _frames(), axis=1.0)
    A, J = out["/pose/right_hand"], out["/pose/right_hand_joints"]
    assert np.allclose(A[:, 0], np.arange(6) / 30.0)
    assert np.allclose(A[:, 1:4], J[:, 0, :])


def test_more_predicted_frames_than_timestamps_is_an_error():
    """A silent truncation here would slide every label off in time."""
    with pytest.raises(ValueError, match="video timestamps"):
        to_world(_pred(n=10), _frames(n=4), axis=1.0)


def test_all_six_pose_keys_are_produced():
    out = to_world(_pred(), _frames(), axis=1.0)
    assert set(POSE_KEYS) <= set(out)


def test_camera_json_matches_the_shape_the_estimator_reads():
    """infer_video.py does cam["frames"][0]["intrinsics"]["fx"] -- exactly this."""
    cam = json.loads(json.dumps(camera_json(K)))
    k = cam["frames"][0]["intrinsics"]
    assert (k["fx"], k["cx"]) == (900.0, 960.0)
    assert (cam["image_width"], cam["image_height"]) == (1920, 1456)


def test_resample_takes_the_nearest_sample_not_an_interpolation():
    t_src = np.array([0.0, 1.0, 2.0])
    X = np.array([10.0, 20.0, 30.0])
    out = _resample(t_src, X, np.array([0.4, 0.6, 1.9]))
    assert np.allclose(out, [10.0, 20.0, 30.0])


def test_aperture_reads_the_thumb_and_index_tips():
    J = np.zeros((1, 21, 3))
    J[0, TIP["index"], 0] = 0.08
    assert np.isclose(aperture(J)[0], 0.08)


def test_joint_layout_is_the_one_both_streams_share():
    """ACE emits OpenPose order, which is the MediaPipe layout features.py uses."""
    assert (MCP["thumb"], MCP["index"], MCP["pinky"]) == (2, 5, 17)
    assert (TIP["thumb"], TIP["index"], TIP["pinky"]) == (4, 8, 20)


def test_attach_is_a_noop_for_the_shipped_source():
    ep = {"name": "x", "/pose/left_hand": "sentinel"}
    assert attach(ep, "shipped")["/pose/left_hand"] == "sentinel"


def test_attach_rejects_an_unknown_source():
    with pytest.raises(ValueError, match="unknown pose source"):
        attach({"name": "x"}, "telepathy")


def test_attach_names_the_command_that_would_fix_a_missing_prediction():
    with pytest.raises(FileNotFoundError, match="pose run"):
        attach({"name": "no_such_episode"}, "ace")


@pytest.mark.skipif(not os.path.isdir(os.path.expanduser("~/ACE-Ego-Hand")),
                    reason="estimator checkout not present")
def test_upstream_joint_order_is_still_openpose():
    """
    Guards the assumption that needs no remap: if upstream ever reorders its
    21 joints, aperture would silently start measuring two other fingers.
    """
    src = open(os.path.expanduser(
        "~/ACE-Ego-Hand/ace_ego_hand/mano_utils.py")).read()
    assert "_MANO_TO_OP = [0, 13, 14, 15, 16, 1, 2, 3, 17," in src


def test_direct_joints_are_taken_as_absolute_not_re_offset():
    """
    Upstream composes joints_cam_direct = rootrel + wrist and dumps the sum.
    Adding the wrist again would put every hand at twice its true depth while
    still looking like a plausible hand -- silent, and fatal to placement.
    """
    from egoannot.pose.ace import camera_joints
    p = _pred()
    J = camera_joints(p, slot=1)
    assert np.allclose(J[:, 0, 2], 0.45)            # not 0.90
    assert np.allclose(J, np.asarray(p["joints_cam_direct"])[:, 1])


# -- chunked inference planning -------------------------------------------

def _planner():
    from egoannot.pose import _ace_batch as B
    return B


def test_trim_yields_a_length_the_causal_vae_accepts():
    B = _planner()
    for n in range(5, 200):
        t = B.trim(n)
        assert t <= n and (t - 1) % 4 == 0


def test_encode_size_snaps_both_axes_to_the_latent_grid():
    B = _planner()
    w, h = B.encode_size(1920, 1456, 1280)
    assert (w % 32, h % 32) == (0, 0)
    assert w == 1280 and abs(h / w - 1456 / 1920) < 0.02


def test_short_clips_are_a_single_chunk():
    B = _planner()
    assert B.plan_chunks(81, 241, 81) == [(0, 81)]


def test_chunks_overlap_by_at_least_one_training_window():
    B = _planner()
    spans = B.plan_chunks(681, 241, 81)
    assert len(spans) > 1
    for (s0, l0), (s1, _) in zip(spans, spans[1:]):
        assert s0 + l0 - s1 >= B.WINDOW, (s0, l0, s1)


def test_keep_windows_tile_the_clip_exactly_once():
    """Every frame decoded once: no gap (missing pose) and no double-write."""
    B = _planner()
    for n in (81, 200, 333, 681, 797):
        spans = B.plan_chunks(n, 241, 81)
        keeps = B.keep_window(spans, B.trim(n))
        covered = []
        for lo, hi in keeps:
            covered.extend(range(lo, hi))
        assert covered == sorted(covered), n
        assert len(covered) == len(set(covered)), f"overlap at n={n}"
        assert covered == list(range(B.trim(n))), f"gap at n={n}"


def test_every_kept_frame_sits_inside_its_own_chunk():
    B = _planner()
    spans = B.plan_chunks(681, 241, 81)
    for (s, ln), (lo, hi) in zip(spans, B.keep_window(spans, B.trim(681))):
        assert s <= lo <= hi <= s + ln


# -- depth calibration ----------------------------------------------------

def _pred_with_depth_error(m=1.25, n=12):
    """A prediction whose 3D head sits m times too far, 2D head correct."""
    from egoannot.pose.ace import camera_joints
    rng = np.random.default_rng(3)
    intr = {"fx": 400.0, "fy": 400.0, "cx": 208.0, "cy": 160.0,
            "image_width": 416, "image_height": 320}
    true = rng.normal(0, 0.03, (n, 2, 21, 3)) + np.array([0.02, 0.0, 0.40])
    # the 2D head sees the TRUE hand
    u = intr["fx"] * true[..., 0] / true[..., 2] + intr["cx"]
    v = intr["fy"] * true[..., 1] / true[..., 2] + intr["cy"]
    j2 = np.stack([u / intr["image_width"], v / intr["image_height"]], -1)
    bad = true.copy()
    bad[..., 2] *= m                       # 3D head over-estimates depth
    return {"joints_cam_direct": bad, "wrist_cam_direct": bad[:, :, 0],
            "joints_2d": j2, "exists_3d": np.ones((n, 2)),
            "exists_2d": np.ones((n, 2)), "intrinsics": intr,
            "go": np.tile(np.eye(3), (n, 2, 1, 1, 1)),
            "cam_trans": np.zeros((n, 2, 3))}


def test_depth_calibration_recovers_a_known_scale_error():
    from egoannot.pose.ace import fit_depth_scale
    cal = fit_depth_scale(_pred_with_depth_error(m=1.25))
    assert abs(cal["scale"] - 1 / 1.25) < 0.02, cal
    assert cal["px_after"] < cal["px_before"] / 3
    assert not cal["at_bound"]


def test_depth_calibration_leaves_a_correct_prediction_alone():
    from egoannot.pose.ace import fit_depth_scale
    cal = fit_depth_scale(_pred_with_depth_error(m=1.0))
    assert abs(cal["scale"] - 1.0) < 0.02, cal


def test_depth_scale_touches_depth_only():
    """Scaling all three axes is a no-op for projection: fx(x/m)/(z/m)=fx x/z."""
    from egoannot.pose.ace import to_world
    p, f = _pred(), _frames()
    a = to_world(p, f, axis=1.0, depth_scale=1.0)["/pose/right_hand_joints"]
    b = to_world(p, f, axis=1.0, depth_scale=0.8)["/pose/right_hand_joints"]
    assert np.allclose(a[..., :2], b[..., :2])          # x, y untouched
    assert not np.allclose(a[..., 2], b[..., 2])        # depth moved


# -- PnP placement and the orientation convention -------------------------

def test_pnp_recovers_a_displaced_hand():
    """
    The whole premise: shape and 2D anchors are good, translation is not.
    Displace the hand along the optical axis and the solve should put it back.
    """
    from egoannot.pose.ace import solve_placement
    p = _pred_with_depth_error(m=1.30, n=6)
    true_z = np.asarray(p["joints_cam_direct"])[:, :, :, 2] / 1.30
    out, st = solve_placement(p, slot=1)
    assert st["rate"] == 1.0, st
    assert st["reproj_px"] < 3.0, st
    # 30% too far, recovered to within 2%: the displacement is removed, and
    # the residual is what PnP can resolve on a hand only a few cm across.
    before = abs(np.median(np.asarray(p["joints_cam_direct"])[:, 1, :, 2])
                 / np.median(true_z[:, 1]) - 1)
    after = abs(np.median(out[:, :, 2]) / np.median(true_z[:, 1]) - 1)
    assert before > 0.25 and after < 0.02, (before, after)


def test_pnp_keeps_the_hand_shape_it_was_given():
    """Placement is re-solved; the finger configuration must survive intact."""
    from egoannot.pose.ace import solve_placement, camera_joints
    p = _pred_with_depth_error(m=1.30, n=6)
    before = camera_joints(p, 1)
    after, _ = solve_placement(p, slot=1)
    d0 = np.linalg.norm(before - before[:, :1], axis=-1)
    d1 = np.linalg.norm(after - after[:, :1], axis=-1)
    assert np.allclose(d0, d1, atol=1e-3)      # bone lengths unchanged


def test_a_bad_frame_keeps_its_original_prediction():
    from egoannot.pose.ace import solve_placement, camera_joints
    p = _pred_with_depth_error(m=1.0, n=6)
    p["joints_2d"] = np.full_like(np.asarray(p["joints_2d"]), np.nan)
    out, st = solve_placement(p, slot=1)
    assert st["solved"] == 0
    assert np.allclose(out, camera_joints(p, 1), equal_nan=True)


def test_orientation_offsets_are_rotations():
    """A non-orthonormal offset would shear the hand frame silently."""
    from egoannot.pose.ace import ORIENT_OFFSET
    for side, R in ORIENT_OFFSET.items():
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-4), side
        assert abs(np.linalg.det(R) - 1) < 1e-4, side


def test_orient_offset_can_be_disabled():
    from egoannot.pose.ace import to_world
    p, f = _pred(), _frames()
    on = to_world(p, f, 1.0, orient_offset=True)["/pose/right_hand_quat"]
    off = to_world(p, f, 1.0, orient_offset=False)["/pose/right_hand_quat"]
    assert not np.allclose(on[:, 1:], off[:, 1:])
