"""Core numerics: smoothing, run finding, minima, rotations, projection."""
import numpy as np
import pytest

from egoannot.core.geometry import in_rect, project, quat_to_R, quats_to_R
from egoannot.core.signal import local_minima, pct_rank, runs, smooth, speed


def test_smooth_preserves_a_constant_at_the_edges():
    """The old convolve/n form depressed the first and last n/2 samples 20-40%."""
    x = np.full(200, 2.0)
    out = smooth(x, 5)
    assert np.allclose(out, 2.0), out[:5]


def test_smooth_is_a_noop_below_the_window():
    x = np.array([1.0, 2.0])
    assert np.allclose(smooth(x, 5), x)


def test_smooth_handles_2d():
    x = np.tile(np.array([1.0, 3.0]), (100, 1))
    out = smooth(x, 7)
    assert out.shape == x.shape
    assert np.allclose(out[:, 0], 1.0) and np.allclose(out[:, 1], 3.0)


@pytest.mark.parametrize("mask,expected", [
    ([0, 0, 0], []),
    ([1, 1, 1], [(0, 2)]),
    ([0, 1, 1, 0, 1, 0], [(1, 2), (4, 4)]),
    ([1, 0, 0, 1], [(0, 0), (3, 3)]),
])
def test_runs(mask, expected):
    assert [tuple(map(int, r)) for r in runs(np.array(mask, bool))] == expected


def test_local_minima_respects_prominence():
    v = np.array([5, 3, 5, 4.8, 4.7, 4.8, 5, 1, 5.0])
    found = local_minima(v, prominence=1.0)
    assert 1 in found and 7 in found
    assert 4 not in found          # shallow dip inside a plateau


def test_local_minima_enforces_a_time_gap():
    """Minima closer than the gap are dropped, keeping the earlier one."""
    t = np.arange(9) * 0.1
    v = np.array([5, 1, 5, 1, 5, 1, 5, 1, 5.0])
    assert list(local_minima(v, 1.0, min_gap_samples=0.0)) == [1, 3, 5, 7]
    # t=0.1 and t=0.7 are 0.6 s apart, so both survive a 0.5 s gap; 0.3 and
    # 0.5 fall inside it.
    assert list(local_minima(v, 1.0, min_gap_samples=0.5, t=t)) == [1, 7]
    assert list(local_minima(v, 1.0, min_gap_samples=0.9, t=t)) == [1]


def test_quats_to_R_matches_the_scalar_path():
    Q = np.random.default_rng(0).normal(size=(64, 4))
    assert np.allclose(quats_to_R(Q), np.array([quat_to_R(*q) for q in Q]))


def test_quat_to_R_handles_a_zero_quaternion():
    assert np.allclose(quat_to_R(0, 0, 0, 0), np.eye(3))


def test_project_puts_a_point_on_the_principal_point():
    K = dict(fx=100.0, fy=100.0, cx=50.0, cy=40.0, w=100, h=80)
    u, v, z, ok = project(np.array([[0.0, 0.0, 1.0]]), np.eye(3),
                          np.zeros(3), K)
    assert ok[0] and np.isclose(u[0], 50.0) and np.isclose(v[0], 40.0)


def test_project_marks_points_behind_the_camera():
    K = dict(fx=100.0, fy=100.0, cx=50.0, cy=40.0, w=100, h=80)
    _, _, _, ok = project(np.array([[0.0, 0.0, -1.0]]), np.eye(3), np.zeros(3), K)
    assert not ok[0]


def test_in_rect_inset_excludes_the_border():
    K = dict(fx=1.0, fy=1.0, cx=0.0, cy=0.0, w=100, h=100)
    u = np.array([5.0, 50.0])
    v = np.array([5.0, 50.0])
    assert list(in_rect(u, v, K, 0.0)) == [True, True]
    assert list(in_rect(u, v, K, 0.25)) == [False, True]


def test_speed_of_a_straight_line():
    t = np.arange(11) * 0.1
    P = np.stack([t, np.zeros(11), np.zeros(11)], axis=1)
    tv, v = speed(t, P)
    assert len(tv) == 10 and np.allclose(v, 1.0)


def test_pct_rank_is_monotone_and_keeps_nans():
    x = np.array([3.0, 1.0, 2.0, np.nan] + [0.5] * 8)
    r = pct_rank(x)
    assert np.isnan(r[3])
    assert r[1] < r[2] < r[0]
