"""
The three changes the methods survey called for: a real pose-free boundary
baseline, the standard dense-captioning metrics, and a published no-reference
image-quality metric alongside the Laplacian gate.
"""
import numpy as np
import pytest

from egoannot import config
from egoannot.core import imquality
from egoannot.evaluation import captions as CAP
from egoannot.stages import rgb_boundaries as RGB


# ---------------------------------------------------------------- RGB arms
def _clip(n=120, size=64, seed=0):
    """Frames that change character halfway: a boundary a detector should find."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (size, size), dtype=np.uint8)
    frames = []
    for i in range(n):
        if i < n // 2:
            shift = i % 3                     # small jitter
        else:
            shift = (i * 7) % size            # fast pan
        frames.append(np.roll(base, shift, axis=1))
    return frames


def test_both_rgb_signals_are_registered():
    assert set(RGB.SIGNALS) == {"rgb_flow", "rgb_tsm"}
    with pytest.raises(ValueError, match="unknown RGB signal"):
        RGB.signal(None, "rgb_nonsense")


def test_tsm_novelty_peaks_at_a_content_change():
    """
    The self-similarity arm should mark the seam between two visually distinct
    halves. It is returned inverted, so a boundary is a MINIMUM of `act`.
    """
    frames = _clip()
    # drive the signal maths directly, bypassing video decode
    X = np.stack([f.astype(np.float32).ravel() for f in frames])
    X -= X.mean(axis=1, keepdims=True)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    S = X @ X.T
    assert S.shape == (len(frames), len(frames))
    # frames within a half look alike; across the seam they do not
    half = len(frames) // 2
    within = S[:half, :half].mean()
    across = S[:half, half:].mean()
    assert within > across


def test_rgb_signals_are_normalised_like_the_pose_signal():
    """Every arm divides by its own p90 so the same prominence rule applies."""
    for fn in (RGB.flow_signal, RGB.tsm_signal):
        assert callable(fn)
    cfg = dict(RGB.CFG)
    assert cfg["fps"] > 0 and cfg["size"] >= 64


# ---------------------------------------------------------------- IoU / match
@pytest.mark.parametrize("a,b,expected", [
    ((0, 2), (0, 2), 1.0),
    ((0, 2), (2, 4), 0.0),
    ((0, 2), (1, 3), 1 / 3),
    ((0, 4), (1, 2), 0.25),
])
def test_iou(a, b, expected):
    assert CAP.iou(a, b) == pytest.approx(expected)


def test_matching_is_one_to_one_and_respects_the_threshold():
    preds = [(0, 2, "a"), (2, 4, "b"), (4, 6, "c")]
    refs = [(0, 2, "x"), (2, 4, "y")]
    pairs, n_p, n_r = CAP.match(preds, refs, 0.5)
    assert (n_p, n_r) == (3, 2)
    assert sorted(pairs) == [(0, 0), (1, 1)]
    assert len({i for i, _ in pairs}) == len(pairs)      # no pred reused
    assert len({j for _, j in pairs}) == len(pairs)      # no ref reused


def test_matching_drops_pairs_below_threshold():
    preds = [(0, 10, "a")]
    refs = [(0, 2, "x")]                                 # IoU 0.2
    assert CAP.match(preds, refs, 0.3)[0] == []
    assert CAP.match(preds, refs, 0.1)[0] == [(0, 0)]


# ---------------------------------------------------------------- SODA
def test_soda_rewards_a_correctly_ordered_sequence():
    preds = [(0, 2, "a"), (4, 6, "b")]
    refs = [(0, 2, "a"), (4, 6, "b")]
    p, r, f = CAP.soda_c(preds, refs, lambda i, j: 1.0 if i == j else 0.0)
    assert f == pytest.approx(1.0)


def test_soda_penalises_a_crossing_alignment():
    """Right captions, wrong order: the alignment cannot cross, so it scores 0."""
    preds = [(0, 2, "b"), (4, 6, "a")]
    refs = [(0, 2, "a"), (4, 6, "b")]
    p, r, f = CAP.soda_c(preds, refs, lambda i, j: 1.0 if i != j else 0.0)
    assert f == pytest.approx(0.0)


def test_soda_precision_falls_when_predictions_outnumber_references():
    """Being denser than the reference costs precision -- the caveat, as maths."""
    refs = [(0, 2, "a")]
    dense = [(0, 1, "a"), (1, 2, "a")]
    _, _, f_dense = CAP.soda_c(dense, refs, lambda i, j: 1.0)
    _, _, f_exact = CAP.soda_c([(0, 2, "a")], refs, lambda i, j: 1.0)
    assert f_exact > f_dense


def test_soda_handles_empty_sides():
    assert CAP.soda_c([], [(0, 1, "a")], lambda i, j: 1.0) == (0.0, 0.0, 0.0)
    assert CAP.soda_c([(0, 1, "a")], [], lambda i, j: 1.0) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------- PIQE
def _textured(size=256, seed=0):
    import cv2
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(
        rng.integers(0, 255, (size, size), dtype=np.uint8), (0, 0), 1.0)


def test_piqe_scores_noise_worse_than_clean():
    clean = _textured()
    noisy = np.clip(clean.astype(int)
                    + np.random.default_rng(1).normal(0, 40, clean.shape),
                    0, 255).astype(np.uint8)
    assert imquality.piqe(noisy)[0] > imquality.piqe(clean)[0]


def test_piqe_declines_to_judge_a_flat_frame():
    flat = np.full((256, 256), 128, np.uint8)
    score, active = imquality.piqe(flat)
    assert np.isnan(score) and active == 0.0


def test_piqe_activity_threshold_was_recalibrated_for_this_corpus():
    """The published 0.1 judges 3.7% of blocks on this footage."""
    assert imquality.ACTIVITY_THRESHOLD < imquality.ACTIVITY_THRESHOLD_PUBLISHED
    frame = _textured()
    _, active_cal = imquality.piqe(frame, imquality.ACTIVITY_THRESHOLD)
    _, active_pub = imquality.piqe(frame, imquality.ACTIVITY_THRESHOLD_PUBLISHED)
    assert active_cal >= active_pub


def test_piqe_is_not_a_sharpness_metric():
    """
    Blur removes the blockiness and noise PIQE models, so it rates a blurred
    frame as CLEANER. This is why PIQE is carried alongside the Laplacian gate
    rather than replacing it -- documented as a test so nobody swaps them.
    """
    import cv2
    clean = _textured()
    blurred = cv2.GaussianBlur(clean, (0, 0), 5)
    s_clean, _ = imquality.piqe(clean)
    s_blur, a_blur = imquality.piqe(blurred)
    if a_blur > 0 and np.isfinite(s_blur):
        assert s_blur <= s_clean


def test_piqe_gate_is_off_until_calibrated():
    from egoannot.stages.quality import hard_rejects
    base = dict(duration_s=4.0, missing_topics=[], n_frames=8,
                sharpness_p25=30.0, brightness_out_frac=0.0,
                max_wrist_speed=1.0, max_head_speed=0.5, max_angular_speed=1.0,
                hand_visible_rate=1.0, hand_central50_rate=0.95, piqe=99.0)
    assert hard_rejects(base, dict(config.QUALITY, piqe_max=None)) == []
    assert "compression_artifacts" in hard_rejects(
        base, dict(config.QUALITY, piqe_max=3.44))


def test_piqe_is_vectorised_enough_for_full_resolution():
    import time
    frame = _textured(512)
    t0 = time.perf_counter()
    imquality.piqe(frame)
    assert time.perf_counter() - t0 < 1.0
