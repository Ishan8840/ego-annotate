"""
No-reference image quality, for the quality tier.

T1 scores sharpness with tiled Laplacian variance and a calibrated floor. That
works and it is auditable, but it measures one thing -- high-frequency energy --
and calls it quality. The curation pipelines built around egocentric and robot
data use published no-reference metrics instead (PIQE and aesthetic scores are
the common pair), and learned NR-VQA models like DOVER and FAST-VQA correlate
about 0.85 with human opinion scores on LSVQ.

This implements PIQE (Venkatanath et al., 2015), which sits between the two:
no training and no weights to download, but a real distortion model rather than
a single gradient statistic. It scores 0-100, LOWER being better, by measuring
blockiness and noise only in blocks with enough spatial activity to judge --
so, unlike Laplacian variance, a flat or de-identified region is excluded
rather than counted as blur.

This is an implementation of the published algorithm, not a port of the
reference MATLAB, so treat the absolute values as this codebase's own scale and
compare within it.
"""
from __future__ import annotations

import numpy as np

BLOCK = 16
# The published activity threshold is 0.1. On this corpus that judges 3.7% of
# blocks -- retail and domestic interiors, de-identified and heavily
# compressed, simply do not carry that much local contrast -- so PIQE as
# published has almost nothing to score. Calibrated the way the blur floor is:
# the p25 of MSCN block variance measured over 262,080 blocks of real footage,
# which keeps the most textured three quarters of each frame judgeable.
ACTIVITY_THRESHOLD = 0.008    # MSCN block variance above which a block is judged
ACTIVITY_THRESHOLD_PUBLISHED = 0.1
NOISE_THRESHOLD = 2.0         # MSCN magnitude counted as noise
BLOCK_THRESHOLD = 0.1         # gradient step across a block edge counted as blocky


def mscn(gray, sigma=7 / 6, window=7):
    """Mean-subtracted contrast-normalised coefficients."""
    import cv2
    img = np.asarray(gray, np.float64)
    k = (window, window)
    mu = cv2.GaussianBlur(img, k, sigma)
    mu_sq = cv2.GaussianBlur(img * img, k, sigma)
    sigma_map = np.sqrt(np.abs(mu_sq - mu * mu))
    return (img - mu) / (sigma_map + 1.0)


def _blocks(coeffs):
    """(n_blocks, BLOCK, BLOCK) view of a frame, dropping any ragged edge."""
    h, w = coeffs.shape
    rows, cols = h // BLOCK, w // BLOCK
    if rows == 0 or cols == 0:
        return None
    trimmed = coeffs[:rows * BLOCK, :cols * BLOCK]
    return (trimmed.reshape(rows, BLOCK, cols, BLOCK)
            .transpose(0, 2, 1, 3)
            .reshape(rows * cols, BLOCK, BLOCK))


def piqe(gray, activity_threshold=None):
    """
    PIQE score for one grayscale frame: 0-100, lower is better.

    Returns (score, active_fraction). `active_fraction` is the share of blocks
    with enough spatial activity to be judged, and it carries information of
    its own: a frame with almost nothing judgeable is either flat or blurred,
    and its score should be trusted less.

    Fully vectorised -- a 1920x1456 frame is 10,920 blocks, and scoring those
    in a Python loop is slower than decoding the video.
    """
    thr = ACTIVITY_THRESHOLD if activity_threshold is None else activity_threshold
    img = np.asarray(gray)
    if img.ndim != 2 or min(img.shape) < BLOCK:
        return float("nan"), 0.0
    blocks = _blocks(mscn(img))
    if blocks is None:
        return float("nan"), 0.0

    active = blocks.var(axis=(1, 2)) > thr
    if not active.any():
        return float("nan"), 0.0
    b = blocks[active]

    dh = np.abs(np.diff(b, axis=2))                    # (n, BLOCK, BLOCK-1)
    dv = np.abs(np.diff(b, axis=1))                    # (n, BLOCK-1, BLOCK)
    edge = np.concatenate([dh[:, :, 0], dh[:, :, -1],
                           dv[:, 0, :], dv[:, -1, :]], axis=1).mean(axis=1)
    interior = np.concatenate([dh[:, :, 1:-1].reshape(len(b), -1),
                               dv[:, 1:-1, :].reshape(len(b), -1)],
                              axis=1).mean(axis=1)
    blockiness = np.maximum(edge - interior, 0.0)
    noise = (np.abs(b) > NOISE_THRESHOLD).mean(axis=(1, 2))
    score = 100.0 * float(np.minimum(blockiness + noise, 1.0).mean())
    return score, float(active.mean())


def piqe_series(frames, activity_threshold=None):
    """PIQE over a sequence; returns (scores, active_fractions) as arrays."""
    out = [piqe(f, activity_threshold) for f in frames]
    return (np.array([s for s, _ in out], float),
            np.array([a for _, a in out], float))
