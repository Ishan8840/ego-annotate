"""
Pose-free boundary detection from RGB alone.

The pipeline cuts spans at troughs in a pose-derived activity signal, and the
only thing that had ever been compared against it was a VLM asked to segment
the clip itself. That is a weak control: the vision-only arm front-loads badly
and produces one 16-second "action" per window, so beating it says little.

The task this stage implements has a name and a literature. Generic Event
Boundary Detection (GEBD) asks for class-agnostic boundaries between
semantically different parts of a video, and the strongest *unsupervised*
methods on Kinetics-GEBD use RGB only -- FlowGEBD works from optical flow,
GraphGEBD from self-similarity of frame features -- reaching F1@0.05 around
0.71-0.73 with no training at all.

Two RGB-only detectors live here, both non-parametric and both operating on
exactly the footage the pose-guided cutter sees:

  flow    optical-flow magnitude and its rate of change, in the spirit of the
          FlowGEBD family. Boundaries fall where the flow field settles.
  tsm     temporal self-similarity: a Foote novelty score along the diagonal
          of a frame-similarity matrix, which is the classical form of the
          idea GraphGEBD builds on.

Neither is the published method -- these are independent implementations of
the same primitives, written to run on this corpus without a model download.
They are here to be a fair opponent, not to reproduce a paper's number.
"""
from __future__ import annotations

import numpy as np

from .. import config
from ..core.signal import local_minima, smooth
from ..core.video import decode_stream

CFG = dict(
    fps=8.0,           # frames per second sampled for the RGB signal
    size=192,          # working resolution, square
    smooth_s=0.15,     # matches the pose stage, so the signals are comparable
    tsm_kernel_s=1.0,  # half-width of the Foote novelty kernel
)


def _frames(ep, cfg=CFG):
    """Grayscale frames at the working rate and resolution, plus their times."""
    size = int(cfg["size"])
    frames = list(decode_stream(ep["vid"], ep["src_fps"], cfg["fps"],
                                size=(size, size)))
    t = np.arange(len(frames)) / cfg["fps"]
    return t, frames


def flow_signal(ep, cfg=CFG):
    """
    Activity from optical flow: mean magnitude, smoothed.

    Boundaries in the GEBD sense fall where the flow field settles between two
    differently-moving stretches, which is the same shape of cue the pose
    signal uses -- so the two are directly comparable, one read from pixels and
    one from joints.
    """
    import cv2
    t, frames = _frames(ep, cfg)
    if len(frames) < 3:
        return np.zeros(0), np.zeros(0)
    mags = []
    for a, b in zip(frames, frames[1:]):
        f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mags.append(float(np.linalg.norm(f, axis=2).mean()))
    mid = (np.arange(len(mags)) + 0.5) / cfg["fps"]
    win = max(2, int(round(cfg["smooth_s"] * cfg["fps"])))
    act = smooth(np.array(mags), win)
    act = act / max(np.percentile(act, 90), 1e-9)     # same p90 norm as the pose signal
    return mid, act


def tsm_signal(ep, cfg=CFG):
    """
    Foote novelty along the diagonal of a frame self-similarity matrix.

    High novelty means the frames before and after this instant look unlike
    each other -- a scene or activity change. Inverted to an "activity" signal
    so that, like the flow and pose signals, boundaries sit at MINIMA and the
    same trough finder and band policy apply unchanged.
    """
    t, frames = _frames(ep, cfg)
    if len(frames) < 8:
        return np.zeros(0), np.zeros(0)
    X = np.stack([f.astype(np.float32).ravel() for f in frames])
    X -= X.mean(axis=1, keepdims=True)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    S = X @ X.T                                       # cosine self-similarity

    k = max(2, int(round(cfg["tsm_kernel_s"] * cfg["fps"])))
    # checkerboard kernel: within-block similarity minus across-block
    kern = np.ones((2 * k, 2 * k))
    kern[:k, k:] = -1.0
    kern[k:, :k] = -1.0
    novelty = np.zeros(len(frames))
    for i in range(k, len(frames) - k):
        novelty[i] = float((S[i - k:i + k, i - k:i + k] * kern).sum())
    novelty[:k] = novelty[k]
    novelty[-k:] = novelty[-k - 1]
    win = max(2, int(round(cfg["smooth_s"] * cfg["fps"])))
    novelty = smooth(novelty, win)
    # invert: boundaries are novelty PEAKS, and the rest of the pipeline cuts
    # at activity MINIMA.
    act = -novelty
    act = act - act.min()
    act = act / max(np.percentile(act, 90), 1e-9)
    return t, act


SIGNALS = {"rgb_flow": flow_signal, "rgb_tsm": tsm_signal}


def signal(ep, kind, cfg=CFG):
    """(timebase, normalised activity) for one RGB detector."""
    if kind not in SIGNALS:
        raise ValueError(f"unknown RGB signal {kind!r}; have {sorted(SIGNALS)}")
    return SIGNALS[kind](ep, cfg)


def boundaries(ep, kind, span_cfg=None, cfg=CFG):
    """Boundary times from an RGB detector, using the pose stage's trough rule."""
    span_cfg = span_cfg or config.SPANS_CFG
    t, act = signal(ep, kind, cfg)
    if not len(t):
        return np.zeros(0), np.zeros(0), np.zeros(0)
    idx = local_minima(act, span_cfg["prominence"],
                       min_gap_samples=span_cfg["min_gap_s"], t=t)
    return (t[idx] if len(idx) else np.zeros(0)), t, act
