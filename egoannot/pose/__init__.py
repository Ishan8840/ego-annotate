"""
Where hand pose comes from.

Every stage downstream of this package reads exactly six arrays per episode --
``/pose/{left,right}_hand``, ``_joints`` and ``_quat`` -- and nothing else.
That is a narrow enough interface to swap the SOURCE of the pose without
touching a single consumer, which is what this package exists to do:

    shipped   the 21-joint hand pose annotated in the mcap (the default)
    ace       ACE-Ego-Hand run on the head camera's own imagery

The same trick the spans stage already uses for boundaries, where
``--signal rgb_flow`` substitutes a pose-free detector and everything
downstream is none the wiser. A source that cannot be swapped cannot be
measured against an alternative, and the point of a second pose source is to
find out which one is right.

The shipped pose is the incumbent, not the ground truth. Two findings in this
repository say so: it is near-rigidly coupled to the head camera (world spread
12-65 cm against 3-12 cm in camera frame, ``docs/quality.md``), and stereo
depth at the projected hand pixel does not agree with the depth the pose
itself predicts (correlation ~ 0, MAE 75-251 mm, ``evaluation/stereo.py``).
So ``evaluate`` reports agreement between the two sources, and separately the
stereo check that neither of them owns.
"""
from __future__ import annotations

import os

import numpy as np

from .. import config
from ..core.mcap_io import read_episode

SOURCES = ("shipped", "ace")
POSE_KEYS = tuple(f"/pose/{s}_hand{k}" for s in ("left", "right")
                  for k in ("", "_joints", "_quat"))


def predictions_dir() -> str:
    return str(config.ARTIFACTS / "pose" / "world")


def _ace_path(name: str) -> str:
    return os.path.join(predictions_dir(), name + ".npz")


def available(name: str, source: str) -> bool:
    return source == "shipped" or os.path.exists(_ace_path(name))


def attach(ep: dict, source: str = "shipped") -> dict:
    """
    Replace the episode's hand pose in place with the chosen source's.

    Only the hand arrays move. Intrinsics, extrinsics, head pose and upper
    body stay as shipped, because those are sensor calibration and device
    tracking rather than hand annotation -- replacing them would be measuring
    a different thing.
    """
    if source == "shipped":
        return ep
    if source not in SOURCES:
        raise ValueError(f"unknown pose source {source!r}; have {list(SOURCES)}")
    path = _ace_path(ep["name"])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no {source} pose for episode {ep['name']!r} at {path} - run "
            f"`python -m egoannot pose run {ep['name']}` first")
    z = np.load(path)
    for key in POSE_KEYS:
        flat = key.replace("/pose/", "").replace("/", "_")
        if flat not in z:
            raise KeyError(f"{path}: missing {flat}")
        ep[key] = z[flat]
    ep["pose_source"] = source
    return ep


def read(path, source: str = "shipped", want_video: bool = True,
         topics=None) -> dict:
    """`read_episode`, with the hand pose taken from the requested source."""
    return attach(read_episode(path, want_video=want_video, topics=topics),
                  source)
