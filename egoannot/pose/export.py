"""
Episode -> the inputs a monocular hand estimator needs: an mp4 and a camera.

The pose stages read hand joints out of the mcap. An RGB estimator cannot: it
needs the head camera's imagery as a normal video file and its calibration as
JSON. This module writes both, plus the sidecar that makes the round trip
exact.

The video is REMUXED, never re-encoded. The mcap carries an H.264 elementary
stream; wrapping it in an mp4 container copies the bitstream through, so the
frames an estimator sees are the frames the quality stage measured, with no
second generation of compression loss in between.

The sidecar matters as much as the video. `read_episode` timestamps every
video message (`vts`), and the pose streams live on that same clock, so frame
i of the exported mp4 is episode time vts[i]. Without that mapping the
estimator's output cannot be put back on the episode's timebase, and every
downstream comparison would be off by an unknown offset.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile

import numpy as np

from .. import config
from ..core.mcap_io import read_episode


def _remux(h264: bytes, src_fps: float, out_path: str) -> None:
    """Wrap a raw H.264 elementary stream in mp4 without re-encoding."""
    with tempfile.NamedTemporaryFile(suffix=".h264", delete=False) as f:
        f.write(h264)
        tmp = f.name
    try:
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "h264", "-r", str(src_fps),
               "-i", tmp, "-c", "copy", out_path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg remux failed: {r.stderr.strip()[:400]}")
    finally:
        os.unlink(tmp)


def camera_json(K: dict) -> dict:
    """
    The calibration in the shape ACE-Ego-Hand's --camera flag expects.

    Intrinsics are written in the sensor's OWN pixels and tagged with the
    resolution they were calibrated at; the consumer rescales them to whatever
    it encodes at. Writing pre-scaled numbers here would silently break the
    moment the encode width changed.
    """
    return {
        "image_width": int(K["w"]),
        "image_height": int(K["h"]),
        "model": "pinhole",
        "frames": [{"intrinsics": {"fx": float(K["fx"]), "fy": float(K["fy"]),
                                   "cx": float(K["cx"]), "cy": float(K["cy"])}}],
    }


def export(path, outdir=None) -> dict:
    """Write <episode>.mp4, <episode>.cam.json and <episode>.frames.npz."""
    outdir = str(outdir or config.ARTIFACTS / "pose" / "clips")
    os.makedirs(outdir, exist_ok=True)
    ep = read_episode(path, want_video=True)
    name = ep["name"]
    if not ep["vid"]:
        raise RuntimeError(f"{name}: no head_left video in the mcap")
    if ep["K"] is None:
        raise RuntimeError(f"{name}: no head_left intrinsics in the mcap")

    mp4 = os.path.join(outdir, name + ".mp4")
    _remux(ep["vid"], ep["src_fps"], mp4)

    cam = os.path.join(outdir, name + ".cam.json")
    json.dump(camera_json(ep["K"]), open(cam, "w"), indent=1)

    # The frame->time map, and the extrinsics needed to lift a camera-frame
    # prediction into the world frame the shipped pose lives in.
    side = os.path.join(outdir, name + ".frames.npz")
    np.savez_compressed(side, vts=ep["vts"], extr=ep["extr"],
                        src_fps=ep["src_fps"], duration_s=ep["duration_s"])

    info = dict(episode=name, mp4=mp4, camera=cam, frames=side,
                n_video_msgs=len(ep["vts"]), src_fps=ep["src_fps"],
                duration_s=round(ep["duration_s"], 2),
                size=[int(ep["K"]["w"]), int(ep["K"]["h"])],
                bytes=os.path.getsize(mp4))
    del ep
    return info


def export_all(paths, outdir=None) -> list[dict]:
    out = []
    for i, p in enumerate(paths, 1):
        info = export(p, outdir)
        out.append(info)
        print(f"[{i:2d}/{len(paths)}] {info['episode']:<32s} "
              f"{info['size'][0]}x{info['size'][1]} @ {info['src_fps']:g} fps  "
              f"{info['n_video_msgs']:5d} frames  {info['bytes']/1e6:6.1f} MB",
              flush=True)
    return out
