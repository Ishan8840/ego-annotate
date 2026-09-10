"""
Drive ACE-Ego-Hand over the exported clips and land the result in world frame.

The estimator lives in its own repository and its own environment -- it pins
torch 2.5.1 against this project's much newer stack -- so it is invoked as a
subprocess rather than imported. That keeps the dependency wall where it
belongs: this package knows the estimator's CLI and its output schema, and
nothing else about it.

Paths are configurable because nothing here should hardcode one machine's
layout, which is the mistake the corpus discovery in `config.py` exists to
correct:

    ACE_HOME   checkout of ggxxii/ACE-Ego-Hand   (default ~/ACE-Ego-Hand)
    ACE_ENV    conda environment name            (default ace-ego-hand)
    ACE_CKPT   checkpoint to run                 (default the K-given one)

The K-GIVEN checkpoint is the default, not K-free. EgoStandard ships real
per-frame intrinsics for the head camera, and the whole point of the K-free
configuration is to work without them; handing the model a calibration it can
trust is strictly more information, and the paper's own border analysis shows
the fitted camera costs translation accuracy toward the frame edge.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

from .. import config
from ..core.mcap_io import read_episode
from . import predictions_dir
from .ace import convert, load_pkl, measured_axis, selfcheck
from .export import export

ACE_HOME = Path(os.environ.get("ACE_HOME", Path.home() / "ACE-Ego-Hand"))
ACE_ENV = os.environ.get("ACE_ENV", "ace-ego-hand")
ACE_CKPT = os.environ.get("ACE_CKPT", "checkpoints/ace_ego_hand_k.pt")
ACE_OPT = os.environ.get("ACE_OPT", "options/ace_ego_hand_k.yml")


def clips_dir() -> str:
    return str(config.ARTIFACTS / "pose" / "clips")


def raw_dir() -> str:
    return str(config.ARTIFACTS / "pose" / "raw")


def infer(name: str, encode_w: int = 832, mode: str = "tiled",
          max_frames: int = 0) -> str:
    """Run the estimator on one exported clip; returns the prediction pkl."""
    mp4 = os.path.join(clips_dir(), name + ".mp4")
    cam = os.path.join(clips_dir(), name + ".cam.json")
    for p in (mp4, cam):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} missing - run `python -m egoannot pose export {name}`")
    out = os.path.join(raw_dir(), name)
    os.makedirs(out, exist_ok=True)
    cmd = ["conda", "run", "--no-capture-output", "-n", ACE_ENV,
           "python", "infer_video.py",
           "--video", os.path.abspath(mp4),
           "--camera", os.path.abspath(cam),
           "--opt", ACE_OPT, "--ckpt", ACE_CKPT,
           "--out", os.path.abspath(out),
           "--encode_w", str(encode_w), "--mode", mode]
    if max_frames:
        cmd += ["--max_frames", str(max_frames)]
    print("  $", " ".join(cmd[6:]), flush=True)
    r = subprocess.run(cmd, cwd=str(ACE_HOME))
    if r.returncode != 0:
        raise RuntimeError(f"{name}: ACE-Ego-Hand exited {r.returncode}")
    pkl = os.path.join(out, name + ".pkl")
    if not os.path.exists(pkl):
        raise RuntimeError(f"{name}: no prediction written to {pkl}")
    return pkl


def run_batch(paths, encode_w: int = 1280, chunk: int = 241, overlap: int = 81,
              mode: str = "tiled", skip_existing: bool = True) -> list[dict]:
    """
    Every episode in ONE process, with each clip chunked inside it.

    Replaces the per-episode subprocess below, which reloaded the 5 GB backbone
    each time, VAE-encoded the whole clip in one call -- so resolution traded
    against clip length and 1280 died on the long episodes -- and abandoned the
    remaining episodes the moment one failed.
    """
    os.makedirs(predictions_dir(), exist_ok=True)
    todo = []
    for path in paths:
        name = Path(path).stem
        if skip_existing and os.path.exists(
                os.path.join(predictions_dir(), name + ".npz")):
            print(f"have {name}")
            continue
        if not os.path.exists(os.path.join(clips_dir(), name + ".mp4")):
            export(path)
        todo.append(dict(
            name=name,
            video=os.path.abspath(os.path.join(clips_dir(), name + ".mp4")),
            camera=os.path.abspath(os.path.join(clips_dir(), name + ".cam.json")),
            out=os.path.abspath(os.path.join(raw_dir(), name))))
    if not todo:
        return []

    manifest = str(config.artifact("pose", "batch_manifest.json"))
    report = str(config.artifact("pose", "batch_report.json"))
    json.dump(todo, open(manifest, "w"), indent=1)
    entry = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         "_ace_batch.py"))
    cmd = ["conda", "run", "--no-capture-output", "-n", ACE_ENV, "python",
           entry, "--manifest", manifest, "--home", str(ACE_HOME),
           "--opt", str(ACE_HOME / ACE_OPT), "--ckpt", str(ACE_HOME / ACE_CKPT),
           "--encode-w", str(encode_w), "--chunk", str(chunk),
           "--overlap", str(overlap), "--mode", mode, "--report", report]
    r = subprocess.run(cmd, cwd=str(ACE_HOME))
    if r.returncode != 0:
        raise RuntimeError(f"batch runner exited {r.returncode}")

    out = []
    rows = json.load(open(report))
    for row in rows:
        if not row.get("ok"):
            print(f"SKIP {row['episode']}: {row.get('error')}")
            continue
        name = row["episode"]
        ep = read_episode(str(config.episode_path(name)), want_video=False)
        axis = measured_axis(ep)
        world = os.path.join(predictions_dir(), name + ".npz")
        convert(row["pkl"], os.path.join(clips_dir(), name + ".frames.npz"),
                ep=ep, axis=axis, out=world)
        chk = selfcheck(load_pkl(row["pkl"]))
        out.append(dict(row, axis=axis, world=world, reproj_selfcheck_px=chk))
        print(f"  {name:<32s} axis={axis:+.0f}  -> {world}")
        del ep
    json.dump(out, open(config.artifact("pose", "run.json"), "w"), indent=1)
    print(f"[pose] {len(out)}/{len(rows)} episodes converted")
    return out


def run(paths, encode_w: int = 832, mode: str = "tiled", max_frames: int = 0,
        skip_existing: bool = True) -> list[dict]:
    """Export, infer and convert, for each episode. Returns per-episode info."""
    os.makedirs(predictions_dir(), exist_ok=True)
    rows = []
    for i, path in enumerate(paths, 1):
        name = Path(path).stem
        world = os.path.join(predictions_dir(), name + ".npz")
        if skip_existing and os.path.exists(world):
            print(f"[{i:2d}/{len(paths)}] {name:<32s} have {world}")
            continue
        print(f"[{i:2d}/{len(paths)}] {name}", flush=True)
        clip = os.path.join(clips_dir(), name + ".mp4")
        if not os.path.exists(clip):
            export(path)
        pkl = infer(name, encode_w, mode, max_frames)

        ep = read_episode(path, want_video=False)
        axis = measured_axis(ep)
        pred = load_pkl(pkl)
        chk = selfcheck(pred)
        frames = os.path.join(clips_dir(), name + ".frames.npz")
        convert(pkl, frames, ep=ep, axis=axis, out=world)
        n = len(np.asarray(pred["cam_trans"]))
        info = dict(episode=name, frames=n, axis=axis, pkl=pkl, world=world,
                    reproj_selfcheck_px=chk)
        rows.append(info)
        print(f"    {n} frames  axis={axis:+.0f}  "
              f"reprojection self-check: median {chk.get('px_p50', float('nan')):.1f} px "
              f"on {chk.get('n', 0)} frames", flush=True)
        del ep
    out = config.artifact("pose", "run.json")
    json.dump(rows, open(out, "w"), indent=1)
    return rows
