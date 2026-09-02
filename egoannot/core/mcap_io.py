"""
Single-pass mcap reader shared by every stage.

Two things worth knowing about this corpus:

* Frame rate is not constant. It contains both 30 fps and 25 fps episodes.
  `session/metadata` declares the rate per camera; we read it and cross-check
  against the message rate. Assuming one rate skews the decoded timeline -- a
  25 fps episode read as 30 fps drifts frame-to-clip mapping by 17% and the
  clip tail decodes as empty.
* Depth is declared in metadata and never delivered, so `undelivered` is a
  real field and not a bug in the reader.

`want_video=False` skips accumulating the H.264 bitstream, which is most of the
resident memory for a long episode (~100 MB on shampoo.mcap). Pose-only
consumers should pass it.
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np

REQUIRED_TOPICS = [
    "/sensor/camera/head_left/video",
    "/sensor/camera/head_left/intrinsic",
    "/sensor/camera/head_left/extrinsic",
    "/pose/left_hand", "/pose/right_hand", "/pose/head",
]

OPTIONAL_TOPICS = [
    "/session/metadata",
    "/annotation/bad_frame/pose/hand",
    "/annotation/semantic_segments",
    "/pose/upper_body",
]

POSE_TOPICS = ("/pose/left_hand", "/pose/right_hand", "/pose/head", "/pose/upper_body")
JOINT_TOPICS = ("/pose/left_hand", "/pose/right_hand", "/pose/upper_body")
QUAT_TOPICS = ("/pose/left_hand", "/pose/right_hand")


def _flag(msg, field) -> bool:
    """A scalar-or-repeated bool field -> single bool."""
    v = getattr(msg, field, None)
    if v is None:
        return False
    try:
        return bool(any(v))          # repeated
    except TypeError:
        return bool(v)               # scalar


def _list(msg, field) -> list:
    """A scalar-or-repeated field -> plain list."""
    v = getattr(msg, field, None)
    if v is None:
        return []
    try:
        return [x if isinstance(x, (int, float, str)) else str(x) for x in v]
    except TypeError:
        return [v]


def read_episode(path, want_video: bool = True, topics: Iterable[str] | None = None) -> dict:
    """One pass over the mcap; returns pose arrays, calibration and video bytes."""
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    path = str(path)
    with open(path, "rb") as fh:
        reader = make_reader(fh, decoder_factories=[DecoderFactory()])
        summary = reader.get_summary()
        stats = summary.statistics
        schemas = {k: v.name for k, v in summary.schemas.items()}
        chans = {c.topic: schemas.get(c.schema_id, "?") for c in summary.channels.values()}
        counts = {c.topic: stats.channel_message_counts.get(cid, 0)
                  for cid, c in summary.channels.items()}
        t0 = stats.message_start_time
        duration = (stats.message_end_time - t0) / 1e9

        wanted = list(topics) if topics else (REQUIRED_TOPICS + OPTIONAL_TOPICS)
        if not want_video:
            wanted = [t for t in wanted if not t.endswith("/video")]
        wanted = [t for t in wanted if t in chans]

        pose = {k: [] for k in POSE_TOPICS}
        joints = {k: [] for k in JOINT_TOPICS}
        quats: dict[str, list] = {}
        extr, K, meta, bad_frames, segments = [], None, None, [], []
        video = bytearray()
        video_ts = []

        for _, chan, msg, dec in reader.iter_decoded_messages(topics=wanted):
            t = (msg.log_time - t0) / 1e9
            topic = chan.topic
            if topic in pose:
                p = dec.transforms[0].pos
                pose[topic].append((t, p.x, p.y, p.z))
                if topic in joints:
                    joints[topic].append([[q.pos.x, q.pos.y, q.pos.z]
                                          for q in dec.transforms])
                if topic in QUAT_TOPICS:
                    # NOTE: do not rebind the reader here. The phase-3 original
                    # assigned this to `r`, the same name as the mcap reader, and
                    # survived only because the generator already held its own
                    # reference.
                    rot = dec.transforms[0].quat
                    quats.setdefault(topic, []).append((t, rot.w, rot.x, rot.y, rot.z))
            elif topic == "/sensor/camera/head_left/video":
                video += dec.data
                video_ts.append(t)
            elif topic == "/sensor/camera/head_left/extrinsic":
                tr = dec.transforms[0]
                q, tl = tr.rotation, tr.translation
                extr.append((t, tl.x, tl.y, tl.z, q.w, q.x, q.y, q.z))
            elif topic == "/sensor/camera/head_left/intrinsic" and K is None:
                K = dict(fx=dec.K[0], fy=dec.K[4], cx=dec.K[2], cy=dec.K[5],
                         w=dec.width, h=dec.height)
            elif topic == "/session/metadata":
                meta = dec
            elif topic == "/annotation/bad_frame/pose/hand":
                bad_frames.append((t, _flag(dec, "is_bad"), _list(dec, "problem_type")))
            elif topic == "/annotation/semantic_segments":
                segments.append((dec.segment.start_time, dec.segment.end_time,
                                 dec.segment.subtask_description))

    meta_fps = None
    if meta is not None:
        for d in meta.devices:
            if d.device_type == "head_left_camera":
                try:
                    meta_fps = float(d.fps)
                except (TypeError, ValueError):
                    meta_fps = None
    msg_fps = (len(video_ts) / duration) if duration > 0 and video_ts else None
    src_fps = meta_fps or msg_fps or 30.0

    ep = dict(
        path=path, name=os.path.basename(path).rsplit(".", 1)[0],
        duration_s=duration, chans=chans, counts=counts, K=K,
        vid=bytes(video), vts=np.array(video_ts),
        badf=bad_frames, segs=segments,
        src_fps=src_fps, meta_fps=meta_fps,
        msg_fps=round(msg_fps, 2) if msg_fps else None,
        fps_mismatch=bool(meta_fps and msg_fps and abs(meta_fps - msg_fps) > 1.0),
    )
    for k, v in pose.items():
        ep[k] = np.array(v) if v else np.zeros((0, 4))
    ep["extr"] = np.array(extr) if extr else np.zeros((0, 8))
    for k, v in joints.items():
        ep[k + "_joints"] = np.array(v) if v else np.zeros((0, 21, 3))
    for k, v in quats.items():
        ep[k + "_quat"] = np.array(v)

    if meta is not None:
        ti = meta.task_info
        ep["meta"] = dict(episode_uuid=ti.episode_uuid, task_name=ti.task_name,
                          environment_id=ti.environment_id, scene_id=ti.scene_id,
                          paradigm=meta.operator.collection_method.paradigm,
                          devices=[(d.device_type, d.fps, d.modality)
                                   for d in meta.devices])
    else:
        ep["meta"] = {}

    # Several stages index the left and right hand joint arrays with a single
    # frame counter, which is only valid on a shared timebase. It holds for
    # this corpus (verified identical on every episode checked) but it is an
    # assumption, so record it rather than leaving it silent.
    tl, tr = ep["/pose/left_hand"], ep["/pose/right_hand"]
    n = min(len(tl), len(tr))
    ep["hands_share_timebase"] = bool(
        n and np.allclose(tl[:n, 0], tr[:n, 0], atol=1e-6))
    return ep


def modality_audit(ep: dict) -> tuple[list[str], list[str]]:
    """(missing required topics, declared-but-undelivered devices)."""
    missing = [t for t in REQUIRED_TOPICS if t not in ep["chans"]]
    declared = {d[0] for d in ep["meta"].get("devices", [])}
    present = set()
    if "/sensor/camera/head_left/video" in ep["chans"]:
        present.add("head_left_camera")
    if "/sensor/camera/head_right/video" in ep["chans"]:
        present.add("head_right_camera")
    if any("depth" in t for t in ep["chans"]):
        present.add("head_depth_camera")
    return missing, sorted(declared - present)
