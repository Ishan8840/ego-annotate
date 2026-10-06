"""Native HOT3D-Clips -> timestamped world poses and unmirrored camera video.

UmeTrack FK is supplied by Meta's optional hand_tracking_toolkit. No MANO
assets or image estimator are needed. Missing thumb CMC stays NaN; these poses
are annotation inputs, not an evaluation of an image-based pose estimator.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import numpy as np

from .geometry import quat_to_R

# UmeTrack canonical landmarks: five tips, wrist, then each finger's joints.
MP_TO_UME = [5, None, 6, 7, 0, 8, 9, 10, 1, 11, 12, 13, 2,
             14, 15, 16, 3, 17, 18, 19, 4]


def map_landmarks(world):
    world = np.asarray(world, float)
    if world.shape != (20, 3) or not np.isfinite(world).all():
        raise ValueError("expected 20 finite world-space UmeTrack landmarks")
    joints = np.full((21, 3), np.nan)
    for target, source in enumerate(MP_TO_UME):
        if source is not None:
            joints[target] = world[source]
    return joints


def timeline(timestamps_ns):
    """Integer subtraction avoids loss of precision on long device clocks."""
    ts = np.asarray(timestamps_ns, dtype=np.int64)
    if len(ts) < 2 or np.any(np.diff(ts) <= 0):
        raise ValueError("HOT3D timestamps must be strictly increasing")
    t = (ts - ts[0]) / 1e9
    dt = float(np.median(np.diff(t)))
    # Encoding at CFR is only valid on a uniform grid; never hide dropped frames.
    if not np.allclose(t, np.arange(len(t)) * dt, atol=0.001, rtol=0):
        raise ValueError("nonuniform HOT3D timestamps: split/resample explicitly before import")
    return t, 1 / dt, float(t[-1] + dt)


def _transform(pose):
    q = np.asarray(pose["quaternion_wxyz"], float)
    xyz = np.asarray(pose["translation_xyz"], float)
    if q.shape != (4,) or xyz.shape != (3,) or not np.isfinite(np.r_[q, xyz]).all():
        raise ValueError("invalid HOT3D transform")
    if not np.isclose(np.linalg.norm(q), 1, atol=1e-4):
        raise ValueError("HOT3D quaternion is not normalized")
    T = np.eye(4)
    T[:3, :3] = quat_to_R(*q)
    T[:3, 3] = xyz
    return T, q, xyz


def _json(archive, name):
    fh = archive.extractfile(name)
    if fh is None:
        raise ValueError(f"missing HOT3D member {name}")
    with fh:
        return json.load(fh)


def prepare(clips, out, stream=None, rotate=0):
    import cv2
    import torch
    from hand_tracking_toolkit.hand_models.umetrack_hand_model import (
        UmeTrackHandPose, from_json, forward_kinematics)

    out = Path(out)
    if rotate not in (0, 90, 180, 270):
        raise ValueError('rotate must be 0, 90, 180 or 270 clockwise degrees')
    (out / "segments").mkdir(parents=True, exist_ok=True)
    definitions, manifests = [], []
    names = [Path(p).stem for p in clips]
    if len(set(names)) != len(names):
        raise ValueError("clip basenames must be unique")
    for path in map(Path, clips):
        name = path.stem
        with tarfile.open(path) as archive:
            members = archive.getnames()
            ids = sorted(int(n.split('.')[0]) for n in members
                         if n.endswith('.info.json') and n.split('.')[0].isdigit())
            infos = [_json(archive, f"{i:06d}.info.json") for i in ids]
            if len({i["sequence_id"] for i in infos}) != 1:
                raise ValueError("one imported clip must belong to one source sequence")
            cameras0 = _json(archive, f"{ids[0]:06d}.cameras.json")
            sid = stream
            if sid is None:
                rgb = [s for s, c in cameras0.items() if 'rgb' in c['calibration']['label']]
                left = [s for s, c in cameras0.items() if c['calibration']['label'].endswith('left')]
                choices = rgb or left
                if len(choices) != 1:
                    raise ValueError("ambiguous camera: specify --stream")
                sid = choices[0]
            timestamps = [i['image_timestamps_ns'][sid] for i in infos]
            if any(abs(ts - i['ref_timestamp_ns']) > 1_000_000 for ts, i in zip(timestamps, infos)):
                raise ValueError("image and annotation timestamps differ by more than 1 ms")
            t, fps, duration = timeline(timestamps)
            shapes = _json(archive, '__hand_shapes.json__')
            model = from_json(shapes['umetrack'])
            for key, value in vars(model).items():
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    setattr(model, key, value.to(torch.float64))
            arrays = {'vts': t, 'timestamps_ns': np.array(timestamps, dtype=np.int64)}
            pose, joints, quats = ({s: [] for s in ('left', 'right')} for _ in range(3))
            extr, visibility = [], {s: [] for s in ('left', 'right')}
            cal = cameras0[sid]['calibration']
            w, h = cal['image_width'], cal['image_height']
            # Full field of view, aspect preserved. Never mirror physical handedness.
            video = out / 'segments' / f'{name}.mp4'
            filters = {0: '', 90: 'transpose=1,', 180: 'hflip,vflip,',
                       270: 'transpose=2,'}[rotate] + 'scale=768:-2'
            cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                   '-s', f'{w}x{h}', '-r', str(fps), '-i', '-', '-an',
                   '-vf', filters, '-c:v', 'libx264', '-crf', '18',
                   '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            try:
                for frame_id, seconds in zip(ids, t):
                    key = f'{frame_id:06d}'
                    camera = _json(archive, key + '.cameras.json')[sid]
                    _, q, xyz = _transform(camera['T_world_from_camera'])
                    extr.append([seconds, *xyz, *q])
                    hands = _json(archive, key + '.hands.json')
                    for side in ('left', 'right'):
                        entry = hands.get(side, {})
                        p = entry.get('umetrack_pose')
                        visibility[side].append(entry.get('visibilities_modeled', {}).get(sid))
                        if not p:
                            continue
                        if 'T_world_from_wrist' not in p:
                            raise ValueError('expected named T_world_from_wrist transform')
                        T, qh, _ = _transform(p['T_world_from_wrist'])
                        angles = np.asarray(p['joint_angles'], float)
                        if angles.shape != (len(model.joint_rotation_axes),) or not np.isfinite(angles).all():
                            raise ValueError('invalid UmeTrack joint angles')
                        hp = UmeTrackHandPose(torch.tensor(int(side == 'right')),
                                             torch.tensor(angles), torch.tensor(T))
                        with torch.no_grad():
                            world, _, _ = forward_kinematics(hp, model, requires_mesh=False)
                        J = map_landmarks(world.numpy())
                        pose[side].append([seconds, *J[0]])
                        joints[side].append(J)
                        quats[side].append([seconds, *qh])
                    with archive.extractfile(f'{key}.image_{sid}.jpg') as fh:
                        image = cv2.imdecode(np.frombuffer(fh.read(), np.uint8), cv2.IMREAD_COLOR)
                    if image is None or image.shape[:2] != (h, w):
                        raise ValueError(f'{key}: image/calibration mismatch')
                    proc.stdin.write(image.tobytes())
            finally:
                proc.stdin.close()
                code = proc.wait()
            if code:
                raise RuntimeError(f'ffmpeg failed ({code}) for {path}')
            arrays['extr'] = np.asarray(extr)
            arrays['/pose/head'] = arrays['extr'][:, :4]
            arrays['/pose/upper_body'] = np.zeros((0, 4))
            arrays['/pose/upper_body_joints'] = np.zeros((0, 21, 3))
            for side in ('left', 'right'):
                arrays[f'/pose/{side}_hand'] = np.asarray(pose[side]).reshape(-1, 4)
                arrays[f'/pose/{side}_hand_joints'] = np.asarray(joints[side]).reshape(-1, 21, 3)
                arrays[f'/pose/{side}_hand_quat'] = np.asarray(quats[side]).reshape(-1, 5)
            with path.open('rb') as fh:
                digest = hashlib.file_digest(fh, 'sha256').hexdigest()
            meta = dict(dataset='HOT3D-Clips', sequence_id=infos[0]['sequence_id'],
                        device=infos[0]['device'], stream=sid, source_file=path.name,
                        source_sha256=digest, frame_count=len(t), fps=fps,
                        duration_s=duration, timestamp_origin_ns=int(timestamps[0]),
                        coordinate_frame='world', units='meters', pose_source='umetrack_annotation',
                        missing_joints=['thumb_cmc'], calibration=cal,
                        video_rotation_clockwise_deg=rotate,
                        camera_forward_axis=1, visibility=visibility,
                        hand_frames={s: len(pose[s]) for s in pose})
            arrays['metadata_json'] = np.array(json.dumps(meta))
            np.savez_compressed(out / f'{name}.npz', **arrays)
            definitions.append(dict(id=name, source=f'{name}.npz', t0=0, t1=duration,
                                    cls='manipulation'))
            manifests.append(meta)
            print(f'{name}: {len(t)} frames, {duration:.3f}s, {sid}, hands={meta["hand_frames"]}', flush=True)
    (out / 'segments.json').write_text(json.dumps(definitions, indent=2) + '\n')
    (out / 'manifest.json').write_text(json.dumps(manifests, indent=2) + '\n')
    return definitions


def read_prepared(path, want_video=True):
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['metadata_json']))
        if meta.get('dataset') != 'HOT3D-Clips':
            raise ValueError('not a prepared HOT3D episode')
        ep = {key: z[key] for key in z.files if key != 'metadata_json'}
    tl, tr = ep['/pose/left_hand'], ep['/pose/right_hand']
    ep.update(path=str(path), name=path.stem, duration_s=meta['duration_s'],
              src_fps=meta['fps'], meta_fps=meta['fps'], msg_fps=meta['fps'],
              fps_mismatch=False, K=None, badf=[], segs=[], meta=meta,
              camera_forward_axis=1, chans={}, counts={}, vid=b'',
              hands_share_timebase=bool(len(tl) and len(tl) == len(tr)
                  and np.allclose(tl[:, 0], tr[:, 0], rtol=0, atol=1e-6)))
    # Preserve the fisheye calibration; never pretend it is pinhole intrinsics.
    if want_video:
        ep['vid'] = subprocess.check_output([
            'ffmpeg', '-v', 'error', '-i', str(path.parent / 'segments' / f'{path.stem}.mp4'),
            '-c:v', 'copy', '-bsf:v', 'h264_mp4toannexb', '-f', 'h264', '-'])
    return ep
