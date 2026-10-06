"""Export actual HOT3D annotation projections for the Preload product film."""
import argparse
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egoannot.core.hot3d import read_prepared
from hand_tracking_toolkit.camera import from_json


def export(corpus, clips, out):
    result = {}
    for path in map(Path, clips):
        ep = read_prepared(Path(corpus) / (path.stem + '.npz'), want_video=False)
        meta = ep['meta']
        if meta['video_rotation_clockwise_deg'] != 90:
            raise ValueError('This film uses the upright, 90-degree HOT3D import.')
        width, height = meta['calibration']['image_width'], meta['calibration']['image_height']
        rows = []
        with tarfile.open(path) as archive:
            for frame, seconds in enumerate(ep['vts']):
                camera = from_json(json.load(archive.extractfile(f'{frame:06d}.cameras.json'))[meta['stream']])
                row = dict(time_s=float(seconds))
                for side in ('left', 'right'):
                    times = ep[f'/pose/{side}_hand'][:, 0]
                    nearest = np.argmin(abs(times - seconds)) if len(times) else None
                    if nearest is None or abs(times[nearest] - seconds) > .02:
                        row[side] = [None] * 21
                        continue
                    joints = ep[f'/pose/{side}_hand_joints'][nearest]
                    finite = np.isfinite(joints).all(axis=1)
                    uv = np.full((21, 2), np.nan)
                    uv[finite] = camera.world_to_window(joints[finite])
                    # Clockwise display rotation and aspect-preserving video resize.
                    row[side] = [[round(float((height - .5 - y) / height), 6),
                                  round(float((x + .5) / width), 6)] if np.isfinite([x, y]).all() else None
                                 for x, y in uv]
                rows.append(row)
        result[path.stem] = dict(source='HOT3D annotated UmeTrack poses',
                                 coordinates='normalized upright video pixels', frames=rows)
    Path(out).write_text(json.dumps(result, separators=(',', ':'), allow_nan=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--corpus', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('clips', nargs='+')
    a = p.parse_args()
    export(a.corpus, a.clips, a.out)
