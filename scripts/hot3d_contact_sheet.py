"""Audit imported world-space hand landmarks against native fisheye images."""
import argparse
import json
from pathlib import Path
import sys
import tarfile
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egoannot.core.hot3d import read_prepared
from hand_tracking_toolkit.camera import from_json

p = argparse.ArgumentParser(__doc__)
p.add_argument('--corpus', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('clips', nargs='+', type=Path)
a = p.parse_args()
rows = []
for path in a.clips:
    ep = read_prepared(a.corpus / (path.stem + '.npz'), want_video=False)
    sid = ep['meta']['stream']
    cells = []
    with tarfile.open(path) as tar:
        for frame in (0, 45, 90, 135):
            key = f'{frame:06d}'
            camera = from_json(json.load(tar.extractfile(key + '.cameras.json'))[sid])
            image = cv2.imdecode(np.frombuffer(tar.extractfile(f'{key}.image_{sid}.jpg').read(), np.uint8), cv2.IMREAD_COLOR)
            for side, color in [('left', (255, 255, 0)), ('right', (0, 255, 255))]:
                times = ep[f'/pose/{side}_hand'][:, 0]
                if not len(times):
                    continue
                idx = np.argmin(abs(times - ep['vts'][frame]))
                if abs(times[idx] - ep['vts'][frame]) > 0.02:
                    continue
                J = ep[f'/pose/{side}_hand_joints'][idx]
                good = np.isfinite(J).all(axis=1)
                uv = np.full((21, 2), np.nan)
                uv[good] = camera.world_to_window(J[good])
                for x, y in uv[good]:
                    cv2.circle(image, (round(x), round(y)), 5, color, -1)
                for chain in ([0, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
                              [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]):
                    for u, v in zip(chain, chain[1:]):
                        cv2.line(image, tuple(np.rint(uv[u]).astype(int)), tuple(np.rint(uv[v]).astype(int)), color, 2)
            image = cv2.resize(image, (512, 410))
            cv2.putText(image, f'{path.stem} {frame/30:.1f}s', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 240, 50), 2)
            cells.append(image)
    rows.append(np.hstack(cells))
a.out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(a.out), np.vstack(rows))
