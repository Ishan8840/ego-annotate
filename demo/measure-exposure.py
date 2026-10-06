"""Measure exposure-range pixel coverage on the same frames as the quality report.

This is a visual diagnostic, not an overall quality or usability score.
Uses the existing report's 320px grayscale measurement protocol and thresholds.
Requires numpy and opencv-python.
"""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np

root = Path(__file__).resolve().parents[1]
source = root/'artifacts/color-demo'
report = json.loads((source/'quality/analysis.json').read_text())
clips = []
for original in report['clips']:
    path = source/(original['id']+'.mp4')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original['sha256']
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {path}')
    frames = pixels = dark = light = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h,w = frame.shape[:2]
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray,(320,max(1,round(h*320/w))),interpolation=cv2.INTER_AREA)
        frames += 1
        pixels += gray.size
        dark += int(np.count_nonzero(gray<=5))
        light += int(np.count_nonzero(gray>=250))
    cap.release()
    assert frames == original['frame_count'], 'Decoded-frame coverage differs from quality report.'
    clips.append(dict(id=original['id'],sha256=original['sha256'],frames=frames,
                      analyzed_pixels=pixels,near_black_pixels=dark,near_white_pixels=light,
                      in_range_pixels=pixels-dark-light,
                      in_range_percent=100*(pixels-dark-light)/pixels))
total = sum(c['analyzed_pixels'] for c in clips)
in_range = sum(c['in_range_pixels'] for c in clips)
result = dict(metric='Pixels in exposure range',unit='percent of analyzed grayscale pixels',
              in_range_percent=100*in_range/total,analyzed_pixels=total,in_range_pixels=in_range,
              frames=sum(c['frames'] for c in clips),clips=clips,
              protocol=dict(metric_width=320,grayscale='OpenCV COLOR_BGR2GRAY',resize='INTER_AREA',
                            lower_inclusive=6,upper_inclusive=249,bit_depth=8,all_decoded_frames=True),
              interpretation='Fraction of analyzed pixels outside near-black (0–5) and near-white (250–255) ranges. '
              'Intentional shadows and highlights affect this diagnostic. It does not measure blur, noise, '
              'visibility, task success or overall video quality; localized glare may still be present.')
assert result['frames'] == report['summary']['total_frames']
assert 0 <= result['in_range_percent'] <= 100
out=source/'quality/exposure.json'
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
