"""Benchmark sparse hand detection/tracking and save source-bound overlay demos."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(path, out, detector, profile, hz, overlay=False, flow=False):
    import cv2
    from egoannot.quality.hand_tracking import HandVisibility
    from egoannot.realtime import SourceProbeCache
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    cache = SourceProbeCache()
    data, code, error = cache.get(path)
    if code or error:
        raise ValueError('Unreliable source probe: ' + error)
    stream = data['streams'][0]
    times = [float(f['best_effort_timestamp_time']) for f in data['frames']]
    times = [t - times[0] for t in times]
    a, b = map(float, stream['avg_frame_rate'].split('/'))
    fps = a / b
    tracker = HandVisibility(detector, hz, profile, flow=flow)
    cap = cv2.VideoCapture(str(path))
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index >= len(times):
                raise ValueError('Frame/PTS count mismatch')
            tracker(index, times[index], frame)
            index += 1
    finally:
        cap.release()
    if index != len(times):
        raise ValueError('Incomplete decode')
    duration = times[-1] + 1 / fps
    result = tracker.finish(duration)
    result.update(source=str(path), width=int(stream['width']), height=int(stream['height']),
                  fps=fps, duration_s=duration, frame_count=index,
                  wall_seconds=time.perf_counter() - start)
    from hashlib import file_digest
    with path.open('rb') as handle:
        result['source_sha256'] = file_digest(handle, 'sha256').hexdigest()
    (out / 'hands.json').write_text(json.dumps(result, indent=2) + '\n')
    with (out / 'samples.csv').open('w', newline='') as handle:
        keys = ['frame_id', 'time_s', 'detection_ran', 'strong_count', 'weak_count', 'crop_candidate', 'recovery_calls']
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: r[k] for k in keys} for r in result['frames'])
    if overlay:
        render_overlay(path, result, out / 'overlay.mp4')
    summary = {k: v for k, v in result.items() if k != 'frames'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    return summary


def render_overlay(path, result, out):
    import cv2
    width, height = result['width'], result['height']
    writer = subprocess.Popen(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}', '-r', str(result['fps']), '-i', '-', '-an', '-c:v', 'libx264',
        '-threads', '4', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)], stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(str(path))
    try:
        for row in result['frames']:
            ok, frame = cap.read()
            if not ok:
                raise ValueError('Overlay source incomplete')
            scale = max(.5, width / 1280)
            for tr in row['tracks']:
                x1, y1, x2, y2 = map(lambda x: int(round(x)), tr['box'])
                strong = tr['status'] == 'observed_strong'
                color = (90, 230, 140) if strong else (80, 185, 255)
                if tr['observed']:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                else:
                    # Dashed boxes are predictions; visibly separate from observations.
                    for x in range(x1, x2, 18):
                        cv2.line(frame, (x, y1), (min(x + 9, x2), y1), color, 1)
                        cv2.line(frame, (x, y2), (min(x + 9, x2), y2), color, 1)
                    for y in range(y1, y2, 18):
                        cv2.line(frame, (x1, y), (x1, min(y + 9, y2)), color, 1)
                        cv2.line(frame, (x2, y), (x2, min(y + 9, y2)), color, 1)
                text = f"ID {tr['track_id']} {tr['score']:.2f} " + ('observed' if tr['observed'] else 'predicted / unverified')
                cv2.putText(frame, text, (max(0, x1), max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .5 * scale, color, 1, cv2.LINE_AA)
            cv2.rectangle(frame, (0, 0), (width, round(78 * scale)), (17, 35, 24), -1)
            text = f"HAND VISIBILITY | {row['time_s']:.2f}s | {result['sample_hz']:g} Hz detector + tracking"
            cv2.putText(frame, text, (15, round(28 * scale)), cv2.FONT_HERSHEY_SIMPLEX, .65 * scale, (160, 238, 186), 1, cv2.LINE_AA)
            status = ('Observed hands: ' + str(row['strong_count'] + row['weak_count'])) if row['detection_ran'] else 'Between detector samples: predictions only'
            if row['detection_ran'] and not row['strong_count'] + row['weak_count']:
                status = 'No hand detected: out of view, occluded, or detector miss'
            cv2.putText(frame, status, (15, round(58 * scale)), cv2.FONT_HERSHEY_SIMPLEX, .58 * scale, (210, 222, 214), 1, cv2.LINE_AA)
            writer.stdin.write(frame.tobytes())
    finally:
        cap.release()
        writer.stdin.close()
        code = writer.wait()
    if code:
        raise RuntimeError('Overlay encoder failed')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('videos', type=Path, nargs='+')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--model', type=Path, default=Path('models/owlv2-hand'))
    p.add_argument('--detector', choices=['owl', 'grounded', 'nano'], default='owl')
    p.add_argument('--profile', choices=['baseline', 'adaptive'], default='baseline')
    p.add_argument('--hz', type=float, default=2.)
    p.add_argument('--size', type=int, default=960)
    p.add_argument('--precision', choices=['fp32', 'fp16'], default='fp16')
    p.add_argument('--flow', action='store_true')
    p.add_argument('--accuracy', choices=['fast', 'balanced', 'high'], default='balanced',
                   help='fast reproduces the previous detector; balanced/high add image-space discovery')
    p.add_argument('--overlay', action='store_true')
    args = p.parse_args()
    if args.out.exists():
        p.error('Use a new output directory')
    os.environ['EGO_VIDEO_THREADS'] = '4'
    os.environ['OPENCV_FFMPEG_THREADS'] = '4'
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    from egoannot.quality.hand_tracking import HandDetector
    start = time.perf_counter()
    if args.detector == 'owl':
        if args.size != 960:
            p.error('OWLv2 requires its native 960px profile')
        from egoannot.quality.hand_recovery import make_hand_detector
        detector = make_hand_detector(args.model, precision=args.precision, accuracy=args.accuracy)
    elif args.detector == 'grounded':
        from egoannot.quality.grounded_hand import GroundedHandDetector
        detector = GroundedHandDetector(args.model, size=args.size, precision=args.precision)
    else:
        detector = HandDetector(args.model)
    detector(np.zeros((320, 320, 3), np.uint8))
    startup = time.perf_counter() - start
    args.out.mkdir(parents=True)
    rows = [run(path.resolve(), args.out / path.stem, detector, args.profile, args.hz, args.overlay, args.flow) for path in args.videos]
    (args.out / 'summary.json').write_text(json.dumps(dict(startup_seconds=startup, clips=rows,
        timing_scope='Probe, decode, detection and tracking; overlay rendering and source hashing are excluded. Quality VLM is not included.',
        model_bytes=sum(p.stat().st_size for p in args.model.glob('*') if p.is_file()) if args.model.is_dir() else args.model.stat().st_size), indent=2) + '\n')


if __name__ == '__main__':
    main()
