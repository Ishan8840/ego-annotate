"""Compare episode concurrency with one resident detector; no VLM or captions.

No frame skipping, detector threshold changes, resolution changes or result cache.
Sequential/control and concurrent runs use the existing process_quality_fast.run.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def canonical(root, stem):
    quality = json.loads((root / stem / 'quality.json').read_text())
    # Only execution-duration fields are excluded. Every measurement, sampled
    # frame, track, confidence, source hash and validity field is compared.
    hands = root / stem / 'hands.json'
    if hands.exists():
        hands = json.loads(hands.read_text())
        for key in ['inference_seconds', 'tracking_stage_seconds']:
            hands.pop(key)
            quality['hand_visibility'].pop(key)
    else:
        hands = None
    evidence = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((root / stem / 'evidence').glob('*.jpg'))}
    return dict(quality=quality, hands=hands, evidence=evidence)


def compare(reference, candidate, videos):
    reports = []
    for path in videos:
        a, b = canonical(reference, path.stem), canonical(candidate, path.stem)
        measured_equal = a['quality']['measured'] == b['quality']['measured']
        motion_equal = a['quality']['motion'] == b['quality']['motion']
        hand_frames_changed, sampled_counts_changed, tracks_changed = [], [], []
        max_box_delta = 0.
        if a['hands'] and b['hands']:
            ar, br = a['hands']['frames'], b['hands']['frames']
            if len(ar) != len(br):
                raise ValueError('Dropped hand frames')
            for x, y in zip(ar, br):
                if x != y:
                    hand_frames_changed.append(x['frame_id'])
                if any(x[k] != y[k] for k in ['detection_ran', 'strong_count', 'weak_count']):
                    sampled_counts_changed.append(x['frame_id'])
                if len(x['tracks']) != len(y['tracks']):
                    tracks_changed.append(x['frame_id'])
                for tx, ty in zip(x['tracks'], y['tracks']):
                    max_box_delta = max(max_box_delta, *(abs(v-w) for v,w in zip(tx['box'], ty['box'])))
        reports.append(dict(clip=path.stem, exact_equal=a == b,
            measured_equal=measured_equal, motion_equal=motion_equal,
            evidence_pixels_equal=a['evidence'] == b['evidence'],
            changed_hand_frames=len(hand_frames_changed), changed_count_frames=len(sampled_counts_changed),
            changed_track_count_frames=len(tracks_changed), max_paired_box_delta_px=max_box_delta,
            first_changed_frames=hand_frames_changed[:10],
            evidence_error=b['quality']['evidence_error'], hand_error=b['quality']['hand_error']))
    return reports


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('videos', type=Path, nargs='+')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', type=Path, default=ROOT/'models/owlv2-hand')
    p.add_argument('--configs', nargs='+', default=['cpu1','cpu2','cpu4','hand1','hand2','hand4','batch4'])
    p.add_argument('--repeats', type=int, default=1)
    a=p.parse_args()
    configurations={'cpu1':(1,0), 'cpu2':(2,0), 'cpu4':(4,0),
                    'hand1':(1,1), 'hand2':(2,1), 'hand4':(4,1), 'batch4':(4,4)}
    if a.out.exists() or a.repeats < 1 or any(c not in configurations for c in a.configs):
        p.error('Use a new output directory, valid configs, and positive repeats')
    if any(not x.is_file() for x in a.videos) or len({x.stem for x in a.videos})!=len(a.videos):
        p.error('Videos must exist and have unique stems')
    if any(c.startswith('cpu') for c in a.configs) and 'cpu1' not in a.configs:
        p.error('Include cpu1 as the comparison control')
    if any(not c.startswith('cpu') for c in a.configs) and 'hand1' not in a.configs:
        p.error('Include hand1 as the comparison control')
    os.environ['EGO_VIDEO_THREADS']='4'
    os.environ['OPENCV_FFMPEG_THREADS']='4'
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    from process_quality_fast import run
    from egoannot.quality.hand_recovery import make_hand_detector, RecoveryHandDetector
    from egoannot.quality.shared_hand import SharedHandExecutor
    a.out.mkdir(parents=True)
    a.videos=[v.resolve() for v in a.videos]
    start=time.perf_counter()
    detector=None
    if any(configurations[c][1] for c in a.configs):
        detector=make_hand_detector(a.model, accuracy='balanced')
        detector(np.zeros((720,1280,3),np.uint8))
        import torch
    startup=time.perf_counter()-start
    rows=[]
    for repeat in range(a.repeats):
        # Alternate direction to reduce systematic first/last-run bias.
        order=a.configs if repeat % 2 == 0 else list(reversed(a.configs))
        for config in order:
            workers,batch=configurations[config]
            out=a.out/f'{config}-r{repeat}'
            out.mkdir()
            server=None
            if batch:
                torch.cuda.reset_peak_memory_stats()
            if workers>1 and batch:
                server=SharedHandExecutor(detector.detector, max_batch=batch)
            def one(path):
                local=(RecoveryHandDetector(server.client(),discovery_every=2) if server else detector) if batch else None
                return run(path,out/path.stem,hand_detector=local)
            began=time.perf_counter()
            try:
                if workers==1:
                    timings=[one(v) for v in a.videos]
                else:
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        timings=list(pool.map(one,a.videos))
            finally:
                if server:
                    server.close()
            elapsed=time.perf_counter()-began
            if any(t['evidence_error'] or t['hand_error'] or (batch and not t['hand_tracking_complete']) for t in timings):
                raise RuntimeError('Incomplete output; inspect saved diagnostics')
            duration=sum(t['video_seconds'] for t in timings)
            row=dict(config=config,repeat=repeat,episode_workers=workers,max_gpu_batch=batch,
                batch_wall_seconds=elapsed,video_seconds=duration,
                realtime_throughput=duration/elapsed,episode_latency_seconds=[t['wall_seconds'] for t in timings],
                process_rss_highwater_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if batch else None,
                gpu_batches=server.stats if server else [], directory=str(out))
            rows.append(row)
            save(out/'batch.json',row)
            save(a.out/'progress.json',rows)
            print('BATCH',config,repeat,round(elapsed,3),'seconds',flush=True)
    for row in rows:
        control='hand1' if row['max_gpu_batch'] else 'cpu1'
        row['comparison']=compare(a.out/f'{control}-r0',Path(row['directory']),a.videos)
        row['exact_equal']=all(c['exact_equal'] for c in row['comparison'])
    summary={}
    for config in a.configs:
        selected=[r for r in rows if r['config']==config]
        summary[config]=dict(median_batch_seconds=statistics.median(r['batch_wall_seconds'] for r in selected),
            trials=[r['batch_wall_seconds'] for r in selected],
            exact_equal=all(r['exact_equal'] for r in selected))
    save(a.out/'benchmark.json',dict(startup_seconds=startup,config=vars(a)|{'videos':list(map(str,a.videos)),
        'out':str(a.out),'model':str(a.model)},summary=summary,runs=rows,
        scope='All-frame measured quality, motion/evidence, optional balanced hands; no VLM, captions, overlay encoding, response caching or skipped frames. Warm batch wall includes output writes; startup is separate.'))
    print('SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
