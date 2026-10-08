"""All-frame QC plus one compact context request per window; separate candidate profile."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run(path, out, engine=None, workers=8, hand_detector=None, hand_hz=2., hand_flow=True):
    from egoannot.quality.analysis import measure_video
    from egoannot.quality.fast import EvidenceCollector, audit_context, CONTEXT, SYSTEM
    from egoannot.realtime import SourceProbeCache
    out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    cache = SourceProbeCache()
    stamp = cache.stamp(path)
    stages = {}
    collector, evidence_error, initialized = None, None, False
    hands, hand_error = None, None
    if hand_detector is not None:
        from egoannot.quality.hand_tracking import HandVisibility
        hands = HandVisibility(hand_detector, hand_hz, profile='baseline', flow=hand_flow)
    start = time.perf_counter()
    def observe(*values):
        nonlocal evidence_error, collector, initialized, hand_error
        if hands is not None and hand_error is None:
            try:
                hands(*values[:3])
            except Exception as exc:
                hand_error = f'{type(exc).__name__}: {exc}'
        if not initialized:
            initialized = True
            try:
                # measure_video already populated this cache. Delaying the
                # evidence plan lets strict decode overlap the initial PTS probe.
                data, _, _ = cache.get(path)
                raw = [float(f.get('best_effort_timestamp_time', 'nan')) for f in data.get('frames', [])]
                collector = EvidenceCollector(raw, out / 'evidence')
            except ValueError as exc:
                evidence_error = str(exc)
        if collector is not None and evidence_error is None:
            try:
                collector(*values)
            except Exception as exc:
                # Preserve the integrity/measurement report even when RGB context
                # evidence cannot safely be bound. Never infer context from it.
                evidence_error = f'{type(exc).__name__}: {exc}'
    clip = measure_video(path, fast=True, probe_cache=cache, frame_observer=observe)
    stages['probe_all_frame_qc_motion_evidence'] = time.perf_counter() - start
    if not initialized:
        evidence_error = 'No decoded frames; context evidence unavailable'
    motion = None
    if collector:
        try:
            motion = collector.finish(clip)
        except ValueError as exc:
            evidence_error = str(exc)
    hand_summary = None
    if hands is not None and hand_error is None:
        try:
            hand_report = hands.finish(clip['duration_s'])
            hand_report['source_sha256'] = clip['sha256']
            write(out / 'hands.json', hand_report)
            hand_summary = {k: v for k, v in hand_report.items() if k != 'frames'}
        except Exception as exc:
            hand_error = f'{type(exc).__name__}: {exc}'
    start = time.perf_counter()
    windows = audit_context(collector, engine, workers) if engine and collector and not evidence_error else []
    stages['context_audit'] = time.perf_counter() - start
    if cache.stamp(path) != stamp:
        raise ValueError('Source changed during evaluation')
    expected = len(collector.windows) * len(CONTEXT) if collector else 0
    assessed = sum(len(w['rows']) for w in windows)
    unavailable = {
        'instruction_compliance': 'No task specification supplied.',
        'task_success': 'No requested final state; success not established.',
        'action_correctness': 'No required sequence supplied.',
        'privacy_safety': 'No collection rules supplied.',
        'continuous_hand_tracking': 'No hand detector/tracker is run by this profile.',
        'physical_contact': 'Not measured from RGB stills.',
        'normal_action_speed': 'PTS and motion are measurements, not an action-speed judgment.',
        'calibrated_quality_score': 'No human calibration; no composite score is produced.',
        'diversity_descriptors': 'Not part of this per-episode quality profile.',
    }
    if engine is None:
        unavailable.update({key: 'Measured-only mode; context not assessed.' for key in CONTEXT})
    if hand_summary is not None:
        unavailable['continuous_hand_tracking'] = 'Sparse hand observations plus unverified optical-flow/box predictions; see hand_visibility and hands.json.'
    elif hand_error:
        unavailable['continuous_hand_tracking'] = hand_error
    result = dict(schema_version=1, profile='compact_context' if engine else 'measured_motion',
                  source_sha256=clip['sha256'], measured=clip, motion=motion,
                  evidence_error=evidence_error, windows=windows,
                  hand_visibility=hand_summary, hand_error=hand_error,
                  evidence_plan=collector.windows if collector else [],
                  context_dimensions=CONTEXT, context_system=SYSTEM,
                  context_coverage=dict(assessed=assessed, requested=expected if engine else 0,
                                        unknown=sum(r['status'] == 'U' for w in windows for r in w['rows'].values())),
                  context_complete=(assessed == expected and expected > 0 and not evidence_error) if engine else None,
                  unavailable=unavailable,
                  caveat='New context rubric: human validation pending. Complete replies do not prove correct judgments. All-frame measurements preserve the original scope.')
    write(out / 'quality.json', result)
    timing = dict(source=str(path), source_sha256=clip['sha256'], video_seconds=clip['duration_s'],
                  wall_seconds=time.perf_counter() - began, stages=stages,
                  context_calls=sum(len(w['calls']) for w in windows), context_coverage=result['context_coverage'],
                  context_complete=result['context_complete'],
                  evidence_error=evidence_error, profile=result['profile'],
                  includes_captions=False, includes_caption_verification=False,
                  includes_hand_pose=False, response_cache=False)
    timing.update(includes_hand_tracking=hand_detector is not None, hand_tracking_complete=hand_summary is not None,
                  hand_error=hand_error, hand_tracking_seconds=hands.seconds if hands else 0.)
    timing['real_time_factor'] = timing['wall_seconds'] / clip['duration_s'] if clip['duration_s'] else None
    write(out / 'timing.json', timing)
    print('RESULT', json.dumps(timing), flush=True)
    return timing


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('videos', nargs='+', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--mode', choices=['hybrid', 'measured'], default='hybrid')
    parser.add_argument('--model', default='models/qwen3-vl-8b')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-output-tokens', type=int, default=768)
    parser.add_argument('--decoding', choices=['plain', 'regex', 'json'], default='regex')
    parser.add_argument('--hands', action='store_true', help='Sparse Apache-2.0 OWLv2 hand detection + optical flow')
    parser.add_argument('--hand-model', default='models/owlv2-hand')
    parser.add_argument('--hand-hz', type=float, default=2.)
    parser.add_argument('--hand-precision', choices=['fp32', 'fp16'], default='fp16')
    parser.add_argument('--hand-accuracy', choices=['fast', 'balanced', 'high'], default='balanced')
    args = parser.parse_args()
    if (args.out.exists() or any(not p.is_file() for p in args.videos) or
            len({p.stem for p in args.videos}) != len(args.videos) or
            args.workers < 1 or args.max_output_tokens < 1):
        parser.error('Use existing unique-basename sources, a new output directory, and positive limits')
    os.environ['CAPTION_GREEDY'] = '1'
    os.environ['EGO_VIDEO_THREADS'] = '4'
    os.environ['OPENCV_FFMPEG_THREADS'] = '4'
    import cv2
    cv2.setNumThreads(1)
    args.out.mkdir(parents=True)
    began = time.perf_counter()
    runtime = None
    hand_detector = None
    hand_start = time.perf_counter()
    if args.hands:
        import numpy as np
        from egoannot.quality.hand_recovery import make_hand_detector
        hand_detector = make_hand_detector(args.hand_model, precision=args.hand_precision, accuracy=args.hand_accuracy)
        hand_detector(np.zeros((720, 1280, 3), dtype=np.uint8))
    hand_startup = time.perf_counter() - hand_start
    if args.mode == 'hybrid':
        from egoannot.qwen_runtime import VLLMRuntime
        from egoannot.quality.fast import context_schema, context_regex
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        import numpy as np
        runtime = VLLMRuntime(args.model, max_batch_size=args.workers, cpu_threads=8)
        # Warm-up uses unrelated pixels. No benchmark replies are cached.
        # Optional JSON constraints affect format, not factual correctness.
        constraint = (StructuredOutputsParams(regex=context_regex()) if args.decoding == 'regex' else
                      StructuredOutputsParams(json=context_schema()) if args.decoding == 'json' else None)
        runtime.sampling = SamplingParams(temperature=0, max_tokens=args.max_output_tokens,
                                         structured_outputs=constraint)
        _, jpeg = cv2.imencode('.jpg', np.zeros((64, 64, 3), dtype=np.uint8))
        runtime('Reply with one word.', [('image', jpeg.tobytes())], [])
        runtime.stats.clear()
    ready = time.perf_counter()
    try:
        rows = []
        for path in args.videos:
            if runtime:
                runtime.llm.reset_prefix_cache()
                runtime.llm.reset_mm_cache()
            rows.append(run(path.resolve(), args.out / path.stem, runtime, args.workers, hand_detector, args.hand_hz))
    finally:
        if runtime:
            runtime.close()
    result = dict(config={**vars(args), 'videos': list(map(str, args.videos)), 'out': str(args.out)},
                  warm_seconds=time.perf_counter() - ready, cold_seconds=time.perf_counter() - began,
                  load_seconds=runtime.load_seconds if runtime else 0.,
                  hand_startup_seconds=hand_startup,
                  clips=rows, model_batches=runtime.stats if runtime else [])
    write(args.out / 'summary.json', result)
    with (args.out / 'timing.csv').open('w', newline='') as handle:
        keys = ['source', 'video_seconds', 'wall_seconds', 'real_time_factor', 'context_calls', 'profile']
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: row[k] for k in keys} for row in rows)
    if args.mode == 'hybrid' and any(not row['context_complete'] for row in rows):
        raise SystemExit('Context coverage incomplete; see raw errors and unavailable fields in outputs')
    if args.hands and any(not row['hand_tracking_complete'] for row in rows):
        raise SystemExit('Hand tracking incomplete; see hand_error in outputs')


if __name__ == '__main__':
    main()
