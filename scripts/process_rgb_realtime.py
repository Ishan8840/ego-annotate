"""Batched RGB captions, verification of every caption, full measured QC.

This candidate omits the full pipeline's two subjective VLM quality rubrics.
Its compact verifier is an uncalibrated second opinion, not human ground truth.
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import json
import os
import math
import subprocess
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def process(path, out, engine, domain, modality, workers, verification_mode='compact'):
    from egoannot.rgb_caption import prepare_sources, system_prompt, caption_batch
    from egoannot.core.video import SegmentFrames
    from egoannot.quality.analysis import analyze, visual_frames, probe
    from egoannot.quality.annotations import audit, verify_label
    from egoannot.realtime import verify_compact, verification_summary, EpisodeFrames, SourceProbeCache, merge_verifications

    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    (out / path.name).symlink_to(path.resolve())
    capout = out / 'captions'
    capout.mkdir()
    stage_times = {}
    probe_cache = SourceProbeCache()

    def stage(name, fn):
        began = time.perf_counter()
        result = fn()
        stage_times[name] = time.perf_counter() - began
        print('STAGE', path.stem, name, round(stage_times[name], 3), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=2) as independent:
        quality_job = independent.submit(stage, 'measured_quality', lambda: analyze(
            [path], out / 'quality', visual='none', probe_cache=probe_cache))
        rows, sources = stage('boundaries', lambda: prepare_sources([path], domain, modality))
        source = sources[0]
        frames = EpisodeFrames(path.parent, probe_cache=probe_cache)
        def preload():
            frames.prepare(rows)
        stage('caption_evidence', preload)
        system = system_prompt(domain, modality)
        def captions():
            with ThreadPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(lambda row: caption_batch(
                    [row], source, frames, engine, system, domain), rows))
        batches = stage('captions', captions)
        labels = [label for batch, _ in batches for label in batch]
        calls = [call for _, batch_calls in batches for call in batch_calls]
        for name, data in [('spans', rows), ('captions', labels)]:
            (capout / (name + '.jsonl')).write_text(''.join(json.dumps(row) + '\n' for row in data))
        write_json(capout / 'run.json', dict(system_prompt=system, sources=sources, calls=calls))
        write_json(capout / 'summary.json', dict(sources=sources, spans=len(rows), captions=len(labels),
            seconds=source['duration_s'], format_valid=sum(not row['validation_errors'] for row in labels),
            uncertain=sum(row['uncertain'] for row in labels),
            time_coverage=sum(row['end_ts'] - row['start_ts'] for row in labels) / source['duration_s']))
        structural_job = independent.submit(stage, 'structural_audit', lambda: audit(
            capout / 'captions.jsonl', capout / 'spans.jsonl', out, out / 'audit', visual='none',
            probe_cache=probe_cache, frame_store=frames))
        span_map = {row['span_id']: row for row in rows}
        def evidence():
            return [frames.verification_frames(span_map[label['span_id']]) for label in labels]
        samples = stage('verification_evidence', evidence)
        def verify():
            def one(pair):
                label, sample = pair
                if verification_mode == 'compact':
                    return verify_compact(label, *sample, engine)
                result = verify_label(label, span_map[label['span_id']], engine, evidence=sample, max_retries=1)
                return dict(result, span_id=label['span_id'], source_video_sha256=source['sha256'])
            with ThreadPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(one, zip(labels, samples)))
        verification = stage('visual_verification', verify)
        quality_job.result()
        structural = structural_job.result()
        frames.release(source['id'])

    write_json(out / 'audit/structural.json', structural)
    combined = merge_verifications(structural, verification)
    combined['visual_backend'] = 'vllm-' + verification_mode
    write_json(out / 'audit/audit.json', combined)
    (out / 'audit/corrections.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in combined['correction_queue']))
    (out / 'audit/summary.md').write_text('# Annotation audit\n\n' + json.dumps(combined['summary'], indent=2)
        + '\n\nModel findings are uncalibrated second opinions. See audit.json for all evidence and findings.\n')
    audit_summary = verification_summary(verification, len(rows), len(labels))
    write_json(out / 'audit/verification.json', dict(mode=verification_mode,
        summary=audit_summary, annotations=verification,
        limitations=['Same-model second opinion, not calibrated accuracy; sampled stills may miss events.']))
    result = dict(source=str(path), sha256=source['sha256'], video_seconds=source['duration_s'],
        wall_seconds=time.perf_counter() - start, stages=stage_times,
        captions=len(labels), format_valid=sum(not row['validation_errors'] for row in labels),
        audit=audit_summary, subjective_quality_score=None,
        verification_mode=verification_mode,
        subjective_quality_status='not_computed_in_realtime_mode',
        measured_quality='all frames; same measurements as full pipeline',
        deferred_work=0, frame_cache_peak_bytes=frames.bytes_peak)
    result['real_time_factor'] = result['wall_seconds'] / result['video_seconds']
    write_json(out / 'timing.json', result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('videos', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--opencv-threads', type=int, default=1)
    parser.add_argument('--decode-threads', type=int, default=8)
    parser.add_argument('--domain', default='food_preparation')
    parser.add_argument('--camera-modality', choices=['rgb', 'monochrome'], default='rgb')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--verification', choices=['full', 'compact'], default='compact')
    parser.add_argument('--max-episode-seconds', type=float, default=600)
    args = parser.parse_args()
    if min(args.batch_size, args.cpu_threads, args.opencv_threads, args.decode_threads) < 1:
        parser.error('Thread and batch limits must be positive')
    if args.out.exists() or any(not path.is_file() or path.suffix.lower() != '.mp4' for path in args.videos):
        parser.error('Use existing MP4 sources and a new output directory')
    if len({p.stem for p in args.videos}) != len(args.videos):
        parser.error('Source stems must be unique')
    if not math.isfinite(args.max_episode_seconds) or args.max_episode_seconds <= 0:
        parser.error('Episode limit must be finite and positive')
    for path in args.videos:
        metadata = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_format',
                              '-show_streams', '-of', 'json', str(path)]))
        duration = float(metadata['format']['duration'])
        video = next(s for s in metadata['streams'] if s['codec_type'] == 'video')
        if video['codec_name'] != 'h264' or not 0 < duration <= args.max_episode_seconds:
            parser.error('Use H.264 MP4 episodes within --max-episode-seconds; split longer recordings first')
    from egoannot.labels.domains import PACKS
    if args.domain not in PACKS:
        parser.error('Unknown domain')
    os.environ['CAPTION_GREEDY'] = '1'
    os.environ['EGO_VIDEO_THREADS'] = str(args.decode_threads)
    os.environ['OPENCV_FFMPEG_THREADS'] = str(args.decode_threads)
    import cv2
    cv2.setNumThreads(args.opencv_threads)
    from egoannot.qwen_runtime import VLLMRuntime
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    runtime = VLLMRuntime(args.model, max_batch_size=args.batch_size, cpu_threads=args.cpu_threads,
                          trace_dir=args.out / 'requests' if args.trace else None)
    # Warm unrelated evidence, then clear both input caches before timed new data.
    import numpy as np
    _, encoded = cv2.imencode('.jpg', np.zeros((64, 64, 3), dtype=np.uint8))
    runtime('Return one word.', [('text', 'Describe this image.'), ('image', encoded.tobytes())], [])
    runtime.llm.reset_prefix_cache()
    runtime.llm.reset_mm_cache()
    runtime.stats.clear()
    warm_start = time.perf_counter()
    try:
        rows = []
        for path in args.videos:
            runtime.llm.reset_prefix_cache()
            runtime.llm.reset_mm_cache()
            rows.append(process(path.resolve(), args.out / path.stem, runtime,
                                args.domain, args.camera_modality, args.batch_size, args.verification))
    finally:
        runtime.close()
    result = dict(config={**vars(args), 'out': str(args.out), 'videos': [str(p) for p in args.videos]},
        load_seconds=runtime.load_seconds, cold_seconds=time.perf_counter() - started,
        warm_seconds=time.perf_counter() - warm_start, video_seconds=sum(r['video_seconds'] for r in rows),
        clips=rows, model_batches=runtime.stats, response_cache=False, deferred_work=0,
        scope='RGB captions, source/format checks, all-caption verification, measured quality; no 3D reconstruction')
    write_json(args.out / 'summary.json', result)
    with (args.out / 'timing.csv').open('w', newline='') as handle:
        fields = ['source', 'video_seconds', 'wall_seconds', 'real_time_factor',
                  'captions', 'format_valid', 'verification_mode', 'verification_coverage']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row['audit']['verification_coverage'] if key == 'verification_coverage'
                             else row[key] for key in fields})


if __name__ == '__main__':
    main()
