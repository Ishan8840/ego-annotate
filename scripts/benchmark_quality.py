"""Time quality alone: measured QC, descriptive VLM report, composite rubric.

No caption generation or caption verification is included. Preserve raw failures
and coverage; an incomplete score is never reported as a successful full run.
"""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run(path, out, runtime, model, mode, workers=1, parallel_groups=False):
    from egoannot.quality.analysis import analyze, describe_video, summarize, duplicate_pairs
    from egoannot.quality.report import write_reports
    from egoannot.realtime import SourceProbeCache
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    (out / path.name).symlink_to(path.resolve())
    times, error = {}, None
    parallel = mode == 'full' and workers > 1
    probe_cache = SourceProbeCache() if parallel else None
    began = time.perf_counter()
    report = analyze([out / path.name], out / 'quality',
            visual='qwen-local' if mode == 'full' and not parallel else 'none',
            model=model, engine=runtime if not parallel else None, visual_workers=workers, probe_cache=probe_cache)
    times['measured_plus_descriptive_vlm' if mode == 'full' and not parallel else 'measured_checks'] = time.perf_counter() - began
    if mode == 'full':
        spec = importlib.util.spec_from_file_location('quality_composite', ROOT / 'demo/score-quality.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        def composite():
            began = time.perf_counter()
            try:
                module.main(['--source', str(out), '--model', model], engine=runtime,
                            workers=workers, parallel_groups=parallel_groups)
            finally:
                times['composite_rubric'] = time.perf_counter() - began

        def descriptive():
            began = time.perf_counter()
            for clip in report['clips']:
                clip['visual'] = describe_video(clip, runtime, workers=workers, probe_cache=probe_cache,
                                                parallel_groups=parallel_groups)
            report['protocol']['visual_backend'] = 'qwen-local'
            report['summary'] = summarize(report['clips'], duplicate_pairs(report['clips']), {})
            # The composite consumes only measured clip metadata. Publish the
            # enriched report atomically in case it is being opened concurrently.
            temporary = out / 'quality/analysis.updated.json'
            write(temporary, report)
            temporary.replace(out / 'quality/analysis.json')
            write(out / 'quality/clips.partial.json', report['clips'])
            write_reports(report, out / 'quality')
            times['descriptive_vlm'] = time.perf_counter() - began

        try:
            if parallel:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    descriptive_job = pool.submit(descriptive)
                    composite_job = pool.submit(composite)
                    descriptive_job.result()
                    composite_job.result()
            else:
                composite()
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    report = json.loads((out / 'quality/analysis.json').read_text())
    clip = report['clips'][0]
    visual = clip.get('visual', {})
    composite_path = out / 'quality/composite.json'
    composite = json.loads(composite_path.read_text()) if composite_path.exists() else {}
    result = dict(source=str(path), source_sha256=clip['sha256'], video_seconds=clip['duration_s'],
        wall_seconds=time.perf_counter() - started, mode=mode, stages=times, error=error,
        descriptive_windows=len(visual.get('windows', [])),
        descriptive_dimension_windows=sum(len(w['analyzed_dimensions']) for w in visual.get('windows', [])),
        descriptive_unavailable=visual.get('unavailable', {}),
        descriptive_calls=len(visual.get('calls', [])), composite_calls=len(composite.get('calls', [])),
        composite_score=composite.get('score_percent'),
        composite_assessed=composite.get('assessed_dimension_windows'),
        composite_total=composite.get('total_dimension_windows'),
        includes_captions=False, includes_caption_audit=False)
    result['real_time_factor'] = result['wall_seconds'] / result['video_seconds']
    write(out / 'timing.json', result)
    print('RESULT', json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('videos', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=['measured', 'full'], default='measured')
    parser.add_argument('--backend', choices=['transformers', 'vllm'], default='vllm')
    parser.add_argument('--model', default='models/qwen3-vl-8b')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--parallel-groups', action='store_true')
    parser.add_argument('--max-output-tokens', type=int, default=1024)
    args = parser.parse_args()
    if args.out.exists() or any(not p.is_file() for p in args.videos):
        parser.error('Use existing sources and a new output directory')
    if len({p.stem for p in args.videos}) != len(args.videos):
        parser.error('Source basenames must be unique')
    if min(args.workers, args.max_output_tokens) < 1:
        parser.error('workers and output-token limit must be positive')
    if args.backend == 'transformers' and args.max_output_tokens != 1024:
        parser.error('Custom output-token limit is currently supported only with vllm')
    os.environ['CAPTION_GREEDY'] = '1'
    os.environ['EGO_VIDEO_THREADS'] = '8'
    os.environ['OPENCV_FFMPEG_THREADS'] = '8'
    import cv2
    cv2.setNumThreads(1)
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    runtime = None
    load_seconds = 0.
    if args.mode == 'full':
        from egoannot.qwen_runtime import QwenRuntime, VLLMRuntime
        cls = VLLMRuntime if args.backend == 'vllm' else QwenRuntime
        runtime = cls(args.model, max_batch_size=args.workers * 2 if args.workers > 1 else 1, cpu_threads=8)
        if args.backend == 'vllm':
            from vllm import SamplingParams
            runtime.sampling = SamplingParams(temperature=0, max_tokens=args.max_output_tokens)
        load_seconds = runtime.load_seconds
        import numpy as np
        _, jpeg = cv2.imencode('.jpg', np.zeros((64, 64, 3), dtype=np.uint8))
        runtime('Reply with one word.', [('text', 'Describe this image.'), ('image', jpeg.tobytes())], [])
        runtime.stats.clear()
    ready = time.perf_counter()
    try:
        rows = []
        for path in args.videos:
            if runtime and args.backend == 'vllm':
                runtime.llm.reset_prefix_cache()
                runtime.llm.reset_mm_cache()
            rows.append(run(path.resolve(), args.out / path.stem, runtime, args.model, args.mode,
                            args.workers, args.parallel_groups))
    finally:
        if runtime:
            runtime.close()
    result = dict(config={**vars(args), 'videos': list(map(str, args.videos)), 'out': str(args.out)},
        load_seconds=load_seconds, warm_seconds=time.perf_counter() - ready,
        cold_seconds=time.perf_counter() - started, clips=rows,
        model_batches=runtime.stats if runtime else [], response_cache=False)
    write(args.out / 'summary.json', result)
    with (args.out / 'timing.csv').open('w', newline='') as handle:
        fields = ['source', 'video_seconds', 'wall_seconds', 'real_time_factor', 'mode', 'error']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


if __name__ == '__main__':
    main()
