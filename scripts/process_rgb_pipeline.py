"""Run all RGB annotation and quality checks with one persistent model.

The conservative profile uses independent, sequential model requests. GPU
batching and alternate inference engines require explicit experimental opt-in.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def process(path, out, engine, model, domain, modality):
    from egoannot.quality.analysis import analyze
    from egoannot.quality.annotations import audit

    caption = module('rgb_caption', ROOT / 'scripts/annotate_rgb_demo.py')
    composite = module('rgb_composite', ROOT / 'demo/score-quality.py')
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    (out / path.name).symlink_to(path.resolve())
    times = {}

    def stage(name, fn):
        started = time.perf_counter()
        result = fn()
        times[name] = time.perf_counter() - started
        print('STAGE', path.stem, name, round(times[name], 3), flush=True)
        return result

    def captions_and_audit():
        stage('caption', lambda: caption.main([
            str(out / path.name), '--out', str(out / 'captions'),
            '--model', model, '--domain', domain, '--camera-modality', modality,
        ], engine=engine))
        return stage('audit', lambda: audit(
            out / 'captions/captions.jsonl', out / 'captions/spans.jsonl',
            out, out / 'audit', visual='qwen-local', model=model, engine=engine,
        ))

    def quality_and_composite():
        stage('quality', lambda: analyze(
            [out / path.name], out / 'quality', visual='qwen-local',
            model=model, engine=engine,
        ))
        stage('composite', lambda: composite.main([
            '--source', str(out), '--model', model,
        ], engine=engine))

    # Only independent stage chains overlap. Caption verification still waits
    # for captions, and the composite still waits for the quality analysis.
    with ThreadPoolExecutor(max_workers=2) as stages:
        caption_job = stages.submit(captions_and_audit)
        quality_job = stages.submit(quality_and_composite)
        result = caption_job.result()
        quality_job.result()

    summary = json.loads((out / 'captions/summary.json').read_text())
    row = dict(
        source=str(path), sha256=file_sha256(path),
        video_seconds=summary['seconds'], wall_seconds=time.perf_counter() - start,
        stages=times, captions=summary['captions'],
        format_valid=summary['format_valid'], uncertain=summary['uncertain'],
        audit=result['summary'],
        quality_score=json.loads((out / 'quality/composite.json').read_text())['score_percent'],
    )
    (out / 'timing.json').write_text(json.dumps(row, indent=2) + '\n')
    return row


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('videos', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--domain', default='food_preparation')
    parser.add_argument('--camera-modality', choices=['rgb', 'monochrome'], default='rgb')
    parser.add_argument('--backend', choices=['transformers', 'vllm'], default='transformers')
    parser.add_argument('--prompt-lookup', type=int, default=0)
    parser.add_argument('--experimental', action='store_true')
    args = parser.parse_args()
    from egoannot.labels.domains import PACKS
    if args.domain not in PACKS:
        parser.error('Unknown domain; choose one of: ' + ', '.join(sorted(PACKS)))
    experimental = args.backend == 'vllm' or args.batch_size > 1 or args.prompt_lookup > 0
    if experimental and not args.experimental:
        parser.error('Unvalidated inference changes require --experimental; compare outputs before deployment')
    if args.workers < 1 or args.cpu_threads < 1 or args.batch_size < 1 or args.prompt_lookup < 0:
        parser.error('Runtime limits must be positive and prompt-lookup nonnegative')
    if args.backend == 'vllm' and args.prompt_lookup:
        parser.error('prompt-lookup is supported only by the Transformers backend')
    if any(not path.is_file() for path in args.videos):
        parser.error('Every source must be an existing video file')
    if args.out.exists():
        parser.error('Use a new output directory to preserve previous results')
    if len({path.stem for path in args.videos}) != len(args.videos):
        parser.error('Source stems must be unique')
    args.out.mkdir(parents=True)

    from egoannot.qwen_runtime import QwenRuntime, VLLMRuntime
    start = time.perf_counter()
    runtime_class = VLLMRuntime if args.backend == 'vllm' else QwenRuntime
    runtime = runtime_class(
        args.model, max_batch_size=args.batch_size, cpu_threads=args.cpu_threads,
        trace_dir=args.out / 'requests' if args.trace else None,
    )
    runtime.prompt_lookup_tokens = args.prompt_lookup
    ready = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(lambda path: process(
                path, args.out / path.stem, runtime, args.model,
                args.domain, args.camera_modality,
            ), args.videos))
    finally:
        runtime.close()
    result = dict(
        config=vars(args) | {'videos': [str(path) for path in args.videos], 'out': str(args.out)},
        load_seconds=runtime.load_seconds, cold_seconds=time.perf_counter() - start,
        warm_seconds=time.perf_counter() - ready,
        video_seconds=sum(row['video_seconds'] for row in rows), clips=rows,
        model_batches=runtime.stats, response_cache=False, all_four_stages=True,
    )
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ['model_batches', 'clips']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
