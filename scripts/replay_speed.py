"""Replay saved independent requests and compare exact model replies.

Input images and prompts stay fixed. No generated-response cache is used.
Warm-up is excluded from workload time; model load is reported separately.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', default='models/qwen3-vl-8b')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--backend', choices=['transformers', 'vllm', 'original'], default='transformers')
    parser.add_argument('--prompt-lookup', type=int, default=0)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a new output directory')
    if any(size < 1 for size in args.batch_sizes) or args.limit < 0 or args.prompt_lookup < 0:
        parser.error('Batch sizes must be positive; limit and prompt-lookup must be nonnegative')
    if args.backend == 'original' and args.batch_sizes != [1]:
        parser.error('Original backend requires --batch-sizes 1')
    if args.backend != 'transformers' and args.prompt_lookup:
        parser.error('prompt-lookup requires the Transformers backend')
    from egoannot.config import CAPTION
    if not CAPTION.get('greedy'):
        parser.error('Exact replay comparisons require CAPTION_GREEDY=1')
    paths = sorted(args.requests.glob('[0-9]*.json'))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        parser.error('No saved requests found')
    rows = [json.loads(path.read_text()) for path in paths]
    requests = [
        (row['system'], [
            (kind, (args.requests / value).read_bytes() if kind == 'image' else value)
            for kind, value in row['parts']
        ], row['batch']) for row in rows
    ]
    args.out.mkdir(parents=True)
    from egoannot.qwen_runtime import QwenRuntime, VLLMRuntime
    runtime_class = VLLMRuntime if args.backend == 'vllm' else QwenRuntime
    engine = runtime_class(
        args.model, cpu_threads=8, batch_wait_ms=20,
        max_batch_size=max(args.batch_sizes) if args.backend == 'vllm' else 1,
    )
    if args.backend == 'original':
        # Invoke the unchanged incumbent implementation as a wrapper parity test.
        def original(batch):
            result = []
            for request in batch:
                _, usage = engine.engine(request.system, request.parts, request.batch)
                result.append((engine.engine.last_raw, usage))
            return result
        engine._infer = original
    engine.prompt_lookup_tokens = args.prompt_lookup
    try:
        warmup_start = time.perf_counter()
        engine(*requests[0])
        warmup_seconds = time.perf_counter() - warmup_start
        results = []
        for size in args.batch_sizes:
            engine.max_batch_size = size
            engine.images.clear()
            engine.cache_bytes = 0
            engine.stats = []
            if args.backend == 'vllm':
                # Each timed workload represents new data. Preserve only natural
                # reuse within that workload, not prompts warmed by earlier runs.
                engine.llm.reset_prefix_cache()
                engine.llm.reset_mm_cache()

            def run(item):
                index, request = item
                _, usage = engine(*request)
                raw = engine.last_raw
                return dict(index=rows[index]['index'], raw=raw, usage=usage,
                            exact_match=raw == rows[index]['raw'])

            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=size) as pool:
                outputs = list(pool.map(run, enumerate(requests)))
            result = dict(
                backend=args.backend, prompt_lookup=args.prompt_lookup, input_caches_reset=True,
                batch_size=size, wall_seconds=time.perf_counter() - started,
                load_seconds=engine.load_seconds, warmup_seconds=warmup_seconds,
                requests=len(outputs), exact_matches=sum(row['exact_match'] for row in outputs),
                input_tokens=sum(row['usage']['input_tokens'] for row in outputs),
                output_tokens=sum(row['usage']['output_tokens'] for row in outputs),
                batches=engine.stats, outputs=outputs,
            )
            results.append(result)
            (args.out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
            print(json.dumps({key: value for key, value in result.items()
                              if key not in ('outputs', 'batches')}), flush=True)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
