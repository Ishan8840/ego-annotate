# Episode concurrency benchmark

The optional `--episode-workers 4` overlaps independent episodes while sharing
one resident OWLv2 model. Each episode keeps its own hand recovery cadence,
tracker, optical flow, source probe cache, evidence collector and output directory.
GPU inference remains one image request at a time. Sequential processing remains
the default; no thresholds, image sizes, frame schedules or checkpoints changed.

## Measured results

A100-SXM4-40GB, 23.04 CPU-core quota, Python 3.12, torch 2.11.0+cu128,
Transformers 4.57.3. Five existing public clips: EPIC P03 (60 s), EPIC P03 (24 s),
EPIC P02 (24 s), HOT3D Quest 000100 (5 s) and 000050 (5 s): **118 seconds / 3,540
frames** in total. Input hashes are recorded in the benchmark report.

| Configuration | Median batch wall time | Range | Throughput gain | Runs | Exact outputs preserved |
|---|---:|---:|---:|---:|---|
| Measured quality, sequential | 32.12 s | 31.69–33.32 s | 1.00× | 3 | Yes |
| Measured quality, 2 episodes | 17.27 s | 17.27–17.27 s | 1.86× | 1 | Yes |
| Measured quality, 4 episodes | 14.09 s | 13.91–14.83 s | 2.28× | 3 | Yes |
| Quality + hands, sequential | 92.09 s | 89.49–93.06 s | 1.00× | 3 | Yes |
| Quality + hands, 2 episodes | 59.21 s | 59.21–59.21 s | 1.56× | 1 | Yes |
| Quality + hands, 4 episodes | 56.89 s | 55.94–57.14 s | 1.62× | 3 | Yes |
| Quality + hands, 4 episodes + GPU image batching | 57.27 s | 57.27–57.27 s | 1.61× | 1 | No; experimental only |

The accepted four-worker configuration processed video at
**8.38× real time** for measured quality and
**2.07× real time** with hands. These are batch
throughput figures, not the completion time of an individual one-minute episode.
Individual clips can take longer while sharing resources. The test uses short
excerpts, not a sustained hours-long workload.

Warm batch wall time includes source probing, strict decoding, every-frame image
measurements, motion, evidence JPEG writes, optional balanced hand detection and
tracking, and JSON reports. Model loading/warm-up is separate (about 6 s in the
first sweep). No captions, VLM judgments, demo encoding, downloads or result
comparison time are included. Model initialization is amortized across episodes;
there is no inference response cache.

Three runs cover the selected 1/4-worker configurations. The two-worker and GPU
image-batching configurations have one exploratory run. Repeat order reverses to
reduce first/last-run bias. These are small benchmark samples, not confidence
intervals or a guarantee on other hardware and footage.

## Output checks

Each candidate was compared against the original sequential control by source.
Quality JSON, motion, hand JSON (including every frame, box, confidence, track
status and validity/coverage field), and evidence JPEG hashes must match exactly.
Only `inference_seconds` and `tracking_stage_seconds` are excluded from report
comparison. All accepted runs matched; no frames were skipped or confidence
thresholds changed. This establishes output preservation on these inputs, not
human-verified detection accuracy.

GPU image batching was rejected for the user-facing path: it did not beat the
four-worker serialized-GPU configuration and changed hand outputs on all five
clips. In the exploratory run, hand-count classifications changed on one frame,
track counts on 23 frames, and paired box coordinates differed by up to 26.53
pixels. The script retains this experiment explicitly as `batch4`.

The serialized shared-model run peaked at about 1.54 GiB of allocated PyTorch GPU
memory, close to the sequential run; image batching reached about 4.39 GiB.
These are allocator peaks, not total device memory including CUDA context/cache.
Per-episode inference time under sharing records allocated GPU service time;
tracking and episode wall times also include queue waits. Use `summary.json`'s
`warm_seconds` for batch throughput instead of adding overlapping clip times.

## Use

```bash
python ego.py quality clip1.mp4 clip2.mp4 clip3.mp4 clip4.mp4 \
  --hands --episode-workers 4 --out outputs/concurrent-hands

# CPU measurements and motion only:
python ego.py quality clip1.mp4 clip2.mp4 clip3.mp4 clip4.mp4 \
  --episode-workers 4 --out outputs/concurrent-quality
```

Supported worker counts: 1 (default), 2, 4. For fewer available CPU cores, try 2.
The new option is limited to measured quality, with optional hands. Hybrid/VLM
concurrency is rejected before processing; the replacement VM does not have the
Qwen environment/weights, and annotation/VLM speedups were not measured here.

## Reproduce

```bash
cd /workspace/ego-hands
.venv/bin/python scripts/benchmark_episode_concurrency.py \
  artifacts/speed/public/epic-P03-120-60s.mp4 \
  artifacts/speed/public/epic-P03-60.mp4 \
  artifacts/speed/public/epic-P02-60.mp4 \
  artifacts/speed/public/hot3d-quest-000100.mp4 \
  artifacts/speed/public/hot3d-quest-000050.mp4 \
  --configs cpu1 cpu4 hand1 hand4 --repeats 3 --out outputs/concurrency-benchmark
```

`benchmark.json` contains trial timings, per-episode latency, comparison results,
GPU batch sizes and peak allocated memory; each run retains normal pipeline
reports and evidence. Add `cpu2 hand2 batch4` to the configs to reproduce the
exploratory settings. All configs need their sequential control (`cpu1`/`hand1`).
The benchmark intentionally compares outputs after timed processing.

The original sweep plus repeat data, `comparison.json`, `timing.csv` and logs are
saved locally under `artifacts/realtime/episode-concurrency/` (ignored by Git).
Regression validation: **289 passed, 12 existing missing-corpus skips**.
No new model or dependency was added. Existing commercial weight licenses and
separate source-footage terms remain unchanged; see [hand tracking](hand-tracking.md).
