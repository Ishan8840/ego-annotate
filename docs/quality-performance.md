# Standalone quality pipeline benchmarks

These are quality-only measurements on the supplied NVIDIA A100-SXM4-80GB VM
(15.36-core CPU quota), using a real contiguous minute of EPIC-KITCHENS P03_02,
original seconds 120–180, 1280×720 at 30 fps. They exclude caption generation,
caption verification, stereo depth, and 3D hand reconstruction.

Measured checks alone took **14.32 seconds per one-minute clip**. A separate
24-second P02 clip took **6.08 seconds**. This mode runs CPU image and integrity
checks over every frame, including sharpness/detail, exposure, frame change,
duplicates, timestamps, and decoder diagnostics. It does not load the VLM.

The complete quality-only pipeline with the existing sequential request schedule
took **362.74 seconds (6:03)** with a warm model. That comprises 280.81 seconds
for measured checks plus the descriptive VLM report, and 81.92 seconds for the
composite rubric. It used vLLM with the existing Qwen3-VL-8B-Instruct BF16 weights;
this is not a new timing of the older Transformers backend.

Parallel windows plus overlapping the independent descriptive and composite
reports reduced this to **115.95 seconds (1:56)**, preserving 88/88 descriptive
dimension-window assessments and 64/64 composite ratings. The composite estimate
changed from 87.50 to 87.08. These are uncalibrated model estimates, not accuracy
or the percentage of usable footage. Same criteria and coverage do not establish
identical judgments or accuracy parity.

A 90.75-second trial also parallelized the criterion groups but lost four
descriptive assessments when a reply exceeded the 1,024-token allowance. That
trial is retained as **rejected for reduced coverage**, not presented as a
successful full-quality speedup.

The final run parallelized the groups and allowed up to 2,048 output tokens so
long JSON replies could finish. It took **92.37 seconds (1:32)** with all 88/88
descriptive assessments and 64/64 composite ratings present: **3.93× faster**
than the sequential baseline. The complete cold run took 138.38 seconds,
including initialization and warm-up, versus 407.09 seconds for the baseline.

| Mode | Warm time for 60 s of video | Scope |
|---|---:|---|
| Measured quality only | 14.32 s | Every-frame CPU checks; no VLM |
| Full quality, sequential | 362.74 s | Measured checks, descriptive report, composite rubric |
| Full quality, batched windows | 115.95 s | Same criteria and coverage |
| Full quality, batched windows/groups, larger output allowance | **92.37 s** | Same criteria and coverage |

The final comparison confirms identical retained measurements, sampled evidence
schedules, first prompts, all 64 ordinal composite ratings, and the resulting
87.50 composite estimate. However, only 49/88 descriptive rows match exactly;
other wording, confidence, concerns, or cited evidence changes. This is not
proof of equivalent real-world accuracy. The original sequential path remains
available and defaults are preserved. The final suite passed **226 tests** with
12 missing-corpus skips. Unit tests verify order, evidence, error propagation,
and rubric aggregation under parallel execution.

## What the faster schedule preserves

The optimizations batch independent windows and criterion groups, overlap two
independent reports, and reuse the validated source probe. They preserve the
sampled images and timestamps, prompts, dimension definitions, rubric weights,
per-frame measurements, and existing evidence validators. The optional larger
output cap lets long JSON replies finish; it does not loosen validation.

Task compliance, success, action correctness, and privacy/safety require supplied
task/collection specifications. No such specifications were supplied for this
benchmark, so those four descriptive dimensions remain explicitly unavailable
in every profile. The remaining 11 dimensions are evaluated across eight
windows; all eight composite criteria are evaluated across those windows.

Existing APIs keep a single worker by default. The new benchmark command exposes
the optional parallel schedule; the existing production command is not silently
switched to a different inference profile. Failed replies and unavailable
assessments remain visible in JSON and in the comparison gate.

## Reproduce

Use the existing separate environments documented in [performance.md](performance.md).
No new weights or dependencies were added. FFmpeg and FFprobe must be on PATH.
Run from `/workspace/ego-speed` on the supplied VM:

```bash
# Measured checks only; no GPU model is loaded.
.venv/bin/python scripts/benchmark_quality.py \
  artifacts/speed/public/epic-P03-120-60s.mp4 \
  --mode measured --out artifacts/realtime/my-quality-measured

# Full quality, original sequential request schedule.
.venv-vllm/bin/python scripts/benchmark_quality.py \
  artifacts/speed/public/epic-P03-120-60s.mp4 \
  --mode full --workers 1 --model models/qwen3-vl-8b \
  --out artifacts/realtime/my-quality-serial

# Full quality, overlapping reports and batching windows.
.venv-vllm/bin/python scripts/benchmark_quality.py \
  artifacts/speed/public/epic-P03-120-60s.mp4 \
  --mode full --workers 8 --model models/qwen3-vl-8b \
  --out artifacts/realtime/my-quality-batched

# Also parallelize criterion groups and allow longer JSON replies.
.venv-vllm/bin/python scripts/benchmark_quality.py \
  artifacts/speed/public/epic-P03-120-60s.mp4 \
  --mode full --workers 8 --parallel-groups --max-output-tokens 2048 \
  --model models/qwen3-vl-8b --out artifacts/realtime/my-quality-groups

.venv/bin/python scripts/compare_quality_benchmarks.py \
  artifacts/realtime/my-quality-serial/epic-P03-120-60s \
  artifacts/realtime/my-quality-groups/epic-P03-120-60s \
  --out artifacts/realtime/my-quality-comparison.json

.venv/bin/python -m pytest -q
```

Warm timing includes source reads, decoded-frame measurements, image extraction,
all model calls and retries, aggregation, and report creation. Cold time also
includes model initialization and a warm-up on an unrelated synthetic image.
Prefix and multimodal caches are cleared before each input episode; natural
reuse within the episode is allowed. There is no response cache or deferred
quality backlog. The two report stages overlap in parallel profiles: do not add
their stage wall times together. Timings are single-run measurements, not a
latency guarantee for every scene or recording length.

Each output directory contains `summary.json`, `timing.csv`, and per-clip
`timing.json`, measured/descriptive reports, the composite report, raw replies,
and coverage data. The comparison checks measured outputs, evidence schedules,
first prompts, and criterion coverage; it reports changed judgments separately.
It does not treat same-model agreement as independent ground truth.

EPIC footage remains separate research evaluation data under CC BY-NC 4.0.
The Qwen weights and vLLM code retain their Apache 2.0 licenses; see the existing
[license inventory](performance.md#data-provenance-and-licenses). Public footage
was not used for training or included in production data. Generated artifacts
are ignored by Git under `artifacts/realtime/`.
