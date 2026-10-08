# Near-real-time RGB processing candidate

`scripts/process_rgb_realtime.py` processes new H.264 MP4 episodes with one resident
Qwen3-VL-8B-Instruct BF16 model. It is an opt-in throughput candidate. The existing
four-stage `process_rgb_pipeline.py` remains the conservative full-audit path.
These measurements cover RGB annotation and video quality, not stereo depth or
3D hand reconstruction. They measure completed-episode throughput, not streaming
latency from a live camera.

## What runs

- The original optical-flow boundaries, five full-resolution caption images,
  caption prompt, independent span requests, and two format retries.
- Every-frame image-quality measurements, decoded-frame hashes, duplicate checks,
  timestamp/integrity checks, source hashes, and caption/source binding checks.
- Visual verification of every produced caption, using the original auditor's
  exact 4 Hz, at-most-16-frame sampling and JPEG pixels. Missing captions and
  invalid verifier replies remain explicit failures; coverage uses requested
  spans as its denominator.
- Bounded encoded evidence storage (512 MiB per episode), shared validated video
  probes, one sequential evidence decode, bounded CPU threads, and GPU batching.
  A source changed during processing is rejected. No model responses are cached.

The default `--verification compact` returns five verdicts with source frame
indices. `--verification full` uses the original detailed verifier prompt,
including reasons and exact timestamps, plus one repair attempt for invalid
replies. The compact prompt changes judgments; it is not established as equally
accurate. Neither verifier is independent ground truth or a calibrated score.

The near-real-time mode **does not compute the two subjective VLM quality
rubrics or the composite score**. The score is explicitly null, with status
`not_computed_in_realtime_mode`. These reports can still be produced with the
existing full pipeline. There is no deferred queue hidden from the timing.

## Setup and run

On the supplied A100 80 GB VM, run from `/workspace/ego-speed`. Keep the vLLM
environment separate from the original Transformers environment:

```bash
uv venv .venv-vllm --python /venv/main/bin/python
uv pip install --python .venv-vllm/bin/python \
  vllm==0.11.2 transformers==4.57.3 numpy==1.26.4 scipy==1.17.1 \
  opencv-python-headless==4.11.0.86 cupy-cuda12x==13.6.0
.venv/bin/python scripts/download_speed_model.py

.venv-vllm/bin/python scripts/process_rgb_realtime.py \
  recording01.mp4 recording02.mp4 \
  --model models/qwen3-vl-8b --out artifacts/realtime/new-run \
  --batch-size 16 --cpu-threads 8 --decode-threads 8 --opencv-threads 1

# Original detailed caption verifier, with the same measured checks:
.venv-vllm/bin/python scripts/process_rgb_realtime.py recording01.mp4 \
  --model models/qwen3-vl-8b --out artifacts/realtime/detailed-run \
  --verification full --batch-size 32

# Original full pipeline, including both subjective quality rubrics:
CAPTION_GREEDY=1 .venv/bin/python scripts/process_rgb_pipeline.py recording01.mp4 \
  --model models/qwen3-vl-8b --out artifacts/realtime/full-audit

.venv/bin/python -m pytest -q
```

FFmpeg and FFprobe must be installed. See [the original setup](performance.md)
for the reference environment and dependency checks. The new runner sets greedy
decoding itself. Use a new output directory and unique video basenames. The
default episode limit is ten minutes; split longer recordings first. This bounds
the optical-flow preparation and evidence planning rather than loading hundreds
of hours into memory. Multiple episodes reuse the loaded model but are processed
sequentially. `--trace` optionally saves image-bearing request traces.

For Quest monochrome footage add `--domain general_manipulation
--camera-modality monochrome`. RGB-only processing does not fabricate hand pose
or motion-capture measurements.

## Outputs and reproducibility

Each episode has `captions/{captions,spans}.jsonl`, raw caption calls,
`quality/analysis.json` and its HTML report, `audit/audit.json`,
`audit/corrections.jsonl`, and `audit/verification.json`. The original structural
audit is retained separately. `timing.json`, root `summary.json`, and `timing.csv`
report wall time, duration, format validity, and verification coverage.

Warm timing includes boundaries, source decoding, all requested model calls and
retries, measured quality, and output creation. Model loading and a warm-up on
an unrelated synthetic image are separately accounted for. Prefix and image
caches are cleared before each episode; natural reuse within one episode is
allowed. Cold timing includes loading and warm-up. Downloading test data and
installing dependencies are excluded. Stage times overlap and must not be added.

```bash
.venv/bin/python scripts/compare_realtime_outputs.py \
  artifacts/speed/optimized-p02/epic-P02-60 \
  artifacts/realtime/final-epic/epic-P02-60 \
  --out artifacts/realtime/comparison-p02.json

.venv/bin/python scripts/export_epic_reference.py \
  --captions artifacts/realtime/final-epic/epic-P03-120-60s/captions/captions.jsonl \
  --video-id P03_02 --source-offset 120 \
  --out artifacts/realtime/official-review-p03.json
```

The comparison fails on retained QC or span changes, and separately reports
caption changes, format validity, and verifier agreement for identical captions.
Agreement with the old VLM judge is a diagnostic, not an accuracy score. Official
EPIC narrations provide separate review evidence but are sparse; temporal overlap
alone does not establish correct caption details. Independent reviewed labels are
still needed before promoting this candidate as a no-loss production replacement.

## Licenses and test sources

No new model weights or inference dependencies were introduced. Qwen3-VL-8B-
Instruct weights, pinned at `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, and vLLM
0.11.2 code are Apache 2.0. The [existing license inventory](performance.md#data-provenance-and-licenses)
covers the remaining code and CUDA runtime dependencies.

Public evaluation is separate from production data. EPIC-KITCHENS is CC BY-NC
4.0; clips and official narrations are research evaluation only. HOT3D recordings
are CC BY-SA 4.0; no hand ground truth was loaded for this RGB benchmark. Sources,
offsets, preprocessing, and hashes are retained in the artifact manifests. The
one-minute test uses P03_02 seconds 120–180, encoded at 1280×720 and 30 fps; it is
a real contiguous minute, not repeated short footage. P03_02 seconds 60–84 and
HOT3D 000050 provide additional clips; their contents were not used to tune prompts.

## Measured results

Final compact-verifier run on one A100-SXM4-80GB, with a 15.36-core CPU quota:

| Source | Video | Complete warm processing | Verified/requested spans | Format-valid captions |
|---|---:|---:|---:|---:|
| EPIC P02_01, seconds 60–84 | 24 s | 19.42 s | 14/14 | 9/14 |
| EPIC P03_02, seconds 120–180 | 60 s | **49.93 s** | 36/36 | 13/36 |
| EPIC P03_02, seconds 60–84 | 24 s | 20.31 s | 14/14 | 2/14 |
| HOT3D Quest 000050 | 5 s | 9.93 s | 3/3 | 0/3 |
| HOT3D Quest 000100 | 5 s | 10.03 s | 3/3 | 0/3 |

The three EPIC clips (108 video seconds) took 89.79 seconds warm and 137.04
seconds including loading and warm-up. The two short HOT3D clips took 20.07
seconds warm. Short episodes do not meet real time individually; there are too
few requests to amortize per-call latency. Model initialization plus warm-up adds
about 46–47 seconds per process on this VM; pass many episodes in one invocation.
Across all five clips, every requested span received a caption and a parseable
verification. That is **coverage, not correctness**.

Retained measured QC and span boundaries match the original pipeline exactly on
the paired P02 and HOT3D 000100 clips. All three HOT3D captions match. On P02,
13/14 sentence texts and all action verbs/hand labels match, but format-valid
captions decreased from 10 to 9. The compact verifier agrees with the original
on 44/60 field judgments for identical P02 captions and 13/15 for HOT3D; ten
previously contradicted fields across these clips become supported. The old
judge may itself be wrong, so neither disagreement nor agreement measures real
accuracy. **The no-loss production gate has not passed.**

The unchanged single-request captioner was also run on the new one-minute P03
clip as a control. It produced 10/36 format-valid captions versus 13/36 in the
fast candidate. Sentence text matched on 28/36 spans; verbs, nouns, and hands
each matched on 32/36. Thus the frequent format failures are substantially an
existing model limitation, while semantic-field differences still need review.
On the additional 24-second P03 clip, the original produced 3/14 format-valid captions versus 2/14 in the candidate; 12/14 sentences matched. This control times captioning only and is not a full-pipeline speed baseline.

```bash
CAPTION_GREEDY=1 .venv/bin/python scripts/benchmark_rgb_caption_reference.py \
  artifacts/speed/public/epic-P03-120-60s.mp4 \
  --out artifacts/realtime/my-caption-control --model models/qwen3-vl-8b
```

The detailed verifier experiment took 61.57 seconds for the one-minute clip at
batch 32, with 33/36 valid verifications. Batch 64 plus evidence-format retries
took 67.02 seconds with 36/36 valid verifications. It is available as an explicit
option; the faster compact verifier is not being presented as equivalent.

The final regression suite passed **222 tests**, with 12 missing-corpus skips.
Tests cover original behavior, exact caption/verifier JPEG parity, source-probe
sharing and mutation rejection, strict evidence parsing, retries, coverage
denominators, and preservation of failures in the combined audit.

See `artifacts/realtime/benchmark-report.json` and the final per-episode timing
files. Results are single-machine measurements, not guarantees
for every scene, resolution, GPU, or recording length. Full accuracy parity is
unproven; every retained format failure and model disagreement is reported.

The additional minute can be reproduced separately from production data:

```bash
ffmpeg -nostdin -v error -rw_timeout 60000000 -ss 120 \
  -i https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/P03/P03_02.MP4 \
  -t 60 -an -vf scale=1280:720,fps=30 -c:v libx264 -preset fast -crf 18 \
  -threads 4 artifacts/speed/public/epic-P03-120-60s.mp4
```
