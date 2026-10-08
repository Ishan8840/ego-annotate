# Public ego-video performance evaluation

This benchmark covers **RGB video annotation and quality analysis** in
`ego-annotate`. It does not time or change the separate stereo 3D hand-pose
pipeline. All timings below use one NVIDIA A100-SXM4-80GB. The container exposes
128 host CPU threads but has a 15.36-core CPU quota. Cold time includes model
loading; warm time starts with the model ready. Dataset download and environment
installation are excluded.

## Measured results

Full four-stage EPIC P02_01 run (24 seconds, 720 frames, 14 caption spans):

| Profile | Warm processing | Cold processing | Semantic output parity |
|---|---:|---:|---|
| Reference, 128 CPU threads | 844.02 s | 852.79 s | Reference |
| Conservative, 8 CPU threads | 672.79 s | 682.07 s | All five files identical |

The warm speedup is **1.25×**, or **20.3% less processing time**. Both runs retain
59 model requests, all input frames, and all existing checks. These are measured
single-run timings, not latency guarantees. The reference already uses a shared
model and overlapping independent stage chains; this comparison therefore does
not quantify the additional benefit of eliminating repeated model loads from
separate legacy commands. It is not comparable to the older caption-only demo
timing. Full processing remains much slower than real time on this profile.

Fixed 12-request replay from the public EPIC run:

| Transformers batch size | Workload time | Exact raw replies |
|---|---:|---:|
| 1 | 100.54 s | 12/12 |
| 2 | 142.75 s | 8/12 |
| 4 | 146.58 s | 7/12 |
| 8 | 141.89 s | 7/12 |

The unchanged original `QwenLocal` implementation also reproduced all 12 replies
(99.98 seconds), validating the persistent wrapper against the incumbent.
Prompt lookup with five proposed tokens took 69.75 seconds but matched only 8/12
replies. It remains experimental despite the speedup.

Batching is rejected as a default: it was slower and changed outputs. The
first 12 reference requests used 135.62 seconds of logged service time; replay
excludes an explicit warm-up, so the complete-run comparison above is the stronger
speed measurement.

Held-out HOT3D Quest clip 000100 (5 seconds, 150 frames):

| Profile | Warm processing | Cold processing | Semantic output parity |
|---|---:|---:|---|
| Reference, 128 CPU threads | 228.07 s | 237.86 s | Reference |
| Conservative, 8 CPU threads | 174.74 s | 183.70 s | All five files identical |

The held-out warm speedup is **1.31×**, or **23.4% less processing time**.
All **80 model request/reply records** also match exactly after accounting for
concurrent request ordering (`artifacts/speed/report/request-parity.json`).
Together, these paired runs cover 29 seconds and 870 frames across two datasets;
they do not establish performance or accuracy on every future recording. P03_02
and HOT3D 000050 were downloaded as additional fixtures but are not included in
these measured results. The final test suite passed **203 tests**, with **12
skipped** because their corpus fixtures are absent.

The EPIC baseline retained four format-invalid captions out of fourteen and
existing visual-audit findings. An unchanged output can still be incorrect.

Final vLLM 0.11.2 replay, with prefix and image-feature caches reset before each
workload (`artifacts/speed/replay-vllm-new-data/results.json`):

| Engine / batch | 12-request time | Exact raw replies | Decision |
|---|---:|---:|---|
| Transformers / 1 | 100.54 s | 12/12 | Conservative default |
| vLLM / 1 | 36.05 s | 6/12 | Experimental |
| vLLM / 4 | 23.81 s | 6/12 | Experimental |

The fastest candidate gives **4.22× inference throughput** on this fixed workload,
but half the replies change. This is not a demonstrated full-pipeline speedup or
an accuracy-preserving replacement. The changed replies include structured
observations/evidence, not merely whitespace. vLLM initialization took 34.27
seconds with compiled kernels already cached; its excluded warm-up took 4.68
seconds. Earlier diagnostic runs with warmed prompt caches are retained in the
artifacts but excluded from these results. No faster experimental mode was
promoted to the production default.

## Execution and regression gate

`scripts/process_rgb_pipeline.py` runs all four existing stages with one resident
Qwen3-VL-8B-Instruct BF16 model:

1. RGB motion boundaries, five images per caption span, and existing retries.
2. Measured video checks over all frames plus the existing visual quality analysis.
3. The existing eight-dimension composite rubric.
4. Source hash/timestamp checks and the existing image-based caption audit.

The two independent stage chains can overlap CPU preparation. The default GPU
queue processes one request at a time. Decoded-image caching is bounded to 128 MiB
and keyed by the JPEG contents; model replies are never cached. Each caller has
its own raw reply, preventing concurrent stages from reading another stage's
response. Model weights, prompts, input images, resolution, output token limit,
retry rules, and audit criteria remain the same. Stage wall times include queue
waiting and overlap, so they must not be added together.

`compare_speed_outputs.py` fails closed unless captions, spans, measured and
visual quality results, composite scores, and audit results match. It excludes
only generated timestamps, output paths, elapsed time, and usage accounting.
Source hashes, evidence timestamps, uncertainty, errors, and failed checks are
retained. Exact output parity is evidence against a regression on the evaluated
inputs, **not proof that the incumbent labels or its VLM quality score are
correct**. There is no independent human semantic ground truth in this speed
benchmark.

Alternate inference engines, multi-request GPU batches, and prompt lookup require
`--experimental`; none should be deployed merely because they are faster. Their
raw-output differences are recorded by `replay_speed.py`. The replay workload
uses the exact saved prompt and image bytes, one excluded warm-up request, and no
response cache. The final vLLM replay resets both prefix and multimodal caches
between timed workloads, retaining only natural input reuse within a workload. Its timings are inference-only, not full episode processing.

## Reproduce on the supplied VM

Run from the repository root. FFmpeg and FFprobe must be on PATH. The tested VM
already has torch 2.11.0+cu128 and torchvision 0.26.0+cu128 in `/venv/main`.
Use `python -m pip` for this inherited environment so its installed base packages
are included in dependency resolution:

```bash
/venv/main/bin/python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-speed.txt
.venv/bin/python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__)'
.venv/bin/python scripts/download_speed_model.py
.venv/bin/python scripts/download_speed_data.py
.venv/bin/python -m pytest -q

CAPTION_GREEDY=1 .venv/bin/python scripts/process_rgb_pipeline.py \
  artifacts/speed/public/epic-P02-60.mp4 \
  --model models/qwen3-vl-8b --out artifacts/speed/my-epic-run \
  --cpu-threads 8 --trace

CAPTION_GREEDY=1 .venv/bin/python scripts/process_rgb_pipeline.py \
  artifacts/speed/public/hot3d-quest-000100.mp4 \
  --model models/qwen3-vl-8b --out artifacts/speed/my-hot3d-run \
  --cpu-threads 8 --domain general_manipulation --camera-modality monochrome --trace

.venv/bin/python scripts/compare_speed_outputs.py \
  artifacts/speed/reference-p02/epic-P02-60 \
  artifacts/speed/my-epic-run/epic-P02-60 \
  --out artifacts/speed/my-parity.json
```

Each output directory must be new. Multiple videos can be supplied in one command
to amortize loading. `--workers` overlaps independent episodes; it does not bypass
the single-request default GPU queue. Explicit `CAPTION_GREEDY=1` is required for
reproducible parity comparisons. Stochastic generations cannot be expected to
match when request scheduling changes. Use `--trace` only for benchmarking: it
saves source images and prompts, which need the same access controls as raw data.

## Data, provenance, and licenses

`artifacts/speed/public/manifest.json` records original URLs, source selection,
preprocessing, hashes, and development/held-out assignments. Selection happened
before inference results were inspected. Evaluation videos are separate from the
older demo. They were not used for training.

- EPIC-KITCHENS P02_01 and P03_02: seconds 60–84, encoded at 1280×720, 30 fps.
  [Official downloader](https://github.com/epic-kitchens/epic-kitchens-download-scripts);
  [dataset license](https://epic-kitchens.github.io/2025.html): CC BY-NC 4.0.
  These clips are kept in separate research evaluation artifacts, not production
  training data or commercial deliverables. Commercial use requires appropriate
  permission from the rights holder.
- HOT3D Quest3 clips 000050 and 000100: five seconds each, left camera, rotated
  upright. [Official clip documentation](https://github.com/facebookresearch/hot3d/blob/main/hot3d/clips/README.md).
  Recording license: CC BY-SA 4.0. Only images and camera metadata are read; hand
  annotations are excluded. The dataset revision is pinned in the downloader.
- Qwen3-VL-8B-Instruct weights: Apache 2.0, pinned revision
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
  [Model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).
  Transformers and Accelerate inference code: Apache 2.0; PyTorch: BSD-style;
  NumPy/SciPy: BSD; OpenCV 4.11: Apache 2.0. No new model weights are introduced.
- Optional vLLM 0.11.2 inference code: [Apache 2.0](https://github.com/vllm-project/vllm/blob/v0.11.2/LICENSE).
  It uses the same Qwen weights, with their Apache 2.0 license. Keep it in a
  separate environment: its torch dependency differs from the reference VM.
  CuPy is [MIT-licensed](https://github.com/cupy/cupy/blob/v13.6.0/LICENSE);
  use its NumPy-1.26-compatible 13.6.0 release in this environment.
  NVIDIA CUDA runtime dependencies retain their vendor licenses.

Full environment capture, request traces, logs, source hashes, and machine details
live under `artifacts/speed/`. Large videos, traces, and weights are ignored by Git.

Optional inference experiments (separate environment; not the production default):

```bash
uv venv .venv-vllm --python /venv/main/bin/python
uv pip install --python .venv-vllm/bin/python \
  vllm==0.11.2 transformers==4.57.3 numpy==1.26.4 scipy==1.17.1 opencv-python-headless==4.11.0.86 cupy-cuda12x==13.6.0
CAPTION_GREEDY=1 .venv-vllm/bin/python scripts/replay_speed.py \
  --requests artifacts/speed/reference-p02/requests \
  --out artifacts/speed/my-vllm-replay --backend vllm \
  --limit 12 --batch-sizes 1 4
CAPTION_GREEDY=1 .venv/bin/python scripts/replay_speed.py \
  --requests artifacts/speed/reference-p02/requests \
  --out artifacts/speed/my-batch-replay --limit 12 --batch-sizes 1 2 4 8
```

## Independent review references

`artifacts/speed/epic-official-review.json` aligns the official EPIC participant
narrations to the generated P02 captions using the 60-second source offset.
The reference is fetched **after** inference, never included in model prompts.
The official annotation revision and source CSV hash are recorded. Narrations
are sparse and do not label every caption detail; temporal overlap is not a
semantic accuracy metric. This file supports a later independent human review:

```bash
.venv/bin/python scripts/export_epic_reference.py \
  --captions artifacts/speed/reference-p02/epic-P02-60/captions/captions.jsonl \
  --video-id P02_01 --source-offset 60 \
  --out artifacts/speed/epic-official-review.json
```

The [official annotation repository](https://github.com/epic-kitchens/epic-kitchens-100-annotations)
and its annotations are CC BY-NC 4.0; the same research-only restriction applies.

## Final environment validation

A final dependency audit removed three local setup overrides (`cuda-bindings`,
`cuda-toolkit`, and `setuptools`) so the environment inherits the VM's compatible
12.9.7, 12.8.1, and 81.0.0 versions. PyTorch, model weights, and processor versions
were unchanged. `python -m pip check` reports no broken requirements. The setup
command was checked with `pip install --dry-run`, and the final suite again passed
203 tests with 12 fixture-dependent skips. A further 12-request replay reproduced
all 12 original replies exactly in 98.35 seconds. Both environment snapshots and
the final check logs are retained; full-episode timings above are from the earlier
paired runs.
