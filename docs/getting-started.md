# Getting started

Run commands from the repository root. Use an activated environment, or replace
`python` with its full path. `python ego.py --help` lists the everyday commands;
each command also accepts `--help`. The CLI never installs dependencies or
silently downloads models during processing.

## CPU quality

Requires Python 3.11 or 3.12 (3.12 tested). Install FFmpeg using your operating system's
package manager; both `ffmpeg` and `ffprobe` must be on PATH. On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg python3-venv
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-quality.txt
python ego.py doctor
python ego.py quality /path/to/clip.mp4 --out outputs/first-quality
python ego.py review quality outputs/first-quality --out outputs/first-review
```

Open `outputs/first-review/index.html`. Copy the whole review directory to move
it to another computer. Quality review copies the source videos; budget disk
space accordingly. No server, API key or cloud service is required.

## GPU hands

The supplied VM already has the correct environment and model:

```bash
cd /workspace/ego-hands
source .venv/bin/activate
python ego.py doctor --profile hands
python ego.py hands artifacts/speed/public/hot3d-quest-000100.mp4 --out outputs/quest-demo
```

For a new machine with the same preinstalled PyTorch environment, reuse it:

```bash
/venv/main/bin/python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements-hands.txt
python ego.py download-hands
python ego.py doctor --profile hands
```

For a clean NVIDIA CUDA 12.8-compatible environment, the tested torch pair is
2.11.0 / torchvision 0.26.0. Install it before the hand dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-hands.txt
python ego.py download-hands
python ego.py doctor --profile hands
```

The dependency files reuse existing audited packages. They deliberately do not
replace PyTorch or install drivers. `doctor` imports the runtime, executes a
small CUDA operation and checks model presence. Full checkpoint hashes are
verified by the existing detector when it loads.

```bash
# Balanced detector, 2 Hz and optical flow; also builds index.html
python ego.py hands input.mp4 --out outputs/demo

# Same tracking without rendering an MP4 or review page
python ego.py hands input.mp4 --no-overlay --out outputs/tracks

# Previous faster detector, or more frequent high-profile detection
python ego.py hands input.mp4 --accuracy fast --out outputs/fast-demo
python ego.py hands input.mp4 --accuracy high --hz 4 --out outputs/high-demo

# Quality + hand tracking in one decode pass, without a VLM
python ego.py quality input.mp4 --hands --out outputs/quality-hands
```

`--accuracy high` alone keeps 2 Hz; specify `--hz 4` to reproduce the higher
cadence experiment. Track predictions between detector samples do not count
as observed coverage. [Hand detector details](hand-tracking.md).

## Concurrent episodes

```bash
# Measured checks + motion, four episodes in flight
python ego.py quality clip1.mp4 clip2.mp4 clip3.mp4 clip4.mp4 \
  --episode-workers 4 --out outputs/concurrent-quality

# Add the balanced hand detector; one model shared across episodes
python ego.py quality clip1.mp4 clip2.mp4 clip3.mp4 clip4.mp4 \
  --hands --episode-workers 4 --out outputs/concurrent-hands
```

Use 1 (default), 2 or 4 episode workers. This improves total batch throughput;
an individual episode can take longer while sharing resources. Hand recovery,
tracks, source probes and output files remain independent for each episode.
The model still processes one image request at a time, preserving the tested
numerics. GPU image batching remains a separate benchmark experiment because
it changed outputs and did not improve throughput on the test set.

Concurrency is supported for measured quality, including optional hands.
`--mode hybrid --episode-workers 2/4` is rejected before model loading;
captioning and VLM concurrency have not been benchmarked on the replacement VM.
See [measurements and reproduction commands](episode-concurrency.md).

## Optional VLM context

Use this when you need the compact visual judgments in addition to measurements.
It does not produce action captions or calibrated task-success scores. The
replacement VM has no VLM environment/model installed yet.

Keep vLLM separate because its PyTorch version differs from the hand benchmark
runtime. Reuse the existing tested inference package set:

```bash
python3.12 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install vllm==0.11.2 transformers==4.57.3 numpy==1.26.4 scipy==1.17.1 opencv-python-headless==4.11.0.86 cupy-cuda12x==13.6.0
python scripts/download_speed_model.py
python ego.py doctor --profile hybrid
python ego.py quality input.mp4 --mode hybrid --out outputs/context
```

To include hands in this environment, also run `python ego.py download-hands`,
then add `--hands` to the quality command. `doctor --profile hands` checks that
checkpoint separately. Defaults resolve models inside the checkout;
`--model /path/to/qwen` and `--hand-model /path/to/owlv2` override them.

This is the existing compact-context candidate; human accuracy validation is
still pending. [Evidence, benchmarks and review](fast-quality.md).

## Outputs and failures

- Input/output paths are relative to your current directory; default models
  resolve relative to the checkout, even if you invoke `ego.py` from elsewhere.
- Pass multiple videos in one call to reuse a resident model. Same-stem inputs
  are rejected before model loading so they cannot collide in the output tree.
- Existing output directories are rejected. Preserve a failed run's diagnostics
  and choose a new output path when retrying.
- Incomplete hand or VLM analysis returns a nonzero exit status and preserves
  available diagnostics. Check `hand_error`, `evidence_error`, `unavailable`
  and coverage fields before using results.
- `hands` creates overlays and a review page automatically. With `--no-overlay`,
  it only writes reports; `review hands` requires existing overlay videos.
- `review quality` reads quality reports and copies source video; it does not
  rerun inference. Export human labels from the page to retain your judgments.

| Symptom | Next step |
|---|---|
| Import error | Activate the intended environment; run `python ego.py doctor` with the relevant profile |
| CUDA unavailable | Use the configured GPU environment; CPU-only quality still works |
| OWLv2 missing or hash mismatch | Use the pinned `download-hands` command; do not substitute unverified weights |
| No video / duplicate basename | Correct paths or give inputs distinct filenames |
| Out-of-memory with VLM + hands | Run stages separately; use existing advanced scripts for runtime tuning |
| Empty context in measured mode | Expected; add `--mode hybrid` with its configured environment to request it |

## Development

Use the existing full VM environment for the complete suite:

```bash
cd /workspace/ego-hands
.venv/bin/python -m pytest -q
```

A clean development environment also needs the corpus readers used by legacy
modules and pytest. After the GPU setup above:

```bash
python -m pip install mcap==1.5.0 mcap-protobuf-support==0.5.4 pytest==9.1.1
python -m pytest -q
```

Corpus-dependent tests skip when their fixtures are unavailable. For the older
caption-metric workflow, `requirements.txt` includes pycocoevalcap (METEOR needs
a JRE); avoid installing two OpenCV variants in one environment. The machine
snapshot in `requirements-speed.txt` is retained for benchmark reproduction,
not the default clean-install guide.

## Licenses

No new checkpoint is introduced by these shortcuts. The existing OWLv2 weights
and Transformers code are Apache-2.0; the pinned downloader records the model
card, license and file hashes. Qwen3-VL-8B-Instruct weights and vLLM code are
Apache-2.0. NumPy/SciPy and PyTorch retain their existing BSD licenses, OpenCV
its Apache-2.0 license, and CUDA/FFmpeg their build-specific terms. See the
[model audit](hand-tracking.md#license-and-model-provenance) and
[dependency/data inventory](performance.md#data-provenance-and-licenses).

ACE/MANO pose experiments are separate non-commercial research paths, outside
these everyday commands. EPIC evaluation footage is non-commercial; HOT3D has
separate CC BY-SA terms. Your own recordings retain their own rights; review
pages are not a grant of a dataset license.
