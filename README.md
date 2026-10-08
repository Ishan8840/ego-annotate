# ego-annotate

Process egocentric video into quality measurements, hand-visibility tracks,
and offline review pages. Start with `python ego.py --help`.

The current hand detector is **OWLv2, balanced recovery, 2 Hz + optical flow**.
It is the same detector used in the latest green demos. The video workflow uses
commercially licensed model weights; older ACE/MANO research commands remain
separate. See [licenses and research boundaries](docs/getting-started.md#licenses).

## Setup once

Python 3.11 or 3.12 (3.12 tested) and FFmpeg are required. For CPU quality checks:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-quality.txt
python ego.py doctor
```

For hand detection, use a CUDA environment and follow the
[GPU setup](docs/getting-started.md#gpu-hands). On the configured VM, just activate
`/workspace/ego-hands/.venv` and run commands from `/workspace/ego-hands`.

## Run

```bash
# Quality measurements + image motion; no model or GPU needed
python ego.py quality input.mp4 --out outputs/quality

# Add the current hand detector in the same decoding pass (GPU)
python ego.py quality input.mp4 --hands --out outputs/quality-hands

# Green hand-overlay video + offline review page (GPU)
python ego.py hands input.mp4 --out outputs/hands

# Turn quality reports into an offline human review page
python ego.py review quality outputs/quality --out outputs/review
```

Open `outputs/hands/index.html` or `outputs/review/index.html` in your browser.
Use a **new output directory for each run**. Pass several videos in one command
to load the model once; their filenames must have different stems:

```bash
python ego.py quality clip-a.mp4 clip-b.mp4 --hands --out outputs/batch
```

Optional VLM context uses `--mode hybrid` and the separate
[VLM environment](docs/getting-started.md#optional-vlm-context). The new shortcut
defaults to `measured`; the existing scripts keep their original defaults.

## What you get

| Output | Contents |
|---|---|
| `quality/<clip>/quality.json` | Image/integrity measurements, motion, evidence, unavailable checks; optional VLM context and hand summary |
| `quality/<clip>/hands.json` | Hand tracks when `--hands` is enabled |
| `quality/timing.csv`, `quality/summary.json` | Per-video timing and batch startup information |
| `hands/<clip>/overlay.mp4` | Green hand demo; dashed boxes are unverified predictions |
| `hands/<clip>/hands.json`, `samples.csv` | Tracks, detector observations, confidence and sampling records |
| `hands/index.html`, `review/index.html` | Local review pages with exportable human labels |

Here `quality` and `hands` mean the output directories chosen above. These hand
outputs are **2D boxes and track IDs**, not anatomical left/right labels or 3D
joints. Missing detection can mean occlusion, out-of-frame hands, or a model miss.
Quality measurements and VLM opinions are not a calibrated acceptance score.

The latest measured hand stage took **36.8 seconds per minute** on an A100 40 GB,
excluding model startup, overlay encoding and the quality VLM. Full updated
quality-plus-VLM timing has not been measured on that VM.
[Benchmark scope and limitations](docs/hand-tracking.md).

## Find your way around

| Path | Purpose |
|---|---|
| `ego.py` | Everyday video commands and setup checks |
| `egoannot/` | Reusable processing, geometry, tracking and evaluation code |
| `scripts/` | Advanced runners, benchmarks and experiment tools; [index](scripts/README.md) |
| `tests/` | Unit and integration checks |
| `docs/` | [Setup, workflow references and research history](docs/README.md) |
| `demo/`, `assets/` | Review templates and demo assets |
| `data/` | Small fixtures, label definitions and corpus metadata |
| `models/`, `outputs/` | Local checkpoints and new runs; ignored by Git |
| `artifacts/` | Existing benchmark evidence and generated demos; retained |

The original `python -m egoannot ...`, `make all`, and `scripts/...` commands
remain available. For MCAP captioning and historical pose experiments, use the
[legacy workflow](docs/research-history.md) and [pipeline reference](docs/pipeline.md).

## Develop

```bash
# In the existing full development environment:
python -m pytest -q
# Or: make test PY=.venv/bin/python
```

See [development setup](docs/getting-started.md#development) for dependencies.
The minimal CPU install intentionally does not include all research/test extras.
