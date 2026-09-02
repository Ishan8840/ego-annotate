# Three changes the methods survey called for

A survey of temporal action segmentation, dense video captioning and
no-reference quality assessment turned up three concrete gaps between this
pipeline and how the field works. All three are now implemented and measured on
the same eight held-out episodes used in
[`heldout_refinement.md`](heldout_refinement.md).

---

## 1. A real pose-free boundary baseline

**The gap.** The only thing the pose-guided cutter had ever been compared
against was a VLM asked to segment the clip itself. That arm front-loads badly
and emits one 16-second "action" per window — beating it says little.

The task has a name: **Generic Event Boundary Detection**. On Kinetics-GEBD
(55K videos, 1.3M boundaries) the strongest *unsupervised* methods use RGB only
— FlowGEBD from optical flow, GraphGEBD from frame self-similarity — reaching
F1@0.05 of 0.71–0.73 with no training.

**What was built.** `egoannot/stages/rgb_boundaries.py`, two pose-free
detectors sharing the pipeline's own trough rule, duration band, refinement and
pose facts, so swapping the signal isolates *where boundaries come from*:

- `rgb_flow` — optical-flow magnitude, the FlowGEBD primitive
- `rgb_tsm` — Foote novelty along a frame self-similarity matrix, the
  primitive GraphGEBD builds on

Both are independent implementations of the published primitives, written to
run without a model download. They are here to be a fair opponent, not to
reproduce a paper's number.

```bash
python -m egoannot spans build --signal rgb_flow
python -m egoannot spans build --signal rgb_tsm
```

**The result — pose is not clearly necessary for boundaries.**

| boundary source | labels | F1@0.3 | F1@0.5 | CIDEr@0.3 | METEOR@0.3 | SODA_c |
|---|---:|---:|---:|---:|---:|---:|
| pose (activity) | 75 | 38% | 27% | **0.736** | **0.166** | 0.053 |
| rgb_flow | 63 | **52%** | 17% | 0.598 | 0.141 | 0.051 |
| rgb_tsm | 68 | **52%** | **31%** | 0.715 | 0.157 | **0.060** |
| VLM segments itself | 91 | 22% | 15% | 0.348 | 0.117 | 0.025 |

A training-free RGB self-similarity detector **matches or beats the pose-guided
cutter** on localisation and sequence coherence. Pose keeps a lead on caption
text quality (CIDEr, METEOR). The old VLM-segments arm is far behind all three,
which confirms it was the strawman the survey said it was — and that the
architecture's central claim had never actually been tested.

Part of the F1@0.3 gap is density: the RGB arms emit fewer labels, so precision
comes easier. That does not explain F1@0.5 or SODA_c, where `rgb_tsm` still
leads.

---

## 2. Standard dense-captioning metrics

**The gap.** The scorer measured format, diversity and grounding — none of
which asks whether a caption is *true*. The pipeline called this an open
problem. It is not; dense video captioning has a standard protocol.

**What was built.** `egoannot/evaluation/captions.py`:

- segments matched to references at IoU ∈ {0.3, 0.5, 0.7, 0.9}, with
  precision / recall / F1 at each
- **CIDEr, BLEU-4, METEOR** over matched pairs, via `pycocoevalcap`
- **SODA_c** — a monotonic alignment over the whole sequence, so the right
  captions in the wrong order score worse than a coherent story

```bash
python -m egoannot dense-eval caps_a.jsonl caps_b.jsonl --labels pose rgb
```

**Read the granularity caveat.** References are the `semantic_segments` the
episodes ship: human-written and timestamped, but *subtask*-level — 29 segments
against 75 atomic labels, so predictions are 2.6× denser than their own
reference. Strict IoU penalises the pipeline for being dense, which is the
thing it is built to be. These numbers compare arms on equal terms; they are
not an absolute grade, and every report prints the granularity ratio to keep
that visible.

---

## 3. PIQE alongside the Laplacian gate — not instead of it

**The gap.** T1 scores sharpness with tiled Laplacian variance. Ego and robot
curation pipelines use published no-reference metrics (PIQE, aesthetic scores);
learned NR-VQA models like DOVER and FAST-VQA correlate ≈0.85 with human
opinion scores.

**What was built.** `egoannot/core/imquality.py` — PIQE (Venkatanath et al.,
2015), vectorised to 59 ms on a 1920×1456 frame, carried as a per-clip channel
(`piqe`, `piqe_active_frac`).

**Two findings that changed the plan.**

*The published threshold is inert here.* PIQE judges only blocks with enough
spatial activity, at a default MSCN block-variance threshold of 0.1. On this
footage — de-identified, compressed retail and domestic interiors — that judges
**3.7% of blocks**. Recalibrated the way the blur floor is, from 262,080 blocks
of real footage, the threshold is **0.008** and 75% of blocks become judgeable.

*It is not a replacement.* Measured on real frames with the same synthetic
degradation the blur floor uses:

| condition | Laplacian | PIQE |
|---|---:|---:|
| real | 5.3 | 2.89 |
| blur σ=3 | 1.1 | 1.87 |
| blur σ=5 | 1.0 | 1.25 |
| noise σ=15 | **4526** | **3.87** |

The Laplacian gate falls correctly with blur but is **catastrophically fooled
by noise** — it rates a noisy frame 4526 against 5.3 for the clean original.
PIQE catches noise but models blockiness and noise rather than sharpness, so it
rates a *blurred* frame as cleaner. They are orthogonal, each covering the
other's blind spot, so PIQE is carried as a second channel rather than swapped
in. There is a test asserting the blur behaviour so nobody swaps them later.

`quality calibrate` now derives a noise gate the same way it derives the blur
floor, with synthetic noise as the positive control:

```
PIQE real frames:      p5=2.56  p50=2.85  p95=3.18
PIQE with σ=15 noise:  p5=3.69  p50=3.88
-> recommended piqe_max = 3.44
```

`piqe_max` ships as `None` — measured, not gated, until calibrated on the
footage in hand.

---

## What this means for the architecture

The pose-first design was justified by beating a baseline the literature would
not consider serious. Against a proper one it is **no longer clearly ahead on
boundaries**, though it still leads on caption text. Combined with the held-out
finding that the vision-only grounding advantage does not replicate, the
defensible claim is narrower than the one the README made: pose gives
boundaries that are *free, deterministic and independently checkable*, not
boundaries that are *better*.

The cheapest next test is whether `rgb_tsm` and pose disagree in the same
places — if they are complementary rather than redundant, an ensemble is worth
more than either.
