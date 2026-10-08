# Sparse hand detector and visibility tracking

The current default is **OWLv2 with 2 Hz full-image detection, 1 Hz overlapping discovery views, verified candidate crops, and forward/backward-checked optical flow**. Use `--accuracy fast` / `--hand-accuracy fast` to reproduce the previous version. No MediaPipe, MANO, ACE, or non-commercial weights are used. The smaller RTMDet model and Grounding DINO remain comparison options, not the selected profile.

## License and model provenance

The following candidates were audited before downloading. The selected OWLv2 checkpoint is published under Apache-2.0, as is its Transformers implementation. This permits commercial use under the license terms; source footage has separate terms. No model was trained on these evaluation clips.

- Detector code provenance: OpenMMLab RTMDet/MMDetection and MMPose, Apache-2.0.
  This project implements its own small ONNX preprocessing/postprocessing adapter.
- Checkpoint: `bukuroo/RTMDet-ONNX`, `rtmdet-n-hand.onnx`.
  Distributor model-card license is explicitly **Apache-2.0**, separately verified
  before downloading weights. Repository revision `aa91dbccc283db36b2c250a1d91ba9518db553da`.
  SHA256 `568d3ea97a5b142488366b67e036b6a5cb0a1fef9087a710cb8e66b6979fbac2`.
  Source: https://huggingface.co/bukuroo/RTMDet-ONNX/tree/aa91dbccc283db36b2c250a1d91ba9518db553da .
- Upstream hand detector: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose/rtmdet/hand .
  Model release: https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose/README.md#hand-2d .
  Code license: https://github.com/open-mmlab/mmpose/blob/main/LICENSE .
- Runtime: ONNX Runtime 1.23.2, MIT; optional ONNX 1.19.1 inspection tooling,
  Apache-2.0. Existing NumPy, OpenCV and SciPy remain under existing licenses.
  Runtime license: https://github.com/microsoft/onnxruntime/blob/main/LICENSE .
- The decision to use inference weights relies on the distributor's published
  Apache-2.0 release. It does not assert rights to upstream training datasets.
  No training data is downloaded or used to fine-tune this detector.
- Existing EPIC clips are separate non-commercial research evaluation footage;
  HOT3D clip terms and all existing source hashes remain in their manifests.

A missed detection cannot distinguish absence, occlusion, or detector failure.
Track prediction may bridge a short gap for identity association, but never
counts as an observed visible hand. Image-border proximity is a crop *candidate*,
not confirmation that fingertips are outside the frame.

Additional runtime dependencies: coloredlogs 15.0.1 and humanfriendly 10.0 (MIT), flatbuffers 25.9.23 (Apache-2.0). License sources: https://github.com/xolox/python-coloredlogs/blob/master/LICENSE.txt , https://github.com/xolox/python-humanfriendly/blob/master/LICENSE.txt , https://github.com/google/flatbuffers/blob/master/LICENSE . The temporary ONNX inspection package is removed because it is not needed by the runtime.

## Stronger detector candidate (audited before download)

Grounding DINO Tiny, `IDEA-Research/grounding-dino-tiny`, revision
`a2bb814dd30d776dcf7e30523b00659f4f141c71`: official checkpoint card declares
Apache-2.0; implementation in the already installed Transformers 4.57.3 is
Apache-2.0, upstream GroundingDINO code Apache-2.0.
Sources: https://huggingface.co/IDEA-Research/grounding-dino-tiny ,
https://github.com/IDEA-Research/GroundingDINO/blob/main/LICENSE ,
https://github.com/huggingface/transformers/blob/main/LICENSE .
This candidate is evaluated at sparse cadence; it is a text-conditioned box
detector, not a generative quality judge. No new runtime dependency is required.

OWLv2 candidate (audited before download): google/owlv2-base-patch16-ensemble,
pinned revision cfd3195ba4ea9592eec887ded089f4c08eff231d. Weight card Apache-2.0:
https://huggingface.co/google/owlv2-base-patch16-ensemble/blob/cfd3195ba4ea9592eec887ded089f4c08eff231d/README.md .
Runtime uses existing Transformers Apache-2.0 implementation; no remote code.

## Setup and commands

Use the existing GPU environment from `requirements-speed.txt` (Python 3.12,
Transformers 4.57.3, PyTorch 2.11.0+cu128, torchvision 0.26.0, NumPy 1.26.4,
OpenCV 4.11, SciPy 1.17). The selected profile adds no new dependency to that
environment. The quality VLM's existing `.venv-vllm` environment can run it too.
ONNX Runtime is needed only for the rejected RTMDet comparison, not OWLv2.

```bash
cd /workspace/ego-hands  # replacement accuracy VM; or your local checkout
.venv/bin/python scripts/download_owl_hand.py
.venv/bin/python scripts/benchmark_hand_tracking.py input.mp4 \
  --out artifacts/hands-example --hz 2 --flow --overlay
.venv/bin/python scripts/build_hand_review.py artifacts/hands-example
# Open artifacts/hands-example/index.html locally after copying the directory.

# Existing all-frame numerical/motion QC + hands in the SAME decoding pass:
.venv/bin/python scripts/process_quality_fast.py input.mp4 \
  --out artifacts/quality-hands --mode measured --hands

# Includes the existing compact quality VLM:
.venv-vllm/bin/python scripts/process_quality_fast.py input.mp4 \
  --out artifacts/quality-hands-hybrid --mode hybrid --hands

.venv/bin/python -m pytest -q
```

The downloader pins revision `cfd3195ba4ea9592eec887ded089f4c08eff231d`, saves the
published model card/license, and hashes all local model assets. Inference is
local-only, uses safetensors and verifies the manifest. Weight SHA256:
`e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7`.

## Detection and tracking semantics

- Native 960px OWLv2 preprocessing; source BGR is explicitly converted to RGB.
  The published processor handles resizing/padding; its postprocessor returns
  source pixel `xyxy` boxes. Use no hand world coordinates or depth estimates.
- Fixed prompt: `a photo of a human hand`. This is a text-conditioned detector,
  not a generative judge. Similarity scores are not correctness probabilities.
- A score of 0.15 can start a track; 0.08–0.15 can update a recently confirmed,
  geometrically associated track. Weak detections alone cannot start or sustain
  a track indefinitely. Ordinary IoU NMS plus containment filtering removes
  lower-scoring arm-sized duplicate boxes around a tighter hand box.
- Optical flow groups all track features into one forward/backward KLT call,
  rejects inconsistent features and excessive motion. IDs have no anatomical
  handedness. Motion predictions never count as observed detections.
- At 2 Hz, tracks expire after 0.75 seconds without a strong observation. A
  short gap can retain identity; a long exit/reentry gets a new ID. Occlusion,
  object texture, blur and crossing hands can still cause drift or ID switches.
- `--profile adaptive` in the standalone benchmark permits at most two extra
  recovery crops when a recent track lacks a strong full-frame match. Cropped
  detections must overlap the prior track. Full-image search always continues.
- Sampled coverage divides by detector samples, not propagated video frames.
  The reported intervals are sampling intervals, not exact visibility boundaries.
  Events shorter than 0.5 seconds may be missed at 2 Hz. Increase `--hz` /
  `--hand-hz` for finer temporal resolution and measure the added cost.
- Proximity to an image edge is a *crop candidate*. Neither absence nor border
  proximity can establish that task-relevant fingers were outside the view.

Reports contain every frame's timestamp, boxes, track IDs, observed/predicted
status, source (`full` or `recovery_crop`), confidence, crop candidate and age.
Standalone runs write `hands.json`, `samples.csv`, `summary.json` and optional
`overlay.mp4`. The quality CLI adds `hand_visibility` to `quality.json`, a separate
`hands.json` and hand-stage timing. Detector errors remain explicit; they cannot
silently turn into a no-hands result. Existing quality measurements are preserved.

## Validation and limits

EPIC and HOT3D clips here are development/research examples, not an independent
labeled detection benchmark. Do not call detection coverage precision/recall.
No 3D pose accuracy or occlusion-invariance is established by this work.
The offline review opens raw samples with boxes hidden and exports visible-hand
counts plus notes. Watching model-overlay video marks later counts on that clip
as non-blind. Counts are not bounding-box ground truth; accurate precision/recall
requires independently labeled boxes and a predeclared IoU criterion.

Initial visual comparison: RTMDet Nano missed many clear hands, particularly in
Quest grayscale frames. Grounding DINO found more hands but produced obvious
false boxes on a refrigerator handle and loose gloves. OWLv2 avoided those
particular errors and found partly obscured hands. It still misses some hands
and can produce loose boxes; this is a candidate with inspectable evidence,
not a claim of universally best detection.

On 30 fixed frames, switching OWLv2 from CPU to GPU preprocessing reduced mean
inference from 321 ms to 81 ms. All 43 boxes above the 0.15 threshold remained;
mean best-match IoU was 0.99797 (minimum 0.98818), and no frame's box count changed.
These are optimization-agreement statistics, not detection accuracy against GT.
Both runs used FP16; FP32 can be selected with `--precision fp32` or
`--hand-precision fp32`. The final duplicate filter was added after this paired
preprocessing check and is separately tested.

## Isolated hand-stage benchmark

A100-SXM4-80GB; existing CPU quota ~15 cores; OpenCV one worker and decoder four
threads. Warm model, no response cache; 2 Hz detections and optical flow on every
video frame. Wall time includes probe/decode/detection/tracking, excludes overlay
encoding, source hashing and the quality VLM. One-time model load/warmup was
6.36 seconds for the baseline run. These are individual runs, not p95 latency.

| Clip | Duration | Default hand-stage time | With recovery crops | Default sampled detection coverage | With crops |
|---|---:|---:|---:|---:|---:|
| EPIC P02 | 24 s | 8.42 s | 9.16 s | 56.2% | 56.2% |
| EPIC P03 short | 24 s | 8.54 s | 10.35 s | 64.6% | 66.7% |
| EPIC P03 long | 60 s | 23.07 s | 27.05 s | 91.7% | 92.5% |
| HOT3D 000050 | 5 s | 4.41 s | 4.11 s | 100% | 100% |
| HOT3D 000100 | 5 s | 4.11 s | 4.37 s | 100% | 100% |

Coverage means at least one detector-observed hand at a sampled instant. It says
nothing about missed second hands, false positives or total visible-hand recall.
The small runtime reversal on the 5-second clip reflects run noise/overhead, not
an optimization by adding crops. The selected default uses full-frame detection
without additional recovery calls; the demo also exposes the optional crop mode.

All 263 tests passed (12 environment-dependent skips), including added tests for
flow correctness/abstention, confidence gating, expiry/reacquisition, missing
observations, duplicate suppression, and preservation of existing quality and
motion measurements when tracking is enabled or fails.

## Complete quality pipeline with hands

The existing `.venv-vllm` runtime (Torch 2.9.0, Transformers 4.57.3, vLLM 0.11.2)
processed the 60-second EPIC P03 clip in **45.39 seconds warm** with `--hands`:
32.26 seconds for probe, all-frame QC, motion, evidence and hand tracking;
13.05 seconds for the compact context VLM. Hand tracking occupied 18.94 seconds
inside the first stage; do not add that nested time again. All 72 requested
context rows were assessed, with no unknowns or reported hand/evidence errors.
Reply completeness is not factual accuracy. This timing excludes startup,
captioning, 3D pose and demo video encoding. The standalone hand runtime above
uses Torch 2.11 and should not be added to another runtime's quality timing.

Cold end-to-end startup plus processing was 115.83 seconds on the single-clip
run; the 45.39-second figure assumes resident models. Numerical QC, motion
outputs and all context status labels matched the previous quality-only report
on this clip. Five overlay videos played successfully in the browser check;
blind/exposed labeling and JSON export were also verified.

## Accuracy update: discovering a previously unseen hand

The old recovery stage needed an existing strong track. A second hand that never
crossed the starting threshold could therefore remain missing indefinitely.
The new `balanced` profile is the default for hand-enabled commands:

1. Keep the same full-image detector and 0.15 starting threshold.
2. Every other detector sample, search two overlapping views along the image's
   long axis (65% spans). At 2 Hz this adds discovery once per second. These
   views can discover a hand without an existing track. A discovery-only box
   needs score >=0.20, and boxes clipped by an artificial crop edge are rejected.
3. Up to two low-confidence full-image proposals per sample get a closer view.
   A proposal is accepted as a new observation only when a >=0.15 crop detection
   overlaps it at IoU >=0.25. Weak scores alone still cannot create a track.
4. Compare hand matches against empty-glove, towel and door-handle queries in
   the same model call; only the hand label is accepted. Ungated crops initially
   introduced loose-glove/towel false positives in the EPIC development frames.
   The competing labels removed those inspected errors. This is not a guarantee
   against all lookalikes; worn gloves and unusual hands need separate validation.

No new checkpoint or dependency is introduced. The same pinned Apache-2.0 OWLv2
model and existing license provenance apply. No training on the evaluation data.
No two-hand assumption: an empty frame can return no hands, and visible-hand
counts remain model observations rather than physical ground truth.

```bash
# Updated default, with full-frame search + image-space recovery:
.venv/bin/python scripts/benchmark_hand_tracking.py input.mp4 \
  --out artifacts/hands-balanced --accuracy balanced --hz 2 --flow --overlay

# More frequent discovery and detector refresh, at a higher compute cost:
.venv/bin/python scripts/benchmark_hand_tracking.py input.mp4 \
  --out artifacts/hands-high --accuracy high --hz 4 --flow --overlay

# Previous fast behavior, useful for a matched comparison:
.venv/bin/python scripts/benchmark_hand_tracking.py input.mp4 \
  --out artifacts/hands-fast --accuracy fast --hz 2 --flow

# Quality integration remains in the same decode pass:
.venv/bin/python scripts/process_quality_fast.py input.mp4 \
  --out artifacts/quality-hands-balanced --mode measured --hands \
  --hand-accuracy balanced --hand-hz 2
```

`--accuracy` controls image recovery; `--profile adaptive` is the older optional
track-local recovery and can add more calls. `high` searches discovery crops on
every detector sample; it does not silently change the sampling rate. Frame
propagation still cannot turn an occlusion into an observed hand. Each sequence
resets the discovery schedule, so model warm-up or an earlier clip cannot skip
its first discovery pass. Reports record recovery settings and the prompt list.

The initial 30-frame development inspection recovered previously missed hands
in Quest 000100 at 1.0s and 4.8s, Quest 000050 at 0.2s, and the long EPIC example
at 45s. A hand at Quest 000100 4.0s remained missed in this still-image probe,
and an existing cup false positive persisted at Quest 000050 0.2s. These are
assistant visual checks, not independent human labels or a certified recall
metric. Review the new before/after videos before relying on the model counts.

### Matched rerun on the replacement VM

NVIDIA A100-SXM4-40GB, PyTorch 2.11.0+cu128, Transformers 4.57.3; both profiles
ran at 2 Hz with optical flow, on the same source hashes and sampled frame IDs.
The table counts samples with at least two *strong detector observations*. It
is not precision or recall; it can include existing false positives and misses.

| Clip | Old two-hand rate | New two-hand rate | Old hand-stage time | New hand-stage time |
|---|---:|---:|---:|---:|
| EPIC P03, 60 s | 57.5% | 74.2% | 25.83 s | 36.79 s |
| EPIC P03, 24 s | 31.2% | 33.3% | 9.48 s | 13.63 s |
| Quest 000100, 5 s | 20% | 60% | 4.17 s | 5.46 s |
| Quest 000050, 5 s | 80% | 100% | 4.51 s | 4.95 s |
| EPIC P02, 24 s | 27.1% | 27.1% | 9.38 s | 12.64 s |

Times include probe, decode, hand detection and tracking, but exclude model
startup, video encoding and the quality VLM. The older full-quality 45.39-second
number above does not describe the updated balanced profile. That complete
VLM-plus-hands timing was not rerun on this replacement VM.

271 tests passed, with 12 existing environment-dependent skips. The additional
regressions check discovery of an untracked hand, matching requirements for weak
candidates, artificial-crop truncation, no forced second hand, and cadence reset.
An initial test run on the replacement VM lacked the repository's two lint data
fixtures; after restoring them, the full suite passed.

The optional `high` profile at 4 Hz took 71.58 seconds for the minute-long EPIC
clip and 8.90 seconds for Quest 000100. Two-strong-hand rates across its own
samples were 75.42% and 65%, respectively (different sampling grid from 2 Hz,
so these are not matched-frame accuracy gains). The default stays balanced:
the higher compute cost is substantial for the observed change in detections.
A deadline-scheduling regression was fixed during this test: 4 Hz now yields
exactly 20 scheduled samples over a five-second 30 fps clip instead of drifting
to 19. The existing 2 Hz evaluation grid is unchanged.

Local review bundle: `artifacts/realtime/hand-accuracy-v2/index.html`.
It includes a five-second Quest before/after, a fifteen-second EPIC before/after,
all five full updated overlays, raw sampled frames, every newly gained
second-hand observation, test logs, model manifest and matched-frame JSON.

All 24 sampled frames that gained a second hand observation were subsequently
visually inspected by the assistant; the added observations overlapped visible
hand regions in those contact sheets. This selected development-case inspection
is logged in `validation/visual-audit.json`; it is not independent human box GT
or an audit of every false positive and miss. Both paired comparison videos and
all five full overlays passed local browser playback checks.

The replacement VM uses `/workspace/ego-hands/.venv` for these hand tests. The
older `.venv-vllm` quality environment was not provisioned on this new instance;
the hybrid-quality commands above require an environment with that existing
VLM runtime and checkpoint. Measured-only QC and hand tracking are ready here.
