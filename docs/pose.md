# Hand pose from RGB

The pipeline's hand pose is an annotation that ships inside the mcap. This
stage produces a second one from the head camera's own imagery, so the two can
be compared and either can drive everything downstream.

```
mcap ──▶ export ──▶ ACE-Ego-Hand ──▶ camera→world ──▶ /pose/{side}_hand{,_joints,_quat}
         mp4 + K      21 joints         measured axis
```

```bash
python -m egoannot pose export --all     # mcap -> mp4 + cam.json + frame times
python -m egoannot pose run    --all     # estimate, lift to world frame
python -m egoannot pose eval   --all     # agreement with the shipped pose
python -m egoannot pose stereo --all     # both sources vs stereo depth
```

Estimator: [ACE-Ego-Hand](https://github.com/ggxxii/ACE-Ego-Hand) (MIT), the
**K-given** checkpoint. EgoStandard ships real per-frame intrinsics, and the
K-free configuration exists precisely to work without them — the paper's own
border analysis shows its fitted camera costs 3-5x translation error toward
the frame edge, so handing the model the calibration it can trust is strictly
more information.

## Why the integration is small

Everything downstream of pose reads exactly six arrays. Replace those and the
rest of the pipeline is unchanged — the same move `--signal rgb_flow` makes
for boundaries.

Three conventions had to line up, and only one needed work:

- **Joint order: nothing to do.** ACE emits 21 joints in OpenPose order
  (`mano_utils._MANO_TO_OP`), which is the MediaPipe layout `features.py`
  already assumes. `TIP[thumb]=4` and `TIP[index]=8` address the same two
  points in both streams, so aperture is directly comparable. A test pins the
  upstream constant, because a silent reorder would leave "aperture" measuring
  two different fingers.
- **Hand identity: fixed by construction.** Slot 0 is left, slot 1 is right,
  with no detector in the loop, so the handedness flips that plague egocentric
  hand detection cannot occur.
- **Optical axis: measured, not assumed.** ACE predicts +z-forward;
  `geometry.camera_convention` decides this corpus's sign from where the
  wrists actually sit. It measured +1 on all 8 episodes (19 deg to +z against
  161 deg to -z).

Absent hands stay absent. The features stage propagates validity rather than
imputing it, so a frame the model declines is written as NaN.

## Two bugs worth recording

**`joints_cam_direct` is already absolute.** Upstream composes it as
`direct_rootrel + direct_wrist_cam` (`memory_projector.py:818`) and dumps the
sum, while dumping the same wrist separately alongside. The docstring reads
"root-relative joints plus a directly regressed camera-space wrist", which
describes how it is built, not what it holds. Adding the wrist again placed
every hand at twice its depth — 0.83 m instead of 0.41 m — while still looking
like a perfectly plausible hand. No downstream metric would have caught it.

**Canonical coordinates are dimensionless.** `features.canonical` divides by
hand scale, so shape error scaled by 1000 is not millimetres. That reported
143 mm for what is 12.3 mm.

## Seeing it

```bash
python -m egoannot pose viewer          # -> artifacts/reports/pose-viewer.html
```

One self-contained HTML file: both skeletons drawn on the frames they
describe, a scrubber, per-hand toggles, and the aperture trace with a
playhead. Amber is the shipped annotation, teal the estimate, throughout.
Both streams are reprojected through the same intrinsics, extrinsics and
measured axis sign, so separation on screen is the poses differing rather than
two cameras.

A **before PnP** toggle draws the estimator's original placement as a dashed
ghost in the same colour, so the correction is visible rather than only
tabulated: on `wipe_speaker` it moves the wrist a median 39 display px out of
440, and the two sources go from visibly offset to overlapping. The readout
reports that shift per frame.

**Playback is real-time.** Wall-clock elapsed is mapped onto each episode's own
timestamps, so a 26-second clip takes 26 seconds and a slow decode drops a
frame instead of stretching the timeline. Frames are decimated to `--fps`
(default 12, snapped to an integer divisor of the source rate, so 15 fps on
30 fps footage) purely to keep the page one file -- the timing is real, the
smoothness is what gets traded. That costs size: 31 MB for all 8 episodes at
15 fps, against 7 MB for the earlier 64-frames-per-episode flip-book, which
compressed a 31-second episode and an 11-second one into the same few seconds
of playback.

## Accuracy

Eight held-out episodes, 173 s, both hands, ~100% coverage. Agreement is
reported in three parts because a single MPJPE hides which half is wrong.

**Final configuration — 1280 px encode, per-frame PnP placement:**

| | estimated | shipped |
|---|---|---|
| coverage | 100% | 100% (dense by fiat) |
| shape (canonicalised MPJPE) | 13.3 mm | — |
| aperture MAE / correlation | 11.6 mm / 0.78 | — |
| wrist placement vs shipped | 56.7 mm | — |
| orientation vs shipped | 13° | — |
| **stereo MAE** | 0.129 m | **0.115 m** |
| **stereo correlation** | **+0.359** | +0.299 |
| **stereo within 5 cm** | 47% | 48% |
| wrist jitter | **0.10 mm/frame²** | 1.23 mm/frame² |

Encode width, with the model's raw translation (no PnP), which is how the
resolution sweep was run:

| encode width | placement | shape | aperture MAE | aperture r | stereo MAE | stereo corr | <5 cm |
|---|---|---|---|---|---|---|---|
| 672  | 111.8 mm | 12.8 mm | 12.0 mm | 0.77 | 0.160 m | +0.291 | 12% |
| 832 | 91.3 mm | **12.3 mm** | **10.6 mm** | **0.83** | 0.150 m | +0.255 | 16% |
| 1280 | **74.7 mm** | 13.3 mm | 11.6 mm | 0.78 | **0.149 m** | **+0.307** | **28%** |
| *shipped* | — | — | — | — | *0.115 m* | *+0.299* | *48%* |

The two streams **agree on what the fingers are doing and disagree on where
the hand is**: 12.3 mm of shape error on an 84 mm hand, against 91 mm of
placement error. A single number would have reported "~90 mm, bad". Most of
that placement gap turns out to be one correctable depth error — see below.

### Placement: the estimator's translation is wrong, and replaceable

Drawn on the footage, the estimated skeleton is visibly offset and undersized
against the shipped one — a fixed transform, not jitter. A similarity fit
confirms it: one scale plus shift absorbs 77% of the disagreement, while the
shipped pose needs no such transform to sit on the estimator's own 2D anchors
(19.7 → 20.1 px). So the shipped overlay is right and the estimated one is
displaced.

That splits the estimator cleanly into what works and what does not. Its
root-relative joints agree with the shipped pose to 13 mm, and its 2D anchors
are where the shipped pose reprojects to within 20 px. Only the camera-space
translation it hangs them on is wrong. So keep the first two and recompute the
third: treat the root-relative joints as a rigid body and solve, per frame,
the pose that reprojects them onto the anchors, through the intrinsics the
prediction was made with. That is the paper's mixed-PnP idea applied to the
direct heads, and it uses **only the model's own outputs** — no annotation,
no shipped pose, so it works on footage where neither exists.

Three arms, on identical 1280 px predictions (`pose compare`):

| arm | place mm | \|dx\| | \|dy\| | \|dz\| | stereo MAE | stereo corr | <5 cm |
|---|---|---|---|---|---|---|---|
| raw | 74.7 | 19.9 | 39.5 | 41.8 | 0.149 m | +0.307 | 28% |
| one depth scalar | 116.3 | 19.9 | 39.5 | 75.0 | 0.258 m | +0.238 | 19% |
| **per-frame PnP** | **56.7** | **9.9** | **8.6** | 50.5 | **0.129 m** | **+0.359** | **47%** |
| *shipped* | — | — | — | — | *0.115 m* | *+0.299* | *48%* |

PnP solves 100% of frames at 0.9–2.1 px reprojection residual and reaches
**parity with the shipped annotation on the independent referee**: 47% within
5 cm against 48%, and a better correlation with measured stereo depth. Median
absolute error stays 12% higher, and the residual is depth alone — lateral
error drops to ~9 mm while |dz| stays at 50 mm.

The depth scalar is *worse* than doing nothing at this resolution. At 1280 the
raw bias is already near zero, so fitting one overcorrects. Resolution and
scalar bias correction are substitutes, and applying both is worse than
either.

### The last 12% is a size/depth ambiguity, not a bug

What remains after PnP is depth alone: |dz| 50.5 mm against |dx| 9.9 and
|dy| 8.6. That is not solver noise — at 1280 a 1 px anchor error on a 235 px
hand costs about 2 mm of depth, not 50.

It is a disagreement about how big the hand physically is. PnP places an
object by matching its known metric size to its apparent size, so a size error
becomes a depth error of the same fraction. Measured:

| | estimated | shipped | ratio |
|---|---|---|---|
| hand scale, wrist to middle MCP | 94-99 mm | 81-87 mm | 1.140 |
| placed depth | | | 0.860 |

The product is 0.98. The estimator thinks the hand is 14% bigger and places it
14% nearer, and both readings project to the same pixels — which is exactly
the monocular scale ambiguity, not a defect in either stream.

Correcting the size to match the shipped pose (`hand_scale`) confirms the
coupling and shows it cannot be won on both sides at once:

| hand_scale | aperture MAE | placement | stereo MAE | <5 cm |
|---|---|---|---|---|
| **1.0 (default)** | 11.6 mm | **56.7 mm** | **0.129 m** | **47%** |
| 1/1.140 | **9.47 mm** | 93.1 mm | 0.153 m | 37% |

Shrinking the hand fixes aperture, because aperture is a metric length read
off the hand, and breaks depth, because PnP then places it nearer still. The
default keeps depth parity; `hand_scale=0.877` is the right choice if aperture
is what you care about, which for `events` and `spans` it may well be.

**A real fix needs an external metric reference.** Monocular vision cannot
resolve size from depth — no amount of model improvement will, because the two
are genuinely unobservable separately. The candidates on this corpus are the
stereo pair (absolute depth at the hand pixel, which would fix scale per
episode without touching the shipped annotation) or a measured per-operator
hand size. Neither is implemented here.

## Validated against real ground truth (ARCTIC s05)

Everything above measures *agreement*, not accuracy. ARCTIC settles it: native
mocap MANO ground truth, an egocentric camera, and subject **s05** — protocol
p2's val list, 34 sequences, which is exactly the split the estimator's own
paper reports as its ARCTIC test set, so it is held out rather than memorised.

The same runner, lift and PnP solve execute here; only the input differs.
8 sequences, ~5,900 frames, both hands, medians over on-screen frames using
the paper's gate (a hand counts if any GT joint projects inside the image at
depth > 1 cm):

| | this pipeline | ACE-Ego-Hand paper |
|---|---|---|
| MPJPE | **12.3 mm** | 15.26 mm |
| PA-MPJPE | **6.5 mm** | 7.47 mm |
| EPE2D | 11.5 px | 9.18 px |
| absolute wrist error | 19.3 mm | — |
| depth bias | **+6.2 mm** | — |
| on-screen frames | 89% | — |

Per-hand MPJPE runs 8.6–17.0 mm across 16 hands, so the figure is not carried
by a few easy sequences — and `box_grab_01`, the one sequence measured first,
turned out to be the *worst* of the eight.

Not an identical protocol: the paper reports a coverage-penalised mean over
its full test set, this is a median over on-screen frames of 8 sequences. The
claim is that the pipeline reaches published-quality accuracy, not that it
beats the published numbers.

### Does this improve on the estimator, or just reimplement it?

The estimator has its own translation decode (MANO + mixed-PnP). This project
never used it — the importer was built around the auxiliary `direct` heads
while MANO was still an unmet licence blocker, and the choice was never
revisited once that cleared. So "our PnP beats raw" was measured against the
estimator's *secondary* read, not its headline one.

Scored properly, on the same ARCTIC frames:

| read | MPJPE | PA-MPJPE | wrist |
|---|---|---|---|
| ACE primary (MANO + mixed-PnP) | **11.6 mm** | 6.9 mm | 50.7 mm |
| **this pipeline** (direct + PnP) | 12.3 mm | **6.5 mm** | **19.3 mm** |
| MANO articulation + our PnP | 12.4 mm | 6.9 mm | 22.4 mm |

The contribution is **placement: 50.7 -> 19.3 mm, 2.6x better**, at a cost of
0.7 mm of wrist-aligned MPJPE. Their MANO articulation is slightly the better
of the two; our metric placement is much better.

Mixing them does not help. Feeding MANO's joints through our PnP is worse on
every column, for two reasons: the solve re-fits rotation and its
image-derived orientation is worse than MANO's, and the direct 3D head agrees
with the direct 2D head more closely than MANO's joints do, both being reads
off the same decoder branch.

So for absolute hand position — the thing that matters for placing a hand
relative to an object — this pipeline is meaningfully ahead of the released
model. For articulation it is a wash.

### This overturns the hand-size conclusion

Against true joints the estimator's hand scale is **1.010** — within 1%,
consistently across all 16 hands. On EgoStandard the same estimator measured
**1.140x the shipped pose**. True wrist-to-middle-MCP is 90-92 mm here; the
estimator predicts 90-93 mm on ARCTIC and 94-99 mm on EgoStandard (a
different operator); the shipped EgoStandard pose says 81-87 mm, smaller than
any of them.

So the estimator is right about hand size and **the shipped EgoStandard
annotation is the one that is wrong**, by about 13%. Two consequences:

* `hand_scale` should be left at 1.0. Setting it to 0.877 does not correct an
  error — it corrupts a correct hand size to match a biased reference. The
  apparent aperture improvement it bought (11.6 -> 9.47 mm) was agreement with
  a biased annotation, not accuracy.
* The depth bias is +6.2 mm against truth, against the ~70 mm measured on
  EgoStandard before PnP. An estimator that is near-unbiased on real ground
  truth makes the EgoStandard residual more likely a property of that corpus's
  annotation and its weak stereo referee than of the estimator.

This is why agreement is not accuracy: measured against the shipped pose alone,
the correct stream looks like the broken one.

### Smoothness: marginally better, not the order of magnitude first reported

An earlier version of this document claimed the estimate was 12x smoother
than the annotation. That number was measured in the CAMERA frame and is an
artefact of how it was taken. Our pose is predicted in camera frame and
rotated to world with the shipped extrinsics, so measuring it back in camera
frame cancels that transform exactly and returns the raw network output. The
shipped pose is native world-frame, so the same round trip adds the
extrinsics' own jitter to it. The comparison comes out 12:1 because one side
carried camera noise and the other had it algebraically removed.

Measured on each stream's own timebase in the world frame -- which is the
frame `features.py` reads for wrist speed, acceleration and jerk -- the
median second difference of the wrist is:

| | shipped | estimated | ratio |
|---|---|---|---|
| wrist jitter, mm/frame² | 2.61 | 2.13 | **1.2x** |

Per-episode the ratio runs 1.0x to 2.3x. The estimate is slightly the smoother
of the two, and that is all. Any claim resting on temporal stability should
use this figure.

### Hand orientation agrees to ~13°, once the frames are reconciled

Raw geodesic error between the two orientation streams is 100° (left) and
167° (right), which looks catastrophic and is a convention mismatch. One
fixed rotation per hand side — pooled over all 8 episodes and ~5000 frames
each, so one constant apiece rather than a per-episode fit — brings it to
**12.8°** and **13.4°**. Two different axes for the two hands, which is what
MANO's mirrored left/right canonical frames predict.

Those constants are now applied in the lift (`ace.ORIENT_OFFSET`), so the
quaternion lands in this corpus's hand frame and `spans` can read a twist off
it. They are the one thing in this stage calibrated against the shipped
annotation, because a convention has no ground truth to recover it from: they
express "the frame EgoStandard uses", not "the correct frame", and would need
refitting on another corpus. Joints and aperture never pass through them.

### Resolution moves the bias, not the spread

Fitting one global depth offset across all episodes (fitted and scored on the
same samples, so an upper bound):

| width | offset | MAE raw → corrected | <5 cm raw → corrected |
|---|---|---|---|
| 672 | −0.111 m | 0.163 → **0.119** | 11% → **39%** |
| 832 | −0.070 m | 0.173 → 0.156 | 14% → 35% |
| 1280 | **+0.022 m** | 0.182 → 0.181 | 26% → 25% |
| *shipped* | *−0.013 m* | *0.104 → 0.102* | *53% → 56%* |

The offset is monotonic in resolution and crosses zero near 1150 px. What the
model gets wrong at low resolution is a systematic depth **scale**, consistent
enough that one scalar removes most of it; what it gets wrong at high
resolution is **spread**, which no offset can touch.

So resolution and bias correction are **substitutes, not complements**.
Raising resolution to 1280 removes the bias and leaves nothing for a
correction to do. The best absolute depth from this estimator is 672 with a
−0.111 m offset (0.119 m MAE, 39% within 5 cm) — not the highest resolution,
and not the uncorrected default.

### What this does not establish

**Parity is on two of three stereo measures, not all three.** With PnP
placement the estimate matches the shipped pose on within-5 cm (47% vs 48%)
and beats it on correlation (+0.359 vs +0.299), but median absolute error
stays 12% higher (0.129 vs 0.115 m). Calling it "as accurate" is fair for
coverage, shape, aperture, orientation and temporal stability; on absolute
depth it is close but still behind.

**The referee is a noisy ruler.** Disparity coverage runs ~40% on these
scenes, and the shipped pose only reaches 53% within 5 cm against it, so part
of both arms' error is stereo failing at hand boundaries — the same failure
`docs/pipeline.md` already records. This test can separate a 70 mm systematic
bias from zero; it cannot resolve a 20 mm difference between two good poses.

**832 is not "their" resolution beaten by ours.** ARCTIC trains at 672x480 and
H2O/OakInk2 at width 832, so two of the three arms are trained widths and 1280
is outside anything the model saw. Accuracy does not track distance from the
training distribution: 1280 wins placement, 832 wins articulation, 672 wins
only after correction.

**Two results here came from a single episode before they were checked.**
Both moved substantially once measured on all eight — a 40% MAE improvement at
1280 shrank to under 1%. Single-episode results in this stage have not been
predictive.

## Running it

Requirements beyond the checkout: the Wan2.2-Fun-5B-Control backbone (23 GB),
the two released checkpoints, and MANO (`mano.is.tue.mpg.de`, research
licence, converted once by the estimator's `convert_mano_pkls.py`). MANO is
not optional — `GeoDitModel.__init__` loads it for the mixed-PnP translation
decode.

`_ace_batch.py` runs every episode in one process because the stock
`infer_video.py` does three things that cost measured runs: it VAE-encodes the
whole clip at once (so memory scales with clip length x frame area, and 1280
died on the long episodes at 24 GB), reloads the 5 GB backbone per
invocation, and aborts the batch on one failure. Here the clip is encoded in
241-frame chunks overlapping by one 81-frame training window, each frame taken
from the chunk whose midpoint is nearest, so no frame is decoded without real
context on both sides — the bidirectional context is the estimator's whole
argument for surviving occlusion. Chunk size halves on CUDA OOM.

Throughput at 832 is ~17 fps on one RTX 4090, ~7 fps at 1280 — 34% of that
cost is the overlap, re-encoding shared frames. Peak host RAM during model
load is 24.3 GB, which is the real constraint on a 31 GB machine, not VRAM.
