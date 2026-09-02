# Events stage — contact/release events and actionness

```
python -m egoannot events measure                        # events + state spans + summary
python -m egoannot events calibrate                      # pooled actionness percentiles
python -m egoannot gold                                  # score events against human gold
python -m egoannot stereo <episode>... --every 3.0 --scale 0.5
```

Implementation: `egoannot/stages/events.py`, `egoannot/evaluation/stereo.py`.
Thresholds: `egoannot/config.py` (`EVENTS`).

Outputs, all under `artifacts/events/`: `events.jsonl` (one record per event),
`events.spans.jsonl` (actionness state spans), `events.episodes.json`
(per-episode summary incl. all validation numbers), `stereo_validation.json`.

## 1 · Contact/release from hand pose

EgoStandard ships no object poses and never delivers depth, so "proximity"
cannot mean hand-to-object. What it does ship is a 21-joint rigged hand
(MediaPipe topology, confirmed from bone-length structure), which gives
finger-to-finger proximity: **grasp aperture** = ‖thumb_tip − index_tip‖,
spanning ~1–14 cm in this footage.

Detector: runs of aperture rate beyond an adaptive threshold
(`max(0.040 m/s, p90|ḋ|)`) that move the aperture by ≥10 mm become candidate
closings/openings; a state machine alternates contact → release with a 0.30 s
refractory period; each contact is then snapped onto the nearest wrist **jerk**
peak within ±0.20 s, which is the acceleration discontinuity of impact.

Result over 17 episodes / 13.6 min: **445 events (32.7/min), 208 grasp
intervals, median grasp 2.52 s.**

## 2 · Actionness (non-contact behaviour)

Five states with hysteresis (0.5 s minimum dwell), priority-ordered so that
holding something counts as manipulation even while the torso moves, and
scanning with quiet hands counts as inspecting rather than idle:

`manipulate` → `reach` → `reposition` → `inspect` → `idle`

Features come from upper body where available (**9 of 17 episodes**): joints
0–7 are a rigid spine (bone-length std 0.00000), joints 12/13 sit at exactly
0.000 m from the shipped hand wrists, which pins 8/9 as shoulders and 10/11 as
elbows. Torso translation uses the mean of joints 0–3; reach uses
‖wrist − shoulder‖ rate. Hands-only episodes fall back to head translation,
which conflates leaning with stepping — the source is recorded per episode as
`base_source`.

Thresholds are pooled percentiles, not guesses (`calibrate_actionness`). The
first pass guessed them and produced a degenerate 99% `manipulate`: `w_head_scan`
was set to 1.40 rad/s, above the corpus p95 of 1.115, so `inspect` could never
fire at all.

Time budget (duration-weighted): manipulate 70.8%, idle 24.2%,
reposition 2.5%, inspect 1.3%, reach 1.1%.

There is **no walking anywhere in this sample** — head speed never exceeds
0.86 m/s and torso translation p90 is 0.43 m/s. `reposition` captures leaning
and weight shifts, not locomotion. Do not read it as walking.

## Validation status: the events are NOT confirmed

This is the honest headline. Three independent checks were run and none
confirms that detected events coincide with real contact:

| Check | Result |
|---|---|
| vs shipped coarse-label boundaries | 1/17 episodes beat a uniform-random null at p<0.05 |
| vs wrist-velocity troughs (independent signal) | 1/17; **pooled sign test 11/17, binomial p=0.166 — not significant** |
| jerk snap distance | p50 0.100 s, p95 **0.200 s** = saturating the window edge, i.e. jerk peaks are often *not* near aperture events |

The events are temporally plausible — 32.7/min, median grasp 2.52 s, which
matches the 2.5–3.0 s atomic scale Phase 2 measured independently — but
distributional plausibility is not event-level correctness.

Scored against the 135-event human gold set (`python -m egoannot gold`), the
detector reaches **F1 0.09 on non-prehensile and 0.02 on pick-and-place** at
±150 ms.

**Do not treat these events as ground truth.** They are a candidate proposal
layer, which is why the captioning prompt deliberately withholds them. Nothing
downstream conditions a caption on a contact event. Confirming them needs
frames: ~50 detected events reviewed by a human would settle it decisively and
cheaply, and doubles as gold seed data.

## The state gate has no absolute grounding

`closed_pct`/`open_pct` are per-hand percentiles, so what "closed" means moves
with the episode's own distribution. Measured aperture levels:

| episode | p30 ("closed") | p65 ("open") | separation |
|---|---|---|---|
| noodles | 68.4 mm | 95.6 mm | 27.2 mm |
| np_tissue | 20.6 mm | 38.7 mm | 18.1 mm |
| d_contactlens | 18.4 mm | 29.5 mm | 11.1 mm |
| t_keyboard | 29.5 mm | 36.7 mm | **7.3 mm** |

"Closed" at 68 mm is a wide-open hand, and a 7.3 mm separation is inside the
hand model's own error, so on `t_keyboard` the gate is a coin flip — that
episode also has the worst cross-signal agreement (p=0.995). Two absolute caps
(`closed_abs_max`, `open_abs_min`) stop the degenerate cases, and
`min_state_margin` marks an episode's gate unusable rather than trusting it;
unusable episodes are named in the report and the gate is skipped for them. On
`noodles` this removes 6 of 64 candidate events; on `np_tissue` it changes
nothing.

## Stereo depth: measured, and rejected

Depth is declared in metadata and never delivered, but a calibrated stereo pair
is shipped, so hand-to-surface proximity looked recoverable. The rig checks out:
perfectly rigid (baseline std **0.000000 m** over ~400 sampled frames), 63.9–64.3 mm,
zero distortion coefficients, giving 1.8–4.9 mm of depth per pixel of disparity
at the 0.30–0.55 m the hands actually work at.

The test: a hand occludes what is behind it, so stereo depth at the projected
hand pixel must equal the hand's own predicted depth.

| Episode | n | median err | MAE | corr | ≤5 cm | disparity coverage |
|---|---|---|---|---|---|---|
| test | 79 | +0.006 m | 0.081 | +0.43 | 57% | 65% |
| shampoo | 83 | −0.003 m | 0.098 | +0.34 | 49% | 50% |
| b_ratlam | 29 | +0.006 m | 0.075 | +0.09 | 41% | 55% |
| stockout_b | 47 | −0.048 m | 0.088 | +0.13 | 34% | 52% |
| belts_a | 24 | −0.046 m | 0.161 | −0.11 | 29% | 61% |
| noodles | 79 | +0.049 m | 0.251 | −0.10 | 16% | 52% |

Two findings, pulling opposite ways:

- **The hand pose is registered to the imagery.** Median error is within
  ±5 cm on every episode, −3 mm on shampoo. This partly rebuts the Phase 3
  worry that the pose might be a fictitious head-anchored offset.
- **But it cannot support proximity.** Correlation is ≈0 (and negative on two
  episodes), MAE is 75–251 mm, and disparity coverage is only 36–65% on these
  low-texture retail scenes. Running at full 1920×1456 made it **worse**
  (MAE 0.215 m, corr −0.11, coverage 36%), so the error is stereo failure and
  background bleed-through at hand boundaries — not resolution.

So contact detection stays on aperture. Making the stereo route work needs a
hand mask (a segmentation model, i.e. a learned component) so depth is sampled
on the hand rather than through it — or the depth camera actually being
delivered.
