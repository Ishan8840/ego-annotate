# Quality stage — per-clip quality record

```
python -m egoannot quality calibrate --all   # derive thresholds from footage + synthetic control
python -m egoannot quality selftest  --all   # fault injection: prove T1 fires
python -m egoannot quality measure           # write the clip records
python -m egoannot quality report            # re-print from existing records
```

Implementation: `egoannot/stages/quality.py`. Thresholds: `egoannot/config.py`
(`QUALITY`).

## Tiers

| Tier | What it does | Learned? |
|---|---|---|
| T1 | deterministic hard rejects: duration, modality, blur, exposure, pose jumps | no — by design |
| T2 | head-motion score (optical flow, 4 fps, 256×256); drops the top 30% **within each episode** | no |
| T3 | hand framing from shipped pose + calibration (21 joints, both hands) | no |
| T4 | `clip_quality.jsonl` — one record per clip | — |

T1 is deliberately threshold-only. A learned quality filter at the hard-reject
tier would encode a model's opinion about what egocentric video should look
like and bake that bias into every downstream dataset built from the output.

## Where the thresholds come from

Blur and exposure floors are **calibrated, not guessed** — `qc.py calibrate`
scores real frames, then scores deliberately degraded copies of the same
frames. On this footage:

- tile Laplacian variance p10 = 0.5 → `lowtex_floor = 0.5` (masks flat/de-ID tiles)
- real frame sharpness p5 = 4.8; Gaussian σ=3 blur p95 = 3.8 → `blur_floor = 4.0`
- σ=1.5 blur is **not separable** from this footage's own soft frames (only 71%
  of sharp frames score above its p95). Mild blur is therefore out of reach;
  the filter catches moderate-to-severe blur only. Stated, not papered over.

Thresholds are specific to native resolution (1920×1456) and these sampling
rates. Re-run `calibrate` before applying to different footage.

## Frame rate is not constant

This sample contains both 30 fps and 25 fps episodes. `session/metadata`
declares the rate correctly per camera; `qc.py` reads it and cross-checks
against the message rate (`fps_mismatch` flag). Assuming a single rate skews
the decoded timeline — for a 25 fps episode read as 30 fps, frame-to-clip
mapping drifts by 17% and the clip tail decodes as empty.

## Known limitation: T3 does not discriminate on EgoStandard

The full-frame hand in-FOV rate is **exactly 1.000** in all 11 episodes
(21 joints × 2 hands × every frame). The shipped hand pose is near-rigidly
coupled to the head camera — hand spread is 12–65 cm in world frame but only
3–12 cm in camera frame (ratio 0.09–0.35), in both `robot_gripper_mimic` and
`natural_human_hand` episodes. So the pose stream cannot recover image-space
hand position, and "clips without hands" cannot be identified from it.
`hand_central50_rate` is retained as the only pose-derived variant with any
spread. A real hand-visibility signal needs a detector on pixels, which
belongs in its own declared tier — not in T1.

Note also that in-FOV is not the same as unoccluded: occlusion by the body,
the shelf or the held object is invisible to a projection test.

## T2 ranks within each episode, not across the corpus

Optical-flow magnitude is scene-dependent, so a pooled percentile turns T2 into
an episode selector rather than a clip filter. Measured on this corpus with
pooled ranking:

| episode | clips | dropped by T2 |
|---|---|---|
| belts_a | 4 | 3 (75%) — with 1 T1 reject, the episode lost every clip |
| stockout_b | 6 | 4 (67%) |
| shampoo | 34 | 14 (41%) |
| test | 31 | 1 (3%) |
| cart_wipes, cleaning_tools_a/b | 3 each | 0 (0%) |

`motion_scope="episode"` (the default) makes the 30% drop apply per episode, so
the accepted set keeps every episode's diversity. Set
`motion_scope="global"` in `config.QUALITY` to restore the old behaviour.

## Downstream use

The span stage reads these records and drops spans that sit inside rejected
clips, gating on **T1 only** by default — T1 is a defect finding, whereas T2
removes 30% of every episode by construction. Previously nothing downstream
read this file at all.

## Files

- `egoannot/stages/quality.py` — `calibrate`, `selftest`, `measure`, `report`
- `artifacts/quality/clip_quality.jsonl` — the per-clip records (the deliverable)
- `artifacts/quality/clip_quality.episodes.json` — per-episode summary incl. fps and modality audit
- `artifacts/logs/quality.*.log` — the runs that produced the records
