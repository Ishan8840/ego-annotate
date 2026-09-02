# Bounded boundary refinement, measured on held-out data

Eight episodes pulled fresh from `LightwheelAI/EgoStandard`, chosen so that no
task overlaps the set the pipeline was developed on (kettle, phone case,
badminton shuttlecocks, adhesive removal, coat rack, file folder, speaker, door
handle). 2.9 min, 165 s of segment video, mixed 25/30 fps, one episode with
full-body pose. Episode list and bucket paths: [`../data/heldout_corpus.json`](../data/heldout_corpus.json);
segment definitions: [`../data/segments_heldout.json`](../data/segments_heldout.json).

Captioning used **greedy decoding** (`CAPTION_GREEDY=1`) in both arms, so a
difference between them comes from the boundaries rather than sampling noise.
Everything else — model, prompt, rules, frame budget — is identical.

## Does refinement improve grounding? No.

| metric | refine off | refine on |
|---|---:|---:|
| labels produced | 75 | 75 |
| labels/min of video | 26.0 | 26.0 |
| spans inside the 1.3–4.0 s band | **100%** | **100%** |
| A1 violations | **0** | **0** |
| atomicity (all rules) | 76% | 75% |
| uniqueness | 77.3% | 76.0% |
| **verb vs measured aperture** | **79%** (33/42) | **79%** (34/43) |
| throughput | 0.61 spans/s | 0.61 spans/s |

Fisher exact on the grounding counts: **p = 1.00**. The two arms are
indistinguishable on the metric the change was built to move.

This is not a case of the change failing to fire. Refinement moved 10
boundaries (mean |delta| 0.323 s, max 0.400 s), 19 of 75 spans changed
geometry, and 28 of 75 captions came out different. It did real work; the work
did not show up in grounding. Atomicity and uniqueness each moved by exactly
one caption, which is noise at n = 75.

Band compliance and density held exactly, and the refinement pass costs
0.3 ms/span against 1.6 s/span of VLM inference.

`boundary_shift_s` was **not** tuned in response to this result.

## The premise behind the change does not replicate

Refinement exists because on the development set the vision-only control arm
beat the pose-guided arm on grounding, 79% to 66% — it cuts where the action it
describes actually happens. On this held-out corpus the comparison reverses:

| | pose-guided | vision-only |
|---|---:|---:|
| verb vs measured aperture | **79%** (34/43) | 51% (23/45) |
| atomicity | **76%** | 55% |
| uniqueness | **77.3%** | 65.9% |
| spans inside the band | **100%** | 68% |
| hand matches measured pose | supplied | 71% |
| labels/min, first 16 s of clip | 26.8 | 45.1 |
| labels/min, after 16 s | **24.4** | 5.1 |

Fisher exact on the grounding gap: **p = 0.0076**. On fresh episodes the
pose-guided arm is significantly *ahead* on the very metric it was supposed to
be behind on.

So the 66-vs-79 result that motivated boundary refinement was a property of
those five development episodes, not of the architecture. The vision-only arm's
*other* failures replicate cleanly and strongly — front-loading (45/min over
the first 16 s, then 5/min), 68% band compliance, 29 A1 violations, and
disagreeing with measured handedness on 29% of labels.

## Caveats

- n is small: 42–45 testable captions per arm for grounding. The A/B null is
  weak evidence of no effect; the pose-vs-vision gap is the stronger claim.
- The development-set numbers used sampled decoding, these use greedy, so the
  66% and this 79% are not strictly comparable. The held-out pose-vs-vision
  comparison is internally consistent — same data, same decoding, same packs.
- No domain pack covers these tasks, so both arms fall back to the
  retail-shelf vocabulary. That depresses atomicity here (75–76% against 87% on
  the development set) via A3/A4 vocabulary failures. It applies equally to
  both arms and does not touch grounding.

## Reproducing

```bash
export EGO_CORPUS=/path/to/heldout/mcaps
export EGO_SEGMENT_DEFS=data/segments_heldout.json
export EGO_ARTIFACTS=/tmp/heldout CAPTION_GREEDY=1
python -m egoannot quality measure && python -m egoannot events measure
python -m egoannot segments render
python -m egoannot spans build --no-boundary-refine --out /tmp/heldout/spans/off.jsonl
python -m egoannot spans build                       --out /tmp/heldout/spans/on.jsonl
for a in off on; do
  python -m egoannot caption run --spans /tmp/heldout/spans/$a.jsonl \
      --backend qwen-local --out /tmp/heldout/captions/caps_$a.jsonl
  python -m egoannot score /tmp/heldout/captions/caps_$a.jsonl
done
python -m egoannot baseline run
python -m egoannot baseline compare --pose /tmp/heldout/captions/caps_on.jsonl
```
