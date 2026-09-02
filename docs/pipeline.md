# Pipeline reference

Stage-by-stage detail, with the measurements behind each design choice.
For the short version see the [README](../README.md).

Turns raw `.mcap` episodes (head camera + 21-joint hand pose + calibration)
into dense, validated, atomic action captions — roughly 28 labels/minute — with
every intermediate product measured rather than assumed.

The design rule throughout: **anything pose can measure, pose measures.** The
vision-language model is asked only for what it can actually see (`text`,
`verb`, `noun`, `visibility`). Span boundaries, the acting hand, grasp
aperture, rotation direction and finger state all come from motion capture,
where they are cheaper and more reliable.

```
python -m egoannot paths          # show resolved paths
make all                          # quality -> events -> features -> spans -> caption -> score
make test                         # 65 tests
```

## Layout

```
egoannot/
  config.py            every path and threshold, with its provenance
  core/                shared primitives
    mcap_io.py           single-pass episode reader
    video.py             ffmpeg decode + the JPEG frame store
    geometry.py          rotations, projection, camera-convention test
    signal.py            smoothing, speed, run and minima finding
  labels/
    domains.py           verb/noun vocabulary packs (single source of truth)
    atomicity.py         the executable label rules (A0-A9) + self-test
  stages/
    quality.py           T1-T4 per-clip quality records
    events.py            contact/release events + actionness states
    features.py          per-frame pose features + closure PCA
    spans.py             span cutting, band policy, quality gate, pose facts
    caption.py           prompt, swappable VLM backends, strict id binding
    score.py             format / diversity / grounding metrics
  evaluation/
    gold.py              events vs the human gold set
    stereo.py            hand pose vs stereo depth (the rejected depth route)
  tools/
    make_segments.py     render review clips with pose overlays
    build_annotator.py   single-page gold annotation tool
    build_showcase.py    single-page results showcase
  cli.py                 one entry point for every stage
data/                    version-controlled inputs (gold, segments, vocab fixtures)
artifacts/               everything generated, one subdirectory per stage
docs/                    measurement write-ups: quality, events, label spec
tests/                   unit tests + corpus-backed integration tests
```

## Paths

Nothing hardcodes a machine path. The corpus is resolved in this order:

1. `EGO_CORPUS=/path/to/mcap/root`
2. a `corpus` symlink in the repo root
3. discovery, newest first

Outputs go to `artifacts/`, overridable with `EGO_ARTIFACTS`. Rendered review
clips go to `artifacts/segments/`, overridable with `EGO_SEGMENTS`. Run
`python -m egoannot paths` to see what resolved and which stages have output.

## The stages

### 1. quality — per-clip quality records

Four tiers, no learned filter at any of them, so the accept/reject boundary is
auditable and does not inherit a model's preferences about what egocentric
video should look like.

| Tier | What | Learned? |
|---|---|---|
| T1 | hard rejects: duration, modality, blur, exposure, framing, pose jumps | no, by design |
| T2 | head-motion score (optical flow, 4 fps, 256×256); drops the top 30% | no |
| T3 | hand framing from shipped pose + calibration (21 joints, both hands) | no |
| T4 | the per-clip record | — |

Blur and exposure floors are **calibrated, not guessed**: `quality calibrate`
scores real frames, then scores deliberately degraded copies of the same
frames, and the floor goes between the two distributions. T1 is proved by
fault injection (`quality selftest`), because it rejects nothing on clean
footage and that is the only way to show it works.

T2 ranks **within each episode**. Ranking on the pooled corpus made it an
episode selector instead of a clip filter — measured drop rates ran from 0%
(`test`, `cart_wipes`) to 75% (`belts_a`, which lost every clip it had),
because optical-flow magnitude is scene-dependent.

Details, including the limits of T3 on this corpus: [`docs/quality.md`](docs/quality.md).

### 2. events — contact/release and actionness

No object poses ship and depth is never delivered, so "proximity" cannot mean
hand-to-object. What ships is a 21-joint rigged hand, which gives
finger-to-finger proximity: **grasp aperture**. Aperture-rate runs give
candidate closings and openings; a state machine alternates contact → release;
each contact snaps onto the nearest wrist **jerk** peak, the acceleration
discontinuity of impact.

Alongside it, five actionness states with hysteresis, priority-ordered so that
holding something counts as manipulation even while the torso moves, and
scanning with quiet hands counts as inspecting rather than idle:

`manipulate` → `reach` → `reposition` → `inspect` → `idle`

**The events are not confirmed as ground truth.** Three independent checks were
run and none confirms that detected events coincide with real contact; against
the human gold set the detector sits at F1 0.02–0.09. They are a candidate
proposal layer, which is why nothing downstream conditions a caption on them.
The full negative result is in [`docs/events.md`](docs/events.md).

The "closed"/"open" state gate now carries absolute caps and a usability flag.
Percentiles alone have no grounding: measured on this corpus, p30 "closed" is
68 mm on `noodles` but 18 mm on `d_contactlens`, and on `t_keyboard` the two
levels sit 7 mm apart — inside the hand model's own error, so the gate there
was a coin flip. Episodes whose gate cannot separate anything are reported.

### 3. features — per-frame pose features

Per-finger curl for all five digits, a 1-D closure score (PC1 of the full
21-joint configuration, canonicalised into a hand-local frame first so the PCA
describes *shape*, not orientation), wrist velocity/acceleration/jerk,
per-episode percentile normalisation, and validity flags that are propagated
and never imputed. `features auc` reports univariate discriminability at gold
event times — a sanity check on the jerk-as-primary hypothesis, not a
classifier.

### 4. spans — where the annotation units come from

Boundaries come from a **combined activity signal**, not wrist speed alone.
Wrist velocity is the wrong cue for fine-motor work: unscrewing a cap barely
moves the wrist while the fingers and forearm do everything, so velocity
troughs land in the middle of the action. The combined signal is the max of
normalised wrist speed, aperture rate and twist rate.

Two policies apply after cutting:

**Duration band.** The linter rejects spans outside 1.3–4.0 s (rule A1), and
that is known at cut time, so it is enforced here rather than charged to the
captioner. Long spans are subdivided at interior activity minima; short spans
are merged into whichever neighbour lands closest to the middle of the band, or
dropped when neither merge is legal — dropping is fine because labels must not
overlap but need not tile. Measured on this corpus:

| signal | band policy | spans | short | long | max | in band |
|---|---|---|---|---|---|---|
| activity | off | 307 | 12 | 0 | 3.8 s | 96% |
| activity | **on** | 295 | 0 | 0 | 3.8 s | **100%** |
| velocity | off | 293 | 12 | 8 | 11.6 s | 93% |
| velocity | **on** | 295 | 0 | 0 | 3.9 s | **100%** |

**Quality gate.** Spans that sit inside clips the quality stage rejected are
dropped. Gating defaults to **T1 only**: T1 is a defect finding, while T2 drops
the top 30% of every episode by construction, so including it removes about a
third of all spans whether or not anything is wrong (26 spans gated vs 98).

Every span carries the pose facts the model is never asked for, all read from
the **dominant** hand: `hand`, `acting_side`, `aperture_mm` (p10–p90),
`aperture_end_mm`, `ap_trend`, `rotation`, `fingers`, `wrist_speed`.

### 5. caption — the VLM step

Backends are swappable: `stub` (no model, exercises the whole pipeline),
`anthropic`, `openai` (any OpenAI-compatible server — vLLM, SGLang, llama.cpp,
LM Studio), `qwen-local` (in-process transformers).

The prompt is deliberately **domain-agnostic**. Putting domain vocabulary and
object lists into it was measured to cost 10 points of uniqueness (88.7% →
79.0%) for zero atomicity gain, because the model reaches for the listed words
instead of describing what it sees. Vocabulary belongs in the linter, and both
sides read it from `labels/domains.py` so they cannot disagree — when they did,
one mismatch cost 11 points of atomicity.

Captions bind to spans **only when the binding is unambiguous**: by `span_id`,
or by order when the reply is full-length. The previous positional fallback
bound short replies to the first N spans, which in one measured run attached 8
captions to spans 0–7 of a 10-span batch regardless of which spans they
described. A mislabelled timestamp is worse than a missing one.

Five spans per call is the default, and that number is measured rather than
assumed. On 269 spans with Qwen3-VL-8B, at identical throughput and a 269/269
bind rate either way:

| spans/call | atomicity | uniqueness | captions >15 words | grounding |
|---|---|---|---|---|
| 1 | 67% | 56.1% | 15 | 62% |
| **5** | **75%** | **66.9%** | **0** | 52% |

Batching makes the model differentiate captions within a single reply and
disciplines length — every over-long caption disappeared, and the Heaps
projection at n=10,000 improves from 26% to 40%. It costs a little agreement
with the measured aperture trend. Batching is only safe because binding is
strict; with the old positional fallback it was the failure mode.

Frames come from a planned JPEG store: every frame the run needs is declared up
front, each segment is decoded once, and only those frames are kept, encoded.
Peak memory is **2.5 MB for 1076 frames**; the previous decode-everything cache
held 8.2 GB across the same segments and competed with a local VLM for RAM.

### 6. score — three kinds of metric, kept separate

| Group | What it answers |
|---|---|
| FORMAT | does the caption obey the atomicity rules (A0–A9) |
| DIVERSITY | uniqueness now, plus a Heaps projection of uniqueness at scale |
| GROUNDING | does the caption agree with what pose actually measured |

Grounding compares the emitted verb against the measured aperture trend
(closure verbs should co-occur with fingers closing; `release`/`place` with
opening) and activates rule A9, which forbids a caption contradicting measured
rotation. Both channels were already being computed and then discarded.

**What still is not measured anywhere: caption correctness.** Format and
diversity never check that "white ceramic cup" is a white ceramic cup, and the
human gold set covers only two of the eight action classes in the segment set —
`gold` prints which classes are unscored. Grounding is a partial, pose-derived
substitute, not a replacement for review.

## Label rules

`labels/atomicity.py` is the executable spec: every rule is code, and a label
is atomic iff it raises no ERROR. A0 schema, A1 span, A2 length, A3 exactly one
verb-noun core, A4 field/text consistency, A5 enums, A6 banned phrasing, A7
imperative tense, A8 no overlap, A9 no contradiction of a measurement.

The full rule table and the reasoning behind each is in
[`docs/labeling_spec.md`](docs/labeling_spec.md).

`python -m egoannot lint` runs the self-test: curated positives must all pass,
curated negatives must all fail, **and every exemplar shipped in the prompt
must lint clean**. That last check is not decoration — the kitchen pack's own
worked example used to fail A3, because `tap` is both a core verb and that
pack's own noun, so the prompt was telling the model to imitate a sentence the
scorer rejected, for the pack covering half the output. A3 now ignores a verb
token that is the head of a vocabulary noun unless it opens the sentence.

## Tools

```
python -m egoannot segments render          # crop + encode review clips, precompute overlays
python -m egoannot gold                     # events vs human gold, by action class
python -m egoannot stereo <episode>...      # hand pose vs stereo depth
python -m egoannot caption prompt --pack kitchen
```

`tools/build_annotator.py` and `tools/build_showcase.py` emit single-page HTML
into `artifacts/reports/`.

## Known limits

- **Contact events are unvalidated** and sit at F1 0.02–0.09 against gold. Do
  not treat them as ground truth. ~50 human-reviewed events would settle it.
- **Depth is never delivered.** A calibrated stereo pair ships and was tested:
  the hand pose *is* registered to the imagery (median error within ±5 cm) but
  cannot support proximity (correlation ≈ 0, MAE 75–251 mm, disparity coverage
  36–65% on low-texture retail scenes). Full-resolution made it worse, so the
  error is stereo failure at hand boundaries, not resolution.
- **T3 does not discriminate on this corpus.** Full-frame hand in-FOV rate is
  exactly 1.000 in every episode because the shipped hand pose is near-rigidly
  coupled to the head camera. A real hand-visibility signal needs a detector on
  pixels, which belongs in its own declared tier.
- **No walking anywhere in this sample.** Head speed never exceeds 0.86 m/s;
  `reposition` captures leaning and weight shifts, not locomotion.
- **Gold covers two action classes of eight.** Everything else is unscored for
  correctness.
- Thresholds are specific to 1920×1456 and these sampling rates. Re-run
  `quality calibrate` and `events calibrate` before applying to new footage.
