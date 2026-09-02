# Label spec — what an atomic caption is

The rules are executable, not prose: `egoannot/labels/atomicity.py` is the
spec, and a label is atomic iff it raises no ERROR. This document explains the
intent; the code is authoritative.

```
python -m egoannot lint                      # self-test: positives, negatives, exemplars
python -m egoannot lint labels.jsonl --pack kitchen
```

## The rules

| Rule | Requirement |
|---|---|
| A0 | schema: `start_ts`, `end_ts`, `text`, `verb`, `noun`, `hand`, `visibility` |
| A1 | span duration inside 1.3–4.0 s (measured band); 1.5–2.5 s is the target |
| A2 | 10–15 words, unless `uncertain` is set |
| A3 | exactly one verb-noun action core in the text |
| A4 | `verb` in the closed vocabulary and present in the text; `noun` head present in the text |
| A5 | `hand` and `visibility` are enum members; `hand=NEITHER` needs a contactless verb |
| A6 | no banned phrasing: continuation wording, open-ended enumeration, unverifiable adverbs, aggregate referents, task-completion framing, enumeration indices |
| A7 | imperative present tense, capitalised leading verb (warning) |
| A8 | labels within an episode must not overlap — they need not tile |
| A9 | the text must not contradict a measurement (rotation direction) |

## Why the band is 1.3–4.0 s and not 1.5–2.5 s

The target band comes from the spec. The measured band is what wrist-velocity
segmentation of real footage actually produces: on trough-cut spans, every A1
rejection measured 1.39–1.49 s, a hair under the old 1.5 s floor. Merging those
would invent boundaries the motion did not have, so the floor widened to 1.3 s
and recovered all of them. A1 violations are now prevented upstream by the span
stage's band policy rather than penalised here.

## Why A3 ignores some verb tokens

Several manipulation primitives are also object names in the same domain.
`tap` is a core verb and the kitchen pack's own noun, so

> Rinse the black frying pan under running water from the chrome tap

scored two verb cores and failed — and that sentence is shipped in the prompt
as a worked example. A3 therefore ignores a verb token that is the head word of
a vocabulary noun, unless it opens the sentence (where it is unambiguously the
action). The same collision shape applies to `screw`, `wrap` and `seat`.

The complementary trap is over-broad vocabulary: adding `clean` to the
action-verb set made "the clean frying pan" read as a second action and cost 24
points of atomicity. `EXTRA_ACTION` is deliberately short.

## Vocabulary is a single source of truth

`egoannot/labels/domains.py` holds the verb and noun packs, and both the prompt
and the linter read from it. When they disagreed — `scrub`, `stir` and `pour`
correctly emitted and wrongly rejected, plus seven "two-clause" failures that
were the same legal verbs miscounted — that one mismatch cost 11 points of
atomicity.

The prompt itself stays domain-agnostic. Injecting the domain's object list
into it was measured to cost 10 points of uniqueness (88.7% → 79.0%) for zero
atomicity gain: the model reaches for the listed words instead of describing
what it sees.

## Uncertainty is correct behaviour

A7 asks for specifics; it does not ask the model to invent them. When the
frames do not show an object's colour, brand or identity, the caption should
set `uncertain: true`, set `visibility` accordingly, name the object at
whatever level is actually supported, and drop the modifiers — falling below 10
words is permitted in that case. A vague caption marked uncertain is correct. A
caption that guesses a colour it cannot see is wrong, and worse than vague,
because it cannot be told apart from one that got it right.
