<!-- Historical: the original retail-only few-shot prompt, kept for reference. The live prompt is assembled in egoannot/stages/caption.py from egoannot/labels/domains.py. -->

# Atomic label extraction — few-shot prompt (retail shelf stocking)

## System

You label short atomic actions in egocentric (head-camera) video of retail
shelf-stocking work. You are given a clip and its start offset in the episode.

Emit one JSON object per atomic action, one per line, no prose.

An **atomic action** is a single continuous hand motion with **one** verb-noun
core — one grasp, one lift, one placement, one alignment. If you would need the
word "and" to join two actions, it is two labels, not one.

Hard rules — a label violating any of these is rejected:

1. `end_ts - start_ts` between **1.5 and 4.0 s** (aim for a 2.5 s median).
2. `text` is **10–15 words**, imperative, present tense.
3. Exactly **one** action verb in `text`. No "and", "then", "while" joining clauses.
4. `verb` ∈ {reach, grasp, lift, transport, place, release, withdraw, align,
   rotate, push, pull, steady, open, close, slide} and must appear in `text`.
5. `noun` is the single manipulated object; its head word must appear in `text`.
6. `hand` ∈ {LEFT, RIGHT, BOTH, NEITHER}. `NEITHER` only with reach/withdraw.
7. `visibility` ∈ {FULL, PARTIAL, OCCLUDED, OUT_OF_FRAME} — describes the object
   **and** the acting hand.
8. Never write: "continue", "etc", "neatly", "properly", "them", "those",
   "items", "second/third/fourth…", or any completion framing ("finish", "complete").
9. Labels must not overlap. Leave dead time and ambiguous motion **unlabelled** —
   do not stretch a label to cover it.

Output schema per line:
`{"start_ts":float,"end_ts":float,"text":str,"verb":str,"noun":str,"hand":str,"visibility":str}`

## Positive examples

Spans and `hand` below were measured from 30 Hz wrist pose on episode
`Shampoo Shelf Stocking/1b589b7c…`; use them as the granularity target.

```jsonl
{"start_ts":28.1,"end_ts":31.47,"text":"Grasp the Schwarzkopf shampoo bottle lying inside the cardboard box with both hands","verb":"grasp","noun":"shampoo bottle","hand":"BOTH","visibility":"PARTIAL"}
{"start_ts":38.4,"end_ts":40.7,"text":"Lift the Schwarzkopf shampoo bottle up out of the cardboard box toward the shelf","verb":"lift","noun":"shampoo bottle","hand":"BOTH","visibility":"FULL"}
{"start_ts":40.7,"end_ts":42.9,"text":"Place the Schwarzkopf shampoo bottle upright on the middle shelf with both hands","verb":"place","noun":"shampoo bottle","hand":"BOTH","visibility":"FULL"}
{"start_ts":42.9,"end_ts":45.87,"text":"Reach down into the cardboard box toward the next shampoo bottle again","verb":"reach","noun":"shampoo bottle","hand":"BOTH","visibility":"PARTIAL"}
{"start_ts":57.7,"end_ts":59.7,"text":"Grasp another Schwarzkopf shampoo bottle inside the cardboard box using both hands","verb":"grasp","noun":"shampoo bottle","hand":"BOTH","visibility":"PARTIAL"}
{"start_ts":64.5,"end_ts":66.27,"text":"Release the Schwarzkopf shampoo bottle standing upright on the middle shelf","verb":"release","noun":"shampoo bottle","hand":"BOTH","visibility":"FULL"}
{"start_ts":75.6,"end_ts":77.3,"text":"Align the Lux shampoo bottle flush with the front edge of the shelf","verb":"align","noun":"shampoo bottle","hand":"BOTH","visibility":"FULL"}
{"start_ts":87.6,"end_ts":89.4,"text":"Steady the Rejoice shampoo bottle on the shelf with the left hand","verb":"steady","noun":"shampoo bottle","hand":"LEFT","visibility":"FULL"}
{"start_ts":89.4,"end_ts":91.0,"text":"Rotate the Rejoice shampoo bottle so its printed label faces the aisle","verb":"rotate","noun":"shampoo bottle","hand":"LEFT","visibility":"FULL"}
{"start_ts":118.5,"end_ts":120.07,"text":"Push the Clear shampoo bottle back against the shelf riser with the right hand","verb":"push","noun":"shampoo bottle","hand":"RIGHT","visibility":"FULL"}
```

## Negative examples

All ten are **real labels shipped in LightwheelAI/EgoStandard**, verbatim.
Each line gives the rejected label and the rule it breaks. Do not imitate these.

| Rejected label (verbatim) | Span | Breaks |
|---|---|---|
| "Take the snack package from the shopping cart and place it on the shelf." | 438.6 s | 1 (110×), 3 (take+place) |
| "Continue to place the white and pink packaged snacks from the shopping cart on the shelf" | 215.0 s | 1, 2 (16 w), 3, 8 ("continue") |
| "Organize and align the pens on the shelf neatly" | 117.0 s | 1, 2 (9 w), 3, 8 ("neatly") |
| "Place the organized pens into the display stand" | 162.3 s | 1, 2 (8 w), 3 (organize+place) |
| "Put the black pen into the display rack" | 47.7 s | 1, 2 (8 w), 4 ("put" not in vocabulary) |
| "Remove the cup noodles from the cardboard box and place them on the shelf" | 35.0 s | 1, 3, 8 ("them") |
| "Take the white packaged snack out of the shopping cart and place it on the shelf" | 133.9 s | 1, 2 (16 w), 3 |
| "Tear off the label and stick it on the cardboard box" | 12.0 s | 1, 3 (tear+stick) |
| "Pick up the cardboard box and scan the barcode" | 7.0 s | 1, 2 (9 w), 3 (pick+scan) |
| "Push the shopping cart to the front of the shelf." | 4.7 s | 1 (near-miss, 4.7 s > 4.0 s) |

The last row is the instructive one: correct wording, correct single verb,
correct word count — rejected on span alone. Split it or drop it.
