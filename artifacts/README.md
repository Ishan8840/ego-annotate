Generated output, one directory per stage. Committed so the numbers quoted in
the top-level README can be checked without the episode corpus, which is not
redistributed here.

- `quality/` — per-clip records (T1–T4) for the 17 episodes the segments draw on
- `events/` — contact/release candidates, actionness state spans, stereo validation
- `spans/spans.jsonl` — the 269 annotation units, with their pose-derived fields
- `captions/captions.jsonl` — the deliverable: 269 captions (Qwen3-VL-8B, 5 spans/call)
- `captions/captions_batch1.jsonl` — the same spans at 1 span/call, for the
  batch-size comparison in the README
- `reports/score.txt` — the scored output
- `logs/` — the runs that produced all of the above, including the captioning ablations

Rendered review clips (`segments/`) and HTML reports are generated locally and
git-ignored; rebuild them with `python -m egoannot segments render`.
