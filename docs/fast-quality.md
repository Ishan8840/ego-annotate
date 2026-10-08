# Fast quality candidate and human evaluation

This opt-in quality-only profile retains every original numerical measurement
and strict decode check, adds apparent global image motion, and consolidates
context into one compact Qwen request per eight-second window. It is a new
rubric pending human validation, not a proven accuracy-equivalent replacement
for the original full descriptive report and composite rubric. Existing default
commands remain unchanged. It does not generate captions, verify captions, or
reconstruct hand pose.

## Changes

- Overlap strict FFmpeg diagnostics with CPU frame processing.
- Avoid copying full frames for hashes and avoid temporary float arrays for
  integer-valued residuals. Compute each perceptual-hash median once.
- Reuse the source probe. During the measurement decode, collect the contextual
  JPEG evidence and compute optical flow at 5 Hz, avoiding another decode and
  repeated random seeks for those features.
- Keep the existing Qwen3-VL-8B BF16 weights resident across episodes. Batch
  independent windows and use compact responses instead of duplicate long
  descriptive/composite passes. Do not quantize weights or lower image resolution.
- Use fixed-key-order constrained decoding, strict source-specific evidence
  validation, and at most one repair attempt. Invalid responses remain missing.
- Preserve full-frame integrity, duplicate detection, metric resolution, and
  source hashes. Source mutations and evidence binding failures remain visible.

## Scope

| Property | Evidence in this profile |
|---|---|
| Detail/blur candidate | Original all-frame Laplacian variance at width 320; low texture also scores low |
| Exposure | Original all-frame luma and clipping measurements |
| Noise/compression proxy | Original high-frequency residual, not an SNR measurement |
| Duplicates, gaps, corruption | Original decoded hashes, PTS checks, strict decoder diagnostics |
| Camera motion | 5 Hz forward/backward feature tracking + robust affine fit; unknown on weak support |
| Hand/object/workspace visibility | Compact VLM assessments on bound sampled frames |
| Interaction and demonstration clarity | Compact VLM; physical contact is not established |
| Failures, distractors, boundaries | Compact VLM; intervening events and task completeness remain uncertain |
| Obstruction/glare/artifacts | Compact VLM complements numerical image proxies |
| Task success/compliance/correctness/safety | Unavailable without specifications; this CLI supplies none |
| Diversity descriptors and composite score | Not produced by this profile |

Motion is normalized apparent image displacement in diagonals per second, not
metric camera motion or a calibrated camera-shake score. Local object motion,
rolling shutter, parallax and low texture can bias or invalidate estimates.
Hand visibility remains with the contextual model; we did not substitute an
unvalidated detector-confidence proxy or claim to measure hand trajectories.

Each window keeps twelve evenly spaced base samples (same schedule as the
previous descriptive report for eight-second windows) plus up to eight metric
candidate onset samples, at the original 640-pixel maximum side and JPEG default
quality. Event onsets are collected online and can differ from the previous
postprocessed event ranges, particularly at window boundaries and short static
runs. Exact frame IDs/timestamps and sample budgets are logged. More than eight
events is explicitly counted; not every event or intervening action is visually
reviewed. Full-frame numerical checks continue regardless.

N means no concern observed **in samples**, C means visible concern, U means
unknown. N/C require an in-range cited frame; U may have no frame. These are
unverified judgments; format validity and full coverage do not establish accuracy.
A single cited frame cannot establish absence of problems between samples.

## Measured results (supplied A100, warm model)

| Review source | Video duration | Quality processing | Context assessments |
|---|---:|---:|---:|
| epic-P03-120-60s | 60 s | 19.08 s | 72/72 |
| epic-P02-60 | 24 s | 9.40 s | 27/27 |
| epic-P03-60 | 24 s | 8.85 s | 27/27 |
| hot3d-quest-000050 | 5 s | 6.01 s | 9/9 |
| hot3d-quest-000100 | 5 s | 5.95 s | 9/9 |
| epic-synthetic-controls | 16 s | 6.36 s | 18/18 |

The one-minute episode fell from **92.37 s** for the prior full profile to
**19.08 s** for this compact profile: **4.84× faster**. This is a comparison of
changed quality profiles, not identical VLM workloads. A separate accepted run
of the compact profile took 18.75 s; the last scheduling change did not show a
further measurable improvement. All six clips total 134 seconds of video,
processed in 55.70 warm seconds or 110.42 cold seconds. Startup/warm-up adds
approximately 54.72 seconds once per process. Keep the model resident for
collections of clips. The 5-second HOT3D clips still take about 6 seconds each.

All **162/162** requested contextual rows were returned, with one explicit
unknown. Coverage is not factual accuracy. No human labels have been supplied.
The full numerical outputs exactly match the old saved reports on the minute,
P02, and HOT3D 000100 clips (excluding file paths and VLM/render-added fields).
The isolated CPU measurement benchmark also confirms exact output equality:

| Video | Original measurement path | Optimized measurement path |
|---|---:|---:|
| epic-P03-120-60s | 16.53 s | 11.17 s |
| epic-P02-60 | 7.32 s | 4.98 s |
| hot3d-quest-000100 | 3.74 s | 2.86 s |

These are single paired runs with four decoder threads, not confidence intervals;
the earlier 14.32-second CPU benchmark used a different thread setting. This
raw measurement comparison excludes optical flow, context JPEGs and VLM calls.

The synthetic control's measured candidates exactly include blur at 4–8 s,
darkness at 8–16 s, and near-static footage at 12–16 s. These known interventions
show sensitivity, not natural-defect accuracy. Review the EPIC P03 short 0–8 s
and P02 8–16 s windows: some C labels only describe a visible hand/object without
explaining a defect. They are preserved verbatim for human review. No automatic
repair relabels them as correct.

Validation: **249 tests passed**, 12 skipped for unavailable corpus fixtures.
Browser checks passed for all six source videos, evidence images, initially
hidden model outputs, label export, blind-rating provenance and mobile layout.
The review pack contains no prefilled human verdicts.

Rejected trials remain in the artifacts: unordered JSON-schema decoding stalled;
plain decoding and prompt-only repairs lost assessment coverage. Fixed-order
regex decoding completed the candidate evaluations without repairs. The final
CLI exits nonzero on incomplete contextual coverage after saving diagnostics.
Grammar constraints prevent formatting errors, not hallucinations.

Artifacts are under `artifacts/realtime/quality-fast-final`,
`quality-human-review`, and `quality-fast-benchmark-report.json`.
The evaluation used the archived v5 inference driver; the delivered driver adds
explicit completion metadata and a failure exit for incomplete context, without
changing inference, evidence, or measurement logic.

## Run on the supplied A100 VM

Use the existing vLLM environment from [quality-performance.md](quality-performance.md).
No new production dependencies or weights are installed.

```bash
cd /workspace/ego-speed
.venv-vllm/bin/python scripts/process_quality_fast.py \
  artifacts/speed/public/epic-P03-120-60s.mp4 \
  artifacts/speed/public/epic-P02-60.mp4 \
  --out artifacts/realtime/my-fast-quality

# No VLM: measured checks, motion, and evidence only; context is unassessed.
.venv/bin/python scripts/process_quality_fast.py \
  artifacts/speed/public/epic-P03-120-60s.mp4 \
  --mode measured --out artifacts/realtime/my-measured-quality

# Paired raw measurement timing and exact numerical equivalence.
.venv/bin/python scripts/benchmark_quality_measurements.py \
  artifacts/speed/public/epic-P03-120-60s.mp4 \
  --out artifacts/realtime/my-measurement-comparison.json
```

A new output directory is required. Results include JSON, timing CSV, motion
samples, JPEG evidence, raw model responses/token usage, unavailable criteria,
and assessed/requested/unknown counts. The model remains resident across the
listed videos; cross-episode response caching is disabled. Warm times include
source reads, probing, metrics, evidence writes, all model retries and report
writing. Cold times include model initialization and unrelated-image warm-up.
Review-pack video copying and benchmark data downloads are separate packaging
steps and excluded from inference timings.

## Human review

```bash
.venv/bin/python scripts/build_quality_review.py \
  artifacts/realtime/quality-fast-final \
  --reference artifacts/realtime/quality-full-final \
  --reference artifacts/speed/reference-p02 \
  --out artifacts/realtime/quality-human-review
```

Open the resulting `index.html` directly in a browser; video, frame thumbnails,
reports and embedded data are portable local files, with no CDN/network calls.
Watch each window and label concerns before revealing model answers or the old
report. Provide notes and corrections with timestamps. Use **Export human review
JSON** to save the labels. Blank means unreviewed; U means uncertainty. Local
browser persistence is a convenience, not a substitute for exporting labels.

```bash
.venv/bin/python scripts/summarize_quality_review.py \
  artifacts/realtime/quality-human-review/review-data.json \
  /path/to/preload-human-quality-review.json \
  --out artifacts/realtime/human-agreement.json
```

The summary excludes uncertain and non-blind human ratings by default. It
reports false alarms, missed concerns, and model abstentions separately. Recall
including unresolved model abstentions is provided, so abstaining cannot silently
inflate apparent recall. This selected review set cannot establish dataset-wide
accuracy; a larger independently labeled holdout and adjudication would be needed.
The previous full VLM report is a comparator, never ground truth.

A synthetic control includes re-encoded source, blur, darkness, and frozen dark
frames. Its intervention intervals are explicitly labeled and can be revealed
after rating. Synthetic sensitivity is not natural-defect detection accuracy.

## Licenses and boundaries

Runtime dependencies and checkpoints are reused: Qwen3-VL-8B-Instruct weights and
vLLM code Apache 2.0, OpenCV Apache 2.0, NumPy BSD 3-Clause; existing FFmpeg build
licensing remains as inventoried by the project. No research-only model, new hand
model, or production checkpoint was added. Playwright (Apache 2.0) is temporary
browser-test tooling, not a runtime dependency. vLLM 0.11.2 structured-output API:
https://docs.vllm.ai/en/v0.11.2/features/structured_outputs/ .

EPIC-KITCHENS recordings and derived controls are CC BY-NC 4.0 and are separate
research evaluation data only. HOT3D recordings are CC BY-SA 4.0. The review pack
is local and is not published or added to Git. See the source manifest and existing
[license inventory](performance.md#data-provenance-and-licenses).
