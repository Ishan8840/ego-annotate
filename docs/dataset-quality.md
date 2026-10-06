# Descriptive dataset quality analysis

`quality analyze` describes the footage, its measurable characteristics, visible
issues and gaps in what is known. It produces no PASS/FAIL/REVIEW labels, no
composite quality score, and never filters videos out of the annotation pipeline.
The older `quality measure` command remains the legacy annotation gate.

```sh
# Full-frame measurements and duplicate comparisons, no GPU required
python -m egoannot quality analyze /path/to/videos --out artifacts/quality

# Prepared HOT3D corpus, plus sampled visual observations on a local GPU
CAPTION_GREEDY=1 python -m egoannot quality analyze \
  artifacts/hot3d/corpus_full --out artifacts/hot3d/review/quality \
  --visual qwen-local --model /path/to/Qwen3-VL-8B-Instruct

# Optional task instructions, collection rules, expected formats and diversity labels
python -m egoannot quality analyze /path/to/videos --out artifacts/quality \
  --spec quality-spec.json --visual qwen-local
```

Inputs can be explicit video files, a directory of MP4 files, or the prepared
HOT3D corpus directory. Analyze the native capture export if you want to assess
capture quality: resized/re-encoded imports describe the quality of those
derivatives, not necessarily their originals. Video basenames must be unique.
Dependencies are the base requirements plus `ffmpeg`/`ffprobe`; local visual
analysis uses the optional dependencies in `requirements-hot3d.txt` and a GPU.

The output directory contains:

- `index.html`: dataset summary, dimension coverage, diversity inventory and
  per-clip videos with clickable evidence timestamps.
- `summary.md`: a readable summary and individual findings.
- `analysis.json`: measured distributions, frame fractions, complete evidence,
  provenance, thresholds, specification, raw model replies and missing analyses.
- `clips.partial.json`: completed clip results retained during a long run.
- `videos/`: copies for a portable report. These consume additional disk space.

Serve the directory using `python -m egoannot.tools.review_server --directory
artifacts/quality --port 8081` for reliable seeking. No external API receives
videos: the optional Qwen backend executes locally.
Browser playback requires a supported input codec/container (tested with H.264
MP4). The source-video link remains available for other formats; measurements
can still analyze files that ffmpeg/OpenCV can decode.

## What the report can establish

| Area | Measurements or observations | Limits |
|---|---|---|
| Video integrity | Full decode in ffmpeg and OpenCV, decoder diagnostics, frame count, resolution, FPS, presentation timestamps, duration, expectation mismatches, repeated frames | Timestamp gaps can be variable-rate capture. Missing frames with rewritten timestamps may be undetectable. No expected property is invented. |
| Visual quality | Every decoded frame: luminance, dark/bright pixel fractions, Laplacian detail, high-frequency residual. Sampled visual descriptions of blur, motion blur, glare, noise, compression and obstruction | Metrics use width-320 grayscale thumbnails. Detail depends on scene texture and resolution. Residual is not a calibrated noise estimate. Compression artifacts and motion blur require visual interpretation. |
| Camera/FOV | Adjacent-frame change candidates and sampled positioning/cropping observations | Fast intended motion can resemble bumps or edits. |
| Hand visibility | Source annotation counts and modeled visibility when available; sampled visible cropping/occlusion | Annotation coverage is not image tracking accuracy. Sampled stills cannot establish continuous trackability. |
| Objects, workspace, interaction, clarity and distractors | Separate visual observations with evidence times | Model judgments may be wrong, especially for ambiguous contact and objects. |
| Instructions, success, action sequence | Compare supplied task, final state and required steps with sampled frames | Without the corresponding specification, the dimension remains explicitly unknown. |
| Completeness and failures | Describe the first/last visible state and observed failure candidates | A short excerpt need not show an entire task. Brief events may occur between samples. |
| Temporal quality | Timestamp gaps, nonmonotonic times, frame changes, static intervals and visual observations | Pauses are not automatically defects; speed needs an external capture reference. |
| Privacy/safety | Observations against supplied collection rules | No policy is invented when rules are missing; sampled analysis cannot certify privacy or safety. |
| Duplicates | File SHA-256, decoded-frame hashes, aligned perceptual-hash comparisons | Near matches are candidates. Partial overlaps, crops, time shifts and speed changes can be missed. Low-texture near matches are deliberately excluded. |
| Diversity | Source sequence/participant counts, resolution/FPS/luminance distributions, supplied labels and sampled model descriptions for all seven requested dimensions | Description wording is not an identity label. Sufficiency requires a target distribution and broader sample. |

The visual model sees up to 12 timestamped images per clip, targeting 2 FPS and
including the start and end. On long clips the cap increases sampling gaps; the
exact timestamps appear in every clip report. Confidence is a subjective model
label, not a calibrated probability. Invalid/missing responses are retained as
unavailable analysis rather than favorable findings. Concern counts are counts
of clips where the model expressed a concern, not verified dataset defect rates.

## Optional specification

Global fields apply to every clip; `clips` overrides them by video basename.
Leave a field out when it is unknown. Expected width/height refer to the supplied
video, which may differ from source-camera resolution.

```json
{
  "task": "Move the cup from the tray to the marked spot.",
  "final_state": "Cup upright on the marked spot.",
  "required_steps": ["Grasp cup", "Lift from tray", "Place on marked spot"],
  "collection_rules": ["No outside assistance", "Flag visible personal documents"],
  "expected_video": {"fps": 30, "width": 768, "height": 960},
  "clips": {
    "clip-000000": {
      "diversity": {
        "background": "kitchen A",
        "lighting": "overhead diffuse",
        "object_instances": "cup 03",
        "object_poses": "upright",
        "viewpoint": "egocentric left camera",
        "starting_state": "cup on tray",
        "execution_trajectory": "left to right"
      }
    }
  }
}
```

Use actual collection metadata for these fields; the example is not a task label
for any HOT3D clip. The report does not reuse generated action captions as ground
truth for assessing instruction compliance or success.

## HOT3D run

Tested on the same 11 prepared Quest3 clips used for the caption benchmark:
55 seconds, 1,650 frames, five participants and eight source sequences. The
analyzed MP4s are upright monochrome derivatives at 768×960 and 30 FPS.

- All frames decoded, without decoder diagnostics or metadata mismatches.
- No exact repeated frames or matching/near-duplicate full-clip pairs were found.
- No frames crossed the configured dark/bright/clipping/low-detail thresholds.
  This does not establish absence of motion blur or compression artifacts.
- The model supplied 121 observations across 11 visual dimensions, plus 77
  descriptions across seven diversity dimensions. It raised visual-quality
  concerns in 9/11 clips and boundary-completeness concerns in 8/11 clips.
  These are unverified model judgments; five-second clips are not necessarily
  intended to contain entire tasks. Temporal and tracking overclaims receive
  explicit caveats beside the original text.
- Instruction compliance, task success, action correctness and privacy/safety
  remain unknown because task definitions and collection rules were not supplied.

The full suite passed **167 tests**, with **11 skipped** for unavailable external
fixtures. Browser checks verified all 11 videos, the 15-dimension table,
dimension filtering, interpretation caveats, timestamp seeking and playback,
with no JavaScript errors.

See the [saved summary](../artifacts/hot3d_benchmark/quality_summary.md) and
[full analysis](../artifacts/hot3d_benchmark/quality_analysis.json). The playable
report is served at `http://localhost:18081/quality/` while the SSH tunnel from
the HOT3D setup is active. These findings apply only to this small analyzed
sample, not to the whole HOT3D dataset.
