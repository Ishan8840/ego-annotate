# HOT3D dense annotation

The `hot3d` command imports native HOT3D-Clips `.tar` files into the existing
span → caption → score path. It uses **annotated UmeTrack hand poses**, not
image-estimated poses. These experiments test downstream annotation transfer;
they do not measure hand-pose estimation accuracy.

## Measured result — October 5, 2026

Tested on the supplied A100 80 GB server with Qwen3-VL-8B-Instruct: **11 Quest3
clips, 55 seconds, 1,650 frames, 8 source sequences and 5 participants**.
The original downstream code and updated code consumed identical prepared,
upright videos and the same annotated hand poses.

| Metric | Original | Updated first pass | Updated + one repair |
|---|---:|---:|---:|
| Format passes | 15/22 (68.2%) | 17/22 (77.3%) | **19/22 (86.4%)** |
| Caption coverage | 22/22 | 22/22 | 22/22 |
| Duration violations | 0 | 0 | 0 |
| Unique captions | 22/22 | 21/22 | 21/22 |
| Aperture-consistency proxy | 6/13 (46.2%) | 5/11 (45.5%) | 5/11 (45.5%) |

The updated first pass took 64.0 s; one bounded repair pass took 77.0 s total,
including five additional model calls. Model loading is excluded. All variants
produced 24 labels/minute of video. On the six additional source clips, format
passes rose from 9/12 to 10/12. These clips were evaluated during development;
this is a small development benchmark, not an untouched semantic test set.

**The demonstrated gain is format compliance and reliability.** The grounding
proxy did not improve, and semantic correctness remains unmeasured. Three
captions still contain multiple action verbs and retain explicit validation
errors: `clip-000001#002.067`, `clip-000500#000.000`, and
`clip-000700#000.000`. Inspection also finds material inconsistency: one spoon
is called plastic in one span and wooden in another. Format-valid output still
needs human review.

Validation: **147 passed, 11 skipped**. Ten skips need the unavailable original
MCAP corpus; one needs an ACE-Ego-Hand checkout at its configured test path. The HOT3D
integration test ran against all imported clips. A headless Chrome check
verified 11 videos, 22 captions, run switching, timestamp seeking, decoded
playback, and no JavaScript errors. Byte-range video serving has a regression
test as well.

Measurements, model revision, source hashes, manifests, captions, raw replies,
retry logs, and the system prompt are in
[`artifacts/hot3d_benchmark`](../artifacts/hot3d_benchmark/).
The local portable review is at
[`artifacts/hot3d/review/index.html`](../artifacts/hot3d/review/index.html).
The server runs the review on loopback port 8081 under Supervisor; port 8080
already hosts Jupyter. Reconnect privately with:

```bash
ssh -F /dev/null -p 20066 -N -L 18081:127.0.0.1:8081 root@213.199.247.187
# Open http://localhost:18081
```

Two prompt experiments were rejected: corpus-specific examples encouraged
repeated object captions, and removing examples hurt word-count compliance.
The final prompt keeps the repository's grammar examples, supplies the legal
domain verbs, and gives explicit correction instructions. Repairs are bounded;
failures are recorded rather than hidden.

## Reproduce

Install the repository requirements, FFmpeg, a CUDA-compatible PyTorch /
torchvision pair, and `requirements-hot3d.txt`. The optional importer uses the
[official hand tracking toolkit](https://github.com/facebookresearch/hand_tracking_toolkit)
for UmeTrack forward kinematics. No MANO model assets are required.

```bash
python -m egoannot hot3d /data/clip-*.tar --out artifacts/hot3d/corpus --rotate 90
export EGO_CORPUS="$PWD/artifacts/hot3d/corpus"
export EGO_SEGMENTS="$EGO_CORPUS/segments"
export EGO_SEGMENT_DEFS="$EGO_CORPUS/segments.json"
export EGO_ARTIFACTS="$PWD/artifacts/hot3d/run"
export CAPTION_GREEDY=1
export QWEN_MODEL=Qwen/Qwen3-VL-8B-Instruct

python -m egoannot spans build --no-quality-gate
python -m egoannot caption run --backend qwen-local
python -m egoannot score
EGO_HOT3D_CORPUS="$EGO_CORPUS" python -m pytest tests -q
```

`--rotate 90` means clockwise. It makes the tested Quest3 physical-left camera
images upright. The default is zero: check a source frame before selecting the
rotation for another device/stream. Rotation preserves handedness and the
full field of view. It does not change world poses or overwrite the native
fisheye calibration. Rendered videos preserve aspect ratio at 768 pixels wide.

The original MCAP quality thresholds were calibrated for a different camera.
This experiment explicitly disables that gate. HOT3D fisheye intrinsics are
preserved in metadata and are **not** substituted with a pinhole approximation.
Do not run MCAP quality/segment-render/PCA training commands on these prepared
files as though they were native MCAPs. The supported path is the importer's
rendered video plus span construction (pose activity or RGB boundaries),
captioning, and scoring. The absent thumb CMC prevents the existing complete
21-joint PCA fit; it is intentionally not synthesized.

## Matched comparison

The benchmark loads one Qwen model and runs the original code, updated first
pass, and updated captioner with one repair attempt. All arms see the same
upright frames and use greedy decoding, four frames per span, and the same
span duration policy. The original source snapshots can be recovered from the
repository commit preceding these changes:

```bash
BASE=412fb0aaba6742bdaab57bcb7669628db327df91
git show "$BASE:egoannot/stages/caption.py" > /tmp/original-caption.py
git show "$BASE:egoannot/stages/spans.py" > /tmp/original-spans.py
git show "$BASE:egoannot/labels/domains.py" > /tmp/original-domains.py
python scripts/benchmark_hot3d.py \
  --original-caption /tmp/original-caption.py \
  --original-spans /tmp/original-spans.py \
  --original-domains /tmp/original-domains.py
python -m egoannot.tools.build_hot3d_review \
  --corpus "$EGO_CORPUS" --run "$EGO_ARTIFACTS" \
  --out artifacts/hot3d/review
```

The comparison measures the combined fixes, not the independent contribution
of every change. Adding legal verbs for general manipulation changes the
validation vocabulary; a format-pass increase is not itself proof of better
semantic accuracy. `first_pass` versus `repaired` isolates bounded validation
repair under the same updated vocabulary. The aperture-agreement metric is a
heuristic and is not action ground truth.

## Data integrity and failure handling

- Native capture timestamps are subtracted as integers before conversion to
  seconds. Duplicated, reversed, nonuniform, or image/annotation-misaligned
  timestamps fail instead of silently drifting against a constant-rate video.
- Physical camera selection uses calibration labels, preferring RGB and then
  physical left. `--stream` overrides it. No mirror transform is applied.
- UmeTrack's 20 landmarks map into the repository's MediaPipe topology in world
  meters. Thumb CMC stays NaN. Unavailable hand observations are omitted, not
  replaced with a stationary or zero-valued hand. The tested clips have both
  hands annotated on all frames; partial tracking gaps remain an unvalidated
  case for activity segmentation.
- `manifest.json` records sequence, source SHA-256, camera, timestamps, units,
  missing joints, frame counts, calibration and display rotation.
- Missing video samples raise an error rather than substituting a nearby frame
  from a different action. Model IDs bind uniquely; mixed IDs cannot relabel
  valid replies and conflicting duplicates remain unresolved.
- The captioner retries only missing/invalid outputs, with the original images
  and specific validation errors. Set `CAPTION_MAX_RETRIES=0` to disable it.
  Failed repairs retain the better candidate with `validation_errors`; missing
  captions remain missing. There are no fabricated fallback labels.
- Completed batches are checkpointed. The `.jsonl.run.json` sidecar stores raw
  responses, attempts, errors, coverage, timing, configuration and missing IDs.
  Zero-caption runs exit with an error. An interrupted run can have a partial
  JSONL without its final sidecar; it is not a completed benchmark.

[HOT3D-Clips format](https://github.com/facebookresearch/hot3d/blob/main/hot3d/clips/README.md)
provides the source data contract. The sample tested here uses Quest3 monochrome
images and UmeTrack annotations. It does not validate the Aria path or MANO-only
clips.

Hand assignment still uses wrist path-length ratios. A `BOTH` tag means both
hands contributed motion, not that both touched the same object. Captions and
hand tags should be reviewed together; the current linter does not establish
physical contact or semantic correctness from this heuristic.
