EGO ANNOTATION + QUALITY FILM

30 seconds / 1920x1080 / 30 FPS / H.264 MP4
Text-led and understandable muted. Warm ambient chord score; no voiceover.

Deliverables in artifacts/demo/:
  ego-demo.mp4             Film with music
  ego-demo-silent.mp4      Silent version
  ego-poster.jpg           Cover image
  provenance.json             Sources, edit decisions, and claim definitions
  verification.json           Export validation
  browser-verification.json   Delivery playback and seeking checks

Previous monochrome HOT3D deliverables are preserved in artifacts/demo/v1-hot3d/.

Visual direction: navy, cool white and ice blue; Inter Display + JetBrains Mono.
Font licenses are in demo/fonts/. The film and delivery page are unbranded.

Timeline:
  00–03   Every action. In detail. Color cutting footage and dense timeline.
  03–17   Drawer / knife / washing sequence with changing model captions.
  17–22   Zucchini cutting sequence with changing model captions.
  22–30   Directly into ego quality: cropping/glare observations and combined quality estimate.

The pan segment, density-summary card and branded outro have been removed.

Sources:
  EPIC-KITCHENS P01_01.MP4, original RGB video.
  Prep excerpt: source 41–65 seconds.
  Cutting excerpt: source 90–114 seconds.
  https://epic-kitchens.github.io/
  https://github.com/epic-kitchens/epic-kitchens-download-scripts
  Source video URL is saved in provenance.json.

Annotation workflow:
  python scripts/annotate_rgb_demo.py artifacts/color-demo/epic-prep.mp4 \
    artifacts/color-demo/epic-cook.mp4 --out artifacts/color-demo/dense-v2 \
    --model PATH_TO_QWEN3_VL_8B

The RGB runner reuses optical-flow boundaries, span duration enforcement and
refinement, frame sampling, the Qwen backend, caption binding, and the linter.
It uses the food_preparation domain with a 1.3–2.5-second duration band.
All model replies, validation flags and missing outputs are recorded.
Dataset labels are not passed to the caption model. Handedness, when retained
in the JSON, is a visual model prediction; no measured poses are available.

Build:
  1. Copy both source MP4s and dense-v2 results under artifacts/color-demo/.
     Run quality analyze on the same inputs with --visual qwen-local, output
     artifacts/color-demo/quality/. Saved visual observations remain subjective.
     CAPTION_GREEDY=1 python3 demo/score-quality.py --model PATH_TO_QWEN3_VL_8B
     Saves a normalized 0–100 model estimate across eight equally weighted
     visual criteria, plus per-window evidence, raw replies, and a score breakdown.
     It is an unvalidated estimate, not a probability or percent of usable frames.
  2. python3 demo/prepare-rgb.py
     Checks source hashes and complete caption coverage, extracts JPEG frames,
     and writes film-data.json and provenance.json from actual saved outputs.
  3. node demo/render-film.cjs --stills
  4. node demo/render-film.cjs
  5. python3 demo/make-music.py (requires numpy).
     Slow chord swells, warm harmonics, diffuse stereo reverb; no percussion.
  6. python3 demo/mix-audio.py
     Preserves chord dynamics using a fixed gain targeting -20 LUFS,
     then encodes stereo AAC at 192 kbps / 48 kHz with faststart enabled.
  7. python3 demo/verify-export.py
  8. Copy the film, silent film, poster, provenance, verification and watch.html
     (as index.html) into the private review server's demo/ directory.
  9. node demo/render-film.cjs --verify-delivery

The renderer uses Playwright from PLAYWRIGHT_PATH or /tmp/ego-review-browser.
CHROME_PATH defaults to /opt/google/chrome/chrome.
Serve the repository locally and open demo/preload-film.html for a scrubber.
The renderer refuses a final export when film-data.json is marked preview-only.

Captions are shown verbatim. Density statistics refer only to the two 24-second
excerpts, not the full dataset. Density and format compliance are not measures
of semantic accuracy. Raw outputs retain model errors and uncertainty flags.
The film is an edited presentation of saved outputs, not live inference speed.
No skeleton, hand-pose or trajectory graphics appear in this revision.

Suggested X post:

Every action. In detail.

Turn color egocentric video into dense, time-aligned captions:
action, object, timestamp, context — plus ego-video quality evidence.

Real footage. Real pipeline outputs. Sound on or off.
