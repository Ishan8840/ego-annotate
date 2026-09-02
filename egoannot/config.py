"""
Every path and every threshold in one place.

Paths resolve from the environment first, then from a repo-local `corpus`
symlink, then from a discovery glob. Nothing in the package hardcodes a
session-specific scratchpad directory the way the phase scripts did -- that
was the single biggest source of breakage when a session ended.

Thresholds keep the provenance comments from the original stages. A value
without a comment saying where it came from is a value nobody can defend, so
they travel together.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = Path(os.environ.get("EGO_ARTIFACTS", ROOT / "artifacts"))
DOCS = ROOT / "docs"

# ---------------------------------------------------------------- corpus
_CORPUS_GLOB = "/tmp/claude-*/-home-axibo-post-ego/*/scratchpad/lw"


def _discover_corpus() -> Path | None:
    """
    Where the .mcap episodes live. Order: explicit env var, repo symlink,
    then newest matching scratchpad. Returns None rather than guessing wrong.
    """
    env = os.environ.get("EGO_CORPUS")
    if env:
        return Path(env)
    link = ROOT / "corpus"
    if link.exists():
        return link.resolve()
    hits = [p for p in glob.glob(_CORPUS_GLOB) if os.path.isdir(p)]
    if hits:
        return Path(max(hits, key=os.path.getmtime))
    return None


def corpus() -> Path:
    """The corpus root, or a clear error naming how to set it."""
    p = _discover_corpus()
    if p is None or not p.is_dir():
        raise FileNotFoundError(
            "No episode corpus found. Set EGO_CORPUS=/path/to/mcap/root, or "
            "create a `corpus` symlink in the repo root pointing at it."
        )
    return p


def episode_path(name: str) -> Path:
    """
    Resolve an episode name to a file. The corpus keeps some episodes at the
    root and some under ep/, so try both instead of assuming one layout.
    """
    root = corpus()
    for cand in (name, f"{name}.mcap", f"ep/{name}.mcap", f"ep/{name}"):
        p = root / cand
        if p.exists():
            return p
    raise FileNotFoundError(f"episode {name!r} not found under {root}")


# ---------------------------------------------------------------- artifacts
def artifact(*parts: str) -> Path:
    """A path under artifacts/, with parent directories created."""
    p = ARTIFACTS.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


QUALITY_RECORDS = ARTIFACTS / "quality" / "clip_quality.jsonl"
EVENTS_RECORDS = ARTIFACTS / "events" / "events.jsonl"
FEATURES_DIR = ARTIFACTS / "features"
CLOSURE_PCA = ARTIFACTS / "features" / "closure_pca.npz"
SEGMENTS_DIR = Path(os.environ.get("EGO_SEGMENTS", ARTIFACTS / "segments"))
SPANS = ARTIFACTS / "spans" / "spans.jsonl"
CAPTIONS = ARTIFACTS / "captions" / "captions.jsonl"

# Overridable like the corpus and the artifact root, so a held-out corpus can
# be run end to end without editing the project's own segment definitions.
SEGMENT_DEFS = Path(os.environ.get("EGO_SEGMENT_DEFS", DATA / "segments.json"))
GOLD = DATA / "gold.json"
LINT_POSITIVES = DATA / "lint_positives.jsonl"
LINT_NEGATIVES = DATA / "lint_negatives.jsonl"


# ---------------------------------------------------------------- stage: quality
# Every value is resolution- and rate-specific. Re-run `calibrate` on new
# footage before trusting them; the blur floors in particular were derived
# from this corpus at 1920x1456.
QUALITY = dict(
    clip_len_s=4.0,          # candidate annotation window
    min_clip_s=2.0,          # T1: shorter than this cannot hold an atomic label
    blur_fps=2.0,            # T1: native-resolution sampling rate
    flow_fps=4.0,            # T2: spec rate
    flow_size=256,           # T2: spec size
    tile_grid=(8, 6),        # T1: blur tiling (cols, rows)
    lowtex_floor=0.5,        # T1: calibrated p10 of observed tile variance
    lowtex_max_frac=0.60,    # T1: more low-texture tiles than this => really blurred
    blur_floor=4.0,          # T1: between sigma=3 blur p95 (3.8) and real p5 (4.8)
    bright_lo=25.0,          # T1: mean luma floor (0-255)
    bright_hi=235.0,         # T1: mean luma ceiling
    bright_bad_frac=0.30,    # T1: reject if this fraction of frames is out of range
    max_sat_frac=0.10,       # T1: >10% of pixels at saturation = blown out
    max_dark_frac=0.35,      # T1: >35% of pixels crushed to black
    max_wrist_speed=6.0,     # T1: m/s, above = tracking glitch
    max_head_speed=4.0,      # T1: m/s
    max_ang_speed=15.0,      # T1: rad/s
    # PIQE, the no-reference metric ego/robot curation pipelines actually use.
    # It does NOT replace the Laplacian sharpness gate -- measured on this
    # corpus the two are orthogonal and each covers the other's blind spot:
    # Laplacian falls correctly with blur (5.3 -> 1.0 at sigma=5) but is fooled
    # by noise, rating a noisy frame 4526; PIQE catches noise (2.9 -> 3.9) but
    # models blockiness and noise rather than sharpness, so it rates a BLURRED
    # frame as cleaner (2.9 -> 1.3). Carried as a second channel, not a swap.
    piqe_activity=0.008,     # T1: calibrated; the published 0.1 judges 3.7% of blocks here
    piqe_max=None,           # T1: noise gate; None until calibrated on your footage
    min_hand_vis=0.30,       # T1: full-frame gate; inert on this corpus (always 1.0)
    min_hand_c50=0.60,       # T1: fraction of hand joints inside the central 50%
    drop_motion_frac=0.30,   # T2: drop this top fraction by head motion
    # Rank head motion within each episode, not across the pooled corpus.
    # Pooled ranking made T2 an episode selector: measured drop rates ran from
    # 0% (test, cart_wipes) to 75% (belts_a, which lost every clip it had),
    # because optical-flow magnitude is scene-dependent.
    motion_scope="episode",  # "episode" | "global"
)

# ---------------------------------------------------------------- stage: events
EVENTS = dict(
    smooth_s=0.15,           # aperture/velocity smoothing window
    da_floor=0.040,          # m/s: minimum |d aperture/dt| for an event
    da_pct=90,               # adaptive: also require |da| above this percentile
    min_delta=0.010,         # m: an event must change aperture by at least this
    min_gap_s=0.30,          # refractory period between events on one hand
    # Absolute-state gate. A decrease in aperture is not a grasp: the hand has
    # to END UP closed. Without this, 74% of detected contacts left the hand
    # still open. Note aperture_delta does NOT predict validity, so tightening
    # min_delta would not have helped.
    require_state=True,
    closed_pct=30,           # per-hand percentile defining "closed"
    open_pct=65,             # per-hand percentile defining "open"
    # Percentile gates have no absolute grounding: measured on this corpus,
    # "closed" is <=68 mm on noodles but <=18 mm on d_contactlens, and on
    # t_keyboard the closed and open bands sit 7 mm apart -- inside the hand
    # model's own error. These floors stop the gate degenerating there.
    closed_abs_max=0.055,    # m: "closed" can never mean wider than this
    open_abs_min=0.030,      # m: "open" can never mean narrower than this
    min_state_margin=0.012,  # m: below this closed/open separation, gate is unusable
    state_hold_s=0.40,       # the state must persist this long after the event
    jerk_snap_s=0.20,        # window to snap contact onto a jerk peak
    dwell_s=0.50,            # actionness: minimum state duration (hysteresis)
    # Actionness thresholds are pooled percentiles over 17 episodes, not
    # guesses. The first pass guessed them and produced a degenerate 99%
    # "manipulate": w_head_scan was 1.40 rad/s, above the corpus p95 of 1.115,
    # so "inspect" could never fire at all.
    v_hand_active=0.413,     # m/s wrist speed                     (p70)
    ap_active=0.079,         # m/s aperture rate                   (p70)
    reach_active=0.093,      # m/s arm extension rate (upper body)  (p80)
    v_torso_move=0.430,      # m/s torso translation               (p90)
    w_head_scan=0.871,       # rad/s head angular speed            (p90)
)

# ---------------------------------------------------------------- stage: features
FEATURES = dict(
    smooth_s=0.15,
    max_wrist_speed=6.0,     # m/s, from the quality stage's measurements
    pca_components=6,
    fit_frames_per_hand=800, # subsample per hand per episode when fitting
)

# ---------------------------------------------------------------- stage: spans
SPANS_CFG = dict(
    signal="activity",       # "activity" | "velocity"
    min_gap_s=1.7,           # minimum separation between detected boundaries
    prominence=0.14,         # trough prominence on the normalised activity signal
    # The linter rejects spans outside 1.3-4.0 s (rule A1). That band is known
    # here, so enforce it at cut time instead of letting the captioner take the
    # blame: measured on the previous output, every single A1 failure was a
    # span-builder artifact (two spans under 1.3 s, four over 4.0 s, one 11.6 s).
    # Long spans also drive repetition -- with no single atomic action to
    # describe, the model falls back on a generic caption.
    band=(1.3, 4.0),
    enforce_band=True,
    split_prominence_factor=0.35,   # relaxed prominence when subdividing a long span
    # Boundary refinement. Once cut, a boundary used to be final, and the
    # captioner had to describe whatever the measured boundary happened to
    # contain -- which is where the vision-only control arm beats this one on
    # verb/aperture agreement (79% vs 66%), because it cuts where the action it
    # describes actually occurs. Each interior cut may now move onto a nearby
    # quieter minimum of the same activity signal.
    #
    # 0.4 s is a bound, not a fitted value. It is 2x the events stage's
    # jerk-snap window (0.20 s), the scale at which this corpus's own contact
    # timing is uncertain, and it satisfies 2 * shift < band floor
    # (0.8 < 1.3) so two boundaries moving toward each other can never merge,
    # reorder or invert their spans. It also sits well under min_gap_s = 1.7.
    boundary_refine=True,
    boundary_shift_s=0.4,
    # Drop spans that overlap clips the quality stage rejected. The quality
    # records existed but no downstream stage ever read them.
    quality_gate=True,
    # T1 only by default. T2 drops the top 30% of every episode by head motion
    # by construction, so gating spans on it removes about a third of them
    # whether or not anything is wrong; add "T2" when a calm subset is wanted.
    quality_gate_tiers=("T1",),
    quality_overlap_frac=0.5,       # drop if this much of the span sits in a bad clip
    dominant_ratio=1.6,             # path-length ratio for LEFT/RIGHT vs BOTH
    rotation_min_deg=55.0,          # net accumulated rotation to call a twist
    rotation_min_coherence=0.45,    # net/total: separates a twist from a wobble
    pinch_max_m=0.022,              # absolute ceiling for "thumb-index pinch"
    pinch_max_pct=20,               # and it must be low for THIS episode too
    extended_pct=85,                # a digit counts as extended above this percentile
)

# ---------------------------------------------------------------- stage: caption
CAPTION = dict(
    frames_per_span=int(os.environ.get("FRAMES_PER_SPAN", "4")),
    # Measured on 269 spans with Qwen3-VL-8B, identical throughput (0.62 vs
    # 0.60 spans/s) and a 269/269 bind rate either way:
    #
    #   spans/call   atomicity   uniqueness   >15 words   grounding
    #            1         67%        56.1%          15         62%
    #            5         75%        66.9%           0         52%
    #
    # Batching makes the model produce distinct captions within one reply, and
    # disciplines length -- every over-long caption disappeared. It costs a
    # little agreement with the measured aperture trend. Safe now only because
    # binding is strict: the old positional fallback silently attached a short
    # reply to the first N spans.
    spans_per_call=int(os.environ.get("SPANS_PER_CALL", "5")),
    context_n=int(os.environ.get("CONTEXT_N", "4")),
    context_mode=os.environ.get("CONTEXT_MODE", "sequential"),  # sequential|shuffled
    free_text=os.environ.get("FREE_TEXT", "0") == "1",
    jpeg_quality=80,
    anthropic_model=os.environ.get("CAPTION_MODEL", "claude-opus-5"),
    openai_base=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
    openai_model=os.environ.get("OPENAI_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"),
    qwen_model=os.environ.get("QWEN_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
    temperature=0.4,
    # Greedy decoding makes a run reproducible from the prompt alone. Off in
    # production (sampling reads better), on when A/B-ing something upstream:
    # otherwise sampling noise is confounded with the change under test.
    greedy=os.environ.get("CAPTION_GREEDY", "0") == "1",
)

# ---------------------------------------------------------------- stage: segments
SEGMENT_RENDER = dict(
    overlay_fps=10.0,
    out_w=512,
    out_h=384,            # 4:3
    crf=os.environ.get("CRF", "34"),
    crop_pad=0.55,
)
