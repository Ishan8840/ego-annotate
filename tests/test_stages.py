"""Stage logic that can be tested without touching the corpus."""
import json

import numpy as np
import pytest

from egoannot import config
from egoannot.stages import caption, events, quality, spans


# ---------------------------------------------------------------- quality T1
def clean(**kw):
    m = dict(duration_s=4.0, missing_topics=[], n_frames=8, sharpness_p25=30.0,
             brightness_out_frac=0.0, max_wrist_speed=1.0, max_head_speed=0.5,
             max_angular_speed=1.0, hand_visible_rate=1.0,
             hand_central50_rate=0.95)
    m.update(kw)
    return m


def test_t1_selftest_passes_without_footage():
    assert quality.selftest([])


@pytest.mark.parametrize("fault,reason", [
    (dict(duration_s=1.0), "clip_too_short"),
    (dict(missing_topics=["/pose/left_hand"]), "missing_modality"),
    (dict(n_frames=0), "no_video_frames"),
    (dict(sharpness_p25=1.0), "blurred"),
    (dict(brightness_out_frac=0.4), "brightness_out_of_range"),
    (dict(max_wrist_speed=8.0), "implausible_wrist_velocity"),
    (dict(hand_central50_rate=0.2), "hands_poorly_framed"),
    (dict(hand_visible_rate=0.1), "hands_out_of_view"),
])
def test_t1_fires_on_each_declared_fault(fault, reason):
    assert reason in quality.hard_rejects(clean(**fault))


def test_t1_accepts_a_clean_clip():
    assert quality.hard_rejects(clean()) == []


def test_motion_filter_ranks_within_each_episode():
    """
    Pooled ranking made T2 an episode selector: on this corpus it dropped 0%
    of one episode and 75% of another.
    """
    records = ([dict(episode="calm", head_motion=1.0 + i * 0.01,
                     hard_reject=False, reject_reasons=[]) for i in range(10)]
               + [dict(episode="busy", head_motion=50.0 + i, hard_reject=False,
                       reject_reasons=[]) for i in range(10)])
    quality.apply_motion_filter(records, dict(config.QUALITY,
                                              motion_scope="episode"))
    for name in ("calm", "busy"):
        group = [r for r in records if r["episode"] == name]
        assert sum(r["motion_reject"] for r in group) == 3

    quality.apply_motion_filter(records, dict(config.QUALITY,
                                              motion_scope="global"))
    calm = [r for r in records if r["episode"] == "calm"]
    busy = [r for r in records if r["episode"] == "busy"]
    assert sum(r["motion_reject"] for r in calm) == 0
    assert sum(r["motion_reject"] for r in busy) == 6


# ---------------------------------------------------------------- events
def test_state_levels_caps_a_degenerate_closed_level():
    """A p30 of 68 mm would make "closed" mean a wide-open hand."""
    wide = np.full(100, 0.090)
    lv = events.state_levels(wide)
    assert lv["closed"] <= config.EVENTS["closed_abs_max"]


def test_state_levels_flags_an_unusable_gate():
    """t_keyboard's closed and open levels sit ~7 mm apart, inside pose error."""
    tight = np.linspace(0.029, 0.037, 200)
    assert events.state_levels(tight)["usable"] is False
    wide = np.linspace(0.005, 0.120, 200)
    assert events.state_levels(wide)["usable"] is True


def test_dwell_absorbs_short_runs_and_reconverges():
    """The single-pass version left runs below the dwell minimum behind."""
    t = np.arange(40) * 0.1
    st = np.array(["idle"] * 10 + ["reach"] * 2 + ["idle"] * 3 + ["reach"] * 2
                  + ["manipulate"] * 23, dtype=object)
    out = events._apply_dwell(st.copy(), t, 0.5)
    runs = [(s, sum(1 for _ in g)) for s, g in __import__("itertools").groupby(out)]
    for state, n in runs[1:]:
        assert n * 0.1 >= 0.5 - 1e-9, runs


def test_grasps_pair_contact_with_release():
    ev = [dict(hand="right", type="contact", t=1.0),
          dict(hand="right", type="release", t=3.0)]
    held = events.grasps(ev)
    assert len(held) == 1 and held[0]["duration"] == 2.0


# ---------------------------------------------------------------- spans band
def _activity(n=2001, span=20.0):
    t = np.linspace(0, span, n)
    return t, np.abs(np.sin(t * 1.2)) + 0.05


def test_band_policy_brings_every_span_into_the_linter_band():
    t, act = _activity()
    lo, hi = config.SPANS_CFG["band"]
    intervals = [(0.0, 0.4), (0.4, 11.6), (11.6, 13.0), (13.0, 20.0)]
    kept, stats = spans.enforce_band(intervals, t, act, config.SPANS_CFG)
    d = np.array([z - a for a, z in kept])
    assert (d >= lo).all() and (d <= hi).all()
    assert stats["split"] > 0 and stats["merged"] > 0


def test_band_policy_keeps_spans_non_overlapping_and_ordered():
    t, act = _activity()
    intervals = [(0.0, 6.0), (6.0, 6.5), (6.5, 20.0)]
    kept, _ = spans.enforce_band(intervals, t, act, config.SPANS_CFG)
    for (a1, z1), (a2, z2) in zip(kept, kept[1:]):
        assert z1 <= a2 + 1e-9
        assert a1 < z1


def test_band_policy_drops_a_short_span_it_cannot_legally_merge():
    """Dropping is allowed: labels must not overlap, but need not tile."""
    t, act = _activity(span=8.0)
    lo, hi = config.SPANS_CFG["band"]
    intervals = [(0.0, 3.9), (3.9, 4.1), (4.1, 8.0)]
    kept, stats = spans.enforce_band(intervals, t, act, config.SPANS_CFG)
    d = np.array([z - a for a, z in kept])
    assert (d >= lo).all() and (d <= hi).all()
    assert stats["dropped"] == 1


def test_band_policy_is_idempotent():
    t, act = _activity()
    intervals = [(0.0, 0.4), (0.4, 11.6), (11.6, 20.0)]
    once, _ = spans.enforce_band(intervals, t, act, config.SPANS_CFG)
    twice, stats = spans.enforce_band(once, t, act, config.SPANS_CFG)
    assert once == twice
    assert stats == dict(split=0, merged=0, dropped=0)


# ---------------------------------------------------------------- quality gate
def test_quality_gate_defaults_to_t1_and_ignores_the_t2_ranking():
    records = [
        dict(episode="e", start_ts=0.0, end_ts=4.0, hard_reject=True,
             motion_reject=False),
        dict(episode="e", start_ts=4.0, end_ts=8.0, hard_reject=False,
             motion_reject=True),
    ]
    t1 = spans.rejected_intervals(records, ("T1",))
    assert t1["e"] == [(0.0, 4.0)]
    both = spans.rejected_intervals(records, ("T1", "T2"))
    assert both["e"] == [(0.0, 4.0), (4.0, 8.0)]


def test_bad_overlap_measures_the_covered_fraction():
    bad = [(0.0, 2.0)]
    assert spans.bad_overlap(0.0, 4.0, bad) == pytest.approx(0.5)
    assert spans.bad_overlap(2.0, 4.0, bad) == 0.0
    assert spans.bad_overlap(1.0, 3.0, bad) == pytest.approx(0.5)
    assert spans.bad_overlap(0.0, 4.0, []) == 0.0


def test_acting_sides_follows_the_dominant_hand():
    assert spans.acting_sides("LEFT") == ["left"]
    assert spans.acting_sides("RIGHT") == ["right"]
    assert spans.acting_sides("BOTH") == ["right", "left"]


# ---------------------------------------------------------------- caption bind
def _batch(n=3):
    return [dict(span_id=f"a#{i}") for i in range(n)]


def test_correct_ids_bind():
    raw = "\n".join(json.dumps(dict(span_id=f"a#{i}", text="x")) for i in range(3))
    got = caption.parse_objects(raw, _batch())
    assert [o["span_id"] for o in got] == ["a#0", "a#1", "a#2"]


def test_a_partial_reply_with_bad_ids_binds_nothing():
    """
    The old positional fallback bound a short reply to the first N spans. One
    measured run mapped 8 objects onto spans 0-7 of a 10-span batch regardless
    of which spans they described.
    """
    raw = '{"span_id":"bogus","text":"x"}\n{"span_id":"bogus","text":"y"}'
    assert caption.parse_objects(raw, _batch()) == []


def test_a_full_reply_without_ids_binds_by_order():
    raw = '{"text":"x"}\n{"text":"y"}\n{"text":"z"}'
    got = caption.parse_objects(raw, _batch())
    assert [o["span_id"] for o in got] == ["a#0", "a#1", "a#2"]


def test_a_partial_reply_with_good_ids_binds_only_those():
    raw = '{"span_id":"a#1","text":"y"}'
    got = caption.parse_objects(raw, _batch())
    assert [o["span_id"] for o in got] == ["a#1"]


def test_prose_around_the_json_is_tolerated():
    raw = ('Here are the captions:\n```json\n{"span_id":"a#0","text":"x"}\n```\n'
           'Hope that helps.')
    got = caption.parse_objects(raw, _batch(1))
    assert len(got) == 1 and got[0]["text"] == "x"


def test_system_prompt_carries_the_verbs_and_the_frame_count():
    text = caption.system_prompt("kitchen", frames_per_span=6)
    assert "6 frames" in text
    for verb in ("grasp", "place", "release"):
        assert verb in text


# ---------------------------------------------------------------- frame store
def test_frame_store_refuses_a_missing_segment():
    """
    Silently returning no frames meant the captioner ran blind on that
    segment: prompts with zero images, and nothing in the output saying so.
    """
    from egoannot.core.video import SegmentFrames
    store = SegmentFrames(config.SEGMENTS_DIR)
    with pytest.raises(FileNotFoundError):
        store.plan([dict(segment="does_not_exist", v_start=0.0, v_end=2.0)], 4)


def test_frame_store_plans_deduplicated_indices():
    from egoannot.core.video import SegmentFrames
    store = SegmentFrames(config.SEGMENTS_DIR)
    store._fps["fake"] = 30.0                      # skip the probe
    idx = store.indices_for("fake", 1.0, 3.0, 4)
    assert idx == sorted(set(idx))
    assert all(30 <= i <= 89 for i in idx), idx


# ---------------------------------------------------------------- refinement
def _signal(n=4001, span=40.0, period=1.2):
    """A synthetic activity signal with regular, prominent minima."""
    t = np.linspace(0, span, n)
    return t, np.abs(np.sin(t * period)) + 0.05


def _cfg(**kw):
    return dict(config.SPANS_CFG, **kw)


def test_refinement_moves_a_boundary_onto_a_quieter_minimum():
    t, act = _signal()
    minima = spans._minima_times(t, act, config.SPANS_CFG["prominence"])
    assert len(minima) > 4
    # place a boundary just off a real minimum, with room on both sides
    m = float(minima[3])
    intervals = [(m - 2.2, m + 0.25), (m + 0.25, m + 2.4)]
    out, prov, stats = spans.refine_boundaries(intervals, t, act, _cfg())
    assert stats["shifted"] == 1
    assert prov[1]["shifted"] is True
    assert abs(out[0][1] - m) < abs(intervals[0][1] - m)      # moved toward it
    assert out[0][1] == out[1][0]                              # still shared
    assert abs(prov[1]["delta"]) <= config.SPANS_CFG["boundary_shift_s"] + 1e-9


def test_refinement_never_moves_further_than_the_window():
    t, act = _signal()
    intervals = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0)]
    out, prov, _ = spans.refine_boundaries(intervals, t, act, _cfg())
    shift = config.SPANS_CFG["boundary_shift_s"]
    for p in prov:
        if p:
            assert abs(p["delta"]) <= shift + 1e-9, p


def test_a_shift_that_would_break_the_band_is_rejected():
    """
    The left span is already at the 4.0 s ceiling, so any later boundary is
    illegal however quiet it is.
    """
    t, act = _signal()
    minima = spans._minima_times(t, act, config.SPANS_CFG["prominence"])
    m = float(minima[4])
    lo, hi = config.SPANS_CFG["band"]
    # left span exactly at the ceiling; the quiet minimum sits 0.18 s to the
    # RIGHT, so taking it would push the left span past 4.0 s
    intervals = [(m - 0.18 - hi, m - 0.18), (m - 0.18, m + 2.0)]
    assert abs((intervals[0][1] - intervals[0][0]) - hi) < 1e-9
    out, prov, stats = spans.refine_boundaries(intervals, t, act, _cfg())
    assert out == [tuple(iv) for iv in intervals]
    assert prov[1]["shifted"] is False
    assert "band" in prov[1]["reason"]
    assert stats["shifted"] == 0


def test_a_shift_that_would_violate_min_gap_is_rejected():
    """
    Both spans start above min_gap_s, so neither may be shortened past it even
    though the band floor (1.3 s) would still allow it.
    """
    t, act = _signal()
    minima = spans._minima_times(t, act, config.SPANS_CFG["prominence"])
    gap = config.SPANS_CFG["min_gap_s"]
    m = float(minima[5])
    # right span is exactly at min_gap_s; the quiet minimum lies to the right,
    # which would take the right span below it
    intervals = [(m - 2.4, m - 0.22), (m - 0.22, m - 0.22 + gap)]
    out, prov, stats = spans.refine_boundaries(intervals, t, act, _cfg())
    assert prov[1]["shifted"] is False
    assert "min_gap" in prov[1]["reason"]
    assert out == [tuple(iv) for iv in intervals]


def test_refinement_leaves_every_span_inside_the_band():
    t, act = _signal(span=60.0, n=6001)
    intervals = [(x, x + 2.0) for x in np.arange(0.0, 56.0, 2.0)]
    out, _, _ = spans.refine_boundaries(intervals, t, act, _cfg())
    lo, hi = config.SPANS_CFG["band"]
    d = np.array([z - a for a, z in out])
    assert (d >= lo - 1e-9).all() and (d <= hi + 1e-9).all()


def test_refinement_preserves_order_and_sharing():
    t, act = _signal(span=60.0, n=6001)
    intervals = [(x, x + 2.0) for x in np.arange(0.0, 56.0, 2.0)]
    out, _, _ = spans.refine_boundaries(intervals, t, act, _cfg())
    for (a1, z1), (a2, z2) in zip(out, out[1:]):
        assert a1 < z1
        assert abs(z1 - a2) < 1e-9          # boundaries stay shared, never cross


def test_refinement_is_idempotent():
    """Re-running on its own output must change nothing."""
    t, act = _signal(span=60.0, n=6001)
    intervals = [(x, x + 2.0) for x in np.arange(0.0, 56.0, 2.0)]
    once, prov1, stats1 = spans.refine_boundaries(intervals, t, act, _cfg())
    twice, _, stats2 = spans.refine_boundaries(once, t, act, _cfg())
    assert twice == once
    assert stats2["shifted"] == 0
    assert stats1["shifted"] > 0            # the first pass did do something


def test_refinement_is_deterministic_across_runs():
    t, act = _signal(span=60.0, n=6001)
    intervals = [(x, x + 2.0) for x in np.arange(0.0, 56.0, 2.0)]
    a_out, a_prov, a_stats = spans.refine_boundaries(intervals, t, act, _cfg())
    b_out, b_prov, b_stats = spans.refine_boundaries(intervals, t, act, _cfg())
    assert a_out == b_out
    assert a_prov == b_prov
    assert a_stats == b_stats


def test_refinement_only_moves_boundaries_shared_by_two_spans():
    """A gap edge abuts material the band policy dropped; it stays put."""
    t, act = _signal()
    intervals = [(0.0, 2.0), (5.0, 7.0)]          # not contiguous
    out, prov, stats = spans.refine_boundaries(intervals, t, act, _cfg())
    assert out == [tuple(iv) for iv in intervals]
    assert stats["shifted"] == 0
    assert prov[1] is None


@pytest.mark.parametrize("off", [
    dict(boundary_refine=False),          # what --no-boundary-refine sets
    dict(boundary_shift_s=0.0),           # a zero window
])
def test_refinement_can_be_switched_off(off):
    """Both routes must reproduce the pre-refinement cut exactly."""
    t, act = _signal()
    intervals = [(x, x + 2.0) for x in np.arange(0.0, 36.0, 2.0)]
    baseline, _, base_stats = spans.refine_boundaries(intervals, t, act, _cfg())
    out, prov, stats = spans.refine_boundaries(intervals, t, act, _cfg(**off))
    assert out == [tuple(iv) for iv in intervals]
    assert stats["shifted"] == 0
    assert all(p is None for p in prov)
    assert base_stats["shifted"] > 0          # refinement does move these


def test_too_large_a_window_is_refused_rather_than_inverting_spans():
    t, act = _signal()
    lo, _ = config.SPANS_CFG["band"]
    with pytest.raises(ValueError, match="invert"):
        spans.refine_boundaries([(0.0, 2.0), (2.0, 4.0)], t, act,
                                _cfg(boundary_shift_s=lo / 2 + 0.01))


def test_provenance_records_what_happened_at_every_boundary():
    t, act = _signal(span=40.0)
    intervals = [(x, x + 2.0) for x in np.arange(0.0, 36.0, 2.0)]
    _, prov, _ = spans.refine_boundaries(intervals, t, act, _cfg())
    assert prov[0] is None and prov[-1] is None          # segment endpoints
    interior = [p for p in prov if p]
    assert len(interior) == len(intervals) - 1
    for p in interior:
        assert set(p) == {"orig", "final", "delta", "n_cand", "shifted", "reason"}
        assert p["n_cand"] >= 1                          # the original always counts
        assert (p["reason"] is None) == p["shifted"]
        assert abs((p["final"] - p["orig"]) - p["delta"]) < 2e-3


def test_refinement_declines_to_touch_an_out_of_band_span():
    """
    With --no-band a span can be far over the ceiling. Refinement must not
    silently "fix" it, and must not shift it either: every candidate fails the
    band check, which is recorded rather than hidden.
    """
    t, act = _signal(span=40.0)
    intervals = [(0.0, 11.6), (11.6, 13.6)]
    out, prov, stats = spans.refine_boundaries(intervals, t, act, _cfg())
    assert out == [tuple(iv) for iv in intervals]
    assert prov[1]["shifted"] is False and prov[1]["reason"] == "band"
    assert stats["shifted"] == 0
