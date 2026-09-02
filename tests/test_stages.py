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
