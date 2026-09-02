"""
End-to-end checks that need the episode corpus. Skipped when it is absent, so
the rest of the suite still runs on a machine without the .mcap files.
"""
import numpy as np
import pytest

from egoannot import config

pytestmark = pytest.mark.skipif(
    config._discover_corpus() is None,
    reason="no episode corpus (set EGO_CORPUS or create a `corpus` symlink)")


def need_episode(name):
    """Skip rather than error when a corpus does not carry this episode."""
    try:
        return config.episode_path(name)
    except FileNotFoundError:
        pytest.skip(f"corpus has no episode {name!r}")


def need_segment(sid):
    """Skip rather than error when the segment definitions omit this id."""
    import json
    ids = {s["id"] for s in json.load(open(config.SEGMENT_DEFS))}
    if sid not in ids:
        pytest.skip(f"segment definitions have no {sid!r}")
    return sid


@pytest.fixture(scope="module")
def episode():
    from egoannot.core.mcap_io import read_episode
    return read_episode(need_episode("np_tissue"))


def test_reader_returns_a_coherent_episode(episode):
    assert episode["duration_s"] > 0
    assert episode["src_fps"] in (25.0, 30.0)
    assert episode["K"]["w"] > 0 and episode["K"]["h"] > 0
    for topic in ("/pose/left_hand", "/pose/right_hand", "/pose/head"):
        assert len(episode[topic]) > 0
    assert episode["/pose/right_hand_joints"].shape[1:] == (21, 3)


def test_hand_streams_share_a_timebase(episode):
    """Several stages index both hands with one frame counter."""
    assert episode["hands_share_timebase"]


def test_pose_only_read_skips_the_video_bitstream():
    from egoannot.core.mcap_io import read_episode
    ep = read_episode(need_episode("np_tissue"), want_video=False)
    assert ep["vid"] == b""
    assert len(ep["/pose/right_hand"]) > 0


def test_declared_depth_is_never_delivered(episode):
    """A documented property of this corpus, asserted so a change is noticed."""
    from egoannot.core.mcap_io import modality_audit
    missing, undelivered = modality_audit(episode)
    assert missing == []
    assert "head_depth_camera" in undelivered


def test_hand_visibility_saturates_on_this_corpus(episode):
    """
    The shipped hand pose is near-rigidly coupled to the head camera, so the
    full-frame in-FOV rate is 1.0 and only the central-50% variant has spread.
    """
    from egoannot.stages.quality import hand_visibility
    hv = hand_visibility(episode)
    assert hv is not None
    assert hv["rate"] > 0.99
    assert 0.0 < hv["rate_c50"] <= 1.0


def test_camera_convention_is_recovered_from_geometry(episode):
    from egoannot.core.geometry import camera_convention
    conv = camera_convention(episode["extr"], episode["/pose/right_hand"])
    assert conv["axis"] in (1.0, -1.0)
    front = min(conv["median_angle_pz"], conv["median_angle_mz"])
    assert front < 90.0        # wrists sit in front of the chosen forward axis


def test_spans_land_inside_the_linter_band():
    from egoannot.labels import atomicity as AL
    from egoannot.stages import spans
    need_segment("np_tissue")
    rows = spans.build(only=["np_tissue"],
                       out=config.artifact("spans", "_test_spans.jsonl"))
    assert rows
    lo, hi = AL.SPAN["measured"]
    d = np.array([r["duration"] for r in rows])
    assert (d >= lo).all() and (d <= hi).all()
    for a, b in zip(rows, rows[1:]):
        assert b["start_ts"] >= a["end_ts"] - 1e-6      # rule A8: no overlap


def test_spans_carry_the_pose_facts_the_model_is_not_asked_for():
    from egoannot.stages import spans
    need_segment("dv_contactlens")
    rows = spans.build(only=["dv_contactlens"],
                       out=config.artifact("spans", "_test_spans2.jsonl"))
    assert rows
    for r in rows:
        assert r["hand"] in ("LEFT", "RIGHT", "BOTH")
        assert r["acting_side"] in ("left", "right")
        assert r["ap_trend"] in ("closing", "opening", "flat", None)
        assert r["rotation"] in ("clockwise", "counter-clockwise", None)
    assert any(r["ap_trend"] in ("closing", "opening") for r in rows)


def test_the_stub_backend_captions_every_span_and_scores_clean():
    from egoannot.stages import caption, score, spans
    span_path = config.artifact("spans", "_test_spans3.jsonl")
    need_segment("np_tissue"); need_segment("dv_coffee")
    rows = spans.build(only=["np_tissue", "dv_coffee"], out=span_path)
    out = config.artifact("captions", "_test_caps.jsonl")
    caps = caption.run(span_path, "stub", out)
    assert len(caps) == len(rows)
    result = score.score(out)
    assert result["atomicity"] == 1.0          # the stub emits a valid caption
    assert result["out_of_band"] == 0
    assert result["grounding_agree"] is not None


def test_the_frame_store_stays_small():
    """The old raw-frame cache reached 8.2 GB across the rendered segments."""
    from egoannot.core.video import SegmentFrames
    from egoannot.stages import spans
    need_segment("np_storagebox")
    rows = spans.build(only=["np_storagebox"],
                       out=config.artifact("spans", "_test_spans4.jsonl"))
    store = SegmentFrames(config.SEGMENTS_DIR)
    planned = store.plan(rows, config.CAPTION["frames_per_span"])
    assert sum(planned.values()) < 20 * len(rows)
    for r in rows[:5]:
        assert len(store.get(r, config.CAPTION["frames_per_span"])) > 0
    assert store.bytes_peak < 50e6
    store.release("np_storagebox")
    assert store.bytes_held == 0
