import numpy as np
import pytest

from egoannot.quality.hand_tracking import BoxTracker, HandVisibility, iou, letterbox, suppress


def detection(box=(10, 10, 50, 50), score=.9, source='full'):
    return dict(box=list(box), score=score, source=source)


def test_enclosing_duplicate_removed_but_neighbor_kept():
    from egoannot.quality.hand_tracking import suppress_contained
    result = suppress_contained([detection(), detection((0, 0, 90, 90), .5), detection((60, 10, 90, 50), .6)])
    assert len(result) == 2
    assert [d['score'] for d in result] == [.9, .6]


def test_letterbox_preserves_upstream_bgr_and_scale():
    frame = np.zeros((100, 200, 3), np.uint8)
    frame[:] = [10, 20, 30]
    tensor, ratio = letterbox(frame)
    assert tensor.shape == (1, 3, 320, 320)
    assert ratio == 1.6
    assert tensor[0, :, 20, 20] == pytest.approx((np.array([10, 20, 30]) - [103.53, 116.28, 123.675]) / [57.375, 57.12, 58.395])
    assert tensor[0, :, 200, 20] == pytest.approx((np.array([114] * 3) - [103.53, 116.28, 123.675]) / [57.375, 57.12, 58.395])


def test_gap_prediction_does_not_count_as_visible_and_reacquires_id():
    tracker = BoxTracker()
    observed = tracker.update([detection()], 0.)
    assert list(observed) == [1]
    assert tracker.update([], .1) == {}
    row = tracker.snapshot(.1, 100, 100, detection_ran=True)[0]
    assert row['status'] == 'predicted_gap'
    assert not row['observed']
    assert not row['crop_candidate']
    assert list(tracker.update([detection((12, 10, 52, 50))], .2)) == [1]


def test_weak_observation_never_creates_new_track():
    tracker = BoxTracker()
    assert tracker.update([detection(score=.2)], 0) == {}
    assert not tracker.tracks
    tracker.update([detection()], .1)
    obs = tracker.update([detection(score=.2)], .2)
    assert obs[1]['status'] == 'observed_weak'
    # Weak observations cannot sustain identity forever through an occlusion.
    assert tracker.update([detection(score=.2)], .5) == {}


def test_long_gap_expires_track_and_requires_new_identity():
    tracker = BoxTracker(max_gap=.5)
    tracker.update([detection()], 0)
    assert not tracker.snapshot(.6, 100, 100)
    observed = tracker.update([detection()], .7)
    assert list(observed) == [2]


def test_flow_recovers_translation_without_claiming_detection():
    from egoannot.quality.hand_tracking import SparseFlow
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 256, (160, 200, 3), dtype=np.uint8)
    import cv2
    moved = cv2.warpAffine(frame, np.float32([[1, 0, 4], [0, 1, 2]]), (200, 160))
    tracker = BoxTracker()
    tracker.update([detection((40, 40, 130, 130))], 0.)
    flow = SparseFlow()
    flow.update(frame, [], 0.)
    assert flow.update(moved, tracker.tracks, .033) == {1}
    row = tracker.snapshot(.033, 200, 160)[0]
    assert row['box'] == pytest.approx([44, 42, 134, 132], abs=.5)
    assert not row['observed']
    assert not row['crop_candidate']


def test_blank_flow_cannot_confirm_or_move_a_hand():
    from egoannot.quality.hand_tracking import SparseFlow
    tracker = BoxTracker()
    tracker.update([detection()], 0.)
    flow = SparseFlow()
    blank = np.zeros((100, 100, 3), np.uint8)
    flow.update(blank, [], 0.)
    assert flow.update(blank, tracker.tracks, .1) == set()
    assert tracker.tracks[0].box.tolist() == [10, 10, 50, 50]


def test_sparse_detector_uses_model_thresholds_and_expires():
    class Fake:
        calls, seconds = 0, 0.
        tracker_high, tracker_low = .15, .08
        def __call__(self, *_):
            self.calls += 1
            return [detection(score=.2)] if self.calls == 1 else []
    tracker = HandVisibility(Fake(), sample_hz=2, profile='baseline')
    frame = np.zeros((100, 100, 3), np.uint8)
    for i in range(31):
        row = tracker(i, i / 30, frame)
    assert not row['tracks']
    report = tracker.finish(31 / 30)
    assert report['detector_samples'] == 3
    assert report['observed_any_sample_pct'] == pytest.approx(100 / 3)


def test_two_hands_match_once_each_even_with_reordered_detections():
    tracker = BoxTracker()
    tracker.update([detection(), detection((60, 10, 90, 50))], 0)
    result = tracker.update([detection((62, 10, 92, 50)), detection((12, 10, 52, 50))], .1)
    assert result[1]['box'][0] == 12
    assert result[2]['box'][0] == 62


def test_sample_coverage_excludes_intermediate_predictions():
    class Fake:
        calls, seconds = 0, 0.
        def __call__(self, *_):
            self.calls += 1
            return [detection()] if self.calls == 1 else []
    model = Fake()
    track = HandVisibility(model, sample_hz=10, profile='baseline')
    frame = np.zeros((100, 100, 3), np.uint8)
    for index in range(9):
        track(index, index / 30, frame)
    result = track.finish(.3)
    assert result['detector_samples'] == 3
    assert result['observed_strong_sample_pct'] == pytest.approx(100 / 3)
    assert result['longest_no_detection_gap_s'] == pytest.approx(.2)
    assert result['frames'][1]['tracks'][0]['status'] == 'propagated_between_detections'


def test_recovery_crop_cannot_invent_an_unrelated_hand():
    class Fake:
        calls, seconds = 0, 0.
        def __call__(self, frame, offset=(0, 0), source='full'):
            self.calls += 1
            if self.calls == 1:
                return [detection((100, 100, 150, 150))]
            if source == 'recovery_crop':
                return [detection((400, 400, 450, 450), source=source)]
            return []
    track = HandVisibility(Fake())
    frame = np.zeros((500, 500, 3), np.uint8)
    track(0, 0, frame)
    row = track(1, .1, frame)
    assert row['recovery_calls'] == 1
    assert row['strong_count'] == 0
    assert len(track.tracker.tracks) == 1


def test_suppression_and_border_candidates():
    rows = suppress([detection(), detection((11, 11, 51, 51), score=.7), detection((70, 70, 90, 90))])
    assert len(rows) == 2
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    tracker = BoxTracker()
    obs = tracker.update([detection((0, 10, 40, 50))], 0)
    assert tracker.snapshot(0, 100, 100, obs, True)[0]['crop_candidate']


def test_bad_time_and_bad_cadence_fail():
    tracker = BoxTracker()
    tracker.update([], 0)
    with pytest.raises(ValueError):
        tracker.update([], 0)
    with pytest.raises(ValueError):
        HandVisibility(None, sample_hz=0)
