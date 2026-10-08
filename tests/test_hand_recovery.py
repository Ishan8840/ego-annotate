import numpy as np
import pytest
from egoannot.quality.hand_recovery import RecoveryHandDetector, discovery_regions, valid_crop_box


class Fake:
    model_name, model_sha256, size, precision, path = 'fake', 'test', 960, 'fp32', ''
    tracker_high, tracker_low = .15, .08
    def __init__(self, fn):
        self.fn, self.calls, self.seconds = fn, 0, 0.
    def __call__(self, image, offset=(0,0), source='full'):
        self.calls += 1
        return self.fn(image, offset, source)


def det(box, score, source):
    return dict(box=box, score=score, source=source)


def test_discovery_covers_both_orientations_and_overlaps():
    assert discovery_regions(1000,600)==[(0,0,650,600),(350,0,1000,600)]
    assert discovery_regions(600,1000)==[(0,0,600,650),(0,350,600,1000)]


def test_artificial_crop_edge_rejected_but_real_image_edge_allowed():
    assert not valid_crop_box([349,10,390,50],(350,0,1000,600),1000,600)
    assert valid_crop_box([950,10,1000,50],(350,0,1000,600),1000,600)


def test_discovery_finds_unseen_hand_without_a_previous_track():
    fake=Fake(lambda image,offset,source: [] if source=='full' else [det([400,100,460,170],.24,source)])
    detector=RecoveryHandDetector(fake,verify_weak=False)
    result=detector(np.zeros((600,1000,3),np.uint8))
    assert len(result)==1
    assert result[0]['source']=='discovery_tile'
    assert fake.calls==3


def test_weak_seed_requires_matching_positive_zoom_not_unrelated_object():
    def response(image,offset,source):
        if source=='full':return [det([100,100,160,160],.1,source)]
        return [det([103,101,161,163],.18,source),det([400,400,460,460],.9,source)]
    detector=RecoveryHandDetector(Fake(response),discovery=False)
    result=detector(np.zeros((600,1000,3),np.uint8))
    assert len(result)==1 and result[0]['source']=='verified_weak_crop'
    assert result[0]['score']==.18


def test_missing_hand_does_not_force_a_second_box():
    detector=RecoveryHandDetector(Fake(lambda *args: []))
    assert detector(np.zeros((600,1000,3),np.uint8))==[]


def test_cadence_resets_at_new_sequence():
    fake=Fake(lambda *args: [])
    detector=RecoveryHandDetector(fake,verify_weak=False,discovery_every=2)
    frame=np.zeros((600,1000,3),np.uint8)
    detector(frame);assert fake.calls==3
    detector(frame);assert fake.calls==4
    detector.reset_sequence();detector(frame);assert fake.calls==7


def test_discovery_below_stricter_threshold_cannot_start_track():
    fake=Fake(lambda image,offset,source: [] if source=='full' else [det([400,100,460,170],.19,source)])
    assert RecoveryHandDetector(fake,verify_weak=False)(np.zeros((600,1000,3),np.uint8))==[]


def test_four_hz_keeps_deadlines_on_thirty_fps_video():
    from egoannot.quality.hand_tracking import HandVisibility
    tracker=HandVisibility(Fake(lambda *args: []),sample_hz=4,profile='baseline')
    frame=np.zeros((100,100,3),np.uint8)
    for i in range(150):tracker(i,i/30,frame)
    samples=[r['frame_id'] for r in tracker.rows if r['detection_ran']]
    assert len(samples)==20
    assert samples[:5]==[0,8,15,23,30]
