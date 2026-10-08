from pathlib import Path
import numpy as np

from scripts import process_quality_fast as quality
from egoannot.quality.hand_recovery import RecoveryHandDetector


class Detector:
    model_name='test'; model_sha256='test'; size=960; precision='fp16'; path='test'
    tracker_high=.15; tracker_low=.08; prompts=['hand']
    calls=0; seconds=0.
    def __call__(self, frame, offset=(0,0), source='full'):
        return []


def test_concurrent_episodes_keep_order_and_separate_recovery(tmp_path, monkeypatch):
    sessions=[]
    def run(path,out,hand_detector=None,hand_hz=2.):
        sessions.append(hand_detector)
        hand_detector.reset_sequence()
        hand_detector(np.zeros((100,100,3),dtype=np.uint8))
        hand_detector(np.zeros((100,100,3),dtype=np.uint8))
        assert hand_detector.frames==2
        assert hand_detector.calls==4  # full+2 discovery, then full
        return path.name
    monkeypatch.setattr(quality,'run',run)
    paths=[Path(f'{i}.mp4') for i in range(4)]
    result=quality.run_concurrent(paths,tmp_path,2,RecoveryHandDetector(Detector(),discovery_every=2))
    assert result==[p.name for p in paths]
    assert len({id(s) for s in sessions})==4
    assert len({id(s.detector) for s in sessions})==4


def test_concurrent_cpu_mode_has_no_detector(tmp_path, monkeypatch):
    def run(path,out,hand_detector=None,hand_hz=2.):
        assert hand_detector is None
        return path.name
    monkeypatch.setattr(quality,'run',run)
    assert quality.run_concurrent([Path('a.mp4'),Path('b.mp4')],tmp_path,2)==['a.mp4','b.mp4']
