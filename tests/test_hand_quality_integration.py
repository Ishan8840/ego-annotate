import importlib.util
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('hand_quality_cli', Path(__file__).resolve().parents[1] / 'scripts/process_quality_fast.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def test_optional_hands_preserve_measurements_and_report_failure(tmp_path):
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg required')
    video = tmp_path / 'sample.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (160,120))
    assert writer.isOpened()
    for i in range(10):
        writer.write(np.full((120,160,3),i*20,np.uint8))
    writer.release()
    class Detector:
        calls,seconds=0,0.
        def __call__(self,*args):
            self.calls+=1
            return [dict(box=[20,20,50,50],score=.9,source='full')]
    base=tmp_path/'base';hand=tmp_path/'hand';failed=tmp_path/'failed'
    cli.run(video,base)
    timing=cli.run(video,hand,hand_detector=Detector())
    a=json.loads((base/'quality.json').read_text());b=json.loads((hand/'quality.json').read_text())
    assert a['measured']==b['measured']
    assert a['motion']==b['motion']
    assert b['hand_visibility']['observed_any_sample_pct']==100
    assert timing['includes_hand_tracking'] and timing['hand_tracking_complete']
    assert (hand/'hands.json').is_file()
    class Broken(Detector):
        def __call__(self,*args):
            raise RuntimeError('detector failed')
    timing=cli.run(video,failed,hand_detector=Broken())
    assert timing['hand_error']=='RuntimeError: detector failed'
    assert not timing['hand_tracking_complete']
    assert json.loads((failed/'quality.json').read_text())['measured']==a['measured']
