import json
import shutil

import cv2
import numpy as np
import pytest

from egoannot.quality.analysis import measure_video
from egoannot.quality.fast import CONTEXT, EvidenceCollector, motion_pair, parse_context, audit_context


@pytest.fixture
def video(tmp_path):
    if not shutil.which('ffprobe') or not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg required')
    path = tmp_path / 'signal.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (160, 120))
    assert writer.isOpened()
    rng = np.random.default_rng(2)
    frame = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    for i in range(30):
        writer.write(np.zeros_like(frame) if i < 8 else np.roll(frame, (i // 2) * 2, axis=1))
    writer.release()
    return path


def test_all_measurements_remain_identical_in_fast_mode(video):
    assert measure_video(video) == measure_video(video, fast=True)


def test_corruption_preserves_integrity_failure(tmp_path):
    path = tmp_path / 'bad.mp4'
    path.write_bytes(b'broken')
    fast = measure_video(path, fast=True)
    import re
    baseline = measure_video(path)
    for report in (fast, baseline):
        report['integrity']['diagnostics'] = re.sub(r'0x[0-9a-f]+', '<address>', report['integrity']['diagnostics'])
    assert fast == baseline
    assert fast['integrity']['decode_exit'] != 0
    assert fast['frame_count'] == 0


def test_motion_recovers_known_translation_and_abstains_on_blank():
    rng = np.random.default_rng(4)
    image = rng.integers(0, 256, (180, 320), dtype=np.uint8)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 4], [0, 1, 0]]), (320, 180))
    result = motion_pair(image, shifted, .2)
    assert result['valid']
    assert result['image_speed_diagonals_per_s'] == pytest.approx(4 / np.hypot(320, 180) / .2, rel=.05)
    blank = np.zeros_like(image)
    assert not motion_pair(blank, blank, .2)['valid']
    assert not motion_pair(image, shifted, 0)['valid']


def test_evidence_timestamps_and_event_budget(video, tmp_path):
    from egoannot.quality.analysis import probe
    data, _, _ = probe(video)
    times = [float(f['best_effort_timestamp_time']) for f in data['frames']]
    collector = EvidenceCollector(times, tmp_path / 'evidence', window_seconds=1.)
    clip = measure_video(video, fast=True, frame_observer=collector)
    collector.finish(clip)
    assert len(collector.windows) == 3
    for window in collector.windows:
        assert len(window['samples']) <= 20
        assert window['events_sampled'] <= 8
        for sample in window['samples']:
            assert sample['time_s'] == times[sample['frame_id']] - times[0]
            assert (collector.directory / sample['file']).is_file()
    with pytest.raises(ValueError, match='Incomplete'):
        collector.finish(dict(clip, frame_count=29))


def payload():
    return {key: ['N', 0, 'Visible in this frame'] for key in CONTEXT}


@pytest.mark.parametrize('bad', [['C', -1, 'No frame'], ['N', 2, 'Outside'],
                                 ['G', 0, 'Wrong'], ['C', True, 'Boolean'],
                                 ['N', 0, ''], ['N', 0, 'x' * 181]])
def test_context_rejects_unbound_or_invalid_evidence(bad):
    obj = payload()
    obj['hand_visibility'] = bad
    with pytest.raises(ValueError):
        parse_context(json.dumps(obj), [dict(time_s=.3)])


def test_unknown_is_not_a_pass_and_missing_rows_fail():
    obj = payload()
    obj['hand_visibility'] = ['U', -1, 'Hands not visible']
    result = parse_context(json.dumps(obj), [dict(time_s=.3)])
    assert result['hand_visibility']['status'] == 'U'
    assert result['hand_visibility']['evidence_s'] is None
    del obj['failures']
    with pytest.raises(ValueError):
        parse_context(json.dumps(obj), [dict(time_s=.3)])


def test_context_failures_remain_unassessed(video, tmp_path):
    collector = EvidenceCollector([i / 10 for i in range(30)], tmp_path / 'evidence')
    clip = measure_video(video, fast=True, frame_observer=collector)
    collector.finish(clip)
    class Broken:
        last_raw = '{}'
        def __call__(self, *_):
            return [], {}
    results = audit_context(collector, Broken())
    assert results[0]['rows'] == {}
    assert len(results[0]['calls']) == 2
    assert results[0]['error']


@pytest.mark.parametrize('times', [[], [0, 0], [1, 0], [0, float('nan')]])
def test_unreliable_timestamps_rejected(tmp_path, times):
    with pytest.raises(ValueError):
        EvidenceCollector(times, tmp_path)


def test_fixed_grammar_requires_bound_nonunknown_rows():
    import re
    from egoannot.quality.fast import context_regex
    good = payload()
    assert re.fullmatch(context_regex(), json.dumps(good))
    good['failures'] = ['N', -1, 'No failure seen']
    assert not re.fullmatch(context_regex(), json.dumps(good))
    good['failures'] = ['U', -1, 'Unknown']
    assert re.fullmatch(context_regex(), json.dumps(good))


def test_runner_keeps_corrupt_input_report(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('fast_quality_runner', Path(__file__).parents[1] / 'scripts/process_quality_fast.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / 'bad.mp4'
    source.write_bytes(b'bad video')
    timing = module.run(source, tmp_path / 'result')
    report = json.loads((tmp_path / 'result/quality.json').read_text())
    assert report['measured']['integrity']['decode_exit'] != 0
    assert timing['evidence_error']
    assert not report['windows']
    assert report['measured']['frame_count'] == 0
