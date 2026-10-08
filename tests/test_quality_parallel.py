"""Parallel windows must retain source order, evidence, failures, and aggregation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import threading
import time
import pytest
from egoannot.quality import analysis


class Model:
    def __init__(self):
        self.local = threading.local()

    @property
    def last_raw(self):
        return self.local.raw

    def __call__(self, system, parts, batch):
        prompt = parts[0][1]
        if 'inventory' in prompt:
            value = {key: None for key in analysis.DIVERSITY_FIELDS}
        elif 'Requested dimensions:' in prompt:
            keys = json.loads(prompt.split('Requested dimensions: ')[1].split('\nTask/collection')[0])
            timestamp = float(parts[1][1])
            time.sleep(.01 if timestamp == 0 else .001)
            # A bad window fails both retries. Other windows must survive.
            evidence = 999 if timestamp == 8 else timestamp
            value = dict(observations=[dict(dimension=k, summary='Visible example', concerns=[],
                         confidence='medium', evidence_s=[[evidence, evidence]]) for k in keys])
        else:
            keys = json.loads(prompt.split('criteria: ')[1])
            timestamp = float(parts[1][1])
            time.sleep(.01 if timestamp == 0 else .001)
            value = dict(ratings=[dict(dimension=k, rating=3, reason='Visible example',
                                      evidence_s=[timestamp]) for k in keys])
        self.local.raw = json.dumps(value)
        return [], {}


@pytest.mark.parametrize('parallel_groups', [False, True])
def test_parallel_descriptions_keep_evidence_order_and_failures(monkeypatch, parallel_groups):
    monkeypatch.setattr(analysis, 'probe', lambda path: (
        dict(frames=[dict(best_effort_timestamp_time=i) for i in range(20)]), 0, ''))
    monkeypatch.setattr(analysis, 'visual_frames', lambda *args, **kwargs: (
        [('text', str(kwargs.get('start', 0)))], [float(kwargs.get('start', 0))]))
    clip = dict(path='fake.mp4', duration_s=20., observations=[])
    serial = analysis.describe_video(clip, Model(), workers=1)
    parallel = analysis.describe_video(clip, Model(), workers=3, parallel_groups=parallel_groups)
    assert serial == parallel
    assert parallel['windows'][1]['unavailable']
    assert parallel['windows'][0]['analyzed_dimensions'] == parallel['windows'][2]['analyzed_dimensions']
    assert [w['interval_s'] for w in parallel['windows']] == [[0., 8.], [8., 16.], [16., 20.]]


@pytest.mark.parametrize('parallel_groups', [False, True])
def test_parallel_composite_preserves_rubric_score_and_window_order(tmp_path, monkeypatch, parallel_groups):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('composite', root / 'demo/score-quality.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'frames', lambda path, start, end: ([('text', str(start))], [float(start)]))
    outputs = []
    for workers in [1, 3]:
        out = tmp_path / str(workers)
        (out / 'quality').mkdir(parents=True)
        (out / 'clip.mp4').write_bytes(b'fixture')
        report = dict(clips=[dict(id='clip', duration_s=20., sha256=hashlib.sha256(b'fixture').hexdigest())])
        (out / 'quality/analysis.json').write_text(json.dumps(report))
        module.main(['--source', str(out), '--model', 'fake'], engine=Model(), workers=workers,
                    parallel_groups=parallel_groups)
        outputs.append(json.loads((out / 'quality/composite.json').read_text()))
    assert outputs[0] == outputs[1]
    assert outputs[1]['score_percent'] == 75
    assert outputs[1]['assessed_dimension_windows'] == 24
