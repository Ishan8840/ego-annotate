import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from egoannot.core.video import SegmentFrames
from egoannot.quality.annotations import FIELDS, audit, parse_verification, validate_spans


def span(**changes):
    value = dict(span_id='s#0', segment='s', episode='e', start_ts=10., end_ts=12.,
                 v_start=0., v_end=2., duration=2., hand='RIGHT')
    return dict(value, **changes)


@pytest.mark.parametrize('change,code', [
    ({'v_start': float('nan')}, 'invalid_timestamps'),
    ({'v_start': True}, 'invalid_timestamps'),
    ({'duration': 3}, 'duration_mismatch'),
    ({'v_start': -1}, 'invalid_interval'),
    ({'segment': '../escape'}, 'invalid_segment'),
])
def test_bad_source_metadata_is_explicit(change, code):
    assert code in [i['code'] for i in validate_spans([span(**change)])]


def test_overlap_and_inconsistent_clock_are_detected():
    rows = [span(), span(span_id='s#1', v_start=1., v_end=3., start_ts=20., end_ts=22.)]
    assert {'overlapping_spans', 'timebase_mismatch'} <= {r['code'] for r in validate_spans(rows)}


def test_sampler_does_not_round_before_span_or_accept_reversed_times():
    frames = SegmentFrames('.')
    frames._fps['s'] = 30
    frames._counts['s'] = 90
    assert frames.indices_for('s', .011, .1, 4) == [1, 2]
    for start, end in [(-1, 1), (2, 1), (0, 4), (float('nan'), 1)]:
        with pytest.raises(ValueError):
            frames.indices_for('s', start, end, 4)


def verdict(evidence=None):
    return {f: dict(verdict='supported', reason='Visible in image', evidence_s=[0.] if evidence is None else evidence)
            for f in FIELDS}


@pytest.mark.parametrize('evidence', [[.75], [True], [float('nan')], []])
def test_verifier_cannot_invent_evidence(evidence):
    with pytest.raises(ValueError):
        parse_verification(json.dumps(verdict(evidence)), [0., 1.])


def test_incomplete_verifier_reply_is_not_a_positive_result():
    result = verdict()
    del result['hand']
    with pytest.raises(ValueError):
        parse_verification(json.dumps(result), [0.])


def test_audit_finds_misbinding_missing_and_visual_disagreement(tmp_path):
    # Use actual decoding, a fake model only at the inference boundary.
    path = tmp_path / 's.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (80, 60))
    assert writer.isOpened()
    for i in range(40):
        writer.write(np.full((60, 80, 3), i * 5, np.uint8))
    writer.release()
    spans = [span(), span(span_id='s#1', start_ts=12., end_ts=14., v_start=2., v_end=4.)]
    labels = [dict(span(), text='Grasp the cardboard box on the shelf with the right hand',
                   verb='grasp', noun='box', visibility='FULL', uncertain=False, pack='retail_shelf')]
    source, caps = tmp_path/'spans.jsonl', tmp_path/'captions.jsonl'
    source.write_text(''.join(json.dumps(r)+'\n' for r in spans))
    caps.write_text(''.join(json.dumps(r)+'\n' for r in labels))
    original = caps.read_bytes()
    class Model:
        def __call__(self, *args):
            v = verdict()
            v['object'] = dict(verdict='contradicted', reason='No box visible', evidence_s=[0.])
            self.last_raw = json.dumps(v)
            return [], {}
    result = audit(caps, source, tmp_path, tmp_path/'audit', engine=Model())
    assert result['missing_span_ids'] == ['s#1']
    assert result['summary']['finding_counts']['visual_contradicted'] == 1
    assert result['summary']['verified_annotations'] == 1
    assert result['sources']['s']['sha256']
    assert caps.read_bytes() == original
    labels[0]['start_ts'] = 11
    caps.write_text(json.dumps(labels[0]))
    result = audit(caps, source, tmp_path, tmp_path/'audit')
    assert result['summary']['finding_counts']['source_binding_mismatch'] == 1
    labels[0]['start_ts'] = 10.
    labels[0]['source_video_sha256'] = '0' * 64
    caps.write_text(json.dumps(labels[0]))
    result = audit(caps, source, tmp_path, tmp_path/'audit', engine=Model())
    assert result['summary']['finding_counts']['source_hash_mismatch'] == 1
    assert result['summary']['verified_annotations'] == 0


def test_sampler_rejects_irregular_presentation_timestamps(tmp_path, monkeypatch):
    import subprocess
    from egoannot.core import video
    writer = cv2.VideoWriter(str(tmp_path/'s.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (80, 60))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((60, 80, 3), np.uint8))
    writer.release()
    probe = {'frames': [{'best_effort_timestamp_time': t} for t in [0, .1, .4]]}
    monkeypatch.setattr(video.subprocess, 'run', lambda *a, **k:
                        subprocess.CompletedProcess(a, 0, json.dumps(probe), ''))
    with pytest.raises(ValueError, match='constant-rate'):
        SegmentFrames(tmp_path).indices_for('s', 0, .2, 2)


def test_decimal_string_evidence_is_losslessly_normalized_and_logged():
    raw = verdict(['0.0000'])
    result = parse_verification(json.dumps(raw), [0.])
    assert result['action']['evidence_s'] == [0.]
    assert result['action']['evidence_normalization']
    with pytest.raises(ValueError):
        parse_verification(json.dumps(verdict(['0.9999'])), [0.])
