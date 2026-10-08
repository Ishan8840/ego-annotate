"""Fail closed on invented evidence; report missing verification as missing."""
import json
import pytest
from egoannot.realtime import parse_compact, verify_compact, verification_summary
from egoannot.quality.annotations import FIELDS


def payload():
    return {key: ['S', 1] for key in FIELDS}


def test_compact_evidence_binds_to_exact_sample_time():
    row = payload()
    row['details'] = ['U', None]
    row['hand'] = ['C', 0]
    parsed = parse_compact(json.dumps(row), [1.2, 1.4333])
    assert parsed['action'] == dict(verdict='supported', evidence_s=[1.4333])
    assert parsed['hand'] == dict(verdict='contradicted', evidence_s=[1.2])
    assert parsed['details'] == dict(verdict='unknown', evidence_s=[])


@pytest.mark.parametrize('bad', [['S', None], ['C', -1], ['S', 2], ['S', True],
                                ['S', 1.0], ['S', '1'], ['pass', 1], ['S'], None])
def test_compact_rejects_unbound_or_malformed_evidence(bad):
    row = payload()
    row['action'] = bad
    with pytest.raises(ValueError):
        parse_compact(json.dumps(row), [0., 1.])


def test_compact_requires_all_fields():
    row = payload()
    del row['details']
    with pytest.raises(ValueError):
        parse_compact(json.dumps(row), [0., 1.])


def test_verifier_missing_frames_never_calls_model_or_passes():
    def engine(*args):
        pytest.fail('Must not ask a model to verify missing source evidence')
    result = verify_compact(dict(span_id='s', source_video_sha256='abc'), [], [], engine)
    assert 'error' in result and 'fields' not in result


def test_coverage_denominator_includes_missing_captions_and_failed_checks():
    good = {'fields': parse_compact(json.dumps(payload()), [0., 1.])}
    result = verification_summary([good, {'error': 'truncated'}], requested_spans=4, captions=2)
    assert result['verification_coverage'] == .25
    assert result['verification_failures'] == 1
    assert result['field_verdict_counts']['supported'] == 5


def test_zero_spans_does_not_claim_full_coverage():
    assert verification_summary([], 0, 0)['verification_coverage'] == 0


def test_shared_decode_preserves_caption_and_audit_images(tmp_path):
    import subprocess
    from egoannot.core.video import SegmentFrames
    from egoannot.quality.analysis import visual_frames
    from egoannot.realtime import EpisodeFrames
    path = tmp_path / 'clip.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                    'testsrc2=size=768x432:rate=30:duration=3', '-c:v', 'libx264',
                    '-threads', '2', str(path)], check=True)
    rows = [dict(span_id='clip#0', segment='clip', v_start=0., v_end=1.433),
            dict(span_id='clip#1', segment='clip', v_start=1.433, v_end=3.)]
    original = SegmentFrames(tmp_path, 88)
    original.plan(rows, 5)
    shared = EpisodeFrames(tmp_path)
    shared.prepare(rows)
    for row in rows:
        assert shared.get(row, 5) == original.get(row, 5)
        assert shared.verification_frames(row) == visual_frames(
            path, 3., sample_fps=4, max_frames=16, start=row['v_start'], end=row['v_end'])
    assert shared.bytes_peak > 0
    shared.release('clip')
    assert shared.bytes_held == 0 and not shared.audit_jpegs


def test_failed_verification_retries_without_counting_failure_as_passed():
    class Engine:
        last_raw = ''
        calls = 0
        def __call__(self, system, parts, batch):
            self.calls += 1
            self.last_raw = '{}' if self.calls == 1 else json.dumps({key: ['S', 0] for key in FIELDS})
            return [], {}
    engine = Engine()
    result = verify_compact(dict(span_id='s', source_video_sha256='abc'), [('image', b'jpg')], [0.], engine)
    assert len(result['attempts']) == 2 and 'error' in result['attempts'][0]
    assert 'fields' in result and 'error' not in result


def test_probe_cache_is_shared_but_never_reuses_changed_source(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from egoannot.realtime import SourceProbeCache
    from egoannot.quality import analysis
    path = tmp_path / 'source.mp4'
    path.write_bytes(b'original')
    calls = []
    def probe(p):
        calls.append(p)
        return {'frames': [{'time': 0}]}, 0, ''
    monkeypatch.setattr(analysis, 'probe', probe)
    cache = SourceProbeCache()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(cache.get, [path] * 4))
    assert len(calls) == 1
    results[0][0]['frames'].clear()
    assert len(cache.get(path)[0]['frames']) == 1
    path.write_bytes(b'changed source')
    with pytest.raises(ValueError, match='Source changed'):
        cache.get(path)


def test_full_verifier_preserves_prompt_and_images_with_preloaded_evidence(monkeypatch):
    from egoannot.quality import analysis
    from egoannot.quality.annotations import verify_label
    parts, times = [('text', 'Frame at 0.0000 seconds'), ('image', b'jpeg')], [0.]
    monkeypatch.setattr(analysis, 'visual_frames', lambda *args, **kwargs: (parts, times))
    class Engine:
        last_raw = json.dumps({key: dict(verdict='supported', reason='visible', evidence_s=[0.]) for key in FIELDS})
        def __init__(self): self.calls = []
        def __call__(self, *args): self.calls.append(args); return [], {}
    a, b = Engine(), Engine()
    span = dict(video_path='unused', video_duration_s=1., v_start=0., v_end=1.)
    assert verify_label({}, span, a) == verify_label({}, span, b, evidence=(parts, times))
    assert a.calls == b.calls


def test_merged_audit_keeps_structural_errors_and_failed_verifications():
    from egoannot.realtime import merge_verifications
    structural = dict(annotations=[dict(span_id='a', issues=[dict(code='format_error', detail='bad words')]),
                                  dict(span_id='b', issues=[])], missing_span_ids=['c'], input_issues=[], summary={})
    result = merge_verifications(structural, [dict(span_id='a', fields=parse_compact(json.dumps(payload()), [0., 1.])),
                                           dict(span_id='b', error='invalid evidence')])
    assert result['summary']['verified_annotations'] == 1
    assert result['summary']['finding_counts'] == {'missing_caption': 1, 'format_error': 1, 'verification_unavailable': 1}
    assert len(structural['annotations'][1]['issues']) == 0
    with pytest.raises(ValueError, match='bind uniquely'):
        merge_verifications(structural, [dict(span_id='unknown', fields={})])
