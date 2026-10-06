import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from egoannot.quality.analysis import (analyze, describe_video, duplicate_pairs, measure_video,
                                      parse_visual, ranges)


@pytest.fixture
def videos(tmp_path):
    if not shutil.which('ffprobe') or not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg/ffprobe required')
    def make(name, frames, fps=10):
        path = tmp_path / (name + '.avi')
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), fps,
                                 (frames[0].shape[1], frames[0].shape[0]))
        assert writer.isOpened()
        for frame in frames:
            writer.write(frame)
        writer.release()
        return path
    return make


def test_dark_static_video_has_measured_evidence(videos):
    path = videos('dark', [np.zeros((80, 120, 3), dtype=np.uint8)] * 20)
    c = measure_video(path, expected={'fps': 30, 'width': 200})
    assert c['frame_count'] == 20
    assert c['duration_s'] == pytest.approx(2)
    assert c['candidate_frame_fractions']['dark_frames'] == 1
    assert c['candidate_frame_fractions']['identical_adjacent_frames'] == pytest.approx(19/20)
    assert {v['measure'] for v in c['integrity']['discrepancies']} == {'fps', 'width'}
    intervals = next(o for o in c['observations'] if o['kind'] == 'near_static')['intervals_s']
    assert intervals == [[.1, 2.0]]


def test_unreadable_video_is_recorded_not_silently_skipped(tmp_path):
    path = tmp_path / 'bad.mp4'
    path.write_bytes(b'not a video')
    c = measure_video(path)
    assert c['frame_count'] == 0
    assert c['integrity']['decode_exit'] != 0
    assert any(o['kind'] == 'no_decoded_frames' for o in c['observations'])
    json.dumps(c, allow_nan=False)


def test_duplicates_and_reencoded_near_duplicates(videos, tmp_path):
    rng = np.random.default_rng(6)
    # Structured content survives lossy encoding and slight brightness changes.
    base = cv2.GaussianBlur(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8), (7, 7), 0)
    frames = [np.roll(base, i * 2, axis=1) for i in range(20)]
    a = videos('original', frames)
    b = tmp_path / 'copy.avi'
    shutil.copyfile(a, b)
    reencoded = tmp_path / 'reencoded.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(a), '-c:v', 'libx264', '-crf', '23', str(reencoded)], check=True)
    clips = [measure_video(p) for p in (a, b, reencoded)]
    pairs = duplicate_pairs(clips)
    assert any(p['kind'] == 'identical_file' for p in pairs)
    assert any(p['kind'] == 'near_duplicate_candidate' for p in pairs)


def row(**changes):
    value = dict(dimension='hand_visibility', summary='A hand appears near the bottom edge.',
                 confidence='medium', concerns=['Fingertips cropped'], evidence_s=[[0, 1]])
    value.update(changes)
    return value


@pytest.mark.parametrize('changes', [dict(evidence_s=[[0, 9]]), dict(evidence_s=[]),
                                     dict(confidence=.99), dict(dimension='invented'),
                                     dict(evidence_s=[[True, 1]]), dict(concerns='no')])
def test_model_claims_must_have_valid_schema_and_evidence(changes):
    with pytest.raises(ValueError):
        parse_visual(json.dumps({'observations': [row(**changes)]}), ['hand_visibility'], 2)


def test_missing_model_rows_are_not_fabricated():
    with pytest.raises(ValueError, match='missing'):
        parse_visual(json.dumps({'observations': [row()]}), ['hand_visibility', 'object_visibility'], 2)


def test_missing_task_rules_remain_unknown_and_model_errors_are_logged(videos):
    p = videos('static', [np.full((80, 120, 3), 120, np.uint8)] * 10)
    class BrokenModel:
        last_raw = 'malformed model response'
        def __call__(self, *_):
            return [], {}
    result = describe_video(measure_video(p), BrokenModel())
    assert 'No task supplied' in result['unavailable']['instruction_compliance']
    assert 'No final_state supplied' in result['unavailable']['task_success']
    assert result['observations'] == []
    assert all(c.get('error') for c in result['calls'])


def test_report_is_descriptive_and_retains_all_clips(videos, tmp_path):
    p = videos('scene', [np.full((80, 120, 3), 128, np.uint8)] * 10)
    out = tmp_path / 'report'
    result = analyze([p], out)
    assert result['summary']['clips'] == 1
    assert result['summary']['visual_dimensions']['task_success']['analyzed_clips'] == 0
    assert 'status' not in result and 'score' not in result
    assert (out / 'index.html').is_file()
    assert (out / 'videos/scene.avi').is_file()
    assert 'quality score' in (out / 'summary.md').read_text()  # explicitly says no composite score
    json.loads((out / 'analysis.json').read_text())


def test_ranges_preserve_separate_events_and_last_frame():
    assert ranges([0, .1, .2, .3], [True, False, True, True], .1) == [[0, .1], [.2, .4]]


def test_gap_and_nonmonotonic_timestamps_have_exact_evidence(videos, monkeypatch):
    from egoannot.quality import analysis
    path = videos('timestamps', [np.full((80, 120, 3), 100, np.uint8)] * 4)
    data, code, errors = analysis.probe(path)
    data['frames'] = [{'best_effort_timestamp_time': t} for t in [0, .1, .4, .3]]
    monkeypatch.setattr(analysis, 'probe', lambda _: (data, code, errors))
    result = measure_video(path)
    obs = {o['kind']: o for o in result['observations']}
    assert obs['timestamp_gap_candidate']['intervals_s'] == [[.1, .4]]
    assert obs['nonmonotonic_timestamp']['intervals_s'] == [[.3, .4]]


def test_bright_and_low_detail_are_separate_measurements(videos):
    white = videos('white', [np.full((80, 120, 3), 255, np.uint8)] * 10)
    yy, xx = np.indices((80, 120))
    pattern = (((xx // 8 + yy // 8) % 2) * 255).astype(np.uint8)
    checker = videos('checker', [np.repeat(pattern[:, :, None], 3, axis=2)] * 10)
    bright, detail = measure_video(white), measure_video(checker)
    assert bright['candidate_frame_fractions']['bright_frames'] == 1
    assert detail['candidate_frame_fractions']['bright_frames'] == 0
    assert detail['measurements']['laplacian_variance']['median'] > bright['measurements']['laplacian_variance']['median'] + 100


def test_truncated_mp4_retains_decoder_diagnostic(videos, tmp_path):
    source = videos('source', [np.full((80, 120, 3), i * 10, np.uint8) for i in range(20)])
    mp4 = tmp_path / 'complete.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(source), '-movflags', '+faststart', str(mp4)], check=True)
    truncated = tmp_path / 'truncated.mp4'
    content = mp4.read_bytes()
    truncated.write_bytes(content[:len(content) * 3 // 4])
    result = measure_video(truncated)
    assert result['integrity']['decode_exit'] or result['integrity']['diagnostics']


def test_supplied_task_enables_semantic_comparison(videos):
    from egoannot.quality.analysis import DIVERSITY_FIELDS
    path = videos('task', [np.full((80, 120, 3), 128, np.uint8)] * 10)
    class Model:
        def __call__(self, system, parts, _):
            if 'inventory' in parts[0][1]:
                result = {key: 'visible example' for key in DIVERSITY_FIELDS}
            else:
                requested = json.loads(parts[0][1].split('Requested dimensions: ')[1].split('\nTask/collection')[0])
                result = {'observations': [row(dimension=key, concerns=[]) for key in requested]}
            self.last_raw = json.dumps(result)
            return [], {}
    result = describe_video(measure_video(path), Model(), dict(task='Move cup', final_state='Cup on table',
                            required_steps=['move cup'], collection_rules=['No assistance']))
    assert len(result['observations']) == 15
    assert not result['unavailable']
    assert len(result['diversity_descriptors']) == 7


def test_model_overclaims_keep_raw_text_but_carry_caveats():
    from egoannot.quality.analysis import annotate_visual_limits
    observation = row(summary='Hands are consistently visible and well-tracked throughout.')
    annotated = annotate_visual_limits({'observations': [observation]})['observations'][0]
    assert annotated['summary'] == observation['summary']
    assert annotated['claim_caveats']
    assert 'No image-based hand tracking accuracy' in annotated['interpretation_limit']


def test_diversity_string_arrays_are_descriptions_not_identity_counts():
    from egoannot.quality.analysis import DIVERSITY_FIELDS, parse_diversity
    raw = dict.fromkeys(DIVERSITY_FIELDS, None)
    raw['object_instances'] = ['cup', 'bottle']
    got = parse_diversity(json.dumps(raw))
    assert got['object_instances'] == 'cup; bottle'
    assert got['background'] is None


def test_exact_repeated_content_detected_across_intervening_frames(videos):
    a = np.full((80, 120, 3), 30, np.uint8)
    b = np.full((80, 120, 3), 210, np.uint8)
    path = videos('repeat', [a, b, a, b])
    result = measure_video(path)
    assert result['candidate_frame_fractions']['identical_adjacent_frames'] == 0
    assert result['candidate_frame_fractions']['repeated_nonadjacent_frames'] == .5
