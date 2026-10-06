import numpy as np
import pytest

from egoannot.core.hot3d import map_landmarks, timeline, _transform


def test_umetrack_mapping_preserves_missing_cmc_and_finger_tips():
    source = np.arange(60).reshape(20, 3) / 1000
    got = map_landmarks(source)
    np.testing.assert_array_equal(got[0], source[5])
    np.testing.assert_array_equal(got[[4, 8, 12, 16, 20]], source[:5])
    assert np.isnan(got[1]).all()
    assert np.isfinite(got).all(axis=1).sum() == 20


def test_timeline_uses_capture_timestamps_and_includes_final_frame():
    origin = 43_000_000_000_000
    ts = [origin + round(i * 1e9 / 30) for i in range(150)]
    t, fps, duration = timeline(ts)
    assert t[0] == 0
    assert fps == pytest.approx(30, rel=1e-6)
    assert duration == pytest.approx(5, abs=1e-6)


@pytest.mark.parametrize('ts', [[0, 0, 1], [0, 2, 1], [0, 33_333_333, 100_000_000]])
def test_bad_timestamps_fail_instead_of_silently_drifting(ts):
    with pytest.raises(ValueError):
        timeline(ts)


def test_transform_preserves_world_translation_and_rotation():
    T, _, _ = _transform(dict(quaternion_wxyz=[1, 0, 0, 0], translation_xyz=[1, 2, 3]))
    np.testing.assert_array_equal(T[:3, 3], [1, 2, 3])
    np.testing.assert_array_equal(T[:3, :3], np.eye(3))


def test_prepared_hot3d_runs_through_spans_and_real_frame_sampling(tmp_path, monkeypatch):
    import json
    import os
    from pathlib import Path
    from egoannot import config
    from egoannot.core.hot3d import read_prepared
    from egoannot.core.video import SegmentFrames
    from egoannot.stages import spans
    root = os.environ.get('EGO_HOT3D_CORPUS')
    if not root:
        pytest.skip('set EGO_HOT3D_CORPUS to an imported HOT3D corpus')
    root = Path(root)
    definitions = json.loads((root / 'segments.json').read_text())
    monkeypatch.setenv('EGO_CORPUS', str(root))
    rows = spans.build(segments=definitions, out=tmp_path/'spans.jsonl',
                       cfg=dict(config.SPANS_CFG, quality_gate=False), quality_records=[])
    assert rows and {r['segment'] for r in rows} == {s['id'] for s in definitions}
    store = SegmentFrames(root/'segments')
    store.plan(rows, 4)
    for definition in definitions:
        ep = read_prepared(root / definition['source'], want_video=False)
        assert ep['vid'] == b'' and ep['camera_forward_axis'] == 1
        for side in ('left', 'right'):
            J = ep[f'/pose/{side}_hand_joints']
            assert np.isnan(J[:, 1]).all()
            assert np.isfinite(J[:, [0, 4, 8, 12, 16, 20]]).all()
        group = [r for r in rows if r['segment'] == definition['id']]
        assert sum(r['duration'] for r in group) == pytest.approx(definition['t1'], abs=0.01)
        for row in group:
            assert 1.3 <= row['duration'] <= 4
            assert len(store.get(row, 4)) == 4
        store.release(definition['id'])
    assert store.bytes_held == 0
