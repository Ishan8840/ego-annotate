"""Front-door safety and compatibility; real pipeline behavior has its own tests."""
from pathlib import Path
import subprocess
import sys

import pytest

from egoannot import workflow as w


def video(tmp_path, name='a clip.mp4'):
    path = tmp_path / name
    path.write_bytes(b'fixture')
    return path


@pytest.mark.parametrize('case', ['existing_output', 'missing_video', 'duplicate_stem', 'invalid_hz'])
def test_invalid_input_fails_before_inference(tmp_path, monkeypatch, case):
    source = video(tmp_path)
    out = tmp_path / 'out'
    arguments = ['hands', str(source), '--out', str(out)]
    if case == 'existing_output':
        out.mkdir()
        (out / 'keep.txt').write_text('keep')
    elif case == 'missing_video':
        source.unlink()
    elif case == 'duplicate_stem':
        other = tmp_path / 'nested'
        other.mkdir()
        arguments.insert(2, str(video(other)))
    else:
        arguments += ['--hz', 'nan']
    monkeypatch.setattr(w, 'ready', lambda *a: pytest.fail('loaded dependencies before validation'))
    with pytest.raises(SystemExit) as exc:
        w.main(arguments)
    assert exc.value.code == 2
    if case == 'existing_output':
        assert (out / 'keep.txt').read_text() == 'keep'
    else:
        assert not out.exists()


def test_quality_batch_keeps_current_processing_options(tmp_path, monkeypatch):
    first, second = video(tmp_path), video(tmp_path, 'second.mp4')
    calls = []
    monkeypatch.setattr(w, 'ready', lambda *a: True)
    monkeypatch.setattr(w, 'run_script', lambda name, args: calls.append((name, list(map(str, args)))))
    w.main(['quality', str(first), str(second), '--hands', '--out', str(tmp_path / 'out')])
    name, args = calls[0]
    assert name == 'process_quality_fast.py'
    assert args[:2] == [str(first), str(second)]  # one process, resident detector
    for option, expected in [('--mode', 'measured'), ('--hand-accuracy', 'balanced'), ('--hand-hz', '2.0')]:
        assert args[args.index(option) + 1] == expected
    assert '--hands' in args


def test_child_failure_does_not_generate_success_review(tmp_path, monkeypatch):
    source = video(tmp_path)
    monkeypatch.setattr(w, 'ready', lambda *a: True)
    monkeypatch.setattr(w.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 7))
    monkeypatch.setattr(w, 'build_review', lambda *a: pytest.fail('review after failure'))
    with pytest.raises(SystemExit) as exc:
        w.main(['hands', str(source), '--out', str(tmp_path / 'out')])
    assert exc.value.code == 7


def test_hands_renders_then_builds_review(tmp_path, monkeypatch):
    source = video(tmp_path)
    calls = []
    monkeypatch.setattr(w, 'ready', lambda *a: True)
    monkeypatch.setattr(w, 'run_script', lambda name, args: calls.append((name, args)))
    def review(*args):
        assert calls[0][0] == 'benchmark_hand_tracking.py'
        assert '--flow' in calls[0][1] and '--overlay' in calls[0][1]
        calls.append(('review', args))
        return tmp_path / 'out/index.html'
    monkeypatch.setattr(w, 'build_review', review)
    assert w.main(['hands', str(source), '--out', str(tmp_path / 'out')]) == 0
    assert len(calls) == 2


def test_missing_dependencies_leave_no_output(tmp_path, monkeypatch):
    source = video(tmp_path)
    out = tmp_path / 'out'
    monkeypatch.setattr(w, 'ready', lambda *a: False)
    monkeypatch.setattr(w, 'run_script', lambda *a: pytest.fail('launched while unavailable'))
    assert w.main(['quality', str(source), '--out', str(out)]) == 1
    assert not out.exists()


def test_empty_review_rejected(tmp_path):
    with pytest.raises(SystemExit) as exc:
        w.main(['review', 'quality', str(tmp_path), '--out', str(tmp_path / 'review')])
    assert exc.value.code == 2
    assert not (tmp_path / 'review').exists()


def test_help_from_outside_checkout_without_models(tmp_path):
    result = subprocess.run([sys.executable, str(w.ROOT / 'ego.py'), '--help'], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert 'doctor' in result.stdout and 'quality' in result.stdout


def test_model_defaults_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = w.parser().parse_args(['quality', 'input.mp4', '--hands', '--out', 'out'])
    assert args.hand_model == w.ROOT / 'models/owlv2-hand'
    assert args.videos == [Path('input.mp4')]
