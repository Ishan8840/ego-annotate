"""Small front door to existing pipelines. Inference and report schemas stay there.

Paths supplied by users are relative to their working directory. Default models
and script paths are relative to this checkout. No shell interpolation is used.
"""
import argparse
import importlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('must be finite and greater than zero')
    return value


def parser():
    p = argparse.ArgumentParser(description='Process ego video, track hands, and review results.',
        epilog='Advanced and historical commands remain under scripts/ and python -m egoannot.')
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('doctor', help='check dependencies, GPU and local models')
    d.add_argument('--profile', choices=['measured', 'hands', 'hybrid'], default='measured')
    d.add_argument('--hand-model', type=Path, default=ROOT / 'models/owlv2-hand')
    d.add_argument('--model', type=Path, default=ROOT / 'models/qwen3-vl-8b')
    sub.add_parser('download-hands', help='download the existing pinned Apache-2.0 OWLv2 checkpoint')
    for name, description in [('quality', 'all-frame quality checks; optional hands and VLM'),
                               ('hands', 'hand tracking with a green overlay and review page')]:
        sp = sub.add_parser(name, help=description)
        sp.add_argument('videos', type=Path, nargs='+', help='one or more video files; unique basenames')
        sp.add_argument('--out', type=Path, required=True, help='new output directory (never overwritten)')
        sp.add_argument('--hand-model', type=Path, default=ROOT / 'models/owlv2-hand')
        sp.add_argument('--accuracy', choices=['fast', 'balanced', 'high'], default='balanced',
                        help='hand detector profile (default: balanced)')
        sp.add_argument('--hz', type=positive_float, default=2., help='hand detection Hz (default: 2)')
        if name == 'quality':
            sp.add_argument('--mode', choices=['measured', 'hybrid'], default='measured',
                            help='measured: no VLM (default); hybrid: add compact VLM context')
            sp.add_argument('--hands', action='store_true', help='include hand tracking in the same decode pass')
            sp.add_argument('--model', type=Path, default=ROOT / 'models/qwen3-vl-8b')
        else:
            sp.add_argument('--no-overlay', action='store_true', help='save reports only; skip video/review rendering')
    r = sub.add_parser('review', help='build an offline quality or hand review page')
    r.add_argument('kind', choices=['quality', 'hands'])
    r.add_argument('input', type=Path, help='the output directory from quality or hands')
    r.add_argument('--out', type=Path, help='new review directory; required for quality, unused for hands')
    return p


def checks(profile, hand_model=None, model=None):
    """Lightweight readiness checks, not an inference or checkpoint-integrity test."""
    yield (3, 11) <= sys.version_info[:2] < (3, 13), f'Python 3.11 or 3.12 ({sys.version.split()[0]})'
    for name in ('ffmpeg', 'ffprobe'):
        yield bool(shutil.which(name)), f'{name} on PATH (install FFmpeg if missing)'
    packages = ['numpy', 'cv2']
    if profile in ('hands', 'hybrid'):
        packages += ['torch', 'torchvision', 'transformers', 'scipy']
    if profile == 'hybrid':
        packages += ['vllm']
    loaded = {}
    for name in packages:
        try:
            loaded[name] = importlib.import_module(name)
            yield True, f'{name} {getattr(loaded[name], "__version__", "available")}'
        except Exception as exc:
            yield False, f'{name}: {exc}; see docs/getting-started.md'
    if profile in ('hands', 'hybrid'):
        torch = loaded.get('torch')
        try:
            available = torch is not None and torch.cuda.is_available()
            if available:
                # A real operation catches wheels that cannot run this GPU architecture.
                torch.zeros(1, device='cuda').add_(1).item()
            yield bool(available), 'CUDA GPU usable by this Python environment'
        except Exception as exc:
            yield False, f'CUDA operation failed: {exc}'
    if hand_model is not None:
        path = Path(hand_model)
        try:
            manifest = json.loads((path / 'manifest.json').read_text())
            ready = (manifest.get('revision') == 'cfd3195ba4ea9592eec887ded089f4c08eff231d'
                     and manifest.get('weight_license') == 'Apache-2.0'
                     and (path / 'model.safetensors').is_file())
            yield ready, f'OWLv2 model present: {path} (inference verifies hashes)'
        except (OSError, ValueError):
            yield False, f'OWLv2 missing/invalid: {path}; run python ego.py download-hands'
    if model is not None:
        path = Path(model)
        ready = (path / 'config.json').is_file() and any(path.glob('*.safetensors'))
        yield ready, f'Qwen local model present: {path}; setup: docs/getting-started.md'


def ready(profile, hand_model=None, model=None):
    results = list(checks(profile, hand_model, model))
    for ok, message in results:
        print(f'{"OK" if ok else "MISSING"}  {message}', flush=True)
    return all(ok for ok, _ in results)


def run_script(name, arguments, **kwargs):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts' / name), *map(str, arguments)], **kwargs)
    if result.returncode:
        raise SystemExit(result.returncode)


def validate_sources(p, args):
    if args.out.exists():
        p.error(f'Output already exists: {args.out}. Choose a new --out directory.')
    for path in args.videos:
        if not path.is_file():
            p.error(f'Video does not exist: {path}')
    if len({path.stem for path in args.videos}) != len(args.videos):
        p.error('Video basenames must be unique because each gets its own output folder.')


def build_review(p, kind, source, out=None):
    if not source.is_dir():
        p.error(f'Results directory does not exist: {source}')
    if kind == 'quality':
        if out is None or out.exists():
            p.error('Quality review requires --out pointing to a new directory.')
        if not any(source.glob('*/quality.json')):
            p.error('No quality reports found; supply the parent output directory.')
        run_script('build_quality_review.py', [source, '--out', out])
        return out / 'index.html'
    if out is not None:
        p.error('Hand review is written inside its input directory; omit --out.')
    reports = list(source.glob('*/hands.json'))
    if not reports:
        p.error('No hand reports found; supply the parent output directory.')
    if any(not (path.parent / 'overlay.mp4').is_file() for path in reports):
        p.error('Hand review requires overlays. Run the hands command without --no-overlay.')
    run_script('build_hand_review.py', [source])
    return source / 'index.html'


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.command == 'doctor':
        return 0 if ready(args.profile,
            args.hand_model if args.profile == 'hands' else None,
            args.model if args.profile == 'hybrid' else None) else 1
    if args.command == 'download-hands':
        # The existing audited downloader writes relative to its working directory.
        run_script('download_owl_hand.py', [], cwd=ROOT)
        return 0
    if args.command == 'review':
        page = build_review(p, args.kind, args.input, args.out)
        print(f'Open in your browser: {page.resolve()}')
        return 0
    validate_sources(p, args)
    with_hands = args.command == 'hands' or args.hands
    profile = 'hybrid' if args.command == 'quality' and args.mode == 'hybrid' else ('hands' if with_hands else 'measured')
    if not ready(profile, args.hand_model if with_hands else None,
                 args.model if profile == 'hybrid' else None):
        return 1
    common = [*args.videos, '--out', args.out]
    if args.command == 'hands':
        options = [*common, '--model', args.hand_model, '--accuracy', args.accuracy,
                   '--hz', args.hz, '--flow']
        if not args.no_overlay:
            options.append('--overlay')
        run_script('benchmark_hand_tracking.py', options)
        if not args.no_overlay:
            page = build_review(p, 'hands', args.out)
            print(f'Open in your browser: {page.resolve()}')
    else:
        options = [*common, '--mode', args.mode, '--model', args.model]
        if with_hands:
            options += ['--hands', '--hand-model', args.hand_model,
                        '--hand-accuracy', args.accuracy, '--hand-hz', args.hz]
        run_script('process_quality_fast.py', options)
    print(f'Results: {args.out.resolve()}')
    return 0
