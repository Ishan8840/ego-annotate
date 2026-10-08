"""Run the unchanged single-request captioner as a quality control, not a speed claim."""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('videos', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--domain', default='food_preparation')
    parser.add_argument('--camera-modality', choices=['rgb', 'monochrome'], default='rgb')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a new output directory')
    from egoannot.qwen_runtime import QwenRuntime
    spec = importlib.util.spec_from_file_location('caption_cli', ROOT / 'scripts/annotate_rgb_demo.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime = QwenRuntime(args.model, max_batch_size=1, cpu_threads=8)
    try:
        for video in args.videos:
            module.main([str(video), '--out', str(args.out / video.stem / 'captions'),
                         '--model', args.model, '--domain', args.domain,
                         '--camera-modality', args.camera_modality], engine=runtime)
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
