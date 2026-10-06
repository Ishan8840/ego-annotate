"""Matched HOT3D runs with one loaded VLM; original source snapshots optional.

Set EGO_CORPUS, EGO_SEGMENTS, EGO_SEGMENT_DEFS, EGO_ARTIFACTS, QWEN_MODEL and
CAPTION_GREEDY=1 before launch. Outputs retain raw model replies and provenance.
"""
import argparse
import importlib.util
import json
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egoannot import config
from egoannot.stages import caption, spans, score


def load_stage(path, name):
    spec = importlib.util.spec_from_file_location('egoannot.stages.' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--original-caption')
    p.add_argument('--original-spans')
    p.add_argument('--original-domains')
    p.add_argument('--original-captions', help='reuse a baseline on the identical corpus/frames')
    args = p.parse_args()
    if not config.CAPTION['greedy']:
        raise ValueError('set CAPTION_GREEDY=1 for matched comparisons')
    out = config.ARTIFACTS
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(config.SPANS_CFG, quality_gate=False)
    span_path = out / 'spans.jsonl'
    spans.build(out=span_path, cfg=cfg)
    engine = caption.QwenLocal()
    caption.BACKENDS['qwen-local'] = lambda: engine
    results = {}
    if args.original_captions:
        shutil.copyfile(args.original_captions, out / 'original.jsonl')
        results['original'] = score.score(out / 'original.jsonl')
    if args.original_caption:
        original = load_stage(args.original_caption, '_original_caption')
        if args.original_domains:
            original.DM = load_stage(args.original_domains, '_original_domains')
        old_spans = out / 'original_spans.jsonl'
        if args.original_spans:
            load_stage(args.original_spans, '_original_spans').build(out=old_spans, cfg=cfg)
        else:
            old_spans = span_path
        # Same weights, preprocessing and greedy decoding; original prompt and binding.
        class OriginalEngine:
            name = engine.name
            def __call__(self, system, parts, batch):
                _, usage = engine(system, parts, batch)
                self.last_raw = engine.last_raw
                return original.parse_objects(self.last_raw, batch), usage
        original.BACKENDS['qwen-local'] = OriginalEngine
        caps = out / 'original.jsonl'
        original.run(old_spans, 'qwen-local', caps)
        results['original'] = score.score(caps)
    for label, retries in [('first_pass', 0), ('repaired', 1)]:
        caps = out / f'{label}.jsonl'
        caption.run(span_path, 'qwen-local', caps,
                    cfg=dict(config.CAPTION, max_retries=retries))
        results[label] = score.score(caps)
    results['protocol'] = dict(model=config.CAPTION['qwen_model'], greedy=True,
                               quality_gate=False, corpus=str(config.corpus()),
                               segments=str(config.SEGMENT_DEFS),
                               note='Format and pose consistency are proxies, not human semantic accuracy.')
    (out / 'comparison.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
