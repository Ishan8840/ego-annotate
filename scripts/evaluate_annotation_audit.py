"""Recheck saved evidence without inference, then challenge the verifier on real RGB frames.

Requires existing RGB demo outputs and the local model environment. Injected
claims are synthetic test cases, never changes to the source annotations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egoannot.quality.annotations import audit, verify_label
from egoannot.stages.caption import QwenLocal
from egoannot.quality.analysis import analyze


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--cached-audit', type=Path)
    p.add_argument('--retry-quality', type=Path, help='reuse source-matched quality replies, retry invalid evidence only')
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    root = Path('artifacts/color-demo')
    if a.retry_quality:
        cached = json.loads(a.retry_quality.read_text())
        assert cached['protocol']['model'] == a.model
        assert cached['protocol']['specification'] == {}
        clip = cached['clips'][0]
        with open(clip['path'], 'rb') as fh:
            assert hashlib.file_digest(fh, 'sha256').hexdigest() == clip['sha256']
        used = set()
        class CachedQuality:
            actual = None
            def __call__(self, system, parts, batch):
                prompt = parts[0][1]
                match = None
                for i, call in enumerate(clip['visual']['calls']):
                    if i in used or not call.get('raw'):
                        continue
                    if 'inventory' in prompt and call['dimensions'] == ['dataset_diversity']:
                        match = (i, call)
                        break
                    if 'Requested dimensions: ' in prompt and call['dimensions'] != ['dataset_diversity']:
                        keys = list(json.loads(prompt.split('Requested dimensions: ')[1].split('\nTask/collection')[0]))
                        start, end = call['window_s']
                        if keys == call['dimensions'] and f'Observed window: {start:.4f}–{end:.4f}s.' in prompt:
                            expected = next(w['sample_timestamps_s'] for w in clip['visual']['windows'] if w['interval_s'] == call['window_s'])
                            actual = [float(v.split()[2]) for k, v in parts if k == 'text' and v.startswith('Frame at ')]
                            assert actual == expected, 'cached frame times changed'
                            match = (i, call)
                            break
                if match:
                    i, call = match
                    used.add(i)
                    self.last_raw = call['raw']
                    return [], dict(replayed_saved_reply=True, original_prompt=call['prompt'],
                                    original_system_prompt=clip['visual']['system_prompt'])
                if self.actual is None:
                    self.actual = QwenLocal(a.model)
                result = self.actual(system, parts, batch)
                self.last_raw = self.actual.last_raw
                return result
        from egoannot.stages import caption
        engine = CachedQuality()
        original_factory = caption.QwenLocal
        try:
            caption.QwenLocal = lambda _: engine
            report = analyze([clip['path']], a.out, visual='qwen-local', model=a.model)
        finally:
            caption.QwenLocal = original_factory
        print('QUALITY WINDOWS', [(w['interval_s'], len(w['analyzed_dimensions']), w['unavailable'])
                                   for w in report['clips'][0]['visual']['windows']], flush=True)
        return
    if a.cached_audit is None:
        p.error('--cached-audit is required unless --retry-quality is set')
    previous = json.loads(a.cached_audit.read_text())
    for source in previous['sources'].values():
        with open(source['path'], 'rb') as fh:
            assert hashlib.file_digest(fh, 'sha256').hexdigest() == source['sha256'], 'source changed'
    saved = [r['verification'] for r in previous['annotations'] if r['verification']]
    class Replay:
        def __call__(self, system, parts, _):
            candidate = json.loads(parts[0][1].split('Candidate annotation: ', 1)[1])
            times = [float(v.split()[2]) for k, v in parts if k == 'text' and v.startswith('Frame at ')]
            matches = [c for c in saved if c['candidate'] == candidate and c['sample_timestamps_s'] == times]
            assert len(matches) == 1, 'ambiguous or changed cached request'
            self.last_raw = matches[0]['raw']
            return [], {'replayed_saved_reply': True, 'original_system_prompt': matches[0]['system_prompt']}
    result = audit(root/'dense-v2/captions.jsonl', root/'dense-v2/spans.jsonl', root,
                   a.out/'annotations', 'qwen-local', a.model, Replay())
    for row in result['annotations']:
        call = row.get('verification')
        if call and call.get('usage', {}).get('replayed_saved_reply'):
            call['system_prompt'] = call['usage']['original_system_prompt']
            call['reparsed_from'] = str(a.cached_audit)
    (a.out/'annotations/audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print('ANNOTATION AUDIT', json.dumps(result['summary']), flush=True)
    spans = [json.loads(s) for s in (root/'dense-v2/spans.jsonl').read_text().splitlines()]
    cases = [('epic-prep', 16, 'Type on the laptop keyboard beside the monitor with both hands', 'type', 'laptop keyboard'),
             ('epic-cook', 17, 'Pour water from the blue bottle into the glass with both hands', 'pour', 'bottle')]
    engine = QwenLocal(a.model)
    challenges = []
    for segment, t, text, verb, noun in cases:
        span = next(r for r in spans if r['segment'] == segment and r['v_start'] <= t < r['v_end'])
        label = dict(text=text, verb=verb, noun=noun, hand='BOTH', visibility='FULL', uncertain=False)
        source = result['sources'][segment]
        call = verify_label(label, dict(span, video_path=source['path'], video_duration_s=source['duration_s']), engine)
        challenges.append(dict(span_id=span['span_id'], injected_candidate=label, verification=call))
        print('INJECTED CLAIM', span['span_id'], call.get('fields', call.get('error')), flush=True)
    (a.out/'injected_claims.json').write_text(json.dumps(challenges, indent=2)+'\n')
    # Release the caption model before the quality command creates its engine.
    del engine
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    report = analyze([root/'epic-prep.mp4'], a.out/'quality', visual='qwen-local', model=a.model)
    windows = report['clips'][0]['visual']['windows']
    print('QUALITY WINDOWS', [(w['interval_s'], len(w['analyzed_dimensions']), w['unavailable']) for w in windows], flush=True)


if __name__ == '__main__':
    main()
