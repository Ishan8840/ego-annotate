"""Compare retained QC exactly; report caption changes without claiming accuracy."""
import argparse
from collections import Counter
import json
from pathlib import Path


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def compare(reference, candidate):
    before = json.loads((reference / 'quality/analysis.json').read_text())['clips'][0]
    after = json.loads((candidate / 'quality/analysis.json').read_text())['clips'][0]
    before = {k: v for k, v in before.items() if k not in ('path', 'visual')}
    after = {k: v for k, v in after.items() if k not in ('path', 'visual')}
    left = {r['span_id']: r for r in jsonl(reference / 'captions/captions.jsonl')}
    right = {r['span_id']: r for r in jsonl(candidate / 'captions/captions.jsonl')}
    ids = sorted(left.keys() & right.keys())
    fields = ['text', 'verb', 'noun', 'hand', 'visibility', 'uncertain']
    changes = [dict(span_id=sid, reference={k: left[sid][k] for k in fields},
                    candidate={k: right[sid][k] for k in fields}) for sid in ids
               if any(left[sid][key] != right[sid][key] for key in fields)]
    original = json.loads((reference / 'audit/audit.json').read_text())
    verification_path = candidate / 'audit/verification.json'
    if not verification_path.exists():
        verification_path = candidate / 'audit/compact.json'
    compact = json.loads(verification_path.read_text())
    checks = {row['span_id']: row for row in compact['annotations']}
    unchanged = {sid for sid in ids if all(left[sid][k] == right[sid][k] for k in fields)}
    pairs = []
    for row in original['annotations']:
        if row['span_id'] not in unchanged:
            continue
        old = (row.get('verification') or {}).get('fields', {})
        new = checks.get(row['span_id'], {}).get('fields', {})
        pairs += [(old[k]['verdict'], new[k]['verdict']) for k in old.keys() & new.keys()]
    return dict(source_hash_equal=before['sha256'] == after['sha256'],
        measured_quality_identical=before == after,
        changed_measured_keys=sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k)),
        spans_identical=jsonl(reference / 'captions/spans.jsonl') == jsonl(candidate / 'captions/spans.jsonl'),
        shared_caption_count=len(ids), missing_caption_ids=sorted(left.keys() - right.keys()),
        reference_format_valid=sum(not r['validation_errors'] for r in left.values()),
        candidate_format_valid=sum(not r['validation_errors'] for r in right.values()),
        identical_fields={k: sum(left[sid][k] == right[sid][k] for sid in ids) for k in fields},
        caption_changes=changes,
        audit_proxy_on_identical_captions=dict(captions=len(unchanged), field_pairs=len(pairs),
            identical_verdicts=sum(a == b for a, b in pairs),
            confusion_counts=dict(Counter(a + ' -> ' + b for a, b in pairs))),
        interpretation='Measured parity is a regression check. Caption changes and same-model judge agreement do not establish semantic accuracy.')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'caption_changes'}, indent=2))
    if not all(result[k] for k in ['source_hash_equal', 'measured_quality_identical', 'spans_identical']):
        raise SystemExit('Deterministic regression detected')


if __name__ == '__main__':
    main()
