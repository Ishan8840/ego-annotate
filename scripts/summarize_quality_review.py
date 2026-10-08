"""Compare exported human labels with the compact context judgments.

The report measures agreement with supplied human labels, not calibrated data
utility. Unreviewed, uncertain, and model-abstained rows remain separate.
"""
import argparse
import json
from pathlib import Path


def summarize(data, human, include_revealed=False):
    sources = {c['sha256']: c for c in data['clips']}
    rows, excluded, seen = [], [], set()
    for review in human['reviews']:
        sha, start = review['source_sha256'], review['start_s']
        if sha not in sources:
            raise ValueError('Human labels refer to an unknown video hash')
        matches = [w for w in sources[sha]['report']['windows'] if w['start_s'] == start and w['end_s'] == review['end_s']]
        if len(matches) != 1:
            raise ValueError('Human labels refer to an unknown window')
        window = matches[0]
        for dimension, label in review['ratings'].items():
            identity = (sha, start, dimension)
            if identity in seen:
                raise ValueError('Duplicate human rating')
            seen.add(identity)
            if label['status'] not in ('C', 'N', 'U'):
                raise ValueError('Invalid human rating')
            if dimension not in sources[sha]['report']['context_dimensions']:
                excluded.append(dict(reason='measured_proxy_human_label', dimension=dimension))
                continue
            if label.get('model_revealed_before_rating', True) and not include_revealed:
                excluded.append(dict(reason='not_blind', dimension=dimension))
                continue
            if label['status'] == 'U':
                excluded.append(dict(reason='human_uncertain', dimension=dimension))
                continue
            predicted = window['rows'].get(dimension, {}).get('status', 'missing')
            rows.append(dict(dimension=dimension, human=label['status'], predicted=predicted))
    results = {}
    for dim in sorted({r['dimension'] for r in rows}):
        values = [r for r in rows if r['dimension'] == dim]
        comparable = [r for r in values if r['predicted'] in ('C', 'N')]
        tp = sum(r['human'] == 'C' and r['predicted'] == 'C' for r in comparable)
        fp = sum(r['human'] == 'N' and r['predicted'] == 'C' for r in comparable)
        fn = sum(r['human'] == 'C' and r['predicted'] == 'N' for r in comparable)
        tn = sum(r['human'] == 'N' and r['predicted'] == 'N' for r in comparable)
        results[dim] = dict(human_definite=len(values), model_definite=len(comparable),
            model_unknown_or_missing=len(values) - len(comparable),
            true_concerns=tp, false_alarms=fp, missed_concerns=fn, true_no_concern=tn,
            concern_precision=tp / (tp + fp) if tp + fp else None,
            concern_recall_on_model_definite=tp / (tp + fn) if tp + fn else None,
            # Count model abstentions as unresolved concerns, not successful recall.
            resolved_concern_recall=tp / sum(r['human'] == 'C' for r in values) if any(r['human'] == 'C' for r in values) else None)
    return dict(dimensions=results, excluded=excluded, includes_revealed=include_revealed,
                limitation='Agreement with supplied human ratings only; small selected review set is not a dataset-wide accuracy estimate.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('review_data', type=Path)
    p.add_argument('human_export', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--include-revealed', action='store_true')
    a = p.parse_args()
    result = summarize(json.loads(a.review_data.read_text()), json.loads(a.human_export.read_text()), a.include_revealed)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
