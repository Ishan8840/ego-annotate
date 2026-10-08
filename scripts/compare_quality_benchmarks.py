"""Compare full-quality coverage and inputs; model agreement is not ground truth."""
import argparse
import json
from pathlib import Path


def compare(reference, candidate):
    left = json.loads((reference / 'quality/analysis.json').read_text())['clips'][0]
    right = json.loads((candidate / 'quality/analysis.json').read_text())['clips'][0]
    # Report rendering attaches a relative video link in memory. It is a UI
    # location, not a measurement; source hashes and every metric stay compared.
    measured = lambda clip: {k: v for k, v in clip.items() if k not in ('path', 'review_video', 'visual')}
    plans = lambda clip: [{k: w.get(k) for k in ['interval_s', 'sample_timestamps_s',
            'metric_event_count', 'metric_events_sampled']} for w in clip['visual']['windows']]
    prompts = lambda clip: sorted(json.dumps({k: c.get(k) for k in ['dimensions', 'prompt', 'window_s']}, sort_keys=True)
        for c in clip['visual']['calls'] if c.get('attempt', 0) == 0)
    a = json.loads((reference / 'quality/composite.json').read_text())
    b = json.loads((candidate / 'quality/composite.json').read_text())
    ratings = lambda obj: {(w['start_s'], r['dimension']): r['rating'] for w in obj['windows'] for r in w['ratings']}
    ar, br = ratings(a), ratings(b)
    descriptions = lambda clip: {(tuple(row['window_s']), row['dimension']): row
                                for row in clip['visual']['observations']}
    ad, bd = descriptions(left), descriptions(right)
    return dict(measured_checks_identical=measured(left) == measured(right),
        descriptive_evidence_identical=plans(left) == plans(right),
        descriptive_first_prompts_identical=prompts(left) == prompts(right),
        descriptive_system_prompt_identical=left['visual']['system_prompt'] == right['visual']['system_prompt'],
        descriptive_assessed_reference=sum(len(w['analyzed_dimensions']) for w in left['visual']['windows']),
        descriptive_assessed_candidate=sum(len(w['analyzed_dimensions']) for w in right['visual']['windows']),
        descriptive_row_pairs=len(ad.keys() & bd.keys()),
        identical_descriptive_rows=sum(ad[k] == bd[k] for k in ad.keys() & bd.keys()),
        composite_evidence_identical=[w['sample_times_s'] for w in a['windows']] == [w['sample_times_s'] for w in b['windows']],
        composite_system_prompt_identical=a['system_prompt'] == b['system_prompt'],
        composite_assessed_reference=a['assessed_dimension_windows'],
        composite_assessed_candidate=b['assessed_dimension_windows'],
        composite_rating_pairs=len(ar.keys() & br.keys()),
        identical_composite_ratings=sum(ar[k] == br[k] for k in ar.keys() & br.keys()),
        reference_score=a['score_percent'], candidate_score=b['score_percent'],
        score_delta_percentage_points=b['score_percent'] - a['score_percent'],
        limitations='Same-model agreement is not accuracy. Batching can change judgments even with identical prompts and evidence.')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if any(value is False for key, value in result.items() if key.endswith('_identical')):
        raise SystemExit('Input or deterministic-check regression detected')
    if any(result[f'{part}_assessed_candidate'] < result[f'{part}_assessed_reference']
           for part in ['descriptive', 'composite']):
        raise SystemExit('Quality assessment coverage decreased')


if __name__ == '__main__':
    main()
