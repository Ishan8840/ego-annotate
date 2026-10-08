"""Create a compact, source-bound comparison from completed full-pipeline runs."""
import argparse
import csv
import json
from pathlib import Path
from compare_speed_outputs import compare


def summarize(name, reference, candidate):
    before = json.loads((reference / 'summary.json').read_text())
    after = json.loads((candidate / 'summary.json').read_text())
    old_clips = {Path(row['source']).stem: row for row in before['clips']}
    new_clips = {Path(row['source']).stem: row for row in after['clips']}
    if old_clips.keys() != new_clips.keys():
        raise ValueError('Reference and candidate must contain the same clips')
    parity = {}
    for stem, row in old_clips.items():
        if row['sha256'] != new_clips[stem]['sha256']:
            raise ValueError('Reference and candidate source hashes differ')
        parity[stem] = compare(reference / stem, candidate / stem)
    warm_before = before['warm_seconds']
    warm_after = after['warm_seconds']
    return dict(
        name=name, source_sha256={stem: row['sha256'] for stem, row in old_clips.items()},
        video_seconds=after['video_seconds'],
        reference_cold_seconds=before['cold_seconds'], candidate_cold_seconds=after['cold_seconds'],
        reference_warm_seconds=warm_before, candidate_warm_seconds=warm_after,
        warm_speedup=warm_before / warm_after,
        warm_time_reduction_percent=100 * (1 - warm_after / warm_before),
        candidate_processing_seconds_per_video_second=warm_after / after['video_seconds'],
        all_semantic_outputs_identical=all(row['all_semantic_outputs_identical'] for row in parity.values()),
        parity=parity, reference_clips=before['clips'], candidate_clips=after['clips'],
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--pair', nargs=3, action='append', metavar=('NAME', 'REFERENCE', 'CANDIDATE'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = [summarize(name, Path(before), Path(after)) for name, before, after in args.pair]
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(
        scope='All four RGB annotation/quality stages; no 3D hand-pose reconstruction.',
        interpretation='Exact parity checks regressions on this sample, not absolute semantic correctness.',
        runs=rows,
    )
    (args.out / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    fields = [key for key, value in rows[0].items() if not isinstance(value, (dict, list))]
    with (args.out / 'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps([{key: row[key] for key in fields} for row in rows], indent=2))
    if not all(row['all_semantic_outputs_identical'] for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
