"""Attach official timed EPIC narrations for separate human review, never inference.

Sparse participant narrations are not a complete reference for every generated
caption detail. This exports temporal overlaps, not an automatic accuracy score.
Research evaluation only: source annotations are CC BY-NC 4.0.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request

REVISION = 'ea8b40457a400c3fffa1c7f406ef3dc169cc2522'
URL = f'https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/{REVISION}/EPIC_100_train.csv'


def seconds(timestamp):
    hours, minutes, secs = map(float, timestamp.split(':'))
    return hours * 3600 + minutes * 60 + secs


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--captions', type=Path, required=True)
    parser.add_argument('--video-id', required=True)
    parser.add_argument('--source-offset', type=float, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    data = urllib.request.urlopen(URL, timeout=60).read()
    references = [row for row in csv.DictReader(io.StringIO(data.decode()))
                  if row['video_id'] == args.video_id]
    if not references:
        parser.error('No official train annotations found for that video')
    rows = []
    for line in args.captions.read_text().splitlines():
        caption = json.loads(line)
        start = caption['start_ts'] + args.source_offset
        end = caption['end_ts'] + args.source_offset
        overlaps = []
        for reference in references:
            rs, re = seconds(reference['start_timestamp']), seconds(reference['stop_timestamp'])
            overlap = max(0., min(end, re) - max(start, rs))
            if overlap > 0:
                overlaps.append(dict(
                    narration_id=reference['narration_id'], narration=reference['narration'],
                    verb=reference['verb'], noun=reference['noun'],
                    original_start_s=rs, original_end_s=re, overlap_s=overlap,
                ))
        rows.append(dict(caption=caption, official_temporal_overlaps=overlaps))
    result = dict(
        source_url=URL, revision=REVISION, source_csv_sha256=hashlib.sha256(data).hexdigest(),
        license='CC-BY-NC-4.0; research evaluation only', video_id=args.video_id,
        source_offset_s=args.source_offset,
        interpretation='Temporal overlap is not semantic agreement. Review images and narrations independently.',
        rows=rows,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Exported {len(rows)} caption review rows to {args.out}')


if __name__ == '__main__':
    main()
