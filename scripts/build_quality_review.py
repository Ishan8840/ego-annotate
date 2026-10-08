"""Build a portable, offline human review pack from compact-quality results."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build(inputs, out, references=()):
    out.mkdir(parents=True, exist_ok=False)
    reference = {}
    for root in references:
        for path in root.glob('*/quality/analysis.json'):
            for clip in json.loads(path.read_text())['clips']:
                reference[clip['sha256']] = clip
    clips = []
    for root in inputs:
        for path in sorted(root.glob('*/quality.json')):
            report = json.loads(path.read_text())
            clip = report['measured']
            if any(c['sha256'] == clip['sha256'] for c in clips):
                raise ValueError('Duplicate sources in review inputs')
            source = Path(clip['path'])
            with source.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            if digest != clip['sha256']:
                raise ValueError('Review video differs from evaluated source')
            dest = out / clip['id']
            dest.mkdir()
            shutil.copy2(source, dest / 'video.mp4')
            shutil.copy2(path, dest / 'quality.json')
            if (path.parent / 'evidence').exists():
                shutil.copytree(path.parent / 'evidence', dest / 'evidence')
            old = reference.get(clip['sha256'])
            control_file = source.with_suffix('.controls.json')
            controls = json.loads(control_file.read_text()) if control_file.exists() else None
            if controls:
                shutil.copy2(control_file, dest / 'controls.json')
            clips.append(dict(id=clip['id'], sha256=clip['sha256'], video=f'{clip["id"]}/video.mp4',
                timing=json.loads((path.parent / 'timing.json').read_text()),
                report=report, reference=old, synthetic_controls=controls,
                license='See source dataset manifest or recording owner for usage rights.'))
    payload = dict(schema_version=1, clips=clips, human_reviews=[],
                   note='Human ratings are blank. VLM agreement is not correctness. No automatic acceptance score.')
    (out / 'review-data.json').write_text(json.dumps(payload, indent=2) + '\n')
    template = (ROOT / 'demo/quality-review.html').read_text()
    # Escape script terminators and Unicode line separators in untrusted model text.
    embedded = json.dumps(payload).replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    (out / 'index.html').write_text(template.replace('/* REVIEW_DATA */ null', embedded))
    print(out / 'index.html')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('inputs', type=Path, nargs='+')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reference', type=Path, action='append', default=[])
    args = p.parse_args()
    build(args.inputs, args.out, args.reference)


if __name__ == '__main__':
    main()
