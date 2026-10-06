"""Summarize the six additional source clips and record environment versions."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egoannot.stages import score, caption

p = argparse.ArgumentParser(__doc__)
p.add_argument('--run', required=True, type=Path)
p.add_argument('--corpus', required=True, type=Path)
a = p.parse_args()
pilot = {'clip-000000', 'clip-000001', 'clip-000006', 'clip-000007', 'clip-000008'}
report = json.loads((a.run/'comparison.json').read_text())
report['additional_sequences'] = {}
for variant in ('original', 'first_pass', 'repaired'):
    rows = [json.loads(l) for l in (a.run/f'{variant}.jsonl').read_text().splitlines()]
    subset = [r for r in rows if r['segment'] not in pilot]
    path = a.run/f'{variant}_additional.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in subset))
    with contextlib.redirect_stdout(io.StringIO()):
        report['additional_sequences'][variant] = score.score(path)
    if variant != 'original':
        run = json.loads((a.run/f'{variant}.jsonl.run.json').read_text())
        report[variant].update(coverage=run['coverage'], elapsed_s=run['elapsed_s'],
                              calls=len(run['attempts']),
                              retry_calls=sum(x['attempt'] > 0 for x in run['attempts']))
manifest = json.loads((a.corpus/'manifest.json').read_text())
report['dataset'] = dict(clips=len(manifest), frames=sum(m['frame_count'] for m in manifest),
                         duration_s=sum(m['duration_s'] for m in manifest),
                         sequences=len({m['sequence_id'] for m in manifest}),
                         participants=len({m['sequence_id'].split('_')[0] for m in manifest}),
                         revision='30fe9674782f32e1e5edba98476b6ff4300132c5')
report['environment'] = {name: importlib.metadata.version(name) for name in
                         ('torch', 'torchvision', 'transformers', 'numpy', 'scipy', 'hand-tracking-toolkit')}
report['protocol'].update(model='Qwen/Qwen3-VL-8B-Instruct',
                          model_revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
                          original_commit='412fb0aaba6742bdaab57bcb7669628db327df91',
                          display_rotation_clockwise_deg=90)
root = Path(__file__).resolve().parents[1]
report['source_sha256'] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in [root/'egoannot/stages/caption.py', root/'egoannot/stages/spans.py',
                                        root/'egoannot/core/hot3d.py', root/'egoannot/labels/domains.py']}
(a.run/'system_prompt.txt').write_text(caption.system_prompt('general_manipulation', 4) + '\n')
(a.run/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
