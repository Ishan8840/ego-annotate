"""Build film data from saved RGB pipeline output; never fabricate missing labels."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/color-demo'
ANNOTATIONS = SOURCE / 'dense-v2'
OUT = ROOT / 'artifacts/demo'
OUT.mkdir(parents=True, exist_ok=True)
summary = json.loads((ANNOTATIONS / 'summary.json').read_text())
spans = [json.loads(l) for l in (ANNOTATIONS / 'spans.jsonl').read_text().splitlines()]
captions = [json.loads(l) for l in (ANNOTATIONS / 'captions.jsonl').read_text().splitlines()]
assert len(captions) == len(spans) == summary['captions']
assert {c['span_id'] for c in captions} == {s['span_id'] for s in spans}
assert len({c['span_id'] for c in captions}) == len(captions)
data = dict(captions={}, summary=summary, preview_only=False)
for source in summary['sources']:
    sid = source['id']
    path = SOURCE / f'{sid}.mp4'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256']
    rows = sorted([c for c in captions if c['segment'] == sid], key=lambda c: c['start_ts'])
    assert rows and rows[0]['start_ts'] == 0
    assert abs(rows[-1]['end_ts'] - source['duration_s']) < .002
    assert all(abs(a['end_ts']-b['start_ts']) < .002 for a, b in zip(rows, rows[1:]))
    data['captions'][sid] = rows
    dest = OUT / 'frames' / sid
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(path), '-q:v', '2',
                    '-start_number', '0', str(dest/'%03d.jpg')], check=True)
quality_path = SOURCE / 'quality/analysis.json'
quality = json.loads(quality_path.read_text())
for clip in quality['clips']:
    assert clip['sha256'] == next(s['sha256'] for s in summary['sources'] if s['id'] == clip['id'])
composite_path = SOURCE / 'quality/composite.json'
composite = json.loads(composite_path.read_text())
assert composite['source_sha256'] == {c['id']:c['sha256'] for c in quality['clips']}
assert len(composite['components']) == 8
assert all(c['score_percent'] is not None for c in composite['components'].values())
assert abs(composite['score_percent'] - sum(c['score_percent'] for c in composite['components'].values())/8) < 1e-9
assert 0 <= composite['score_percent'] <= 100
evidence_clip = next(c for c in quality['clips'] if c['id'] == 'epic-prep')
observations = {o['dimension']:o for o in evidence_clip['visual']['observations']}
hand, glare = observations['hand_visibility'], observations['visual_quality']
assert any('cropped' in c.lower() for c in hand['concerns'])
assert any('glare' in c.lower() for c in glare['concerns'])
def time_range(row):
    a,b = row['evidence_s'][0]
    return f'{a:05.2f}–{b:05.2f} s'
data['quality'] = dict(overall_score_percent=composite['score_percent'],score_components=composite['components'],
    score_coverage=dict(assessed=composite['assessed_dimension_windows'],total=composite['total_dimension_windows']),frames=quality['summary']['total_frames'], clips=quality['summary']['clips'],
    decode_error_clips=sum(c['integrity']['decode_exit'] != 0 or bool(c['integrity']['diagnostics']) for c in quality['clips']),
    evidence=dict(clip='epic-prep',frame_s=hand['evidence_s'][0][1],
                  hand_time=time_range(hand),glare_time=time_range(glare),glare_frame_s=glare['evidence_s'][0][1]),
    hand_observation=hand,glare_observation=glare)
(OUT / 'film-data.json').write_text(json.dumps(data, indent=2)+'\n')
provenance = dict(
    brand=None, revision='Unbranded green / zucchini to quality / combined visual-quality estimate', duration_seconds=30,
    resolution=[1920,1080], fps=30,
    source_dataset='EPIC-KITCHENS', original_video='P01_01.MP4',
    source_url='https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/P01/P01_01.MP4',
    source_intervals_seconds={'epic-prep':[41,65], 'epic-cook':[90,114]},
    preprocessing='Original RGB footage, resized to 1280x720 and resampled to 30 FPS. No colorization or generative edits.',
    caption_source='artifacts/color-demo/dense-v2/captions.jsonl',
    raw_model_replies='artifacts/color-demo/dense-v2/run.json',
    model=summary['model'], boundary_source=summary['boundary_source'],
    presentation='Edited motion-graphic presentation of saved outputs. Annotation footage plays at 1x; the quality section holds a timestamped evidence frame; this is not a recording of inference speed. No skeletons or trajectories.',
    caption_handling='Verbatim model captions and structured labels. Source dataset action annotations were used to select footage, never passed to the caption model.',
    film_edits=[dict(film=[0,3],source='epic-cook',source_start=18),
                dict(film=[3,17],source='epic-prep',source_start=10),
                dict(film=[17,22],source='epic-cook',source_start=15),
                dict(film=[22,30],content='Ego quality: model observations and measured video integrity')],
    claims={k:summary[k] for k in ['seconds','captions','annotations_per_minute','median_span_s','time_coverage','format_valid','uncertain']},
    quality_source=str(quality_path.relative_to(ROOT)),
    quality_claims=data['quality'],
    composite_source=str(composite_path.relative_to(ROOT)),
    visual_theme='Forest green, lime and warm white',
    interpretation='Overall quality percentage is a normalized model estimate across eight equally weighted visual criteria. It is unvalidated and is not a probability or a percentage of usable frames. Quality visual findings are sampled model observations, not verified defects. Annotation density is measured on two 24-second excerpts. It is not a semantic accuracy score. Model captions may contain semantic and format errors. All original output and validation flags are retained.',
    credits=dict(footage='EPIC-KITCHENS dataset',dataset_url='https://epic-kitchens.github.io/',
                 annotations_url='https://github.com/epic-kitchens/epic-kitchens-100-annotations',
                 fonts='Inter Display and JetBrains Mono; licenses in demo/fonts',
                 music='Original warm ambient chord score from demo/make-music.py; no percussion or external recordings; mixed at -20 LUFS'),
    input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [ANNOTATIONS/'captions.jsonl',ANNOTATIONS/'run.json',quality_path,composite_path,OUT/'film-data.json']})
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps(provenance['claims'],indent=2))
