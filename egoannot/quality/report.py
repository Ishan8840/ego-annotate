"""Portable dataset summary and timestamped, playable evidence."""
import json
from pathlib import Path
import shutil
from urllib.parse import quote


def _pct(value):
    return 'unavailable' if value is None else f'{value * 100:.2f}%'


def narrative(data):
    s = data['summary']
    lines = [f'The analyzed sample contains {s["clips"]} videos, {s["total_frames"]:,} decoded frames '
             f'and {s["total_seconds"]:.2f} seconds of footage.',
             f'Decoder/probe diagnostics occurred in {len(s["integrity_diagnostic_clips"])} clips; '
             f'metadata or expected-format discrepancies occurred in {len(s["metadata_discrepancy_clips"])} clips.']
    m = s['candidate_metrics']
    lines.append(f'Across decoded frames, {_pct(m["dark_frames"]["frame_fraction"])} have low mean luminance, '
                 f'{_pct(m["light_clipping"]["frame_fraction"])} exceed the bright-pixel threshold, and '
                 f'{_pct(m["low_detail"]["frame_fraction"])} fall below the image-detail threshold. '
                 'These are diagnostic candidates; they do not by themselves establish darkness, glare or blur defects.')
    lines.append(f'{_pct(m["identical_adjacent_frames"]["frame_fraction"])} of decoded frames repeat the preceding frame exactly. '
                 f'Duplicate search found {len(s["duplicate_pairs"])} matching or similar clip pairs.')
    observed = [v['analyzed_clips'] for v in s['visual_dimensions'].values()]
    lines.append(f'Visual analysis produced {sum(observed)} dimension-level observations from sampled frames. '
                 'Counts below describe model observations, not verified defect prevalence. Missing context and unexamined intervals remain unknown.')
    most = sorted(s['visual_dimensions'].items(), key=lambda x: x[1]['clips_with_model_concerns'], reverse=True)
    most = [(dim.replace('_', ' '), row) for dim, row in most[:3] if row['clips_with_model_concerns']]
    if most:
        lines.append('The model most often raised concerns about ' + '; '.join(
            f'{dim} ({row["clips_with_model_concerns"]}/{row["analyzed_clips"]} analyzed clips)' for dim, row in most)
                     + '. Read the descriptions and evidence below to assess their significance.')
    diversity = s['diversity']
    if diversity['participant_count']:
        lines.append(f'Source metadata identifies {diversity["participant_count"]} participants and '
                     f'{len(diversity["sequence_clip_counts"])} sequences. '
                     'This alone does not establish task, object or environment diversity.')
    if data['clips'] and all(c['provenance'].get('dataset') == 'HOT3D-Clips' for c in data['clips']):
        lines.append('These HOT3D inputs are short clips. An action continuing beyond a clip boundary '
                     'does not establish that the original demonstration was recorded incompletely.')
    return lines


def write_reports(data, out):
    out = Path(out)
    paragraphs = narrative(data)
    s = data['summary']
    md = ['# Dataset quality analysis', '', *[p + '\n' for p in paragraphs],
          '## Visual observations and coverage', '',
          '| Dimension | Clips analyzed | Clips with model concerns |', '|---|---:|---:|']
    for dim, row in s['visual_dimensions'].items():
        md.append(f'| {dim.replace("_", " ")} | {row["analyzed_clips"]}/{s["clips"]} | {row["clips_with_model_concerns"]} |')
    md.extend(['', '## Per-clip findings', ''])
    for clip in data['clips']:
        md += [f'### {clip["id"]}', '',
               f'{clip["duration_s"]:.3f}s · {clip["fps"]:.3f} FPS · {clip["width"]}×{clip["height"]} · {clip["frame_count"]} frames', '']
        for obs in clip['observations']:
            times = ', '.join(f'{a:.2f}–{b:.2f}s' for a, b in obs['intervals_s'][:8])
            md.append(f'- {obs["kind"]}: {obs["description"]} {times}')
        for obs in clip.get('visual', {}).get('observations', []):
            md.append(f'- {obs["dimension"]}: {obs["summary"]} Evidence: {obs["evidence_s"]}. Subjective confidence: {obs["confidence"]}.')
            if obs.get('interpretation_limit'):
                md.append('  Limitation: ' + obs['interpretation_limit'])
            if obs.get('claim_caveats'):
                md.append('  Wording caveat: ' + ' '.join(obs['claim_caveats']))
        for dim, why in clip.get('visual', {}).get('unavailable', {}).items():
            md.append(f'- {dim}: {why}')
        md.append('')
    md += ['## Interpretation limits', '', *['- ' + x for x in data['protocol']['limitations']]]
    md += ['', '## Dataset diversity', '']
    for field, values in s['diversity']['dimensions'].items():
        md += ['', f'### {field.replace("_", " ")}', '',
               f'Supplied labels for {values["labeled_clips"]} clips: {values["label_counts"]}', '']
        md += [f'- {v["clip"]}: {v["description"]} (sampled model description)' for v in values['model_descriptions']]
    (out / 'summary.md').write_text('\n'.join(md) + '\n')
    # Copy videos so the HTML works over a basic static server or offline.
    (out / 'videos').mkdir(exist_ok=True)
    for clip in data['clips']:
        dest = out / 'videos' / (clip['id'] + Path(clip['path']).suffix)
        if dest.resolve() != Path(clip['path']).resolve():
            shutil.copyfile(clip['path'], dest)
        clip['review_video'] = 'videos/' + quote(dest.name)
    payload = json.dumps(data, allow_nan=False).replace('<', '\\u003c')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dataset quality analysis</title><style>
*{box-sizing:border-box}body{margin:0;background:#111720;color:#e5ecf3;font:16px/1.55 system-ui}main{max-width:1200px;margin:auto;padding:32px}
h1{font-size:32px;letter-spacing:-.8px}h2{margin-top:32px}h3{margin:0}p{max-width:1000px}a{color:#9cd8ff}
.muted,small{color:#aebbd0}small{display:block}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:10px;border-bottom:1px solid #344154}
th{color:#aebbd0}.cards{display:grid;gap:24px}.card{background:#1d2836;border-radius:12px;padding:20px}.body{display:grid;grid-template-columns:minmax(230px,340px) 1fr;gap:20px;margin-top:16px}
video{width:100%;max-height:480px;background:#000;position:sticky;top:15px}button,select{font:inherit;color:inherit;background:#2b3b50;border:1px solid #5c7390;border-radius:5px;padding:4px 8px;cursor:pointer}
button{margin:3px}.finding{margin:12px 0;border-bottom:1px solid #344154;padding-bottom:12px}.finding p{margin:4px 0}
summary{cursor:pointer;color:#b9dfff}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}label{display:block;margin:20px 0}
@media(max-width:720px){main{padding:18px}.body{grid-template-columns:1fr}video{position:static}table{font-size:12px}td,th{padding:5px}}
</style><main><h1>Dataset quality analysis</h1><div id="intro"></div>
<p><a href="analysis.json">Full measurements and raw model replies</a> · <a href="summary.md">Text report</a></p>
<h2>Visual findings and analysis coverage</h2><p class="muted">Concern counts are model observations from sampled images. They are not verified defect rates. Select a dimension to read the evidence.</p>
<table id="coverage"><thead><tr><th>Dimension</th><th>Clips analyzed</th><th>Clips with concerns</th><th>Example observations</th></tr></thead><tbody></tbody></table>
<h2>Duplicates and dataset diversity</h2><div id="dataset"></div>
<details><summary>Methods, thresholds and limits</summary><div id="limits"></div></details>
<h2>Clip evidence</h2><label>Focus <select id="focus"><option value="all">All dimensions</option><option value="measured">Measured video/image observations</option></select></label>
<div class="cards" id="clips"></div></main><script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.querySelector('#data').textContent),focus=document.querySelector('#focus');
function el(tag,text,parent){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(parent)parent.append(e);return e;}
for(const p of __NARRATIVE__)el('p',p,document.querySelector('#intro'));
for(const [dim,row] of Object.entries(data.summary.visual_dimensions)){
 const tr=el('tr',undefined,document.querySelector('#coverage tbody'));
 const cell=el('td',undefined,tr),b=el('button',dim.replaceAll('_',' '),cell);b.onclick=()=>{focus.value=dim;render();document.querySelector('#focus').scrollIntoView();};
 el('td',row.analyzed_clips+' / '+data.summary.clips,tr);el('td',String(row.clips_with_model_concerns),tr);
 const examples=row.observations.filter(o=>o.concerns.length).slice(0,2);
 el('td',examples.length?examples.map(o=>o.clip+': '+o.concerns.join('; ')).join(' · '):
 (row.observations[0]?.summary??row.unavailable[0]?.reason??'Unknown'),tr);
 const opt=el('option',dim.replaceAll('_',' '),focus);opt.value=dim;}
const dataset=document.querySelector('#dataset');
el('p',data.summary.duplicate_pairs.length?'Matching/similar pairs:':'No matching pairs found by the implemented full-clip comparisons.',dataset);
for(const pair of data.summary.duplicate_pairs)el('p',pair.clips.join(' ↔ ')+' · '+pair.kind,dataset);
const diversity=data.summary.diversity;
el('p','Source sequences: '+Object.keys(diversity.sequence_clip_counts).length+' · Participants from metadata: '+(diversity.participant_count??'unknown')+' · Resolutions: '+JSON.stringify(diversity.resolutions),dataset);
for(const [key,row] of Object.entries(diversity.dimensions)){
 const d=el('details',undefined,dataset);el('summary',key.replaceAll('_',' ')+': '+row.labeled_clips+' labeled clips · '+row.model_descriptions.length+' model descriptions',d);
 el('p','Supplied labels: '+JSON.stringify(row.label_counts),d);
 for(const item of row.model_descriptions)el('p',item.clip+': '+item.description,d);}
el('p','Model descriptions are an inventory, not verified counts of unique objects or environments. A small sample does not establish sufficiency for a target dataset.',dataset);
for(const line of data.protocol.limitations)el('p',line,document.querySelector('#limits'));
el('pre',JSON.stringify(data.protocol,null,2),document.querySelector('#limits'));
function seek(parent,video,spans){for(const [a,b] of spans){const button=el('button',a.toFixed(2)+'–'+b.toFixed(2)+' s',parent);button.onclick=()=>{video.currentTime=a;video.play().catch(()=>{});};}}
function render(){const root=document.querySelector('#clips');root.replaceChildren();
 for(const c of data.clips){const card=el('article',undefined,root);card.className='card';el('h3',c.id,card);
 el('small',c.duration_s.toFixed(2)+' seconds · '+c.fps.toFixed(2)+' FPS · '+c.width+'×'+c.height+' · '+c.frame_count+' frames',card);
 const body=el('div',undefined,card);body.className='body';const player=el('div',undefined,body),video=el('video',undefined,player);video.controls=true;video.preload='metadata';video.src=c.review_video;
 const source=el('a','Source video',player);source.href=c.review_video;
 video.onerror=()=>el('p','Preview unavailable in this browser. Open the source video in a compatible player.',player);
 const findings=el('div',undefined,body),v=c.visual;
 el('small',v?'Visual samples at '+v.sample_timestamps_s.map(t=>t.toFixed(2)).join(', ')+' s':'Visual model not run.',findings);
 if(focus.value==='all'||focus.value==='measured'){
  for(const o of c.observations){const f=el('div',undefined,findings);f.className='finding';el('strong',o.kind.replaceAll('_',' ')+' · '+o.source,f);el('p',o.description,f);seek(f,video,o.intervals_s);}
  if(!c.observations.length)el('p','No candidates under the configured measured thresholds. See visual findings and limits.',findings);
  const d=el('details',undefined,findings);el('summary','Measurements, metadata and hand annotation coverage',d);el('pre',JSON.stringify({measurements:c.measurements,frame_fractions:c.candidate_frame_fractions,integrity:c.integrity,hand_annotations:c.hand_annotations},null,2),d);}
 for(const o of v?.observations??[]){if(focus.value!=='all'&&focus.value!==o.dimension)continue;
  const f=el('div',undefined,findings);f.className='finding';el('strong',o.dimension.replaceAll('_',' '),f);el('p',o.summary,f);
  for(const concern of o.concerns)el('p','Concern: '+concern,f);el('small','Sampled visual judgment · subjective confidence '+o.confidence,f);
  if(o.interpretation_limit)el('small',o.interpretation_limit,f);
  for(const caveat of o.claim_caveats??[])el('small','Wording caveat: '+caveat,f);seek(f,video,o.evidence_s);}
 for(const [dim,why] of Object.entries(v?.unavailable??{})){if(focus.value==='all'||focus.value===dim)el('p',dim.replaceAll('_',' ')+': '+why,findings);}
 }
}
focus.onchange=render;render();</script></html>'''
    page = page.replace('__DATA__', payload).replace('__NARRATIVE__', json.dumps(paragraphs).replace('<', '\\u003c'))
    (out / 'index.html').write_text(page)
