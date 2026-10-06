"""Build a portable, local review page for matched HOT3D caption runs."""
import argparse
import html
import json
from pathlib import Path
import shutil


def build(corpus, run, out):
    corpus, run, out = map(Path, (corpus, run, out))
    (out/'videos').mkdir(parents=True, exist_ok=True)
    definitions = json.loads((corpus/'segments.json').read_text())
    results = json.loads((run/'comparison.json').read_text())
    data = {}
    for variant in ('original', 'first_pass', 'repaired'):
        path = run/f'{variant}.jsonl'
        if path.exists():
            data[variant] = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
            shutil.copyfile(path, out/path.name)
    for clip in definitions:
        shutil.copyfile(corpus/'segments'/f'{clip["id"]}.mp4', out/'videos'/f'{clip["id"]}.mp4')
    shutil.copyfile(run/'comparison.json', out/'comparison.json')
    shutil.copyfile(corpus/'manifest.json', out/'manifest.json')
    cards = ''.join(f'<article data-clip="{html.escape(c["id"])}"><h2>{html.escape(c["id"])}</h2>'
                    f'<video controls preload="metadata" src="videos/{html.escape(c["id"])}.mp4"></video>'
                    '<p class="active"></p><ol class="captions"></ol></article>' for c in definitions)
    payload = json.dumps(dict(captions=data, metrics=results)).replace('<', '\\u003c')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HOT3D annotation review</title><style>
body{font:16px system-ui;margin:0;background:#11161e;color:#e8edf5}main{max-width:1400px;margin:auto;padding:28px}
h1{font-size:28px;margin-bottom:8px}header p{color:#afbed0;max-width:950px;line-height:1.5}
select,a,button{font:inherit;color:inherit}select{background:#273244;padding:8px;border-radius:6px}
a{color:#a1d8ff}#metrics{display:flex;gap:24px;flex-wrap:wrap;margin:24px 0}.metric strong{display:block;font-size:25px}
#clips{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:20px}article{background:#1c2430;padding:16px;border-radius:10px}
h2{font:14px ui-monospace;color:#afbed0}video{width:100%;max-height:430px;background:#050709}.active{min-height:48px;color:#9fe9ce}
ol{padding-left:22px}li{padding:8px 0;line-height:1.45}button{text-align:left;background:none;border:0;cursor:pointer;padding:0}
small{display:block;color:#aeb8c8}.error{color:#ffb7ac}.playing{background:#29394b}
</style><main><header><h1>HOT3D · dense annotation review</h1>
<p>__COUNT__ clips · __SECONDS__ seconds. Click a caption to seek; scrub each video to check the action and timing.
These runs use annotated UmeTrack hand poses and Qwen3-VL-8B. Format compliance and pose consistency are automatic checks;
semantic correctness still needs human review. Clips are independent examples, not one continuous episode.</p>
<label>Run <select id="variant"><option value="repaired">Updated · one repair attempt</option>
<option value="first_pass">Updated · first pass</option><option value="original">Original pipeline</option></select></label>
<p><a href="comparison.json">Metrics JSON</a> · <a href="repaired.jsonl">Updated captions</a> · <a href="manifest.json">Dataset provenance</a></p>
</header><section id="metrics"></section><section id="clips">__CARDS__</section></main>
<script type="application/json" id="data">__PAYLOAD__</script><script>
const data=JSON.parse(document.querySelector('#data').textContent), choose=document.querySelector('#variant');
for(const option of [...choose.options]) if(!(option.value in data.captions)) option.remove();
function render(){
 const key=choose.value, metrics=data.metrics[key], rows=data.captions[key];
 const panel=document.querySelector('#metrics');panel.replaceChildren();
 for(const [label,value] of [['Captions',metrics.n],['Format pass',Math.round(metrics.atomicity*100)+'%'],
 ['Unique text',Math.round(metrics.uniqueness*100)+'%'],['Outside duration band',metrics.out_of_band]]){
 const tile=document.createElement('div');tile.className='metric';const n=document.createElement('strong');n.textContent=value;
 const title=document.createElement('span');title.textContent=label;tile.append(n,title);panel.append(tile);}
 for(const card of document.querySelectorAll('article')){
 const captions=rows.filter(x=>x.segment===card.dataset.clip),list=card.querySelector('ol'),video=card.querySelector('video');list.replaceChildren();
 for(const c of captions){const li=document.createElement('li'),b=document.createElement('button'),detail=document.createElement('small');
 b.textContent=c.start_ts.toFixed(2)+'–'+c.end_ts.toFixed(2)+' s  '+c.text;b.onclick=()=>{video.currentTime=c.start_ts;video.play();};
 detail.textContent='Hand: '+c.hand+' · '+c.visibility+(c.uncertain?' · uncertain':'')+' · '+c.verb;
 li.append(b,detail);if(c.validation_errors?.length){const e=document.createElement('small');e.className='error';e.textContent=c.validation_errors.join('; ');li.append(e);}list.append(li);}
 const update=()=>{const i=captions.findIndex(c=>video.currentTime>=c.start_ts&&video.currentTime<c.end_ts);
 card.querySelector('.active').textContent=i<0?'No caption at this time':captions[i].text;
 [...list.children].forEach((li,k)=>li.classList.toggle('playing',i===k));};video.ontimeupdate=update;update();
 }
}
choose.onchange=render;render();
</script></html>'''
    page = page.replace('__COUNT__', str(len(definitions))).replace('__SECONDS__', f'{sum(c["t1"]-c["t0"] for c in definitions):.0f}')
    page = page.replace('__CARDS__', cards).replace('__PAYLOAD__', payload)
    (out/'index.html').write_text(page)
    print(out/'index.html')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--corpus', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    build(args.corpus, args.run, args.out)
