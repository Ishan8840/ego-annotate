"""Annotation showcase - single-page HTML build. Paths come from egoannot.config."""
import base64, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from egoannot import config

SEG = str(config.SEGMENTS_DIR)
OUT = str(config.artifact("reports", "ego-annotation-showcase.html"))
D=json.load(open(os.path.join(str(config.DATA), "showcase_data.json")))
# `_cmp` carries the pose-guided vs vision-only numbers, not a segment.
ids=[k for k in D if not k.startswith("_")]
V={i:"data:video/mp4;base64,"+base64.b64encode(open(f"{SEG}/{i}.mp4","rb").read()).decode() for i in ids}
O={}
for i in ids:
    o=json.load(open(f"{SEG}/{i}.json")); O[i]=dict(fps=o["fps"],L=o["L"],R=o["R"])

HEAD=r'''<title>Egocentric Annotation Showcase</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Public+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --bg:#0a0c0d; --surf:#14181a; --surf2:#1b2124; --line:#242c2f; --line2:#333e42;
  --ink:#f2f5f5; --ink2:#c3ced0; --mut:#8b9a9c;
  --acc:#ffb454; --ok:#4fb477; --no:#e0715c;
  --d:"Sora","Helvetica Neue",Arial,sans-serif;
  --b:"Public Sans","Helvetica Neue",Arial,sans-serif;
  --m:"IBM Plex Mono",ui-monospace,Menlo,monospace;
}
:root[data-theme="light"]{
  --bg:#f7f8f8; --surf:#ffffff; --surf2:#eef1f2; --line:#e0e5e6; --line2:#c9d2d4;
  --ink:#0c1012; --ink2:#384346; --mut:#6a7679;
  --acc:#a35d00; --ok:#1f7a4d; --no:#a8402c;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:var(--b);font-size:15px;
  line-height:1.6;-webkit-font-smoothing:antialiased}
.w{max-width:1160px;margin:0 auto;padding:0 28px}
h1,h2,h3,.ey,.num{font-family:var(--d)}
h1{font-size:clamp(2.4rem,6vw,4.4rem);line-height:.98;font-weight:700;
  letter-spacing:-.035em;margin:0;text-wrap:balance}
h2{font-size:clamp(1.4rem,2.6vw,1.95rem);font-weight:600;letter-spacing:-.02em;margin:0}
h3{font-size:1rem;font-weight:600;margin:0;letter-spacing:-.01em}
p{margin:0}
.ey{font-family:var(--m);font-size:.68rem;text-transform:uppercase;letter-spacing:.2em;
  color:var(--mut)}
code{font-family:var(--m);font-size:.86em;color:var(--acc)}

/* hero */
header{padding:82px 0 0}
.dek{font-size:clamp(1.05rem,1.7vw,1.3rem);color:var(--ink2);margin-top:26px;
  max-width:64ch;line-height:1.5}
.dek b{color:var(--ink);font-weight:600}
.nums{display:grid;grid-template-columns:repeat(auto-fit,minmax(136px,1fr));
  gap:1px;background:var(--line);border:1px solid var(--line);border-radius:10px;
  overflow:hidden;margin-top:44px}
.nums > div{background:var(--surf);padding:20px 22px 22px}
.nums .n{font-family:var(--d);font-weight:700;font-size:2.1rem;letter-spacing:-.035em;
  line-height:1;font-variant-numeric:tabular-nums}
.nums .n i{font-style:normal;font-size:.8rem;font-weight:500;color:var(--mut);
  letter-spacing:0;margin-left:3px}
.nums .k{font-family:var(--m);font-size:.63rem;text-transform:uppercase;
  letter-spacing:.15em;color:var(--mut);margin-top:9px}

/* domain chips */
.chips{display:flex;flex-wrap:wrap;gap:9px;margin:56px 0 22px}
.chip{background:var(--surf);border:1px solid var(--line);border-radius:9px;
  padding:12px 15px;cursor:pointer;text-align:left;min-width:154px;
  transition:border-color .16s,transform .16s}
.chip:hover{border-color:var(--line2);transform:translateY(-1px)}
.chip.on{border-color:var(--acc);background:var(--surf2)}
.chip .t{font-family:var(--d);font-weight:600;font-size:.87rem;letter-spacing:-.01em}
.chip .s{font-family:var(--m);font-size:.63rem;color:var(--mut);margin-top:5px}
.chip.on .s{color:var(--acc)}

/* stage */
.stage{display:grid;grid-template-columns:minmax(0,1.62fr) minmax(268px,1fr);gap:22px;
  align-items:start}
@media (max-width:940px){.stage{grid-template-columns:1fr}}
.screen{position:relative;background:#000;border-radius:12px;overflow:hidden;
  line-height:0;border:1px solid var(--line)}
.screen video{width:100%;display:block}
.screen canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.sub{position:absolute;left:0;right:0;bottom:0;padding:52px 22px 20px;
  background:linear-gradient(transparent,rgba(0,0,0,.88));line-height:1.35}
.sub .cap{font-family:var(--d);font-size:clamp(.98rem,1.5vw,1.24rem);font-weight:600;
  color:#fff;letter-spacing:-.015em;text-wrap:balance;display:block}
.sub .meta{font-family:var(--m);font-size:.65rem;color:#a9bcbf;margin-top:8px;display:block}
.sub .meta b{color:var(--acc);font-weight:500}
.ctl{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:14px}
button{font-family:var(--m);font-size:.7rem;letter-spacing:.03em;padding:8px 12px;
  border:1px solid var(--line2);background:var(--surf);color:var(--ink);
  border-radius:7px;cursor:pointer;transition:border-color .16s,color .16s}
button:hover{border-color:var(--acc);color:var(--acc)}
.tog{font-family:var(--m);font-size:.66rem;color:var(--mut);display:flex;
  align-items:center;gap:6px;cursor:pointer}
.track{position:relative;height:38px;margin-top:12px;background:var(--surf);
  border:1px solid var(--line);border-radius:8px;overflow:hidden;cursor:pointer}
.blk{position:absolute;top:4px;bottom:15px;background:var(--ok);opacity:.32;
  border-right:1px solid var(--bg);border-radius:2px;transition:opacity .12s}
.blk.no{background:var(--no)}
.blk.on{opacity:.95}
.evt{position:absolute;bottom:4px;width:2px;height:8px;background:var(--acc);opacity:.8}
.tlab{font-family:var(--m);font-size:.6rem;letter-spacing:.12em;text-transform:uppercase;
  color:var(--mut);margin:12px 0 5px}
.track.vt .blk{background:var(--acc)}
.track.vt .blk.no{background:var(--no)}
.arms{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:8px}
@media (max-width:820px){.arms{grid-template-columns:1fr}}
.vs{width:100%;border-collapse:collapse;font-size:.86rem;margin-top:18px}
.vs th,.vs td{padding:9px 12px;border-bottom:1px solid var(--line);text-align:right}
.vs th:first-child,.vs td:first-child{text-align:left}
.vs thead th{font-family:var(--m);font-size:.62rem;text-transform:uppercase;
  letter-spacing:.14em;color:var(--mut);font-weight:500}
.vs td.w{color:var(--ok);font-weight:600}
.vs td.l{color:var(--no)}
.vs tr.hi td{background:var(--surf2)}
.vs .u{font-family:var(--m);font-size:.78rem}
.note{background:var(--surf);border:1px solid var(--line);border-left:3px solid var(--acc);
  border-radius:0 9px 9px 0;padding:15px 18px;margin-top:18px}
.note b{color:var(--acc)}
.head{position:absolute;top:0;bottom:0;width:2px;background:var(--ink);z-index:3}
.legend{display:flex;flex-wrap:wrap;gap:15px;margin-top:11px}
.legend span{font-family:var(--m);font-size:.63rem;color:var(--mut);display:flex;
  align-items:center;gap:6px}
.sw{width:9px;height:9px;border-radius:2px}

/* caption list */
.panel{background:var(--surf);border:1px solid var(--line);border-radius:11px;overflow:hidden}
.panel .ph{padding:13px 16px;border-bottom:1px solid var(--line)}
.panel .ph .ey{font-size:.62rem}
.panel .ph .h{font-family:var(--d);font-weight:600;font-size.95rem;margin-top:4px}
.rows{max-height:466px;overflow-y:auto}
.row{padding:12px 16px;border-bottom:1px solid var(--line);cursor:pointer;
  transition:background .14s}
.row:last-child{border-bottom:0}
.row:hover{background:var(--surf2)}
.row.on{background:var(--surf2);box-shadow:inset 2px 0 0 var(--acc)}
.row .tc{font-family:var(--m);font-size:.62rem;color:var(--mut);display:flex;
  justify-content:space-between;gap:8px}
.row .tx{font-size:.85rem;color:var(--ink2);margin-top:5px;line-height:1.45}
.row.on .tx{color:var(--ink)}
.row .er{font-family:var(--m);font-size:.6rem;color:var(--no);margin-top:5px;display:block}
.dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--ok)}
.dot.no{background:var(--no)}

/* findings */
.finds{display:grid;grid-template-columns:repeat(auto-fit,minmax(268px,1fr));gap:14px;
  margin-top:22px}
.find{background:var(--surf);border:1px solid var(--line);border-radius:11px;padding:22px 24px}
.find .big{font-family:var(--d);font-weight:700;font-size:1.95rem;letter-spacing:-.03em;
  line-height:1;color:var(--acc);font-variant-numeric:tabular-nums}
.find h3{margin:13px 0 7px}
.find p{font-size:.88rem;color:var(--ink2);line-height:1.55}
section{margin-top:76px}
.shead{margin-bottom:24px}
.shead p{color:var(--mut);margin-top:10px;max-width:66ch;font-size:.95rem}
table{width:100%;border-collapse:collapse;font-size:.85rem;margin-top:4px}
th,td{padding:11px 13px;text-align:left;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{padding-left:0}
th{font-family:var(--m);font-size:.62rem;text-transform:uppercase;letter-spacing:.13em;
  color:var(--mut);border-bottom:1px solid var(--line2)}
td{color:var(--ink2)}
td.nm{font-family:var(--d);font-weight:600;color:var(--ink)}
td.n{text-align:right;font-family:var(--m);font-variant-numeric:tabular-nums;white-space:nowrap}
th.n{text-align:right}
.tw{overflow-x:auto}
footer{margin-top:86px;padding:26px 0 74px;border-top:1px solid var(--line);
  font-family:var(--m);font-size:.68rem;color:var(--mut);line-height:1.85}
:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
'''

BODY=r'''<div class="w">
<header>
  <div class="ey">LightwheelAI · EgoStandard · pose-grounded annotation</div>
  <h1>Five everyday tasks,<br>annotated frame-accurately.</h1>
  <p class="dek">Spans cut from <b>wrist motion</b>, captioned by an <b>8B vision model
  running locally</b>. Hands, timing, <b>rotation direction</b> and <b>which fingers are
  pinching</b> come from <b>3D pose measurement</b> — never guessed. Every label is
  machine-checked against a per-domain spec, and the ones that fail are shown failing.</p>
  <div class="nums">
    <div><div class="n">69</div><div class="k">labels</div></div>
    <div><div class="n">27.5<i>/min</i></div><div class="k">density</div></div>
    <div><div class="n">77<i>%</i></div><div class="k">unique</div></div>
    <div><div class="n">87<i>%</i></div><div class="k">pass spec</div></div>
    <div><div class="n">1.3<i>× realtime</i></div><div class="k">one RTX 4090</div></div>
    <div><div class="n">5</div><div class="k">domains</div></div>
  </div>
</header>

<section id="explore">
  <div class="shead">
    <h2>Explore the annotations</h2>
    <p>Pick a task. The caption on screen is the live label for that instant; the bar
    below is every label in the clip, green where it passes the atomicity spec and red
    where it does not. Amber ticks are pose-derived contact events.</p>
  </div>
  <div class="chips" id="chips"></div>
  <div class="stage">
    <div>
      <div class="screen">
        <video id="v" playsinline preload="metadata"></video>
        <canvas id="c" width="512" height="384"></canvas>
        <div class="sub"><span class="cap" id="cap">—</span><span class="meta" id="meta">—</span></div>
      </div>
      <div class="ctl">
        <button id="play">▶&nbsp; Play</button>
        <button id="prev">⟵ prev label</button>
        <button id="next">next label ⟶</button>
        <button id="rate">1×</button>
        <label class="tog"><input type="checkbox" id="skel"> hand pose</label>
        <label class="tog"><input type="checkbox" id="ev" checked> contact events</label>
      </div>
      <div class="tlab">pose-guided spans &mdash; boundaries measured</div>
      <div class="track" id="track"><div class="head" id="head"></div></div>
      <div class="tlab">vision-only spans &mdash; the same model choosing its own boundaries</div>
      <div class="track vt" id="trackv"><div class="head" id="headv"></div></div>
      <div class="legend">
        <span><i class="sw" style="background:var(--ok)"></i> passes spec</span>
        <span><i class="sw" style="background:var(--no)"></i> fails spec</span>
        <span><i class="sw" style="background:var(--acc)"></i> contact event / vision span</span>
        <span>click either bar to seek</span>
      </div>
    </div>
    <div class="panel">
      <div class="ph"><div class="ey">every label · <span id="ppack">—</span></div><div class="h" id="ptitle">—</div></div>
      <div class="rows" id="rows"></div>
    </div>
  </div>
</section>

<section id="upg">
  <div class="shead">
    <h2>Three fixes from watching it back</h2>
    <p>Reviewing the previous pass surfaced three faults. Each had a measurable cause,
    and none of them was the model's fault.</p>
  </div>
  <div class="finds">
    <div class="find">
      <div class="big">89 → 18%</div>
      <h3>Fingers were named far too often</h3>
      <p>A fixed 35 mm pinch threshold sat <i>above</i> the median aperture in fine-motor
      work — 26 mm for the lens case, 32 mm for the toothbrush — so "pinch" fired on
      67–81% of frames and described nothing. Finger state is now reported only when it is
      distinctive for that episode.</p>
    </div>
    <div class="find">
      <div class="big">net°</div>
      <h3>Unscrewing was invisible</h3>
      <p>Rotation was thresholded on instantaneous rate, but frame-level median is only
      0.1–0.3 rad/s: turning a cap is <i>accumulated</i> angle, not speed. It is now net
      degrees about the hand's own axis with a coherence test to reject wobble — so
      "rotate the bottle cap counter-clockwise" appears where it happens.</p>
    </div>
    <div class="find">
      <div class="big">2.00 s</div>
      <h3>Timing was off because the cue was wrong</h3>
      <p>Boundaries came from wrist velocity alone. In fine-motor work the wrist barely
      moves while the fingers do everything, so troughs landed mid-action. The signal is
      now the combined maximum of wrist speed, aperture rate and twist rate — every span
      is 1.7–2.6 s, inside the spec band.</p>
    </div>
  </div>
</section>

<section id="versus">
  <div class="shead">
    <h2>Against a vision-only baseline</h2>
    <p>The premise of this pipeline is that pose should supply span boundaries and
    handedness, and the model only the language. That was an assumption, so here is the
    control arm: the <b>same 8B model, same rules, same verb list, same footage</b>, shown
    frames and asked to segment the clip itself and name the acting hand. It was given a
    <b>larger frame budget</b> than the pose-guided arm &mdash; 384 frames against 276 &mdash;
    so it is not handicapped.</p>
  </div>
  <table class="vs">
    <thead><tr><th>metric</th><th>pose-guided</th><th>vision-only</th><th>&nbsp;</th></tr></thead>
    <tbody id="vsbody"></tbody>
  </table>
  <div class="note" id="vsnote"></div>
  <div class="arms" id="vsarms"></div>
</section>

<section id="tradeoff">
  <div class="shead">
    <h2>Two things worth knowing about the design</h2>
    <p>Both were tested rather than assumed, and both came out against the intuitive
    choice.</p>
  </div>
  <div class="tw"><table>
    <thead><tr><th>Question</th><th>Intuition</th><th>Measured</th></tr></thead>
    <tbody>
      <tr><td class="nm">Let the model use any verb?</td>
        <td>More expressive</td>
        <td><b>Worse on every axis.</b> Pass 76%, unique 77.4%, and finger/direction
        mentions <i>fell</i> from 61% to 18%. Removing the constraint removed discipline;
        the sentence was already free, only the opening verb was fixed.</td></tr>
      <tr><td class="nm">Put domain vocabulary in the prompt?</td>
        <td>Fewer rejections</td>
        <td><b>Costs 10 points of uniqueness</b> for no gain. Naming the objects makes the
        model reach for the listed words instead of describing what it sees. Vocabulary
        belongs in the linter, which is where the original bug was.</td></tr>
      <tr><td class="nm">Can the model contradict a measurement?</td>
        <td>—</td>
        <td>Yes, once in 74 — it wrote "clockwise" where pose measured
        counter-clockwise. That is now rule A9, checked automatically.</td></tr>
    </tbody>
  </table></div>
  <p style="color:var(--mut);margin-top:16px;font-size:.9rem;max-width:70ch">Pass rate and
  uniqueness trade against each other in every variant tested, and at 74 labels the
  run-to-run spread is roughly ±8 points — comparable to the differences above. These are
  directional, not settled.</p>
</section>

<section id="found">
  <div class="shead">
    <h2>What five domains revealed</h2>
    <p>The same pipeline that was tuned on retail shelf-stocking, pointed at a bar, a
    kitchen, a bedroom and two dining rooms. Three things only showed up here.</p>
  </div>
  <div class="finds">
    <div class="find">
      <div class="big">77%</div>
      <h3>Task diversity beats every tuning knob</h3>
      <p>Uniqueness was 70% across four shelf-stocking segments. Five different tasks give
      77% with the same model and a Heaps exponent of 0.910 against 0.959 for
      Lightwheel's own human labels — projecting 49% at 10k labels, where shelf-stocking
      projected 38%. Diversity moved this metric further than every prompt and
      frame-count variant combined.</p>
    </div>
    <div class="find">
      <div class="big">6 packs</div>
      <h3>Vocabulary is per domain</h3>
      <p>Each domain now carries its own verb and object vocabulary, routed automatically
      from the episode's scene and task name. The 15 core manipulation verbs are shared;
      kitchen adds <code>scrub</code>, <code>rinse</code>, <code>wash</code>; assembly
      adds <code>insert</code>, <code>twist</code>, <code>snap</code>. Putting those lists
      in the <i>prompt</i> cost 10 points of uniqueness, so they live in the linter.</p>
    </div>
    <div class="find">
      <div class="big">0 events</div>
      <h3>Fine motor work is invisible to pose</h3>
      <p>Assembling a toothbrush produced zero contact events and read as 100% idle. The
      thresholds come from pooled shelf-stocking percentiles, where hands sweep across
      shelves. Small in-hand manipulation never crosses them.</p>
    </div>
  </div>
</section>

<section id="per">
  <div class="shead"><h2>Per task</h2></div>
  <div class="tw"><table>
    <thead><tr><th>Task</th><th>Scene</th><th>Kind</th><th class="n">Length</th>
      <th class="n">Labels</th><th class="n">Per min</th><th class="n">Pass spec</th>
      <th class="n">Contact events</th></tr></thead>
    <tbody id="tbody"></tbody>
  </table></div>
</section>
</div>

<footer><div class="w">
  Spans from smoothed wrist-velocity troughs · captions from Qwen3-VL-8B-Instruct at bf16
  on one RTX 4090, 10 spans per request, 4 frames per span, previous four labels as rolling
  context · <code>hand</code>, <code>start_ts</code> and <code>end_ts</code> from the shipped
  21-joint hand pose, never from the model · every label checked by an eight-rule linter.<br>
  62 labels over 2.5 minutes. Small sample, shown in full including its failures.
</div></footer>
'''

JS=r'''<script>
const D=__D__, V=__V__, O=__O__;
const CMP=D._cmp; delete D._cmp;
const ids=Object.keys(D);
const BN=[[0,1],[0,5],[0,9],[0,13],[0,17],[5,9],[9,13],[13,17],[1,2],[2,3],[3,4],
 [5,6],[6,7],[7,8],[9,10],[10,11],[11,12],[13,14],[14,15],[15,16],[17,18],[18,19],[19,20]];
let cur=ids[0];
const $=i=>document.getElementById(i);
const v=$("v"), c=$("c"), ctx=c.getContext("2d");
const dur=()=>D[cur].dur, pc=t=>100*t/dur();

function chips(){
  const W=$("chips"); W.innerHTML="";
  ids.forEach(id=>{
    const d=D[id], b=document.createElement("button");
    b.className="chip"+(id===cur?" on":"");
    b.innerHTML='<div class="t">'+d.task+'</div><div class="s">'+d.scene+' · '+
      d.caps.length+' labels · '+d.dur.toFixed(0)+'s</div>';
    b.onclick=()=>load(id); W.appendChild(b);
  });
}
function table(){
  const T=$("tbody"); T.innerHTML="";
  ids.forEach(id=>{
    const d=D[id], ok=d.caps.filter(x=>x.ok).length;
    const r=document.createElement("tr");
    r.innerHTML='<td class="nm">'+d.task+'</td><td>'+d.scene+'</td><td>'+d.cls+
      '</td><td class="n">'+d.dur.toFixed(1)+'s</td><td class="n">'+d.caps.length+
      '</td><td class="n">'+(d.caps.length/(d.dur/60)).toFixed(1)+
      '</td><td class="n">'+ok+'/'+d.caps.length+'</td><td class="n">'+d.ev.length+'</td>';
    T.appendChild(r);
  });
}
function fill(el,caps){
  [...el.querySelectorAll(".blk,.evt")].forEach(n=>n.remove());
  caps.forEach(x=>{
    const e=document.createElement("div");
    e.className="blk"+(x.ok?"":" no");
    e.style.left=pc(x.a)+"%"; e.style.width=Math.max(.5,pc(x.b-x.a))+"%";
    e.title=x.a.toFixed(1)+"-"+x.b.toFixed(1)+"s  "+x.t;
    el.appendChild(e);
  });
}
function track(){
  const T=$("track");
  fill(T, D[cur].caps);
  fill($("trackv"), D[cur].capsv||[]);
  if($("ev").checked) D[cur].ev.forEach(g=>{
    const e=document.createElement("div"); e.className="evt";
    e.style.left=pc(g.t)+"%"; e.title=g.ty+" "+g.h; T.appendChild(e);
  });
}
function vstable(){
  const P=CMP.pose, X=CMP.vision, T=$("vsbody");
  const pct=x=>(100*x).toFixed(0)+"%";
  const R=[
    ["labels produced", P.n, X.n, P.n>X.n?1:-1],
    ["labels / min of video", P.density.toFixed(1), X.density.toFixed(1), 1],
    ["labels / min, first 16s of a clip", P.head_density.toFixed(1), X.head_density.toFixed(1), 0, 1],
    ["labels / min, after 16s", P.tail_density.toFixed(1), X.tail_density.toFixed(1), 1, 1],
    ["median span", P.median_dur.toFixed(2)+"s", X.median_dur.toFixed(2)+"s", 0],
    ["longest span (p90)", P.p90_dur.toFixed(2)+"s", X.p90_dur.toFixed(2)+"s", 0],
    ["passes the atomicity spec", pct(P.atomicity), pct(X.atomicity), 1, 1],
    ["unique captions", pct(P.uniqueness), pct(X.uniqueness), 1],
    ["spans inside the 1.3-4.0s band", pct(P.in_band), pct(X.in_band), 1, 1],
    ["overlapping labels", P.overlaps, X.overlaps, 0],
    ["hand matches measured pose", "supplied", pct(X.hand_agree), 1, 1],
    ["verb matches measured aperture", pct(P.grounding), pct(X.grounding), -1, 1],
  ];
  T.innerHTML="";
  R.forEach(([k,a,b,dir,hi])=>{
    const r=document.createElement("tr");
    if(hi) r.className="hi";
    const ca=dir>0?"w":(dir<0?"l":""), cb=dir>0?"l":(dir<0?"w":"");
    r.innerHTML='<td>'+k+'</td><td class="u '+ca+'">'+a+'</td>'+
      '<td class="u '+cb+'">'+b+'</td><td class="u" style="color:var(--mut)">'+
      (dir>0?"pose":(dir<0?"vision":"\u2014"))+'</td>';
    T.appendChild(r);
  });
  $("vsnote").innerHTML=
    "<b>Two findings, pulling opposite ways.</b> The vision-only arm cannot hold a "+
    "duration band it was told to hold &mdash; "+pct(1-X.in_band)+" of its spans fall "+
    "outside 1.3&ndash;4.0s, which is every one of its "+(X.fails.A1||0)+" spec failures "+
    "under that rule &mdash; and it disagrees with the measured acting hand on "+
    pct(1-X.hand_agree)+" of labels. Worse, it front-loads: "+
    X.head_density.toFixed(0)+" labels/min in the first 16 seconds of a clip, then "+
    X.tail_density.toFixed(1)+" after, emitting a single 16-second \"action\" per window. "+
    "The pose-guided arm is flat at "+P.head_density.toFixed(0)+" and "+
    P.tail_density.toFixed(0)+".<br><br>But the vision-only arm's <b>verbs agree better "+
    "with the measured finger aperture</b> ("+pct(X.grounding)+" vs "+pct(P.grounding)+
    ", n&asymp;28 each). Choosing its own boundaries, it cuts where the action it wants "+
    "to describe actually happens; the pose-guided arm has to caption whatever the "+
    "measured boundary contains. That is a real cost of fixed boundaries, and it is not "+
    "what we expected to find.";
}
function arms(){
  const A=$("vsarms"), d=D[cur];
  const col=(title,caps,sub)=>{
    const box=document.createElement("div"); box.className="panel";
    let h='<div class="ph"><div class="ey">'+sub+'</div><div class="h">'+title+'</div></div><div class="rows">';
    (caps||[]).forEach(x=>{
      h+='<div class="row"><div class="tc"><span>'+x.a.toFixed(1)+'\u2013'+x.b.toFixed(1)+
        's</span><span><i class="dot'+(x.ok?'':' no')+'"></i> '+x.h.toLowerCase()+
        (x.ph&&x.ph!==x.h?' <span style="color:var(--no)">(measured '+x.ph.toLowerCase()+')</span>':'')+
        ' \u00b7 '+x.v+'</span></div><div class="tx">'+x.t+'</div>'+
        (x.ok?'':'<span class="er">'+x.err.join(" \u00b7 ")+'</span>')+'</div>';
    });
    box.innerHTML=h+'</div>';
    return box;
  };
  A.innerHTML="";
  A.appendChild(col(d.task+" \u2014 pose-guided", d.caps, (d.caps||[]).length+" labels"));
  A.appendChild(col(d.task+" \u2014 vision-only", d.capsv, (d.capsv||[]).length+" labels"));
}
function rows(){
  const R=$("rows"); R.innerHTML="";
  D[cur].caps.forEach((x,i)=>{
    const r=document.createElement("div");
    r.className="row"; r.dataset.i=i;
    r.innerHTML='<div class="tc"><span>'+x.a.toFixed(1)+'–'+x.b.toFixed(1)+'s</span>'+
      '<span><i class="dot'+(x.ok?'':' no')+'"></i> '+x.h.toLowerCase()+' · '+x.v+'</span></div>'+
      '<div class="tx">'+x.t+'</div>'+
      ((x.rot||x.fng)?'<div class="tc" style="margin-top:5px;color:var(--acc)">measured: '+
        [x.rot,x.fng].filter(Boolean).join(' · ')+'</div>':'')+
      (x.ok?'':'<span class="er">'+x.err.join(" · ")+'</span>');
    r.onclick=()=>{ v.currentTime=Math.max(0,Math.min(dur()-.05,x.a+.04)); };
    R.appendChild(r);
  });
}
let last=-2;
function draw(){
  const t=v.currentTime, d=D[cur];
  const i=d.caps.findIndex(x=>t>=x.a&&t<x.b);
  if(i!==last){
    last=i; const x=i>=0?d.caps[i]:null;
    $("cap").textContent=x?x.t:"—";
    $("cap").style.color=x&&!x.ok?"#ffc4b4":"#fff";
    $("meta").innerHTML=x?("<b>"+x.v+"</b> · "+x.n+" · hand "+x.h.toLowerCase()+" · "+
      x.vis.toLowerCase()+(x.rot?" · <b>"+x.rot+"</b>":"")+(x.fng?" · <b>"+x.fng+"</b>":"")+
      " · "+x.a.toFixed(1)+"–"+x.b.toFixed(1)+"s"+
      (x.ok?"":" · <b>fails spec</b>")):"—";
    [...$("rows").children].forEach((n,k)=>n.classList.toggle("on",k===i));
    [...$("track").querySelectorAll(".blk")].forEach((n,k)=>n.classList.toggle("on",k===i));
  }
  ctx.clearRect(0,0,c.width,c.height);
  if($("skel").checked){
    const o=O[cur], fi=Math.max(0,Math.min((o.L||[]).length-1,Math.round(t*o.fps)));
    [[o.L,"#7fe3d8"],[o.R,"#ffb454"]].forEach(([arr,col])=>{
      const f=arr&&arr[fi]; if(!f) return;
      const P=[]; for(let j=0;j<21;j++){const X=f[1+2*j],Y=f[2+2*j];P.push(X<=-900?null:[X,Y]);}
      ctx.strokeStyle=col; ctx.lineWidth=1.7; ctx.beginPath();
      BN.forEach(([a,b])=>{if(P[a]&&P[b]){ctx.moveTo(P[a][0],P[a][1]);ctx.lineTo(P[b][0],P[b][1]);}});
      ctx.stroke();
      ctx.fillStyle=col; P.forEach(p=>{if(p){ctx.beginPath();ctx.arc(p[0],p[1],1.9,0,6.3);ctx.fill();}});
    });
  }
  $("head").style.left=pc(t)+"%"; $("headv").style.left=pc(t)+"%";
  requestAnimationFrame(draw);
}
function load(id){
  cur=id; last=-2; v.src=V[id]; v.load();
  $("ptitle").textContent=D[id].task;
  $("ppack").textContent=D[id].pack.replace(/_/g," ")+" vocabulary";
  chips(); track(); rows(); arms();
}
$("play").onclick=()=>{ if(v.paused){v.play();$("play").innerHTML="❙❙&nbsp; Pause";}
                        else{v.pause();$("play").innerHTML="▶&nbsp; Play";} };
v.addEventListener("pause",()=>$("play").innerHTML="▶&nbsp; Play");
v.addEventListener("play",()=>$("play").innerHTML="❙❙&nbsp; Pause");
const RT=[1,.5,2]; let ri=0;
$("rate").onclick=()=>{ri=(ri+1)%RT.length;v.playbackRate=RT[ri];$("rate").textContent=RT[ri]+"×";};
function jump(dir){
  const t=v.currentTime, a=D[cur].caps.map(x=>x.a).filter(x=>dir>0?x>t+.15:x<t-.15);
  if(a.length) v.currentTime=(dir>0?Math.min(...a):Math.max(...a))+.04;
}
$("next").onclick=()=>jump(1); $("prev").onclick=()=>jump(-1);
["track","trackv"].forEach(id=>$(id).onclick=e=>{
  const r=$(id).getBoundingClientRect();
  v.currentTime=Math.max(0,Math.min(dur()-.05,dur()*(e.clientX-r.left)/r.width));});
$("ev").onchange=track; $("skel").onchange=()=>{};
document.addEventListener("keydown",e=>{
  if(e.target.tagName==="INPUT") return;
  if(e.code==="Space"){e.preventDefault();$("play").click();}
  else if(e.key==="ArrowRight"){e.preventDefault();jump(1);}
  else if(e.key==="ArrowLeft"){e.preventDefault();jump(-1);}
});
table(); vstable(); load(ids[0]); requestAnimationFrame(draw);
</script>
'''
JS=JS.replace("__D__",json.dumps(D,separators=(",",":")))
JS=JS.replace("__V__",json.dumps(V,separators=(",",":")))
JS=JS.replace("__O__",json.dumps(O,separators=(",",":")))
open(OUT,"w").write(HEAD+BODY+JS)
print("wrote %s  %.2f MB"%(OUT,os.path.getsize(OUT)/1e6))
