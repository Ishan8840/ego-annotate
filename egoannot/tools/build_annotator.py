"""Contact gold annotator - single-page HTML build. Paths come from egoannot.config."""
import base64, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from egoannot import config

SEG = str(config.SEGMENTS_DIR)
OUT = str(config.artifact("reports", "contact-annotator.html"))
man = json.load(open(os.path.join(SEG, "manifest.json")))

meta, vids = {}, {}
for m in man:
    sid = m["id"]
    ov = json.load(open(os.path.join(SEG, sid + ".json")))
    meta[sid] = ov
    vids[sid] = base64.b64encode(open(os.path.join(SEG, sid + ".mp4"), "rb").read()).decode()

HEAD = r'''<title>Contact Gold Annotator</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root{
  --bg:#eef2f3; --surface:#ffffff; --surface-2:#f6f9f9;
  --ink:#121a1e; --ink-2:#33454c; --muted:#5b6f77;
  --line:#d3dee1; --line-strong:#b6c7cc;
  --teal:#0e6e7c; --steel:#55919e; --alert:#9d3a2b; --good:#1f6b4a; --amber:#8a6410;
  --shadow:0 1px 2px rgba(18,26,30,.06),0 8px 24px -12px rgba(18,26,30,.16);
  --d:"Archivo","Helvetica Neue",Arial,sans-serif;
  --m:"IBM Plex Mono",ui-monospace,Menlo,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0c1215; --surface:#131c20; --surface-2:#18242a;
  --ink:#e8eef0; --ink-2:#c2d0d5; --muted:#93a6ad;
  --line:#24343a; --line-strong:#38505a;
  --teal:#58c3d4; --steel:#4a7f8b; --alert:#e08a72; --good:#63c396; --amber:#d9ab4f;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -12px rgba(0,0,0,.6);
}}
:root[data-theme="dark"]{
  --bg:#0c1215; --surface:#131c20; --surface-2:#18242a;
  --ink:#e8eef0; --ink-2:#c2d0d5; --muted:#93a6ad;
  --line:#24343a; --line-strong:#38505a;
  --teal:#58c3d4; --steel:#4a7f8b; --alert:#e08a72; --good:#63c396; --amber:#d9ab4f;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -12px rgba(0,0,0,.6);
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:var(--d);font-size:14px;line-height:1.5}
.wrap{max-width:1240px;margin:0 auto;padding:22px 20px 70px}
h1{font-size:1.5rem;font-weight:700;letter-spacing:-.02em;margin:0}
h2{font-size:.98rem;font-weight:600;margin:0}
p{margin:0}
code{font-family:var(--m);font-size:.85em;background:var(--surface-2);
  border:1px solid var(--line);border-radius:3px;padding:.05em .3em}
.eyebrow{font-family:var(--m);font-size:.66rem;text-transform:uppercase;
  letter-spacing:.15em;color:var(--muted)}
header{border-bottom:1px solid var(--line);padding-bottom:16px;margin-bottom:18px}
.sub{color:var(--ink-2);margin-top:8px;max-width:88ch;font-size:.92rem}
.grid{display:grid;grid-template-columns:214px minmax(0,1fr) 306px;gap:16px;align-items:start}
@media (max-width:1100px){.grid{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--line);border-radius:6px;
  box-shadow:var(--shadow);overflow:hidden}
.card > .hd{padding:9px 12px;border-bottom:1px solid var(--line);background:var(--surface-2);
  display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.card > .hd .t{font-family:var(--m);font-size:.66rem;text-transform:uppercase;
  letter-spacing:.12em;color:var(--muted)}
.seglist{max-height:none}
.segrow{padding:8px 12px;border-bottom:1px solid var(--line);cursor:pointer;display:grid;
  grid-template-columns:1fr auto;gap:6px;align-items:center}
.segrow:last-child{border-bottom:0}
.segrow:hover{background:var(--surface-2)}
.segrow.on{background:var(--surface-2);box-shadow:inset 3px 0 0 var(--teal)}
.segrow .nm{font-family:var(--m);font-size:.73rem;font-weight:600}
.segrow .cl{font-family:var(--m);font-size:.62rem;color:var(--muted)}
.badge{font-family:var(--m);font-size:.62rem;font-weight:600;padding:2px 5px;border-radius:3px;
  background:var(--line);color:var(--ink-2);white-space:nowrap}
.badge.done{background:var(--good);color:#fff}
.vwrap{position:relative;background:#000;line-height:0}
.vwrap video{width:100%;display:block}
.vwrap canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;padding:10px 12px;
  border-bottom:1px solid var(--line)}
button{font-family:var(--m);font-size:.72rem;font-weight:600;padding:6px 9px;
  border:1px solid var(--line-strong);background:var(--surface);color:var(--ink);
  border-radius:4px;cursor:pointer}
button:hover{border-color:var(--teal);color:var(--teal)}
button.pri{background:var(--teal);border-color:var(--teal);color:#fff}
button.pri:hover{opacity:.88;color:#fff}
button.warn{border-color:var(--alert);color:var(--alert)}
button:disabled{opacity:.45;cursor:not-allowed}
.markrow{display:grid;grid-template-columns:1fr 1fr;gap:6px;padding:10px 12px;
  border-bottom:1px solid var(--line)}
.markrow button{padding:9px 6px;font-size:.74rem}
.mk-c{border-color:var(--good);color:var(--good)}
.mk-r{border-color:var(--alert);color:var(--alert)}
.scrub{padding:10px 12px}
.scrub input[type=range]{width:100%}
.tl{position:relative;height:26px;background:var(--surface-2);border:1px solid var(--line);
  border-radius:4px;margin-top:7px;cursor:pointer}
.tl .ev{position:absolute;top:2px;width:2px;height:9px}
.tl .ev.c{background:var(--good)} .tl .ev.r{background:var(--alert)}
.tl .ev.R{top:13px}
.tl .cd{position:absolute;top:2px;bottom:2px;width:1px;background:var(--amber);opacity:.85}
.tl .ph{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--ink);z-index:5}
.key{font-family:var(--m);font-size:.64rem;color:var(--muted);padding:0 12px 11px;line-height:1.7}
.key b{color:var(--ink-2)}
.evlist{max-height:340px;overflow-y:auto}
.evrow{padding:7px 10px;border-bottom:1px solid var(--line);display:grid;
  grid-template-columns:auto 1fr auto;gap:8px;align-items:center;cursor:pointer;font-size:.72rem}
.evrow:hover{background:var(--surface-2)}
.evrow.cur{background:var(--surface-2);box-shadow:inset 3px 0 0 var(--ink)}
.evrow .tm{font-family:var(--m);font-weight:600;font-variant-numeric:tabular-nums}
.evrow .tg{font-family:var(--m);font-size:.63rem;color:var(--muted);line-height:1.5}
.pill{display:inline-block;font-family:var(--m);font-size:.6rem;font-weight:600;
  padding:1px 4px;border-radius:3px;margin-right:3px}
.pill.c{background:var(--good);color:#fff} .pill.r{background:var(--alert);color:#fff}
.pill.o{background:var(--steel);color:#fff} .pill.g{background:var(--amber);color:#fff}
select{font-family:var(--m);font-size:.66rem;padding:3px 4px;border:1px solid var(--line-strong);
  background:var(--surface);color:var(--ink);border-radius:3px}
.chk{font-family:var(--m);font-size:.64rem;color:var(--muted);display:flex;align-items:center;gap:4px}
.stat{padding:10px 12px;font-family:var(--m);font-size:.68rem;color:var(--ink-2);line-height:1.75}
.stat b{color:var(--ink)}
.warnbox{margin:0 12px 12px;padding:9px 11px;border-left:3px solid var(--amber);
  background:var(--surface-2);font-size:.74rem;color:var(--ink-2);border-radius:0 4px 4px 0}
.hidden{display:none!important}
:focus-visible{outline:2px solid var(--teal);outline-offset:2px}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
'''

BODY = r'''<div class="wrap">
<header>
  <div class="eyebrow">Phase 5 · gold set · LightwheelAI / EgoStandard</div>
  <h1>Contact Gold Annotator</h1>
  <p class="sub">Mark <b>every</b> contact and release in each segment — including ones no
  detector proposed. Exhaustive annotation is what makes recall and timing offset
  measurable; judging a detector's candidates can only ever give precision. The
  detector's proposals stay hidden until you finish a segment, so they cannot anchor you.</p>
</header>
<div class="grid">
  <div class="card">
    <div class="hd"><span class="t">segments</span><span class="t" id="progall">—</span></div>
    <div class="seglist" id="seglist"></div>
    <div class="stat" id="totals">—</div>
  </div>

  <div class="card">
    <div class="hd"><span class="t" id="segtitle">—</span><span class="t" id="segmeta">—</span></div>
    <div class="vwrap">
      <video id="v" playsinline preload="auto"></video>
      <canvas id="c" width="512" height="384"></canvas>
    </div>
    <div class="markrow">
      <button class="mk-c" data-mark="L,contact">Q · contact LEFT</button>
      <button class="mk-c" data-mark="R,contact">O · contact RIGHT</button>
      <button class="mk-r" data-mark="L,release">W · release LEFT</button>
      <button class="mk-r" data-mark="R,release">P · release RIGHT</button>
    </div>
    <div class="bar">
      <button id="play">▶ space</button>
      <button id="b10">⟵ 10f</button>
      <button id="b1">⟵ 1f</button>
      <button id="f1">1f ⟶</button>
      <button id="f10">10f ⟶</button>
      <button id="rate">1×</button>
      <label class="chk"><input type="checkbox" id="skel" checked> skeleton</label>
      <button id="undo">⌫ undo</button>
    </div>
    <div class="scrub">
      <input type="range" id="sc" min="0" max="1000" value="0">
      <div class="tl" id="tl"><div class="ph" id="ph"></div></div>
      <div class="key" style="padding:8px 0 0">
        <b>Q</b>/<b>W</b> left contact/release · <b>O</b>/<b>P</b> right contact/release ·
        <b>←</b>/<b>→</b> step frame (shift = 10) · <b>space</b> play · <b>Z</b> undo<br>
        Mark the frame where the fingers first touch (contact) or first separate (release).
        Step frame-by-frame at the moment — that precision is what the timing-offset
        measurement needs.
      </div>
    </div>
  </div>

  <div>
    <div class="card">
      <div class="hd"><span class="t">marked events</span><span class="t" id="evcount">0</span></div>
      <div class="evlist" id="evlist"></div>
      <div class="bar" style="border-top:1px solid var(--line);border-bottom:0">
        <button class="pri" id="finish">✓ finish segment</button>
        <button id="reveal" disabled>reveal detector</button>
      </div>
      <div class="warnbox" id="revealnote">Detector proposals are hidden. Finish the segment
        first, then reveal to compare — marking after seeing them biases the gold set.</div>
      <div class="stat hidden" id="cmp"></div>
    </div>
    <div class="card" style="margin-top:14px">
      <div class="hd"><span class="t">storage</span></div>
      <div class="stat" id="dbstat">connecting…</div>
      <div class="bar" style="border-bottom:0"><button id="copy">copy all JSON</button></div>
    </div>
  </div>
</div>
</div>
'''

JS = r'''<script>
const META=__META__, VID=__VID__;
const IDS=Object.keys(META);
const BONES=[[0,1],[0,5],[0,9],[0,13],[0,17],[5,9],[9,13],[13,17],
 [1,2],[2,3],[3,4],[5,6],[6,7],[7,8],[9,10],[10,11],[11,12],
 [13,14],[14,15],[15,16],[17,18],[18,19],[19,20]];
const ACTS=["pick-and-place","non-prehensile","tool-mediated","other"];
let cur=IDS[0], DB=null;
const G={};  // segid -> {events:[], done:false, revealed:false}
IDS.forEach(id=>G[id]={events:[],done:false,revealed:false});
const LS="egostd_gold_v1";
try{ const j=JSON.parse(localStorage.getItem(LS)||"{}");
     Object.keys(j).forEach(k=>{ if(G[k]) G[k]=Object.assign(G[k],j[k]); }); }catch(e){}

const v=document.getElementById("v"), c=document.getElementById("c"), ctx=c.getContext("2d");
const $=id=>document.getElementById(id);

function step(){ return 1/(META[cur].src_fps||30); }
function dur(){ return META[cur].t1-META[cur].t0; }
function et(){ return META[cur].t0+v.currentTime; }

function persist(){
  try{ localStorage.setItem(LS,JSON.stringify(G)); }catch(e){}
  if(DB){
    const m=META[cur];
    DB.doc("gold/"+cur).set({segment:cur,episode:m.episode,cls:m.cls,t0:m.t0,t1:m.t1,
      src_fps:m.src_fps,done:!!G[cur].done,events:G[cur].events,
      updated:new Date().toISOString()}).then(()=>{
        $("dbstat").innerHTML="saved to <b>artifact store</b> · gold/"+cur;
      }).catch(e=>{ $("dbstat").innerHTML="store write failed ("+(e&&e.code||"?")+
        ") — <b>browser-local only</b>"; });
  }
}

function loadSeg(id){
  cur=id; const m=META[id];
  v.src=VID[id];
  v.load();
  $("segtitle").textContent=id+" · "+m.cls;
  $("segmeta").textContent=m.episode+" · "+m.t0.toFixed(0)+"–"+m.t1.toFixed(0)+"s · "+
    m.src_fps.toFixed(0)+" fps";
  $("reveal").disabled=!G[id].done;
  $("revealnote").classList.toggle("hidden",!!G[id].revealed);
  renderSegs(); renderEvents(); renderTl(); renderCmp();
}

function renderSegs(){
  const L=$("seglist"); L.innerHTML="";
  let done=0, tot=0;
  IDS.forEach(id=>{
    const g=G[id]; if(g.done) done++; tot+=g.events.length;
    const d=document.createElement("div");
    d.className="segrow"+(id===cur?" on":"");
    d.innerHTML='<div><div class="nm">'+id+'</div><div class="cl">'+META[id].cls+
      ' · '+(META[id].t1-META[id].t0).toFixed(0)+'s</div></div>'+
      '<span class="badge'+(g.done?" done":"")+'">'+g.events.length+(g.done?" ✓":"")+'</span>';
    d.onclick=()=>loadSeg(id);
    L.appendChild(d);
  });
  $("progall").textContent=done+"/"+IDS.length+" done";
  const mins=IDS.reduce((a,id)=>a+(META[id].t1-META[id].t0),0)/60;
  const byc={};
  IDS.forEach(id=>{ const k=META[id].cls; byc[k]=(byc[k]||0)+G[id].events.length; });
  $("totals").innerHTML="<b>"+tot+"</b> events marked over <b>"+mins.toFixed(1)+
    "</b> min<br>"+ACTS.filter(a=>byc[a]).map(a=>a+": <b>"+byc[a]+"</b>").join("<br>");
}

function renderEvents(){
  const L=$("evlist"); L.innerHTML="";
  const g=G[cur]; const now=et();
  g.events.slice().sort((a,b)=>a.t-b.t).forEach(e=>{
    const d=document.createElement("div");
    d.className="evrow"+(Math.abs(e.t-now)<0.25?" cur":"");
    d.innerHTML='<div class="tm">'+e.t.toFixed(2)+'s</div>'+
      '<div><span class="pill '+(e.type==="contact"?"c":"r")+'">'+e.hand+' '+e.type+'</span>'+
      (e.occluded?'<span class="pill o">occl</span>':'')+
      (e.gloved?'<span class="pill g">glove</span>':'')+
      '<div class="tg">'+e.action+'</div></div>'+
      '<div><button data-del="'+e.id+'" class="warn" style="padding:3px 6px">✕</button></div>';
    d.onclick=ev=>{
      const b=ev.target.closest("button[data-del]");
      if(b){ ev.stopPropagation(); g.events=g.events.filter(x=>x.id!==b.dataset.del);
             persist(); renderEvents(); renderTl(); renderSegs(); renderCmp(); return; }
      v.currentTime=Math.max(0,Math.min(dur()-0.01,e.t-META[cur].t0));
    };
    // tag editors
    const sel=document.createElement("select");
    ACTS.forEach(a=>{ const o=document.createElement("option"); o.value=a; o.textContent=a;
      if(a===e.action)o.selected=true; sel.appendChild(o); });
    sel.onchange=ev=>{ ev.stopPropagation(); e.action=sel.value; persist(); renderEvents(); };
    sel.onclick=ev=>ev.stopPropagation();
    const wrap=document.createElement("div");
    wrap.style.cssText="grid-column:1/-1;display:flex;gap:8px;align-items:center;padding-top:4px";
    wrap.appendChild(sel);
    [["occluded","o"],["gloved","g"]].forEach(([k])=>{
      const lb=document.createElement("label"); lb.className="chk";
      const cb=document.createElement("input"); cb.type="checkbox"; cb.checked=!!e[k];
      cb.onclick=ev=>ev.stopPropagation();
      cb.onchange=()=>{ e[k]=cb.checked; persist(); renderEvents(); };
      lb.appendChild(cb); lb.appendChild(document.createTextNode(k)); wrap.appendChild(lb);
    });
    d.appendChild(wrap);
    L.appendChild(d);
  });
  $("evcount").textContent=g.events.length+" marked";
}

function renderTl(){
  const T=$("tl");
  [...T.querySelectorAll(".ev,.cd")].forEach(n=>n.remove());
  const g=G[cur], m=META[cur];
  const pc=t=>100*(t-m.t0)/dur();
  g.events.forEach(e=>{
    const d=document.createElement("div");
    d.className="ev "+(e.type==="contact"?"c":"r")+" "+e.hand;
    d.style.left=pc(e.t)+"%"; T.appendChild(d);
  });
  if(g.revealed){
    (m.candidates||[]).forEach(k=>{
      const d=document.createElement("div"); d.className="cd";
      d.style.left=pc(k.t)+"%"; d.title=k.type+" "+k.hand+" @"+k.t.toFixed(2);
      T.appendChild(d);
    });
  }
}

function renderCmp(){
  const g=G[cur], m=META[cur], box=$("cmp");
  if(!g.revealed){ box.classList.add("hidden"); return; }
  box.classList.remove("hidden");
  const cands=(m.candidates||[]).map(k=>({t:k.t,hand:k.hand==="left"?"L":"R",type:k.type}));
  const tol=0.5;
  let tp=0, offs=[];
  const used=new Set();
  g.events.forEach(e=>{
    let best=-1,bd=1e9;
    cands.forEach((k,i)=>{ if(used.has(i)||k.hand!==e.hand||k.type!==e.type) return;
      const d=Math.abs(k.t-e.t); if(d<bd){bd=d;best=i;} });
    if(best>=0&&bd<=tol){ used.add(best); tp++; offs.push(cands[best].t-e.t); }
  });
  const fp=cands.length-tp, fn=g.events.length-tp;
  const med=offs.length?offs.slice().sort((a,b)=>a-b)[Math.floor(offs.length/2)]:null;
  box.innerHTML="vs detector (±"+tol+"s, hand+type matched)<br>"+
    "matched <b>"+tp+"</b> · detector-only (FP) <b>"+fp+"</b> · missed by detector (FN) <b>"+fn+"</b><br>"+
    "precision <b>"+(cands.length?(100*tp/cands.length).toFixed(0):"—")+"%</b> · recall <b>"+
    (g.events.length?(100*tp/g.events.length).toFixed(0):"—")+"%</b>"+
    (med!==null?"<br>median timing offset <b>"+(1000*med).toFixed(0)+" ms</b> (detector − gold)":"");
}

function frameIdx(){
  const m=META[cur];
  return Math.max(0,Math.min((m.L||[]).length-1, Math.round(v.currentTime*m.fps)));
}
function draw(){
  const m=META[cur];
  ctx.clearRect(0,0,c.width,c.height);
  if($("skel").checked){
    const i=frameIdx();
    [["L",m.L,"#58c3d4"],["R",m.R,"#ffd166"]].forEach(([k,arr,col])=>{
      const f=arr&&arr[i]; if(!f) return;
      const P=[]; for(let j=0;j<21;j++){ const x=f[1+2*j],y=f[2+2*j];
        P.push(x<=-900?null:[x,y]); }
      ctx.strokeStyle=col; ctx.lineWidth=1.8; ctx.beginPath();
      BONES.forEach(([a,b])=>{ if(P[a]&&P[b]){ctx.moveTo(P[a][0],P[a][1]);ctx.lineTo(P[b][0],P[b][1]);} });
      ctx.stroke();
      ctx.fillStyle=col;
      P.forEach(p=>{ if(p){ctx.beginPath();ctx.arc(p[0],p[1],2,0,6.284);ctx.fill();} });
      if(P[4]&&P[8]){ ctx.strokeStyle="#fff"; ctx.lineWidth=1.2; ctx.setLineDash([3,3]);
        ctx.beginPath(); ctx.moveTo(P[4][0],P[4][1]); ctx.lineTo(P[8][0],P[8][1]);
        ctx.stroke(); ctx.setLineDash([]); }
    });
  }
  $("ph").style.left=(100*v.currentTime/dur())+"%";
  $("sc").value=Math.round(1000*v.currentTime/dur());
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);

let lastCur=-1;
setInterval(()=>{ const i=Math.round(et()*4); if(i!==lastCur){lastCur=i;renderEvents();} },250);

function mark(hand,type){
  const g=G[cur];
  const e={id:cur+"_"+hand+"_"+type+"_"+Math.round(et()*1000),
           t:+et().toFixed(3), hand:hand, type:type,
           action:META[cur].cls, occluded:false, gloved:false};
  if(g.events.some(x=>x.hand===hand&&x.type===type&&Math.abs(x.t-e.t)<0.05)) return;
  g.events.push(e); persist(); renderEvents(); renderTl(); renderSegs(); renderCmp();
}
document.querySelectorAll("[data-mark]").forEach(b=>{
  b.onclick=()=>{ const [h,t]=b.dataset.mark.split(","); mark(h,t); };
});
$("play").onclick=()=>{ if(v.paused){v.play();$("play").textContent="❙❙ space";}
                        else{v.pause();$("play").textContent="▶ space";} };
v.addEventListener("pause",()=>$("play").textContent="▶ space");
v.addEventListener("play",()=>$("play").textContent="❙❙ space");
const nudge=n=>{ v.pause(); v.currentTime=Math.max(0,Math.min(dur()-0.01,v.currentTime+n*step())); };
$("b1").onclick=()=>nudge(-1); $("f1").onclick=()=>nudge(1);
$("b10").onclick=()=>nudge(-10); $("f10").onclick=()=>nudge(10);
const RATES=[1,0.5,0.25];
let ri=0;
$("rate").onclick=()=>{ ri=(ri+1)%RATES.length; v.playbackRate=RATES[ri];
  $("rate").textContent=RATES[ri]+"×"; };
$("undo").onclick=()=>{ const g=G[cur]; if(!g.events.length) return;
  g.events.sort((a,b)=>a.t-b.t); g.events.pop();
  persist(); renderEvents(); renderTl(); renderSegs(); renderCmp(); };
$("sc").oninput=()=>{ v.currentTime=dur()*(+$("sc").value)/1000; };
$("tl").onclick=ev=>{ const r=$("tl").getBoundingClientRect();
  v.currentTime=Math.max(0,Math.min(dur()-0.01,dur()*(ev.clientX-r.left)/r.width)); };
$("finish").onclick=()=>{ G[cur].done=true; $("reveal").disabled=false;
  persist(); renderSegs(); };
$("reveal").onclick=()=>{ G[cur].revealed=true; $("revealnote").classList.add("hidden");
  renderTl(); renderCmp(); };
$("skel").onchange=()=>{};
$("copy").onclick=()=>{
  const out=IDS.map(id=>({segment:id,episode:META[id].episode,cls:META[id].cls,
    t0:META[id].t0,t1:META[id].t1,src_fps:META[id].src_fps,
    done:G[id].done,events:G[id].events}));
  navigator.clipboard.writeText(JSON.stringify(out,null,1)).then(
    ()=>{$("copy").textContent="copied ✓";setTimeout(()=>$("copy").textContent="copy all JSON",1500);},
    ()=>{$("copy").textContent="copy failed";});
};
document.addEventListener("keydown",e=>{
  if(e.target.tagName==="INPUT"||e.target.tagName==="SELECT") return;
  const k=e.key.toLowerCase();
  if(k===" "||e.code==="Space"){e.preventDefault();$("play").click();}
  else if(e.key==="ArrowLeft"){e.preventDefault();nudge(e.shiftKey?-10:-1);}
  else if(e.key==="ArrowRight"){e.preventDefault();nudge(e.shiftKey?10:1);}
  else if(k==="q")mark("L","contact"); else if(k==="w")mark("L","release");
  else if(k==="o")mark("R","contact"); else if(k==="p")mark("R","release");
  else if(k==="z")$("undo").click();
});

loadSeg(IDS[0]);
if(window.claude&&claude.use){
  claude.use("db").then(d=>{
    if(!d){ $("dbstat").innerHTML="store unavailable — <b>browser-local only</b>. Use "+
      "<i>copy all JSON</i> to hand the gold set over."; return; }
    DB=d;
    d.collection("gold").get().then(snap=>{
      let n=0;
      (snap.docs||[]).forEach(doc=>{
        const dt=doc.data?doc.data():doc;
        if(dt&&dt.segment&&G[dt.segment]){
          G[dt.segment].events=dt.events||[]; G[dt.segment].done=!!dt.done; n++;
        }
      });
      $("dbstat").innerHTML="connected · loaded <b>"+n+"</b> saved segment(s)";
      renderSegs(); renderEvents(); renderTl();
    }).catch(()=>{ $("dbstat").innerHTML="connected (no saved data yet)"; });
  }).catch(()=>{ $("dbstat").innerHTML="store unavailable — <b>browser-local only</b>"; });
}else{ $("dbstat").innerHTML="<b>browser-local only</b> (no store in this context)"; }
</script>
'''

JS = JS.replace("__META__", json.dumps(meta, separators=(",", ":")))
JS = JS.replace("__VID__", json.dumps({k: "data:video/mp4;base64," + b for k, b in vids.items()},
                                      separators=(",", ":")))
open(OUT, "w").write(HEAD + BODY + JS)
print("wrote %s  %.2f MB" % (OUT, os.path.getsize(OUT) / 1e6))
