"""
Side-by-side hand pose viewer - single-page HTML build.

Two pose streams describe the same hands: the annotation shipped in the mcap
and the one estimated from the head camera (docs/pose.md). They agree on
finger configuration and disagree on placement, and a table of medians is a
poor way to see what that means. This draws both skeletons on the frame they
describe, so the disagreement is visible rather than tabulated.

Both streams are reprojected through the SAME shipped intrinsics, extrinsics
and measured optical-axis sign, so any separation on screen is the poses
differing and not two different cameras.

Frames are embedded as JPEG data so the page is one self-contained file, which
is why it samples a fixed number of frames per episode rather than shipping
the video. Colour follows the house palette: amber is the shipped annotation,
teal the estimate, everywhere on the page.

    python -m egoannot pose viewer [--frames 64] [--width 460]
"""
from __future__ import annotations

import base64
import json
import os

import numpy as np

from .. import config
from ..core.geometry import quats_to_R
from ..core.mcap_io import read_episode
from ..pose import _ace_path
from ..pose.ace import measured_axis
from ..pose.evaluate import _resample, aperture

# MediaPipe/OpenPose topology, shared by both streams (docs/pose.md).
BONES = ([(0, 1), (1, 2), (2, 3), (3, 4)] + [(0, 5), (5, 6), (6, 7), (7, 8)]
         + [(0, 9), (9, 10), (10, 11), (11, 12)]
         + [(0, 13), (13, 14), (14, 15), (15, 16)]
         + [(0, 17), (17, 18), (18, 19), (19, 20)]
         + [(5, 9), (9, 13), (13, 17)])


def _project(P, R, t, K, axis, sx, sy):
    """World joints -> display pixels, through the shipped camera."""
    pc = np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), P - t)
    z = pc[:, 2] * axis
    ok = z > 1e-6
    u = np.full(len(P), np.nan)
    v = np.full(len(P), np.nan)
    u[ok] = (K["fx"] * pc[ok, 0] / z[ok] + K["cx"]) * sx
    v[ok] = (K["fy"] * pc[ok, 1] / z[ok] + K["cy"]) * sy
    return u, v


def _round(x, nd=1):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


QUALITY = 52


def episode_payload(path, play_fps: float, width: int,
                    seconds: float | None = None) -> dict | None:
    import cv2
    ep = read_episode(str(path), want_video=False)
    name = ep["name"]
    pred = _ace_path(name)
    clip = os.path.join(str(config.ARTIFACTS / "pose" / "clips"), name + ".mp4")
    if not (os.path.exists(pred) and os.path.exists(clip)):
        print(f"  skip {name}: need `pose run` and `pose export` first")
        return None

    z = np.load(pred)
    K, extr, axis = ep["K"], ep["extr"], measured_axis(ep)
    tm = z["right_hand"][:, 0]
    # Sample at a fixed rate along the timeline, not a fixed COUNT across it.
    # A fixed count stretches or compresses the clock -- a 30 s episode and a
    # 11 s one would play in the same wall time -- so the viewer showed motion
    # at whatever speed the episode length happened to imply.
    src_fps = float(ep["src_fps"]) or 30.0
    step = max(1, int(round(src_fps / float(play_fps))))
    limit = len(tm) if seconds is None else min(
        len(tm), int(round(float(seconds) * src_fps)) + 1)
    idx = np.arange(0, limit, step)
    times = tm[idx]

    cap = cv2.VideoCapture(clip)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    disp_h = int(round(src_h * (width / src_w) / 2) * 2)
    want, frames, i = set(int(x) for x in idx), [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            f = cv2.resize(f, (width, disp_h), interpolation=cv2.INTER_AREA)
            enc, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
            if enc:
                frames.append(base64.b64encode(buf).decode())
        if i >= max(want):
            break
        i += 1
    cap.release()

    ei = np.clip(np.searchsorted(extr[:, 0], times), 0, len(extr) - 1)
    R, tt = quats_to_R(extr[ei, 4:8]), extr[ei, 1:4]
    sx, sy = width / K["w"], disp_h / K["h"]

    # the pre-PnP arm, drawn as a ghost so the correction is visible
    raw_dir = config.ARTIFACTS / "pose" / "world_raw"
    z_raw = np.load(raw_dir / (name + ".npz")) if (
        raw_dir / (name + ".npz")).exists() else None

    rec = dict(name=name, task=ep["meta"].get("task_name") or name,
               w=width, h=disp_h, duration=round(ep["duration_s"], 1),
               src_fps=src_fps, play_fps=round(src_fps / step, 2),
               times=[round(float(x), 3) for x in times],
               frames=frames, joints={}, aperture={})
    for src in ("shipped", "ace", "ace_raw"):
        if src == "ace_raw" and z_raw is None:
            continue
        rec["joints"][src], rec["aperture"][src] = {}, {}
        for side in ("left", "right"):
            if src == "ace":
                J = z[f"{side}_hand_joints"][idx]
            elif src == "ace_raw":
                Jr = z_raw[f"{side}_hand_joints"]
                J = Jr[idx] if len(Jr) > idx.max() else Jr[:len(idx)]
            else:
                J = _resample(ep[f"/pose/{side}_hand"][:, 0],
                              ep[f"/pose/{side}_hand_joints"], times)
            per = []
            for fi in range(len(times)):
                u, v = _project(J[fi], np.repeat(R[fi:fi + 1], 21, 0),
                                np.repeat(tt[fi:fi + 1], 21, 0), K, axis, sx, sy)
                per.append([[_round(a), _round(b)] for a, b in zip(u, v)])
            rec["joints"][src][side] = per
            rec["aperture"][src][side] = [_round(x * 1000) for x in aperture(J)]
    del ep
    return rec


HEAD = r'''<title>Two Hands, Two Sources</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root{
  --bg:#eef2f3; --surface:#ffffff; --surface-2:#f6f9f9;
  --ink:#121a1e; --ink-2:#33454c; --muted:#5b6f77;
  --line:#d3dee1; --line-strong:#b6c7cc;
  --teal:#0e6e7c; --steel:#55919e; --alert:#9d3a2b; --good:#1f6b4a; --amber:#8a6410;
  --ace:#0e6e7c; --shipped:#8a6410;
  --ace-ink:#0e6e7c; --shipped-ink:#8a6410;
  --shadow:0 1px 2px rgba(18,26,30,.06),0 8px 24px -12px rgba(18,26,30,.16);
  --d:"Archivo","Helvetica Neue",Arial,sans-serif;
  --m:"IBM Plex Mono",ui-monospace,Menlo,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0c1215; --surface:#131c20; --surface-2:#18242a;
  --ink:#e8eef0; --ink-2:#c2d0d5; --muted:#93a6ad;
  --line:#24343a; --line-strong:#38505a;
  --teal:#58c3d4; --steel:#4a7f8b; --alert:#e08a72; --good:#63c396; --amber:#d9ab4f;
  --ace-ink:#58c3d4; --shipped-ink:#d9ab4f;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -12px rgba(0,0,0,.6);
}}
:root[data-theme="dark"]{
  --bg:#0c1215; --surface:#131c20; --surface-2:#18242a;
  --ink:#e8eef0; --ink-2:#c2d0d5; --muted:#93a6ad;
  --line:#24343a; --line-strong:#38505a;
  --teal:#58c3d4; --steel:#4a7f8b; --alert:#e08a72; --good:#63c396; --amber:#d9ab4f;
  --ace-ink:#58c3d4; --shipped-ink:#d9ab4f;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -12px rgba(0,0,0,.6);
}
*{box-sizing:border-box}
html,body{margin:0}
body{background:var(--bg);color:var(--ink);font-family:var(--d);font-size:14px;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding:22px 20px 60px}
h1{font-size:1.5rem;font-weight:700;letter-spacing:-.02em;margin:0;text-wrap:balance}
h2{font-size:.98rem;font-weight:600;margin:0}
p{margin:0}
.eyebrow{font-family:var(--m);font-size:.66rem;text-transform:uppercase;
  letter-spacing:.15em;color:var(--muted)}
header{border-bottom:1px solid var(--line);padding-bottom:16px;margin-bottom:18px}
.sub{color:var(--ink-2);margin-top:8px;max-width:78ch;font-size:.92rem}
.grid{display:grid;grid-template-columns:minmax(0,1fr) 290px;gap:16px;align-items:start}
@media (max-width:960px){.grid{grid-template-columns:1fr}}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:6px;
  padding:14px;box-shadow:var(--shadow)}
.stagewrap{display:flex;flex-direction:column;gap:10px}
.stage{position:relative;background:#000;border-radius:4px;overflow:hidden;
  display:flex;justify-content:center}
canvas{display:block;max-width:100%;height:auto}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
button{font-family:var(--m);font-size:.72rem;font-weight:600;padding:6px 9px;
  background:var(--surface-2);color:var(--ink-2);border:1px solid var(--line);
  border-radius:4px;cursor:pointer}
button:hover{border-color:var(--teal);color:var(--teal)}
button.pri{background:var(--teal);border-color:var(--teal);color:#fff}
button.pri:hover{opacity:.88;color:#fff}
button[aria-pressed="false"]{opacity:.5}
:focus-visible{outline:2px solid var(--teal);outline-offset:2px}
input[type=range]{flex:1;min-width:140px;accent-color:var(--teal)}
select{font-family:var(--m);font-size:.74rem;padding:6px 8px;background:var(--surface-2);
  color:var(--ink);border:1px solid var(--line);border-radius:4px;width:100%}
.swatch{width:10px;height:10px;border-radius:2px;display:inline-block;
  vertical-align:-1px;margin-right:6px}
.readout{display:grid;grid-template-columns:1fr auto;gap:2px 10px;
  font-family:var(--m);font-size:.78rem;font-variant-numeric:tabular-nums}
.readout .k{color:var(--muted)}
.readout .v{text-align:right}
.rule{height:1px;background:var(--line);margin:12px 0}
table{width:100%;border-collapse:collapse;font-family:var(--m);font-size:.72rem;
  font-variant-numeric:tabular-nums}
th{text-align:right;font-weight:600;color:var(--muted);padding:3px 0;
  border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
td{padding:3px 0;border-bottom:1px solid var(--line);color:var(--ink-2)}
.big{font-family:var(--m);font-size:1.5rem;font-weight:600;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums}
.note{font-size:.8rem;color:var(--muted);margin-top:10px;max-width:70ch}
.tracewrap{position:relative}
.legend{display:flex;gap:14px;font-family:var(--m);font-size:.7rem;color:var(--ink-2)}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>'''


BODY = r'''
<div class="wrap">
<header>
  <div class="eyebrow">egoannot &middot; pose</div>
  <h1>Two hands, two sources</h1>
  <p class="sub">The same hands, described twice: <b style="color:var(--shipped-ink)">the pose
  annotated in the mcap</b> and <b style="color:var(--ace-ink)">the pose estimated from
  the head camera</b>. Both are reprojected through the same shipped intrinsics,
  extrinsics and measured optical-axis sign, so any gap you see on screen is the two
  poses differing &mdash; not two different cameras.</p>
  <p class="sub">The estimator&rsquo;s joints and its 2D anchors are both good; only the
  translation it hung them on was wrong, placing hands about 20% too far. That
  translation is now re-solved per frame by fitting the joints onto the model&rsquo;s own
  anchors &mdash; no ground truth involved. Switch on <b>before PnP</b> to see where the
  hand used to sit.</p>
</header>

<div class="grid">
  <div class="stagewrap">
    <div class="panel" style="padding:10px">
      <div class="stage"><canvas id="cv" width="460" height="349"></canvas></div>
      <div class="controls" style="margin-top:10px">
        <button id="play" class="pri">&#9654;&nbsp; Play</button>
        <input type="range" id="scrub" min="0" max="0" value="0" step="1"
               aria-label="frame">
        <span class="eyebrow" id="tpos" style="font-size:.7rem">0.00 s</span>
      </div>
      <div class="controls" style="margin-top:8px">
        <button id="t-shipped" aria-pressed="true">
          <span class="swatch" style="background:var(--shipped-ink)"></span>shipped</button>
        <button id="t-ace" aria-pressed="true">
          <span class="swatch" style="background:var(--ace-ink)"></span>estimated</button>
        <button id="t-ace_raw" aria-pressed="false">
          <span class="swatch" style="background:var(--ace-ink);opacity:.45"></span>before PnP</button>
        <button id="t-left" aria-pressed="true">left hand</button>
        <button id="t-right" aria-pressed="true">right hand</button>
      </div>
    </div>

    <div class="panel">
      <div style="display:flex;justify-content:space-between;align-items:baseline;
                  gap:12px;flex-wrap:wrap">
        <h2>Grasp aperture &mdash; thumb tip to index tip</h2>
        <div class="legend">
          <span><span class="swatch" style="background:var(--shipped-ink)"></span>shipped</span>
          <span><span class="swatch" style="background:var(--ace-ink)"></span>estimated</span>
        </div>
      </div>
      <p class="note" style="margin-top:4px">This is the one scalar the events and spans
      stages actually read, which is why it matters more than joint position.</p>
      <div class="tracewrap" style="margin-top:8px">
        <canvas id="trace" width="1120" height="190"
                style="width:100%;height:auto"></canvas>
      </div>
    </div>
  </div>

  <div>
    <div class="panel">
      <div class="eyebrow">episode</div>
      <select id="ep" style="margin-top:6px"></select>
      <div class="rule"></div>
      <div class="eyebrow">this frame &middot; right hand</div>
      <div class="readout" style="margin-top:8px">
        <span class="k">aperture, shipped</span><span class="v" id="ap-s">&mdash;</span>
        <span class="k">aperture, estimated</span><span class="v" id="ap-a">&mdash;</span>
        <span class="k">difference</span><span class="v" id="ap-d">&mdash;</span>
        <span class="k">wrist separation</span><span class="v" id="wr-d">&mdash;</span>
        <span class="k">PnP moved the wrist</span><span class="v" id="pnp-d">&mdash;</span>
      </div>
      <div class="rule"></div>
      <div class="eyebrow">episode medians</div>
      <div style="margin-top:8px">
        <div class="eyebrow" style="font-size:.62rem">shape disagreement</div>
        <div class="big" id="m-shape">&mdash;</div>
        <div class="eyebrow" style="font-size:.62rem;margin-top:8px">placement disagreement</div>
        <div class="big" id="m-place">&mdash;</div>
      </div>
      <div class="rule"></div>
      <table>
        <thead><tr><th>hand</th><th>ap MAE</th><th>ap r</th><th>cover</th></tr></thead>
        <tbody id="mtab"></tbody>
      </table>
    </div>
    <p class="note">Playback runs at recorded speed &mdash; wall-clock time is mapped
    onto the episode&rsquo;s own timestamps, so a 26-second clip takes 26 seconds.
    Frames are decimated to <span id="pf">&mdash;</span>&#8239;fps to keep the page one
    self-contained file; the timing is real, the smoothness is not.</p>
  </div>
</div>
</div>

<script>
const DATA = __PAYLOAD__;
const $ = id => document.getElementById(id);
const cv = $("cv"), ctx = cv.getContext("2d");
const tc = $("trace"), tctx = tc.getContext("2d");
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

let ep = DATA.episodes[0], fi = 0, playing = false, imgs = [], lastGood = null;
const show = {shipped:true, ace:true, ace_raw:false, left:true, right:true};

DATA.episodes.forEach((e,i) => {
  const o = document.createElement("option");
  o.value = i; o.textContent = e.task + "  (" + e.duration + "s)";
  $("ep").appendChild(o);
});

function loadEpisode(i){
  ep = DATA.episodes[i]; fi = 0; t0 = 0; lastGood = null;
  cv.width = ep.w; cv.height = ep.h;
  $("scrub").max = ep.frames.length - 1;
  $("scrub").value = 0;
  imgs = ep.frames.map(b64 => { const im = new Image(); im.src = "data:image/jpeg;base64," + b64; return im; });
  metrics();
  $("pf").textContent = ep.play_fps;
  // Decode only the opening eagerly. Forcing all of them resolves ~270 MB of
  // bitmap at once on the longer episodes; the rest decode as they are drawn,
  // and a frame that is not ready yet simply keeps the previous one on screen.
  Promise.all(imgs.slice(0, 24).map(im => im.decode().catch(()=>{})))
    .then(() => { draw(); trace(); });
  draw(); trace();
}

function drawHand(pts, colour){
  ctx.lineWidth = 2; ctx.strokeStyle = colour; ctx.fillStyle = colour;
  ctx.lineJoin = "round"; ctx.lineCap = "round";
  for (const [a,b] of DATA.bones){
    const p = pts[a], q = pts[b];
    if (p[0]==null || q[0]==null) continue;
    ctx.beginPath(); ctx.moveTo(p[0],p[1]); ctx.lineTo(q[0],q[1]); ctx.stroke();
  }
  for (let j=0;j<pts.length;j++){
    const p = pts[j]; if (p[0]==null) continue;
    ctx.beginPath(); ctx.arc(p[0],p[1], j===0?3.4:2.2, 0, 6.2832); ctx.fill();
  }
}

function draw(){
  ctx.clearRect(0,0,cv.width,cv.height);
  const im = imgs[fi];
  if (im && im.complete && im.naturalWidth){ ctx.drawImage(im,0,0,cv.width,cv.height); lastGood = im; }
  else if (lastGood){ ctx.drawImage(lastGood,0,0,cv.width,cv.height); }
  else { ctx.fillStyle="#000"; ctx.fillRect(0,0,cv.width,cv.height); }
  // a hairline dark pass under the marks keeps them legible on bright footage
  for (const src of ["ace_raw","shipped","ace"]){          // ghost first, under
    if (!show[src] || !ep.joints[src]) continue;
    const ghost = src === "ace_raw";
    const colour = src === "shipped" ? css("--shipped-ink") : css("--ace-ink");
    for (const side of ["left","right"]){
      if (!show[side]) continue;
      const pts = ep.joints[src][side][fi];
      if (!pts) continue;
      if (ghost){            // thin, dashed, no joints: the same source, before
        ctx.save();
        ctx.globalAlpha=.5; ctx.strokeStyle=colour; ctx.lineWidth=1.5;
        ctx.setLineDash([4,3]); ctx.lineJoin="round"; ctx.lineCap="round";
        for (const [a,b] of DATA.bones){
          const q=pts[a],r=pts[b]; if(q[0]==null||r[0]==null) continue;
          ctx.beginPath(); ctx.moveTo(q[0],q[1]); ctx.lineTo(r[0],r[1]); ctx.stroke();
        }
        ctx.restore(); continue;
      }
      ctx.save(); ctx.globalAlpha=.55; ctx.strokeStyle="rgba(0,0,0,.9)"; ctx.lineWidth=4.5;
      ctx.lineJoin="round"; ctx.lineCap="round";
      for (const [a,b] of DATA.bones){
        const p=pts[a],q=pts[b]; if(p[0]==null||q[0]==null) continue;
        ctx.beginPath(); ctx.moveTo(p[0],p[1]); ctx.lineTo(q[0],q[1]); ctx.stroke();
      }
      ctx.restore();
      drawHand(pts, colour);
    }
  }
  $("tpos").textContent = (ep.times[fi] ?? 0).toFixed(1) + " / "
      + ep.times[ep.times.length-1].toFixed(1) + " s";
  readout();
}

function readout(){
  const s = ep.aperture.shipped.right[fi], a = ep.aperture.ace.right[fi];
  const f = v => v==null ? "&mdash;" : v.toFixed(1) + " mm";
  $("ap-s").innerHTML = f(s); $("ap-a").innerHTML = f(a);
  $("ap-d").innerHTML = (s==null||a==null) ? "&mdash;" : (Math.abs(s-a)).toFixed(1)+" mm";
  const ps = ep.joints.shipped.right[fi]?.[0], pa = ep.joints.ace.right[fi]?.[0];
  $("wr-d").innerHTML = (!ps||!pa||ps[0]==null||pa[0]==null) ? "&mdash;"
    : Math.hypot(ps[0]-pa[0], ps[1]-pa[1]).toFixed(0) + " px";
  const pr = ep.joints.ace_raw?.right?.[fi]?.[0];
  $("pnp-d").innerHTML = (!pr||!pa||pr[0]==null||pa[0]==null) ? "&mdash;"
    : Math.hypot(pr[0]-pa[0], pr[1]-pa[1]).toFixed(0) + " px";
}

function trace(){
  const W = tc.width, H = tc.height, padL = 42, padR = 12, padT = 12, padB = 24;
  tctx.clearRect(0,0,W,H);
  const series = [["shipped", css("--shipped-ink")], ["ace", css("--ace-ink")]];
  let vals = [];
  for (const [k] of series) vals = vals.concat(ep.aperture[k].right.filter(v=>v!=null));
  if (!vals.length) return;
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const y0 = Math.floor(lo/10)*10, y1 = Math.ceil(hi/10)*10 || y0+10;
  const n = ep.times.length;
  const X = i => padL + (W-padL-padR) * (n<2?0:i/(n-1));
  const Y = v => padT + (H-padT-padB) * (1 - (v-y0)/((y1-y0)||1));

  tctx.strokeStyle = css("--line"); tctx.lineWidth = 1;
  tctx.fillStyle = css("--muted"); tctx.font = "11px ui-monospace,monospace";
  tctx.textAlign = "right"; tctx.textBaseline = "middle";
  for (let g=0; g<=2; g++){
    const v = y0 + (y1-y0)*g/2, y = Math.round(Y(v))+.5;
    tctx.beginPath(); tctx.moveTo(padL,y); tctx.lineTo(W-padR,y); tctx.stroke();
    tctx.fillText(v.toFixed(0), padL-8, y);
  }
  tctx.textAlign="left"; tctx.textBaseline="top";
  tctx.fillText("mm", 6, padT-6);

  for (const [k,colour] of series){
    tctx.strokeStyle = colour; tctx.lineWidth = 2;
    tctx.lineJoin="round"; tctx.lineCap="round";
    tctx.beginPath(); let open=false;
    ep.aperture[k].right.forEach((v,i) => {
      if (v==null){ open=false; return; }
      if (!open){ tctx.moveTo(X(i),Y(v)); open=true; } else tctx.lineTo(X(i),Y(v));
    });
    tctx.stroke();
  }
  // playhead
  const px = Math.round(X(fi))+.5;
  tctx.strokeStyle = css("--ink"); tctx.globalAlpha=.35; tctx.lineWidth=1;
  tctx.beginPath(); tctx.moveTo(px,padT); tctx.lineTo(px,H-padB); tctx.stroke();
  tctx.globalAlpha=1;
  for (const [k,colour] of series){
    const v = ep.aperture[k].right[fi]; if (v==null) continue;
    tctx.fillStyle = colour; tctx.beginPath(); tctx.arc(px, Y(v), 4, 0, 6.2832); tctx.fill();
    tctx.strokeStyle = css("--surface"); tctx.lineWidth=2; tctx.stroke();
  }
  tctx.fillStyle = css("--muted"); tctx.textAlign="center"; tctx.textBaseline="top";
  tctx.fillText("0 s", padL, H-padB+6);
  tctx.fillText(ep.times[n-1].toFixed(1)+" s", W-padR-12, H-padB+6);
}

function metrics(){
  const a = DATA.agreement[ep.name];
  const tb = $("mtab"); tb.innerHTML = "";
  if (!a){ $("m-shape").textContent="—"; $("m-place").textContent="—"; return; }
  const hs = Object.entries(a.hands);
  const med = k => { const v = hs.map(([,h])=>h[k]).filter(x=>x!=null); 
                     return v.length ? v.reduce((s,x)=>s+x,0)/v.length : null; };
  const sm = med("shape_mm"), pm = med("placement_mm");
  $("m-shape").innerHTML = sm==null?"&mdash;":sm.toFixed(1)+' <span style="font-size:.8rem;color:var(--muted)">mm</span>';
  $("m-place").innerHTML = pm==null?"&mdash;":pm.toFixed(0)+' <span style="font-size:.8rem;color:var(--muted)">mm</span>';
  for (const [side,h] of hs){
    const tr = document.createElement("tr");
    tr.innerHTML = "<td>"+side+"</td><td>"+(h.aperture_mae_mm?.toFixed(1) ?? "—")+
      "</td><td>"+(h.aperture_corr?.toFixed(2) ?? "—")+"</td><td>"+
      Math.round(100*h.coverage)+"%</td>";
    tb.appendChild(tr);
  }
}

// Playback follows the episode clock, not a frame counter: elapsed wall time
// is mapped onto ep.times, so a clip runs at the speed it was recorded at
// whatever rate the frames were sampled, and a slow decode drops a frame
// instead of stretching the timeline.
let t0 = 0, tStart = 0;
function step(ts){
  if (!playing) return;
  if (!t0){ t0 = ts; tStart = ep.times[fi]; }
  const target = tStart + (ts - t0) / 1000;
  const end = ep.times[ep.times.length - 1];
  let k = fi;
  while (k + 1 < ep.times.length && ep.times[k + 1] <= target) k++;
  if (target >= end){ k = 0; t0 = ts; tStart = ep.times[0]; }
  if (k !== fi){ fi = k; $("scrub").value = fi; draw(); trace(); }
  requestAnimationFrame(step);
}
$("play").onclick = () => {
  playing = !playing;
  $("play").innerHTML = playing ? "&#10073;&#10073;&nbsp; Pause" : "&#9654;&nbsp; Play";
  t0 = 0;                                  // re-anchor the clock on resume
  if (playing) requestAnimationFrame(step);
};
$("scrub").oninput = e => { fi = +e.target.value; t0 = 0; draw(); trace(); };
$("ep").onchange = e => loadEpisode(+e.target.value);
for (const k of ["shipped","ace","ace_raw","left","right"]){
  $("t-"+k).onclick = () => {
    show[k] = !show[k];
    $("t-"+k).setAttribute("aria-pressed", String(show[k]));
    draw();
  };
}
tc.addEventListener("click", e => {
  const r = tc.getBoundingClientRect();
  const x = (e.clientX - r.left) / r.width * tc.width;
  const frac = (x - 42) / (tc.width - 54);
  fi = Math.max(0, Math.min(ep.times.length-1, Math.round(frac*(ep.times.length-1))));
  t0 = 0; $("scrub").value = fi; draw(); trace();
});
matchMedia("(prefers-color-scheme:dark)").addEventListener("change", () => { draw(); trace(); });
loadEpisode(0);
</script>
'''


def build(play_fps: float = 12.0, width: int = 440, out=None,
          seconds: float | None = None, quality: int = 52) -> str:
    out = str(out or config.artifact("reports", "pose-viewer.html"))
    root = config.corpus()
    paths = sorted(root.glob("*.mcap")) + sorted((root / "ep").glob("*.mcap"))
    eps = []
    for p in paths:
        print(f"  {p.stem}", flush=True)
        rec = episode_payload(p, play_fps, width, seconds)
        if rec:
            eps.append(rec)
    if not eps:
        raise SystemExit("no episodes with both a clip and a prediction")

    agree = {}
    ap = config.ARTIFACTS / "pose" / "agreement.json"
    if ap.exists():
        for r in json.load(open(ap)):
            agree[r["episode"]] = r

    payload = json.dumps(dict(episodes=eps, agreement=agree, bones=BONES),
                         separators=(",", ":"))
    html = HEAD + BODY.replace("__PAYLOAD__", payload)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write(html)
    n = sum(len(e["frames"]) for e in eps)
    print(f"\nwrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB, "
          f"{len(eps)} episodes, {n} frames at {eps[0]['play_fps']:g} fps)")
    return out
