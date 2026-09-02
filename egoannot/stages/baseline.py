"""
Vision-only baseline: the VLM does everything.

The pipeline's premise is that pose should supply span boundaries, handedness
and the physical fields, and the VLM only the language. That premise was
asserted, not tested. This stage is the control arm.

The VLM is shown a window of frames and asked to segment it into atomic
actions itself, assigning its own boundaries and its own acting hand. It gets
the same format rules and the same verb vocabulary as the pose-guided arm, so
any difference measures the architecture rather than the prompt.

Fairness matters here, so the frame budget is matched. The pose-guided arm
spends `frames_per_span` frames on each ~2.1 s span, about 1.9 frames per
second of video; this arm samples at `fps` (default 2.0) across each window.

Pose is still used afterwards -- never in the prompt -- to score the arm:
handedness and aperture trend are measured over whatever window the model
proposed, which is what makes the two arms comparable at all.
"""
from __future__ import annotations

import json
import os
import re
import time

import numpy as np

from .. import config
from ..core.mcap_io import read_episode
from ..core.video import SegmentFrames
from ..labels import domains as DM
from . import spans as SP

CFG = dict(
    window_s=16.0,      # how much video the model segments in one call
    fps=2.0,            # frame sampling rate inside a window
    min_actions=3,      # guidance only, not enforced
)


def system_prompt():
    """Same rules and vocabulary as the pose-guided arm, plus segmentation."""
    return (
        "You annotate egocentric (head-camera) video of manipulation work.\n\n"
        "You are shown frames sampled evenly across one continuous clip, with "
        "the clip duration given. YOU decide how to divide the clip into atomic "
        "actions, where each action begins and ends, and which hand performs "
        "it.\n\n"
        "Output one JSON object per action, in time order:\n"
        '  {"start_ts": <s>, "end_ts": <s>, "hand": "LEFT|RIGHT|BOTH", '
        '"text": "...", "verb": "...", "noun": "...", "visibility": "...", '
        '"uncertain": false}\n\n'
        "start_ts and end_ts are seconds from the START of this clip. Actions "
        "must not overlap. Each action must last between 1.3 and 4.0 seconds. "
        "Cover the clip; do not leave long gaps.\n"
        "hand is the hand that performs the action; BOTH only if both "
        "genuinely contribute.\n\n"
        f"{DM.CORE_RULES}\n\n"
        f"The verb must be one of: {', '.join(DM.CORE_VERBS)}\n\n"
        "Worked examples - each is 10-15 words and passes validation:\n"
        + "\n".join("  " + e for e in DM.PROMPT_EXEMPLARS) + "\n\n"
        "Output only the JSON objects, one per line, nothing else.")


def _parse(txt, window_s):
    """Objects with a usable text and a plausible in-window time range."""
    out = []
    for m in re.finditer(r"\{[^{}]*\}", txt):
        try:
            o = json.loads(m.group(0))
        except ValueError:
            continue
        if not o.get("text"):
            continue
        try:
            a = float(o.get("start_ts"))
            b = float(o.get("end_ts"))
        except (TypeError, ValueError):
            continue
        if not (0.0 <= a < b <= window_s + 1e-6):
            continue
        out.append(dict(o, start_ts=a, end_ts=b))
    out.sort(key=lambda o: o["start_ts"])
    return out


def run(segments=None, out=None, backend="qwen-local", cfg=CFG,
        frames_per_call=None):
    """Caption segments with no pose input, then measure the result against pose."""
    from .caption import BACKENDS

    out = str(out or config.artifact("captions", "baseline_vision.jsonl"))
    segs = json.load(open(config.SEGMENT_DEFS))
    if segments:
        wanted = set(segments)
        segs = [s for s in segs if s["id"] in wanted]

    engine = BACKENDS[backend]()
    store = SegmentFrames(config.SEGMENTS_DIR, config.CAPTION["jpeg_quality"])

    rows, t0 = [], time.time()
    n_calls = 0
    for seg in segs:
        sid, t_seg0, t_seg1 = seg["id"], seg["t0"], seg["t1"]
        duration = t_seg1 - t_seg0
        name = os.path.basename(seg["source"]).rsplit(".", 1)[0]
        ep = read_episode(config.episode_path(name), want_video=False)

        starts = np.arange(0.0, duration - 0.5, cfg["window_s"])
        for w_start in starts:
            w_end = min(duration, w_start + cfg["window_s"])
            w_len = w_end - w_start
            n_frames = frames_per_call or max(4, int(round(w_len * cfg["fps"])))
            probe = dict(segment=sid, v_start=w_start, v_end=w_end)
            store.plan([probe], n_frames)
            jpegs = store.get(probe, n_frames)

            parts = [("text",
                      f"Clip: {w_len:.1f} seconds of egocentric manipulation "
                      f"video, {len(jpegs)} frames sampled evenly across it. "
                      f"Divide it into atomic actions.")]
            for jpg in jpegs:
                parts.append(("image", jpg))

            try:
                engine(system_prompt(), parts, [])
                got = _parse(getattr(engine, "last_raw", "") or "", w_len)
            except Exception as e:
                print(f"  CALL FAILED {sid} @{w_start:.0f}s: "
                      f"{type(e).__name__}: {e}", flush=True)
                continue
            n_calls += 1

            for k, o in enumerate(got):
                a = t_seg0 + w_start + o["start_ts"]
                b = t_seg0 + w_start + o["end_ts"]
                # Pose is used ONLY here, after the fact, to score the arm.
                pose_hand = SP.dominant_hand(ep, a, b)
                side = SP.acting_sides(pose_hand)[0]
                ap = SP.aperture_stats(ep, side, a, b) or {}
                rows.append(dict(
                    span_id=f"{sid}#v{w_start + o['start_ts']:07.3f}",
                    segment=sid, episode=name, cls=seg["cls"],
                    pack=DM.pack_for(None, sid, seg["cls"]),
                    start_ts=round(a, 3), end_ts=round(b, 3),
                    # model-proposed
                    hand=(o.get("hand") or "BOTH").upper(),
                    text=o.get("text", ""),
                    verb=(o.get("verb") or "").lower(),
                    noun=(o.get("noun") or "").lower(),
                    visibility=(o.get("visibility") or "FULL").upper(),
                    uncertain=bool(o.get("uncertain")),
                    # measured after the fact, for scoring only
                    pose_hand=pose_hand,
                    ap_trend=ap.get("ap_trend"),
                    aperture_mm=ap.get("aperture_mm"),
                    rotation=SP.rotation(ep, side, a, b),
                    fingers=SP.finger_state(ep, side, a, b),
                    arm="vision_only", backend=engine.name))
            print(f"  {sid:<15s} window {w_start:5.1f}-{w_end:5.1f}s "
                  f"({len(jpegs)} frames) -> {len(got):2d} actions", flush=True)
        store.release(sid)
        del ep

    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    elapsed = time.time() - t0
    total_v = sum(s["t1"] - s["t0"] for s in segs)
    print(f"\n{len(rows)} actions from {n_calls} calls over {total_v:.0f}s of "
          f"video in {elapsed:.0f}s")
    print(f"  density {len(rows) / (total_v / 60):.1f} actions/min  "
          f"frame store peak {store.bytes_peak / 1e6:.1f} MB")
    print("  wrote", out)
    return rows


def compare(pose_path=None, vision_path=None, segments=None):
    """Score both arms on the same footage and print the head-to-head."""
    from .score import CLOSING_VERBS, OPENING_VERBS
    from ..labels import atomicity as AL

    pose = [json.loads(l) for l in open(pose_path or config.CAPTIONS)]
    vision = [json.loads(l) for l in open(
        vision_path or config.artifact("captions", "baseline_vision.jsonl"))]
    if segments:
        keep = set(segments)
    else:
        keep = {r["segment"] for r in vision}
    pose = [c for c in pose if c["segment"] in keep]
    vision = [c for c in vision if c["segment"] in keep]

    segdefs_early = {x["id"]: x for x in json.load(open(config.SEGMENT_DEFS))}

    def measure(caps, arm):
        n = len(caps)
        if not n:
            return None
        fails, ok = {}, 0
        for c in caps:
            AL.use_domain(c.get("pack") or "retail_shelf")
            lab = dict(start_ts=c["start_ts"], end_ts=c["end_ts"], text=c["text"],
                       verb=c["verb"], noun=c["noun"],
                       hand=c["hand"] if c["hand"] in AL.HANDS else "BOTH",
                       visibility=c["visibility"], episode=c["episode"],
                       uncertain=c.get("uncertain", False),
                       rotation=c.get("rotation"))
            errs = [e for e in AL.lint(lab) if e[0] == "ERROR"]
            if errs:
                for _, code, _ in errs:
                    fails[code] = fails.get(code, 0) + 1
            else:
                ok += 1
        AL.use_domain("retail_shelf")

        norm = lambda t: re.sub(r"[^a-z0-9 ]", "", t.lower().strip())
        uniq = len({norm(c["text"]) for c in caps}) / n
        d = np.array([c["end_ts"] - c["start_ts"] for c in caps])
        lo, hi = AL.SPAN["measured"]
        in_band = float(((d >= lo) & (d <= hi)).mean())

        # overlap: labels must not overlap (rule A8)
        overlaps = 0
        by_seg = {}
        for c in caps:
            by_seg.setdefault(c["segment"], []).append(c)
        for group in by_seg.values():
            group.sort(key=lambda c: c["start_ts"])
            for a, b in zip(group, group[1:]):
                if b["start_ts"] < a["end_ts"] - 1e-6:
                    overlaps += 1

        # handedness against the pose measurement
        if arm == "pose":
            hand_agree = 1.0            # true by construction: pose supplied it
        else:
            pairs = [(c["hand"], c.get("pose_hand")) for c in caps
                     if c.get("pose_hand")]
            hand_agree = (sum(1 for a, b in pairs if a == b) / len(pairs)
                          if pairs else None)

        # verb against the measured aperture trend
        agree = conflict = 0
        for c in caps:
            trend, verb = c.get("ap_trend"), c["verb"]
            if trend not in ("closing", "opening"):
                continue
            if verb in CLOSING_VERBS:
                implied = "closing"
            elif verb in OPENING_VERBS:
                implied = "opening"
            else:
                continue
            agree += implied == trend
            conflict += implied != trend
        ground = agree / (agree + conflict) if (agree + conflict) else None

        # Labels per minute must be denominated in WALL-CLOCK video, not in
        # the time the labels happen to span. A single 16 s "action" inflates
        # the latter and makes a sparse arm look dense.
        segdefs = {x["id"]: x for x in json.load(open(config.SEGMENT_DEFS))}
        wall = sum(segdefs[s]["t1"] - segdefs[s]["t0"]
                   for s in {c["segment"] for c in caps} if s in segdefs)
        # union of labelled time, so overlaps and gaps are both accounted for
        union = 0.0
        for group in by_seg.values():
            merged = []
            for c in sorted(group, key=lambda c: c["start_ts"]):
                if merged and c["start_ts"] <= merged[-1][1] + 1e-9:
                    merged[-1][1] = max(merged[-1][1], c["end_ts"])
                else:
                    merged.append([c["start_ts"], c["end_ts"]])
            union += sum(b - a for a, b in merged)
        # Uniformity. A "100% covered" arm can still annotate only the opening
        # of each clip and then emit one span per window for the rest, which is
        # what the vision-only arm does; coverage alone hides that.
        head = tail = 0.0
        n_head = n_tail = 0
        for sid, group in by_seg.items():
            if sid not in segdefs_early:
                continue
            t0 = segdefs_early[sid]["t0"]
            span = segdefs_early[sid]["t1"] - t0
            head += min(span, 16.0)
            tail += max(0.0, span - 16.0)
            for c in group:
                if c["start_ts"] - t0 < 16.0:
                    n_head += 1
                else:
                    n_tail += 1
        return dict(n=n, atomicity=ok / n, uniqueness=uniq, in_band=in_band,
                    head_density=n_head / (head / 60) if head else 0.0,
                    tail_density=n_tail / (tail / 60) if tail else 0.0,
                    overlaps=overlaps, hand_agree=hand_agree, grounding=ground,
                    grounding_n=agree + conflict,
                    median_dur=float(np.median(d)), p90_dur=float(np.percentile(d, 90)),
                    density=n / (wall / 60) if wall else 0.0,
                    wall=wall, union=union, coverage=union / wall if wall else 0.0,
                    fails=fails)

    a = measure(pose, "pose")
    b = measure(vision, "vision")
    if not a or not b:
        print("one arm is empty")
        return {}

    print("=" * 78)
    print(f"POSE-GUIDED vs VISION-ONLY   {len(keep)} segments: "
          f"{', '.join(sorted(keep))}")
    print()
    rows = [
        ("labels produced", f"{a['n']}", f"{b['n']}"),
        ("labels/min of video", f"{a['density']:.1f}", f"{b['density']:.1f}"),
        ("labels/min, first 16s of clip",
         f"{a['head_density']:.1f}", f"{b['head_density']:.1f}"),
        ("labels/min, after 16s",
         f"{a['tail_density']:.1f}", f"{b['tail_density']:.1f}"),
        ("video inside a label", f"{100*a['coverage']:.0f}%", f"{100*b['coverage']:.0f}%"),
        ("median span (s)", f"{a['median_dur']:.2f}", f"{b['median_dur']:.2f}"),
        ("p90 span (s)", f"{a['p90_dur']:.2f}", f"{b['p90_dur']:.2f}"),
        ("atomicity (all rules)", f"{100*a['atomicity']:.0f}%", f"{100*b['atomicity']:.0f}%"),
        ("uniqueness", f"{100*a['uniqueness']:.1f}%", f"{100*b['uniqueness']:.1f}%"),
        ("spans inside 1.3-4.0s band", f"{100*a['in_band']:.0f}%", f"{100*b['in_band']:.0f}%"),
        ("overlapping label pairs", f"{a['overlaps']}", f"{b['overlaps']}"),
        ("hand matches measured pose",
         "100% (supplied)", f"{100*b['hand_agree']:.0f}%" if b['hand_agree'] is not None else "-"),
        ("verb matches measured aperture",
         f"{100*a['grounding']:.0f}% (n={a['grounding_n']})" if a['grounding'] is not None else "-",
         f"{100*b['grounding']:.0f}% (n={b['grounding_n']})" if b['grounding'] is not None else "-"),
    ]
    print(f"  {'metric':<32s} {'pose-guided':>16s} {'vision-only':>16s}")
    for label, x, y in rows:
        print(f"  {label:<32s} {x:>16s} {y:>16s}")
    print()
    print(f"  pose-guided failures : {a['fails']}")
    print(f"  vision-only failures : {b['fails']}")
    return dict(pose=a, vision=b)
