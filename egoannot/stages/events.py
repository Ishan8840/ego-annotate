"""
Contact/release events and an actionness signal, from shipped pose only.

Two deliverables:

  1. contact/release events per hand, derived from the 21-joint hand pose:
     grasp aperture (proximity), wrist velocity, and jerk (the acceleration
     discontinuity of impact) used to place the instant of contact.
  2. an actionness signal covering non-contact behaviour, so clips containing
     inspecting/waiting/repositioning are labelled as such rather than looking
     like empty manipulation.

There are no object poses in this corpus and depth is never delivered, so
"proximity" cannot mean hand-to-object. What is shipped is a 21-joint rigged
hand, which gives finger-to-finger proximity: grasp aperture. That is the
primitive used here, and it is a real one -- aperture spans ~1-14 cm.

VALIDATION STATUS: the events are NOT confirmed. Three independent checks were
run and none confirms that detected events coincide with real contact (see
docs/events.md). They are a candidate proposal layer, not ground truth.
"""
from __future__ import annotations

import collections
import json
import math
import os

import numpy as np

from .. import config
from ..core.mcap_io import read_episode
from ..core.signal import angular_speed, gradient, runs, smooth, speed, window

CFG = config.EVENTS

# MediaPipe 21-landmark topology, confirmed from bone-length structure
WRIST = 0
TIPS = [4, 8, 12, 16, 20]      # thumb, index, middle, ring, pinky
THUMB_TIP, INDEX_TIP = 4, 8

# Upper-body joint roles were identified from the data, not assumed: joints 0-7
# form a rigid spine chain (bone-length std 0.00000), and joints 12/13 sit at
# exactly 0.000 m from the shipped left/right hand wrists, which pins 8/9 as
# shoulders and 10/11 as elbows.
SPINE = [0, 1, 2, 3]
L_SHOULDER, R_SHOULDER = 8, 9
L_WRIST_UB, R_WRIST_UB = 12, 13

STATES = ["manipulate", "reach", "reposition", "inspect", "idle"]


# ---------------------------------------------------------------- signals
def hand_signals(ep, side, cfg=CFG):
    """Aperture, grip curl, wrist speed / acceleration / jerk for one hand."""
    J = ep[f"/pose/{side}_hand_joints"]
    A = ep[f"/pose/{side}_hand"]
    if len(J) < 8:
        return None
    t = A[:, 0]
    win = window(cfg["smooth_s"], ep["src_fps"])
    aperture = smooth(np.linalg.norm(J[:, THUMB_TIP] - J[:, INDEX_TIP], axis=1), win)
    curl = smooth(np.linalg.norm(J[:, TIPS] - J[:, WRIST:WRIST + 1],
                                 axis=2).mean(axis=1), win)
    da = gradient(aperture, t)
    tv, v = speed(t, A[:, 1:4], win)
    acc = gradient(v, tv)
    jerk = np.abs(gradient(acc, tv))
    return dict(t=t, aperture=aperture, curl=curl, da=da,
                tv=tv, v=v, acc=acc, jerk=jerk)


def state_levels(aperture, cfg=CFG):
    """
    The "closed" and "open" aperture levels for one hand, and whether the gate
    is usable at all.

    Percentiles alone have no absolute grounding. Measured on this corpus, a
    p30 "closed" level is 68 mm on noodles but 18 mm on d_contactlens, and on
    t_keyboard p30 and p65 sit 7 mm apart -- inside the hand model's own error,
    so the gate there is a coin flip. Absolute caps stop "closed" meaning a
    wide-open hand, and `usable` records when the two levels are too close to
    separate anything.
    """
    closed = float(np.percentile(aperture, cfg["closed_pct"]))
    open_l = float(np.percentile(aperture, cfg["open_pct"]))
    closed = min(closed, cfg["closed_abs_max"])
    open_l = max(open_l, cfg["open_abs_min"])
    return dict(closed=closed, open=open_l,
                margin=round(open_l - closed, 4),
                usable=bool(open_l - closed >= cfg["min_state_margin"]))


def detect_events(sig, side, cfg=CFG, levels=None):
    """
    Aperture-rate runs give candidate closings and openings; each must move the
    aperture by `min_delta`. A contact is then snapped onto the nearest wrist
    jerk peak, which is where the acceleration discontinuity of impact sits.
    A state machine alternates contact -> release so events come in pairs.
    """
    t, ap, da = sig["t"], sig["aperture"], sig["da"]
    thr = max(cfg["da_floor"], float(np.percentile(np.abs(da), cfg["da_pct"])))
    lv = levels or state_levels(ap, cfg)
    closed, open_l, hold = lv["closed"], lv["open"], cfg["state_hold_s"]

    def sustained(k, kind):
        """Does the hand END UP in the claimed state, over the hold window?"""
        m = (t >= t[k]) & (t <= t[k] + hold)
        if not m.any():
            return None
        med = float(np.median(ap[m]))
        return med <= closed if kind == "contact" else med >= open_l

    candidates = []
    for kind, mask in (("contact", da < -thr), ("release", da > thr)):
        for a, b in runs(mask):
            delta = abs(ap[b] - ap[a])
            if delta < cfg["min_delta"]:
                continue
            k = (a + int(np.argmin(da[a:b + 1]))) if kind == "contact" \
                else (a + int(np.argmax(da[a:b + 1])))
            if cfg["require_state"] and lv["usable"] and sustained(k, kind) is not True:
                continue
            m = (t >= t[k]) & (t <= t[k] + hold)
            candidates.append(dict(
                kind=kind, i=k, t=float(t[k]), delta=float(delta),
                rate=float(abs(da[k])),
                ap_held=round(float(np.median(ap[m])), 4),
                closed=round(closed, 4), open_l=round(open_l, 4)))
    candidates.sort(key=lambda c: c["t"])

    # snap contacts to the nearest jerk peak (acceleration discontinuity)
    tv, jerk = sig["tv"], sig["jerk"]
    for c in candidates:
        if c["kind"] != "contact":
            continue
        m = np.abs(tv - c["t"]) <= cfg["jerk_snap_s"]
        if m.any():
            idx = np.where(m)[0]
            j = idx[int(np.argmax(jerk[idx]))]
            c["t_jerk"] = float(tv[j])
            c["jerk"] = float(jerk[j])
            c["snap_ds"] = round(float(tv[j] - c["t"]), 3)
            c["t"] = c["t_jerk"]

    # alternate contact/release with a refractory period
    events, want, last = [], "contact", -1e9
    for c in candidates:
        if c["kind"] != want or c["t"] - last < cfg["min_gap_s"]:
            continue
        vi = int(np.clip(np.searchsorted(sig["tv"], c["t"]), 0, len(sig["v"]) - 1))
        jerk_val = c.get("jerk")
        events.append(dict(
            hand=side, type=c["kind"], t=round(c["t"], 3),
            aperture_delta=round(c["delta"], 4),
            aperture_rate=round(c["rate"], 4),
            aperture_held=c.get("ap_held"),
            closed_level=c.get("closed"), open_level=c.get("open_l"),
            state_gate_usable=lv["usable"], state_gate_margin=lv["margin"],
            wrist_speed=round(float(sig["v"][vi]), 3),
            jerk=round(jerk_val, 2) if jerk_val is not None
            and not math.isnan(jerk_val) else None,
            jerk_snap_ds=c.get("snap_ds")))
        last = c["t"]
        want = "release" if want == "contact" else "contact"
    return events


def grasps(events):
    """Pair consecutive contact/release into held intervals."""
    out = []
    for a, b in zip(events, events[1:]):
        if a["type"] == "contact" and b["type"] == "release":
            out.append(dict(hand=a["hand"], start=a["t"], end=b["t"],
                            duration=round(b["t"] - a["t"], 3)))
    return out


# ---------------------------------------------------------------- actionness
def upper_body_signals(ep, cfg=CFG):
    """Torso translation and per-arm extension, from /pose/upper_body."""
    UB = ep.get("/pose/upper_body_joints")
    if UB is None or len(UB) < 5:
        return None
    t = ep["/pose/upper_body"][:, 0]
    win = window(cfg["smooth_s"], ep["src_fps"])
    torso = UB[:, SPINE].mean(axis=1)
    t_torso, v_torso = speed(t, torso, win)
    ext = {}
    for side, sh, wr in (("right", R_SHOULDER, R_WRIST_UB),
                         ("left", L_SHOULDER, L_WRIST_UB)):
        e = smooth(np.linalg.norm(UB[:, wr] - UB[:, sh], axis=1), win)
        ext[side] = dict(ext=e, rate=gradient(e, t))
    reach_rate = np.maximum(np.abs(ext["right"]["rate"]), np.abs(ext["left"]["rate"]))
    return dict(t=t, torso=torso, t_torso=t_torso, v_torso=v_torso,
                ext=ext, reach_rate=smooth(reach_rate, win))


def features(ep, cfg=CFG):
    """All actionness features interpolated onto one timebase."""
    tb = ep["/pose/right_hand"][:, 0]
    if not len(tb):
        return None

    def onto(t, v):
        return np.interp(tb, t, v) if len(t) else np.zeros(len(tb))

    sr = hand_signals(ep, "right", cfg)
    sl = hand_signals(ep, "left", cfg)
    zero = np.zeros(len(tb))
    v_hand = np.maximum(onto(sr["tv"], sr["v"]) if sr else zero,
                        onto(sl["tv"], sl["v"]) if sl else zero)
    ap_rate = np.maximum(np.abs(onto(sr["t"], sr["da"])) if sr else zero,
                         np.abs(onto(sl["t"], sl["da"])) if sl else zero)
    t_ang, w_ang = angular_speed(ep["extr"][:, 0], ep["extr"][:, 4:8],
                                 window(cfg["smooth_s"], ep["src_fps"])) \
        if len(ep["extr"]) > 2 else (np.zeros(0), np.zeros(0))
    w_head = onto(t_ang, w_ang)

    ub = upper_body_signals(ep, cfg)
    if ub:
        v_torso = onto(ub["t_torso"], ub["v_torso"])
        reach_rate = onto(ub["t"], ub["reach_rate"])
        arm_ext = onto(ub["t"], ub["ext"]["right"]["ext"])
        source = "upper_body"
    else:
        # Hands-only episodes fall back to head translation, which conflates
        # leaning with stepping. The source is recorded per episode so the
        # difference stays visible downstream.
        H = ep["/pose/head"]
        th, vh = speed(H[:, 0], H[:, 1:4], window(cfg["smooth_s"], ep["src_fps"]))
        v_torso = onto(th, vh)
        reach_rate = np.full(len(tb), np.nan)
        arm_ext = np.full(len(tb), np.nan)
        source = "head_translation_fallback"
    return dict(t=tb, v_hand=v_hand, ap_rate=ap_rate, w_head=w_head,
                v_torso=v_torso, reach_rate=reach_rate, arm_ext=arm_ext,
                torso_source=source, has_upper_body=bool(ub))


def _apply_dwell(states, t, dwell):
    """
    Hysteresis: absorb runs shorter than `dwell` into the previous state, then
    re-check, so a merge that creates another short run is also resolved. The
    original single pass left 3% of emitted spans below the dwell minimum
    (shortest 0.08 s) because it advanced past each merge without re-testing.
    """
    for _ in range(len(states)):
        changed = False
        i = 0
        while i < len(states):
            j = i
            while j < len(states) and states[j] == states[i]:
                j += 1
            end_t = t[j] if j < len(t) else t[-1]
            if i > 0 and (end_t - t[i]) < dwell and states[i] != states[i - 1]:
                states[i:j] = states[i - 1]
                changed = True
            i = j
        if not changed:
            break
    return states


def actionness(ep, held, cfg=CFG):
    """
    Per-frame state. Priority matters: holding something is manipulation even
    while the torso moves; scanning shelves with quiet hands is inspecting, not
    idle. `reach` and `inspect` are the non-contact states -- they are what
    stops purposeful non-manipulation from reading as dead time.
    """
    f = features(ep, cfg)
    if f is None:
        return None
    tb = f["t"]
    in_hold = np.zeros(len(tb), bool)
    for g in held:
        in_hold |= (tb >= g["start"]) & (tb <= g["end"])

    rr = f["reach_rate"]
    states = np.empty(len(tb), dtype=object)
    for i in range(len(tb)):
        if in_hold[i] or (f["ap_rate"][i] > cfg["ap_active"]
                          and f["v_hand"][i] > cfg["v_hand_active"]):
            states[i] = "manipulate"
        elif not np.isnan(rr[i]) and rr[i] > cfg["reach_active"]:
            states[i] = "reach"
        elif f["v_torso"][i] > cfg["v_torso_move"]:
            states[i] = "reposition"
        elif f["w_head"][i] > cfg["w_head_scan"]:
            states[i] = "inspect"
        else:
            states[i] = "idle"
    states = _apply_dwell(states, tb, cfg["dwell_s"])

    score = (np.clip(f["v_hand"] / cfg["v_hand_active"], 0, 1) * 0.40
             + np.clip(f["ap_rate"] / cfg["ap_active"], 0, 1) * 0.25
             + np.clip(np.nan_to_num(rr) / cfg["reach_active"], 0, 1) * 0.15
             + np.clip(f["v_torso"] / cfg["v_torso_move"], 0, 1) * 0.10
             + np.clip(f["w_head"] / cfg["w_head_scan"], 0, 1) * 0.10)
    return dict(t=tb, state=states, score=np.clip(score, 0, 1),
                base_source=f["torso_source"], has_upper_body=f["has_upper_body"],
                feats=f)


def state_spans(act, min_s=0.0):
    out, i = [], 0
    t, st = act["t"], act["state"]
    while i < len(st):
        j = i
        while j < len(st) and st[j] == st[i]:
            j += 1
        a = float(t[i])
        b = float(t[min(j, len(t) - 1)])
        if b - a >= min_s:
            out.append(dict(state=st[i], start=round(a, 3), end=round(b, 3),
                            duration=round(b - a, 3),
                            mean_score=round(float(act["score"][i:j].mean()), 3)))
        i = j
    return out


def calibrate_actionness(paths, cfg=CFG):
    """Pool features across episodes so thresholds are percentiles, not guesses."""
    acc = {k: [] for k in ("v_hand", "ap_rate", "w_head", "v_torso", "reach_rate")}
    n_ub = 0
    for path in paths:
        ep = read_episode(path, want_video=False)
        f = features(ep, cfg)
        if f is None:
            continue
        n_ub += bool(f["has_upper_body"])
        for k in acc:
            v = f[k]
            acc[k].append(v[~np.isnan(v)])
        del ep
    print("=" * 78)
    print(f"ACTIONNESS CALIBRATION  {len(paths)} episodes ({n_ub} with upper body)")
    for k, v in acc.items():
        if not v:
            continue
        a = np.concatenate(v)
        if not len(a):
            print(f"  {k:12s} (no data)")
            continue
        print(f"  {k:12s} " + "  ".join(
            f"p{q}={np.percentile(a, q):.3f}" for q in (50, 60, 70, 75, 80, 90, 95)))
    return acc


# ---------------------------------------------------------------- validation
def boundary_alignment(event_times, boundaries, duration, rng=None, n_null=400):
    """
    Do detected events land near reference boundaries more than chance?

    Null model: the same number of events placed uniformly at random. Reports
    median distance for real vs null and the fraction of null draws that beat
    the real alignment (an empirical p-value).
    """
    if len(event_times) == 0 or len(boundaries) == 0:
        return None
    ev = np.asarray(event_times, float)
    sb = np.asarray(boundaries, float)

    def med(e):
        return float(np.median(np.min(np.abs(sb[:, None] - e[None, :]), axis=1)))

    real = med(ev)
    rng = rng or np.random.default_rng(7)
    null = np.array([med(np.sort(rng.uniform(0, duration, len(ev))))
                     for _ in range(n_null)])
    return dict(median_dist_real=round(real, 3),
                median_dist_null=round(float(np.median(null)), 3),
                p_empirical=round(float((null <= real).mean()), 4),
                n_events=len(ev), n_boundaries=len(sb))


def velocity_troughs(ep, prom=0.06, min_gap=1.5, cfg=CFG):
    """
    Prominent minima of smoothed wrist speed.

    Used here only as an INDEPENDENT check on the aperture detector -- aperture
    comes from finger joint angles, troughs from wrist translation, so
    agreement is evidence rather than tautology. (Both are read off the same
    fitted hand model, which is the residual caveat.)
    """
    from ..core.signal import local_minima
    win = window(cfg["smooth_s"], ep["src_fps"])
    series = []
    for side in ("right", "left"):
        A = ep[f"/pose/{side}_hand"]
        if len(A) < 5:
            continue
        series.append(speed(A[:, 0], A[:, 1:4], win))
    if not series:
        return np.zeros(0)
    tv = series[0][0]
    v = smooth(np.maximum.reduce([np.interp(tv, t, x) for t, x in series]), win)
    idx = local_minima(v, prom, min_gap_samples=min_gap, t=tv)
    return tv[idx] if len(idx) else np.zeros(0)


def episode_analysis(path, cfg=CFG):
    ep = read_episode(path, want_video=False)
    name = ep["name"]
    duration = ep["duration_s"]

    events = []
    gates = {}
    for side in ("right", "left"):
        sig = hand_signals(ep, side, cfg)
        if sig:
            lv = state_levels(sig["aperture"], cfg)
            gates[side] = lv
            events += detect_events(sig, side, cfg, levels=lv)
    events.sort(key=lambda e: e["t"])

    held = (grasps([e for e in events if e["hand"] == "right"])
            + grasps([e for e in events if e["hand"] == "left"]))
    act = actionness(ep, held, cfg)
    spans = state_spans(act) if act else []

    segs = ep["segs"]
    bounds = sorted({round(x, 3) for a, b, _ in segs for x in (a, b)
                     if 0.5 < x < duration - 0.5})          # interior only
    align = boundary_alignment([e["t"] for e in events], bounds, duration)
    # cross-signal check: UNSNAPPED aperture events vs wrist-velocity troughs
    raw_t = [round(e["t"] - (e["jerk_snap_ds"] or 0.0), 3) for e in events]
    troughs = velocity_troughs(ep, cfg=cfg)
    cross = boundary_alignment(raw_t, troughs, duration)

    budget = {}
    if act:
        for s in STATES:
            budget[s] = round(float((act["state"] == s).sum()) / len(act["state"]), 4)

    summary = dict(
        episode=name, duration_s=round(duration, 2), src_fps=ep["src_fps"],
        paradigm=ep["meta"].get("paradigm"), task=ep["meta"].get("task_name"),
        n_events=len(events),
        events_per_min=round(len(events) / (duration / 60), 2) if duration else 0,
        n_grasps=len(held),
        grasp_dur_median=round(float(np.median([g["duration"] for g in held])), 3)
        if held else None,
        state_gates={k: v for k, v in gates.items()},
        base_source=act["base_source"] if act else None,
        state_budget=budget,
        mean_actionness=round(float(act["score"].mean()), 3) if act else None,
        n_shipped_segments=len(segs), alignment=align,
        n_troughs=int(len(troughs)), cross_signal=cross)
    records = [dict(episode=name, **e) for e in events]
    return records, spans, summary, held


def measure(paths, out=None, cfg=CFG, write=True):
    out = str(out or config.EVENTS_RECORDS)
    all_events, all_spans, summaries = [], [], []
    for i, path in enumerate(paths, 1):
        try:
            ev, sp, s, _ = episode_analysis(path, cfg)
            all_events += ev
            all_spans += [dict(episode=s["episode"], **x) for x in sp]
            summaries.append(s)
            al = s["alignment"] or {}
            print(f"[{i}/{len(paths)}] {s['episode']:18s} {s['duration_s']:6.1f}s  "
                  f"{s['n_events']:3d} ev ({s['events_per_min']:5.2f}/min)  "
                  f"{s['n_grasps']:3d} grasps  "
                  f"manip={s['state_budget'].get('manipulate', 0):.2f} "
                  f"insp={s['state_budget'].get('inspect', 0):.2f} "
                  f"idle={s['state_budget'].get('idle', 0):.2f} "
                  f"repo={s['state_budget'].get('reposition', 0):.2f}  "
                  f"align={al.get('median_dist_real', '-')}s vs null "
                  f"{al.get('median_dist_null', '-')}s "
                  f"p={al.get('p_empirical', '-')}", flush=True)
        except Exception as e:
            print(f"[{i}/{len(paths)}] FAIL {path}: {type(e).__name__}: {e}", flush=True)

    if write:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as fh:
            for e in all_events:
                fh.write(json.dumps(e) + "\n")
        with open(out.replace(".jsonl", ".spans.jsonl"), "w") as fh:
            for x in all_spans:
                fh.write(json.dumps(x) + "\n")
        with open(out.replace(".jsonl", ".episodes.json"), "w") as fh:
            json.dump(summaries, fh, indent=1)
        print(f"\nwrote {len(all_events)} events, {len(all_spans)} state spans -> {out}")
    report(summaries)
    return all_events, all_spans, summaries


def report(summaries):
    if not summaries:
        return
    print("=" * 78)
    total_d = sum(s["duration_s"] for s in summaries)
    total_e = sum(s["n_events"] for s in summaries)
    print(f"{len(summaries)} episodes, {total_d / 60:.1f} min, {total_e} events "
          f"({total_e / (total_d / 60):.2f}/min), "
          f"{sum(s['n_grasps'] for s in summaries)} grasp intervals")
    gd = [s["grasp_dur_median"] for s in summaries if s["grasp_dur_median"]]
    if gd:
        print(f"  median grasp duration across episodes: {np.median(gd):.2f}s "
              f"(range {min(gd):.2f}-{max(gd):.2f})")
    print("  time budget (duration-weighted):")
    for st in STATES:
        w = sum(s["state_budget"].get(st, 0) * s["duration_s"]
                for s in summaries) / total_d
        print(f"    {st:12s} {100 * w:5.1f}%")

    unusable = [s["episode"] for s in summaries
                if any(not g["usable"] for g in s.get("state_gates", {}).values())]
    if unusable:
        print(f"  state gate unusable (closed/open levels too close) on "
              f"{len(unusable)} episode(s): {', '.join(unusable[:6])}")

    print("  cross-signal: aperture events vs independent wrist-velocity troughs")
    p_values = []
    for s in summaries:
        c = s.get("cross_signal")
        if c:
            print(f"    {s['episode']:18s} real {c['median_dist_real']:6.3f}s   "
                  f"null {c['median_dist_null']:6.3f}s   p={c['p_empirical']:.3f}   "
                  f"({c['n_events']} ev vs {c['n_boundaries']} troughs)")
            p_values.append(c["p_empirical"])
    if p_values:
        print(f"    episodes beating null at p<0.05: "
              f"{sum(1 for p in p_values if p < 0.05)}/{len(p_values)}")
        # per-episode tests are underpowered (2-120 events); pool with a sign test
        pairs = [(s["cross_signal"]["median_dist_real"],
                  s["cross_signal"]["median_dist_null"])
                 for s in summaries if s.get("cross_signal")]
        better = sum(1 for r, n in pairs if r < n)
        k, N = better, len(pairs)
        p_sign = sum(math.comb(N, i) for i in range(k, N + 1)) / (2 ** N)
        print(f"    POOLED sign test: {k}/{N} episodes closer than null, "
              f"binomial p={p_sign:.3f} -> "
              f"{'significant' if p_sign < 0.05 else 'NOT significant'}")
    print("  shipped coarse-label boundary alignment vs uniform-random null:")
    for s in summaries:
        a = s["alignment"]
        if a:
            print(f"    {s['episode']:18s} real {a['median_dist_real']:6.3f}s   "
                  f"null {a['median_dist_null']:6.3f}s   p={a['p_empirical']:.3f}")
    ps = [s["alignment"]["p_empirical"] for s in summaries if s["alignment"]]
    if ps:
        print(f"    episodes beating null at p<0.05: "
              f"{sum(1 for p in ps if p < 0.05)}/{len(ps)}")
    print("  base-motion source:",
          dict(collections.Counter(s["base_source"] for s in summaries)))
