"""
Cut annotation spans, and attach the pose-derived facts a captioner should
never be asked to guess.

Boundaries come from a COMBINED activity signal, not wrist speed alone. Wrist
velocity is the wrong cue for fine-motor work: unscrewing a cap barely moves
the wrist while the fingers and forearm do everything, so velocity troughs
land in the middle of the action instead of between actions. The combined
signal is the max of normalised wrist speed, aperture rate and twist rate, so
gross transport and in-hand manipulation both register.

Two things this stage does that the original span builder did not:

* It enforces the linter's duration band (rule A1, 1.3-4.0 s) at cut time.
  Every A1 failure in the previous output was a span-builder artifact -- two
  spans under 1.3 s, four over 4.0 s, one of them 11.6 s -- so the captioner
  was being marked down for boundaries it did not choose. Overlong spans also
  drive repetition: with no single atomic action to describe, the model falls
  back on a generic caption, which is where the most-repeated caption in the
  previous run came from.
* It consults the quality stage. Those records existed but nothing downstream
  ever read them, so blurred and high-motion footage was captioned anyway.

Facts are read from the DOMINANT hand. The original mixed hands per channel:
`fingers` used the dominant hand while `aperture` and `ap_trend` pooled both,
so on a single-hand span the reported aperture range was a median 50 mm wide
(p90 80 mm) -- nearly the whole 10-140 mm physiological range, because it
unioned the idle hand. It was labelled "(measured)" in the prompt and
constrained nothing.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .. import config
from ..core.geometry import quat_to_R
from ..core.mcap_io import read_episode
from ..core.signal import gradient, local_minima, smooth, speed, window

CFG = config.SPANS_CFG

MCP = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}
TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
DIGITS = list(MCP)
THUMB_TIP, INDEX_TIP = 4, 8
MIDDLE_MCP = 9


# ---------------------------------------------------------------- activity
def activity_signal(ep, cfg=CFG):
    """
    (timebase, normalised activity) from wrist speed, aperture rate and twist
    rate, each divided by its own p90 so no channel dominates by units.
    """
    fps = ep["src_fps"]
    win = window(0.15, fps)
    channels, base_t = [], None
    for side in ("right", "left"):
        A = ep[f"/pose/{side}_hand"]
        J = ep[f"/pose/{side}_hand_joints"]
        Q = ep.get(f"/pose/{side}_hand_quat")
        if len(A) < 8:
            continue
        t = A[:, 0]
        if base_t is None:
            base_t = t
        tv, v = speed(t, A[:, 1:4], win)
        channels.append(np.interp(base_t, tv, v))
        aperture = smooth(np.linalg.norm(J[:, THUMB_TIP] - J[:, INDEX_TIP], axis=1), win)
        channels.append(np.interp(base_t, t, np.abs(gradient(aperture, t))))
        if Q is not None and len(Q) > 3:
            axis = J[:, MIDDLE_MCP] - J[:, 0]
            axis = axis / np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-9)
            n = min(len(Q), len(J))
            w = np.zeros(n)
            for i in range(1, n):
                R0, R1 = quat_to_R(*Q[i - 1, 1:5]), quat_to_R(*Q[i, 1:5])
                dR = R0.T @ R1
                om = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0],
                               dR[1, 0] - dR[0, 1]]) / 2.0
                w[i] = abs(float(np.dot(R0 @ om, axis[i]))) / max(
                    Q[i, 0] - Q[i - 1, 0], 1e-6)
            channels.append(np.interp(base_t, Q[:n, 0], smooth(w, win)))
    if not channels or base_t is None:
        return np.zeros(0), np.zeros(0)
    normed = [ch / max(np.percentile(ch, 90), 1e-9) for ch in channels]
    return base_t, smooth(np.maximum.reduce(normed), win)


def signal_for(ep, cfg=CFG):
    """
    (timebase, activity) for whichever boundary signal this run selected.

    All arms share one downstream path -- trough finding, the duration band,
    boundary refinement and the pose facts -- so swapping the signal isolates
    the question of where boundaries should come from.
    """
    kind = cfg.get("signal", "activity")
    if kind == "activity":
        return activity_signal(ep, cfg)
    if kind.startswith("rgb_"):
        from .rgb_boundaries import signal as rgb_signal
        return rgb_signal(ep, kind)
    if kind == "velocity":
        # the velocity arm keeps the combined signal for band/refinement work;
        # only its boundary set differs
        return activity_signal(ep, cfg)
    raise ValueError(f"unknown span signal {kind!r}")


def troughs_for(ep, t, act, cfg=CFG):
    """Boundary times for the selected signal."""
    kind = cfg.get("signal", "activity")
    if kind == "velocity":
        from .events import velocity_troughs
        return velocity_troughs(ep)
    if not len(t):
        return np.zeros(0)
    idx = local_minima(act, cfg["prominence"],
                       min_gap_samples=cfg["min_gap_s"], t=t)
    return t[idx] if len(idx) else np.zeros(0)


def activity_troughs(ep, cfg=CFG):
    """Boundary candidates: prominent minima of the combined activity signal."""
    t, act = activity_signal(ep, cfg)
    if not len(t):
        return np.zeros(0)
    idx = local_minima(act, cfg["prominence"], min_gap_samples=cfg["min_gap_s"], t=t)
    return t[idx] if len(idx) else np.zeros(0)


# ---------------------------------------------------------------- band policy
def _split_long(a, z, t, act, lo, hi, prom):
    """
    Interior cut points that bring (a, z) within the ceiling.

    Cuts at the deepest qualifying activity minimum that leaves both sides at
    or above the floor; if no minimum qualifies, cuts at the plain argmin of
    activity in the legal window. Recurses until every piece fits.
    """
    if z - a <= hi:
        return []
    legal = (t > a + lo) & (t < z - lo)
    if not legal.any():
        return []                      # cannot split without breaking the floor
    idx = np.where(legal)[0]
    minima = [i for i in local_minima(act, prom) if legal[i]]
    k = min(minima, key=lambda i: act[i]) if minima else idx[int(np.argmin(act[idx]))]
    cut = float(t[k])
    return sorted(_split_long(a, cut, t, act, lo, hi, prom) + [cut]
                  + _split_long(cut, z, t, act, lo, hi, prom))


def enforce_band(intervals, t, act, cfg=CFG):
    """
    Bring every span inside the linter's duration band.

    Long spans are subdivided at interior activity minima. Short spans are
    merged into whichever neighbour lands the result closest to the middle of
    the band, and dropped when neither merge is legal -- dropping is allowed
    because labels need not tile (rule A8 forbids overlap, not gaps).
    """
    lo, hi = cfg["band"]
    prom = cfg["prominence"] * cfg["split_prominence_factor"]
    target = (lo + hi) / 2.0

    expanded = []
    n_split = 0
    for a, z in intervals:
        cuts = _split_long(a, z, t, act, lo, hi, prom)
        n_split += len(cuts)
        edges = [a] + cuts + [z]
        expanded += list(zip(edges, edges[1:]))

    kept, n_merged, n_dropped = [], 0, 0
    i = 0
    while i < len(expanded):
        a, z = expanded[i]
        if z - a >= lo:
            kept.append((a, z))
            i += 1
            continue
        back = (kept and abs(kept[-1][1] - a) < 1e-9
                and (z - kept[-1][0]) <= hi)
        fwd = (i + 1 < len(expanded) and abs(expanded[i + 1][0] - z) < 1e-9
               and (expanded[i + 1][1] - a) <= hi)
        choose_back = back and (
            not fwd or abs((z - kept[-1][0]) - target)
            <= abs((expanded[i + 1][1] - a) - target))
        if choose_back:
            kept[-1] = (kept[-1][0], z)
            n_merged += 1
        elif fwd:
            expanded[i + 1] = (a, expanded[i + 1][1])
            n_merged += 1
        else:
            n_dropped += 1
        i += 1
    return kept, dict(split=n_split, merged=n_merged, dropped=n_dropped)


# ---------------------------------------------------------------- refinement
def _minima_times(t, act, prominence):
    """Times of every local minimum of the activity signal meeting prominence."""
    idx = local_minima(np.asarray(act, float), prominence)
    return np.asarray(t, float)[idx] if len(idx) else np.zeros(0)


def refine_boundaries(intervals, t, act, cfg=CFG, minima=None):
    """
    Nudge each shared interior cut onto a nearby quieter activity minimum.

    A cut used to be final the moment it was made, so the captioner had to
    describe whatever the measured boundary contained. The vision-only control
    arm beats this one on verb/aperture agreement precisely because it cuts
    where the action it describes actually occurs. This recovers some of that
    freedom without giving up determinism or the duration band.

    For each boundary, candidates are the original cut plus every local minimum
    of the SAME signal within +/- `boundary_shift_s` that meets the same
    prominence criterion. Selection is by activity level -- the quietest
    instant wins, matching what `_split_long` already does -- with ties broken
    by the smaller move and then by the earlier time, so the choice is fully
    reproducible from the signal. There is no learned scorer here on purpose.

    A candidate is rejected unless BOTH adjoining spans land inside the band,
    and unless neither falls below `min(min_gap_s, that span's original
    duration)` -- so a shift can shorten a span toward the gap floor but never
    past it, and never makes an already-short span shorter.

    Boundaries are visited left to right, each decision reading the
    already-committed span on its left and the not-yet-moved boundary on its
    right, which is what makes the result independent of iteration order.
    Only boundaries SHARED by two spans move: segment endpoints are definition,
    not measurement, and a gap edge abuts material the band policy dropped.

    Returns (intervals, provenance, stats). `provenance[i]` describes the
    boundary between span i-1 and span i, or None where it does not move.
    """
    lo, hi = cfg["band"]
    shift = float(cfg["boundary_shift_s"])
    min_gap = float(cfg["min_gap_s"])
    if not cfg.get("boundary_refine", True) or shift <= 0 or len(intervals) < 2:
        return list(intervals), [None] * (len(intervals) + 1), dict(
            shifted=0, rejected=0, considered=0, total_delta=0.0)
    # Ordering can only be preserved if two boundaries closing on each other
    # cannot cross the shortest legal span.
    if 2 * shift >= lo:
        raise ValueError(
            f"boundary_shift_s={shift} too large for band floor {lo}: "
            f"2*shift must stay below the floor or spans can invert")

    if minima is None:
        minima = _minima_times(t, act, cfg["prominence"])
    t = np.asarray(t, float)
    act = np.asarray(act, float)

    def level(x):
        """Activity at time x, from the nearest sample."""
        if not len(t):
            return float("inf")
        return float(act[int(np.clip(np.searchsorted(t, x), 0, len(t) - 1))])

    cur = [list(iv) for iv in intervals]
    orig_dur = [z - a for a, z in intervals]
    prov = [None] * (len(intervals) + 1)
    stats = dict(shifted=0, rejected=0, considered=0, total_delta=0.0)

    for i in range(1, len(cur)):
        # only a boundary genuinely shared by two spans may move
        if abs(cur[i - 1][1] - cur[i][0]) > 1e-9:
            continue
        origin = float(intervals[i - 1][1])
        left_start = cur[i - 1][0]
        right_end = cur[i][1]

        near = minima[np.abs(minima - origin) <= shift] if len(minima) else np.zeros(0)
        cands = [origin] + [float(x) for x in near if abs(x - origin) > 1e-9]
        stats["considered"] += len(cands)

        floor_left = min(min_gap, orig_dur[i - 1])
        floor_right = min(min_gap, orig_dur[i])

        feasible, why = [], set()
        for c in cands:
            left_dur, right_dur = c - left_start, right_end - c
            if not (lo <= left_dur <= hi and lo <= right_dur <= hi):
                why.add("band")
                continue
            if left_dur < floor_left - 1e-9 or right_dur < floor_right - 1e-9:
                why.add("min_gap")
                continue
            feasible.append(c)

        if not feasible:
            # the original cut is always a candidate, so this only happens when
            # the original itself is out of band -- leave it exactly as cut.
            prov[i] = dict(orig=round(origin, 3), final=round(origin, 3), delta=0.0,
                           n_cand=len(cands), shifted=False,
                           reason="+".join(sorted(why)) or "no_candidate")
            stats["rejected"] += 1
            continue

        best = min(feasible, key=lambda c: (level(c), abs(c - origin), c))
        moved = abs(best - origin) > 1e-9
        if moved:
            cur[i - 1][1] = best
            cur[i][0] = best
            stats["shifted"] += 1
            stats["total_delta"] += abs(best - origin)
        elif len(feasible) > 1:
            # other candidates were legal; the original was simply the quietest
            why = {"original_best"}
        elif len(cands) > 1:
            stats["rejected"] += 1      # every alternative was ruled out
        prov[i] = dict(
            orig=round(origin, 3), final=round(best, 3),
            delta=round(best - origin, 3), n_cand=len(cands), shifted=moved,
            reason=None if moved else ("+".join(sorted(why)) or "no_candidate"))

    return [tuple(iv) for iv in cur], prov, stats


# ---------------------------------------------------------------- quality gate
def rejected_intervals(records, tiers=("T1",)):
    """
    {episode: [(start, end), ...]} for clips the quality stage rejected.

    T1 and T2 are different kinds of verdict and should not be gated alike. T1
    is a defect finding -- blur, exposure, framing, a pose glitch -- and a span
    sitting inside one is genuinely unusable. T2 drops the top 30% of each
    episode by head motion BY CONSTRUCTION, so gating on it removes about a
    third of all spans regardless of whether anything is wrong with them.
    Default to T1; ask for T2 explicitly when the goal is a calm subset.
    """
    out: dict[str, list[tuple[float, float]]] = {}
    for r in records:
        bad = (("T1" in tiers and r.get("hard_reject"))
               or ("T2" in tiers and r.get("motion_reject")))
        if bad:
            out.setdefault(r["episode"], []).append((r["start_ts"], r["end_ts"]))
    return out


def bad_overlap(a, z, intervals):
    """Fraction of [a, z) that falls inside any rejected clip."""
    if not intervals or z <= a:
        return 0.0
    covered = sum(max(0.0, min(z, e) - max(a, s)) for s, e in intervals)
    return covered / (z - a)


# ---------------------------------------------------------------- pose facts
def dominant_hand(ep, a, z, cfg=CFG):
    """LEFT / RIGHT / BOTH, by wrist path length over the span."""
    def path_len(A):
        m = (A[:, 0] >= a) & (A[:, 0] < z)
        return float(np.linalg.norm(np.diff(A[m, 1:4], axis=0), axis=1).sum()) \
            if m.sum() > 2 else 0.0

    left = path_len(ep["/pose/left_hand"])
    right = path_len(ep["/pose/right_hand"])
    ratio = cfg["dominant_ratio"]
    if right > ratio * max(left, 1e-9):
        return "RIGHT"
    if left > ratio * max(right, 1e-9):
        return "LEFT"
    return "BOTH"


def acting_sides(hand):
    """Which hand's signals describe the action, most-likely first."""
    if hand == "LEFT":
        return ["left"]
    if hand == "RIGHT":
        return ["right"]
    return ["right", "left"]


def aperture_stats(ep, side, a, z):
    """
    Grasp aperture for one hand over one span: robust range and end state.

    p10-p90 rather than min-max, because a single tracking spike widened the
    reported range enough to make it uninformative, and the end state is what
    distinguishes a grasp from a release.
    """
    J = ep[f"/pose/{side}_hand_joints"]
    A = ep[f"/pose/{side}_hand"]
    if not len(J):
        return None
    m = (A[:, 0] >= a) & (A[:, 0] < z)
    if m.sum() < 3:
        return None
    ap = np.linalg.norm(J[m][:, THUMB_TIP] - J[m][:, INDEX_TIP], axis=1)
    k = max(2, len(ap) // 4)
    start_med = float(np.median(ap[:k]))
    end_med = float(np.median(ap[-k:]))
    delta = end_med - start_med
    trend = "flat" if abs(delta) < 0.006 else ("closing" if delta < 0 else "opening")
    return dict(
        aperture_mm=[int(round(1000 * np.percentile(ap, 10))),
                     int(round(1000 * np.percentile(ap, 90)))],
        aperture_end_mm=int(round(1000 * end_med)),
        ap_trend=trend,
        ap_delta_mm=int(round(1000 * delta)))


def rotation(ep, side, a, z, cfg=CFG):
    """
    NET accumulated rotation over the span, about the hand's own long axis.

    Unscrewing a cap is sustained same-direction rotation -- tens of degrees
    accumulated -- not a high instantaneous rate, which is why a rate threshold
    missed it (frame-level median is only 0.1-0.3 rad/s). Sign is taken about
    the camera axis so "clockwise" means clockwise as the wearer sees it.
    Coherence (net/total) separates a real twist from a hand jittering back and
    forth: near 1 means sustained one-way rotation, near 0 means wobble.
    """
    Q = ep.get(f"/pose/{side}_hand_quat")
    E = ep["extr"]
    J = ep[f"/pose/{side}_hand_joints"]
    if Q is None or len(Q) < 4 or not len(E) or not len(J):
        return None
    t = Q[:, 0]
    m = np.where((t >= a) & (t < z))[0]
    m = m[m < len(J)]
    if len(m) < 4:
        return None
    axis = J[:, MIDDLE_MCP] - J[:, 0]
    axis = axis / np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-9)
    net_hand = net_cam = total_hand = 0.0
    for i in m[1:]:
        R0, R1 = quat_to_R(*Q[i - 1, 1:5]), quat_to_R(*Q[i, 1:5])
        dR = R0.T @ R1
        om = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0],
                       dR[1, 0] - dR[0, 1]]) / 2.0
        wv = R0 @ om
        step = float(np.dot(wv, axis[i]))
        net_hand += step
        total_hand += abs(step)
        k = int(np.clip(np.searchsorted(E[:, 0], t[i]), 0, len(E) - 1))
        net_cam += float((quat_to_R(*E[k, 4:8]).T @ wv)[2])
    degrees = abs(np.degrees(net_hand))
    coherence = abs(net_hand) / max(total_hand, 1e-9)
    if degrees < cfg["rotation_min_deg"] or coherence < cfg["rotation_min_coherence"]:
        return None
    return "clockwise" if net_cam < 0 else "counter-clockwise"


def finger_state(ep, side, a, z, cfg=CFG):
    """
    Which digits are extended, and whether thumb and index are pinched.

    Only reported when DISTINCTIVE for this episode. An absolute 35 mm pinch
    threshold fired on 67-81% of frames in fine-motor tasks, where the median
    aperture is 26-32 mm -- so it described nothing, and feeding it to the
    captioner on 56% of spans is what produced the most-repeated caption in the
    previous run.
    """
    J = ep[f"/pose/{side}_hand_joints"]
    A = ep[f"/pose/{side}_hand"]
    if not len(J):
        return None
    t = A[:, 0]
    m = (t >= a) & (t < z)
    if m.sum() < 3:
        return None
    scale = np.maximum(np.linalg.norm(J[:, MCP["middle"]] - J[:, 0], axis=1), 1e-9)
    curl = {d: np.linalg.norm(J[:, TIP[d]] - J[:, MCP[d]], axis=1) / scale
            for d in DIGITS}
    extended = [d for d in DIGITS
                if np.median(curl[d][m]) > np.percentile(curl[d], cfg["extended_pct"])]
    ap = np.linalg.norm(J[:, THUMB_TIP] - J[:, INDEX_TIP], axis=1)
    med = float(np.median(ap[m]))
    pinched = (med < cfg["pinch_max_m"]
               and med < np.percentile(ap, cfg["pinch_max_pct"]))
    # mutually exclusive: a hand cannot be pinching AND wrapped around something
    if pinched:
        return "thumb-index pinch"
    if len(extended) >= 4:
        return "whole hand wrapped"
    if set(extended) in ({"index"}, {"index", "thumb"}):
        return "index finger extended"
    return None          # a partial spread of digits is not worth naming


def wrist_speed(ep, sides, a, z):
    out = 0.0
    for side in sides:
        A = ep[f"/pose/{side}_hand"]
        m = (A[:, 0] >= a) & (A[:, 0] < z)
        if m.sum() > 2:
            dt = np.diff(A[m, 0])
            ok = dt > 1e-9
            if ok.any():
                out = max(out, float(np.median(
                    np.linalg.norm(np.diff(A[m, 1:4], axis=0)[ok], axis=1) / dt[ok])))
    return round(out, 3)


# ---------------------------------------------------------------- build
def build(segments=None, out=None, only=None, cfg=CFG,
          quality_records=None, events_file=None):
    """Cut spans for the given segments and write them with their pose facts."""
    out = str(out or config.SPANS)
    segs = segments or json.load(open(config.SEGMENT_DEFS))
    if only:
        wanted = set(only)
        segs = [s for s in segs if s["id"] in wanted]

    if quality_records is None and cfg["quality_gate"]:
        from .quality import load_records
        quality_records = load_records()
    bad_by_ep = rejected_intervals(quality_records or [], cfg["quality_gate_tiers"])

    events_file = str(events_file or config.EVENTS_RECORDS)
    candidates = ([json.loads(l) for l in open(events_file)]
                  if os.path.exists(events_file) else [])

    rows, stats = [], dict(split=0, merged=0, dropped=0, gated=0,
                           shifted=0, rejected=0, considered=0, total_delta=0.0)
    for seg in segs:
        name = os.path.basename(seg["source"]).rsplit(".", 1)[0]
        needs_video = str(cfg.get("signal", "activity")).startswith("rgb_")
        try:
            ep = read_episode(config.episode_path(name), want_video=needs_video)
        except FileNotFoundError as e:
            print("MISSING", e)
            continue
        t, act = signal_for(ep, cfg)
        if not len(t):
            print("no boundary signal for", seg["id"])
            del ep
            continue
        troughs = troughs_for(ep, t, act, cfg)

        t0, t1 = seg["t0"], seg["t1"]
        edges = np.concatenate([[t0], troughs[(troughs > t0) & (troughs < t1)], [t1]])
        intervals = list(zip(edges, edges[1:]))
        if cfg["enforce_band"]:
            intervals, counts = enforce_band(intervals, t, act, cfg)
            for k in counts:
                stats[k] += counts[k]
        else:
            intervals = [(a, z) for a, z in intervals if z - a >= 0.5]

        # Refine before the quality gate, so a gate decision is made on the
        # span that will actually be captioned.
        prov = [None] * (len(intervals) + 1)
        if cfg.get("boundary_refine"):
            intervals, prov, rstats = refine_boundaries(intervals, t, act, cfg)
            for k, v in rstats.items():
                stats[k] += v

        bad = bad_by_ep.get(name, [])
        n_seg = 0
        for k_iv, (a, z) in enumerate(intervals):
            if cfg["quality_gate"] and bad_overlap(a, z, bad) >= cfg["quality_overlap_frac"]:
                stats["gated"] += 1
                continue
            hand = dominant_hand(ep, a, z, cfg)
            sides = acting_sides(hand)
            ap = next((x for x in (aperture_stats(ep, s, a, z) for s in sides)
                       if x is not None), None)
            rot = next((r for r in (rotation(ep, s, a, z, cfg) for s in sides)
                        if r is not None), None)
            fingers = next((f for f in (finger_state(ep, s, a, z, cfg) for s in sides)
                            if f is not None), None)
            rows.append(dict(
                span_id=f"{seg['id']}#{a - t0:07.3f}", segment=seg["id"],
                episode=name, cls=seg["cls"],
                start_ts=round(float(a), 3), end_ts=round(float(z), 3),
                duration=round(float(z - a), 3),
                # pose-derived, authoritative -- never asked of the model
                boundary_signal=cfg.get("signal", "activity"),
                hand=hand, acting_side=sides[0],
                rotation=rot, fingers=fingers,
                aperture_mm=(ap or {}).get("aperture_mm"),
                aperture_end_mm=(ap or {}).get("aperture_end_mm"),
                ap_trend=(ap or {}).get("ap_trend"),
                ap_delta_mm=(ap or {}).get("ap_delta_mm"),
                wrist_speed=wrist_speed(ep, sides, a, z),
                contact_events=[dict(t=round(k["t"] - t0, 3), hand=k["hand"],
                                     type=k["type"])
                                for k in candidates
                                if k["episode"] == name and a <= k["t"] < z],
                # how this span's own boundaries were placed
                start_refine=prov[k_iv], end_refine=prov[k_iv + 1],
                # video-local time, for frame extraction from the segment mp4
                v_start=round(float(a - t0), 3), v_end=round(float(z - t0), 3)))
            n_seg += 1
        del ep

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    report(rows, stats, out)
    return rows


def report(rows, stats=None, out=None):
    if not rows:
        print("no spans built")
        return
    lo, hi = CFG["band"]
    d = np.array([r["duration"] for r in rows])
    print(f"built {len(rows)} spans over {len({r['segment'] for r in rows})} "
          f"segments" + (f" -> {out}" if out else ""))
    if stats:
        print(f"  band policy: {stats['split']} long spans subdivided, "
              f"{stats['merged']} short spans merged, {stats['dropped']} dropped; "
              f"{stats['gated']} spans gated by clip quality")
        if stats.get("considered"):
            n_sh = stats["shifted"]
            mean_d = stats["total_delta"] / n_sh if n_sh else 0.0
            print(f"  refinement : {n_sh} boundaries moved "
                  f"(mean |delta| {mean_d:.3f}s), {stats['rejected']} left in place, "
                  f"{stats['considered']} candidates considered")
    print(f"  duration: median {np.median(d):.2f}s  "
          f"in-band [{lo}, {hi}]: {100 * ((d >= lo) & (d <= hi)).mean():.0f}%  "
          f"({(d < lo).sum()} short, {(d > hi).sum()} long)")
    for key in ("rotation", "fingers", "ap_trend"):
        vals = [r.get(key) for r in rows]
        known = sum(1 for v in vals if v)
        print(f"  {key:9s} populated on {known}/{len(vals)} spans "
              f"({100 * known / len(vals):.0f}%)")
    print(f"  {'segment':<15s} {'class':<15s} {'spans':>5s} {'sec':>6s} "
          f"{'/min':>6s} {'median':>7s}")
    for sid in sorted({r["segment"] for r in rows}):
        group = [r for r in rows if r["segment"] == sid]
        total = sum(x["duration"] for x in group)
        print(f"  {sid:<15s} {group[0]['cls']:<15s} {len(group):5d} {total:6.1f} "
              f"{len(group) / (total / 60):6.1f} "
              f"{np.median([x['duration'] for x in group]):7.2f}")
