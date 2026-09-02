"""
Score the threshold event detector against the human gold set, broken down by
action type.

Greedy one-to-one matching on (hand, type) within a tolerance; unmatched gold
= FN, unmatched candidate = FP.

Coverage caveat, printed by the report itself: gold exists for only some of the
action classes in the segment set. Anything captioned outside those classes has
never been scored against ground truth by any stage of this pipeline.
"""
from __future__ import annotations

import json

import numpy as np

from .. import config

HAND_CODE = {"left": "L", "right": "R"}
TOLERANCES = [0.15, 0.25, 0.50, 1.00]


def match(gold, candidates, tol):
    """Greedy nearest-neighbour, one-to-one, same hand and same type."""
    used = set()
    pairs = []
    for g in sorted(gold, key=lambda x: x["t"]):
        best, best_d = -1, 1e9
        for i, k in enumerate(candidates):
            if i in used or k["hand"] != g["hand"] or k["type"] != g["type"]:
                continue
            d = abs(k["t"] - g["t"])
            if d < best_d:
                best_d, best = d, i
        if best >= 0 and best_d <= tol:
            used.add(best)
            pairs.append((g, candidates[best], candidates[best]["t"] - g["t"]))
    fp = [k for i, k in enumerate(candidates) if i not in used]
    matched = {id(p[0]) for p in pairs}
    fn = [g for g in gold if id(g) not in matched]
    return pairs, fp, fn


def prf(tp, n_fp, n_fn):
    p = tp / (tp + n_fp) if tp + n_fp else float("nan")
    r = tp / (tp + n_fn) if tp + n_fn else float("nan")
    f = 2 * p * r / (p + r) if p + r else float("nan")
    return p, r, f


def evaluate(gold_path=None, candidates_path=None, segment_defs=None):
    gold_all = json.load(open(gold_path or config.GOLD))
    candidates = [json.loads(l) for l in open(candidates_path or config.EVENTS_RECORDS)]
    segs = json.load(open(segment_defs or config.SEGMENT_DEFS))

    done = [g for g in gold_all if g.get("done") and g.get("events")]
    print("=" * 78)
    print(f"GOLD SET - {len(done)} annotated segment(s)")
    total_s = total_e = 0
    for g in done:
        n = len(g["events"])
        d = g["t1"] - g["t0"]
        total_s += d
        total_e += n
        print(f"  {g['segment']:<14s} {g['cls']:<15s} {d:5.1f}s  {n:3d} gold events  "
              f"{n / (d / 60):5.1f}/min")
    if total_s:
        print(f"  {'TOTAL':<14s} {'':<15s} {total_s:5.1f}s  {total_e:3d} total"
              f"          {total_e / (total_s / 60):5.1f}/min")
    missing = sorted({s["cls"] for s in segs} - {g["cls"] for g in done})
    if missing:
        print("  NO GOLD YET for: " + ", ".join(missing))
        print("  -> captions produced for those classes are unscored for correctness")

    rows = []
    for g in done:
        cd = [dict(t=k["t"], hand=HAND_CODE[k["hand"]], type=k["type"],
                   aperture_delta=k["aperture_delta"], wrist_speed=k["wrist_speed"])
              for k in candidates
              if k["episode"] == g["episode"] and g["t0"] <= k["t"] < g["t1"]]
        rows.append((g, cd))

    for tol in TOLERANCES:
        print("\n" + "=" * 78)
        print(f"TOLERANCE +/-{1000 * tol:.0f} ms")
        print(f"{'segment':<14s} {'class':<15s} {'gold':>5s} {'cand':>5s} {'TP':>5s} "
              f"{'FP':>5s} {'FN':>6s} {'prec':>6s} {'rec':>6s} {'offset':>9s}")
        by_class = {}
        for g, cd in rows:
            pairs, fp, fn = match(g["events"], cd, tol)
            p, r, _ = prf(len(pairs), len(fp), len(fn))
            offsets = [d for _, _, d in pairs]
            med = np.median(offsets) if offsets else float("nan")
            print(f"{g['segment']:<14s} {g['cls']:<15s} {len(g['events']):5d} "
                  f"{len(cd):5d} {len(pairs):5d} {len(fp):5d} {len(fn):6d} "
                  f"{100 * p:5.0f}% {100 * r:5.0f}% {1000 * med:+7.0f} ms")
            b = by_class.setdefault(g["cls"], dict(gold=0, cand=0, tp=0, fp=0,
                                                   fn=0, offs=[]))
            b["gold"] += len(g["events"])
            b["cand"] += len(cd)
            b["tp"] += len(pairs)
            b["fp"] += len(fp)
            b["fn"] += len(fn)
            b["offs"] += offsets
        print("-" * 78)
        for cls, b in by_class.items():
            p, r, f = prf(b["tp"], b["fp"], b["fn"])
            med = np.median(b["offs"]) if b["offs"] else float("nan")
            print(f"{'POOLED':<14s} {cls:<15s} {b['gold']:5d} {b['cand']:5d} "
                  f"{b['tp']:5d} {b['fp']:5d} {b['fn']:6d} {100 * p:5.0f}% "
                  f"{100 * r:5.0f}% {1000 * med:+7.0f} ms   F1 {f:.2f}")

    tol = 0.50
    print("\n" + "=" * 78)
    print("BREAKDOWN at +/-500 ms")
    for key, getter in (("event type", lambda x: x["type"]),
                        ("hand", lambda x: x["hand"])):
        agg = {}
        for g, cd in rows:
            pairs, fp, fn = match(g["events"], cd, tol)
            for _, k, d in pairs:
                a = agg.setdefault(getter(k), dict(tp=0, fp=0, fn=0, offs=[]))
                a["tp"] += 1
                a["offs"].append(d)
            for k in fp:
                agg.setdefault(getter(k), dict(tp=0, fp=0, fn=0, offs=[]))["fp"] += 1
            for k in fn:
                agg.setdefault(getter(k), dict(tp=0, fp=0, fn=0, offs=[]))["fn"] += 1
        print(f"  by {key}:")
        for k, a in sorted(agg.items()):
            p, r, _ = prf(a["tp"], a["fp"], a["fn"])
            med = np.median(a["offs"]) if a["offs"] else float("nan")
            print(f"    {k:<10s} TP {a['tp']:3d}  FP {a['fp']:3d}  FN {a['fn']:3d}   "
                  f"prec {100 * p:4.0f}%  rec {100 * r:4.0f}%  "
                  f"offset {1000 * med:+6.0f} ms")

    print("\n" + "=" * 78)
    print("BIMANUAL STRUCTURE (why per-hand matching may understate the detector)")
    for g, cd in rows:
        def partnered(events):
            return sum(1 for a in events
                       if any(b is not a and b["hand"] != a["hand"]
                              and b["type"] == a["type"]
                              and abs(b["t"] - a["t"]) <= 0.20 for b in events))

        gt, ct = g["events"], cd
        print(f"  {g['segment']:<14s} gold: {len(gt):3d} events, {partnered(gt):3d} "
              f"({100 * partnered(gt) / max(len(gt), 1):3.0f}%) have a same-type "
              f"partner on the other hand within 200 ms")
        print(f"  {'':<14s} cand: {len(ct):3d} events, {partnered(ct):3d} "
              f"({100 * partnered(ct) / max(len(ct), 1):3.0f}%) likewise")
