"""
Dense video captioning metrics, against the references the corpus ships.

Until now the scorer measured format (does a caption obey the rules), diversity
(is it different from its neighbours) and grounding (does its verb agree with
measured finger motion). None of those asks whether the caption is *true*, and
the pipeline said so but treated it as an open problem. It is not: dense video
captioning has a standard protocol, and this implements it.

  localisation   segments matched to references at IoU in {0.3, 0.5, 0.7, 0.9};
                 precision, recall and F1 at each
  caption        CIDEr, BLEU-4 and METEOR over the matched pairs
  SODA_c         a monotonic alignment over the whole sequence, so a set of
                 individually plausible captions in the wrong order scores
                 worse than one that tells the story in sequence

READ THE GRANULARITY WARNING. The references are the `semantic_segments` the
episodes ship: human-written, timestamped, but *subtask*-level -- about 10 per
minute against the 26-29 atomic labels per minute this pipeline emits. Strict
IoU matching therefore penalises the pipeline for being denser than its
reference, which is the thing it is built to be. These numbers compare arms
against each other on equal terms; they are not an absolute grade, and the
reported `granularity` ratio is there to keep that visible.
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np

from .. import config

IOU_THRESHOLDS = (0.3, 0.5, 0.7, 0.9)


# ---------------------------------------------------------------- references
def load_references(episodes=None, segments=None):
    """
    {segment_id: [(start, end, text), ...]} from the shipped semantic segments,
    clipped to each segment's window and expressed in segment-local time.
    """
    from ..core.mcap_io import read_episode

    segs = segments or json.load(open(config.SEGMENT_DEFS))
    if episodes:
        segs = [s for s in segs if s["id"] in set(episodes)]
    out = {}
    for seg in segs:
        name = os.path.basename(seg["source"]).rsplit(".", 1)[0]
        try:
            ep = read_episode(config.episode_path(name), want_video=False)
        except FileNotFoundError:
            continue
        t0, t1 = seg["t0"], seg["t1"]
        rows = []
        for a, b, text in ep["segs"]:
            a, b = max(a, t0), min(b, t1)
            if b - a > 0.2 and text:
                rows.append((float(a), float(b), str(text)))
        out[seg["id"]] = sorted(rows)
        del ep
    return out


# ---------------------------------------------------------------- matching
def iou(a, b):
    """Temporal IoU of two (start, end) intervals."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def match(preds, refs, threshold):
    """
    Greedy one-to-one matching by descending IoU.

    Returns (pairs, n_pred, n_ref) where each pair is (pred_index, ref_index).
    """
    scored = sorted(
        ((iou(p[:2], r[:2]), i, j) for i, p in enumerate(preds)
         for j, r in enumerate(refs)),
        key=lambda x: (-x[0], x[1], x[2]))
    used_p, used_r, pairs = set(), set(), []
    for score, i, j in scored:
        if score < threshold or score <= 0:
            break
        if i in used_p or j in used_r:
            continue
        used_p.add(i)
        used_r.add(j)
        pairs.append((i, j))
    return pairs, len(preds), len(refs)


# ---------------------------------------------------------------- SODA
def soda_c(preds, refs, pair_score):
    """
    SODA_c: the best monotonic alignment between prediction and reference
    sequences, scored by caption similarity weighted by temporal IoU.

    Monotonic means an alignment cannot cross itself, so getting the right
    captions in the wrong order costs you -- which is the point of the metric
    and what separates it from matching each segment independently.
    """
    n, m = len(preds), len(refs)
    if n == 0 or m == 0:
        return 0.0, 0.0, 0.0
    score = np.zeros((n, m))
    for i, p in enumerate(preds):
        for j, r in enumerate(refs):
            ov = iou(p[:2], r[:2])
            if ov > 0:
                score[i, j] = ov * pair_score(i, j)

    # DP over a monotonic alignment: match (i,j), or skip either side
    dp = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = max(dp[i - 1, j], dp[i, j - 1],
                           dp[i - 1, j - 1] + score[i - 1, j - 1])
    total = float(dp[n, m])
    precision = total / n
    recall = total / m
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0.0)
    return precision, recall, f1


# ---------------------------------------------------------------- evaluation
class _quiet:
    """pycocoevalcap prints internal diagnostics to stdout; keep them out."""

    def __enter__(self):
        import contextlib, io
        self._buf = io.StringIO()
        self._ctx = contextlib.redirect_stdout(self._buf)
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


def _caption_scores(hyps, refs_for_hyp):
    """CIDEr / BLEU-4 / METEOR over aligned caption lists."""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    if not hyps:
        return dict(CIDEr=float("nan"), BLEU4=float("nan"), METEOR=float("nan"))
    gts = {str(i): [{"caption": r}] for i, r in enumerate(refs_for_hyp)}
    res = {str(i): [{"caption": h}] for i, h in enumerate(hyps)}
    tok = PTBTokenizer()
    try:
        gts, res = tok.tokenize(gts), tok.tokenize(res)
    except Exception:                       # tokenizer needs java; fall back
        gts = {k: [v[0]["caption"].lower()] for k, v in gts.items()}
        res = {k: [v[0]["caption"].lower()] for k, v in res.items()}
    out = {}
    with _quiet():
        try:
            out["CIDEr"] = float(Cider().compute_score(gts, res)[0])
        except Exception:
            out["CIDEr"] = float("nan")
        try:
            out["BLEU4"] = float(Bleu(4).compute_score(gts, res)[0][3])
        except Exception:
            out["BLEU4"] = float("nan")
        try:
            out["METEOR"] = float(Meteor().compute_score(gts, res)[0])
        except Exception:
            out["METEOR"] = float("nan")
    return out


def evaluate(caps_path=None, references=None, thresholds=IOU_THRESHOLDS,
             label=None):
    """Score one caption file against the shipped references."""
    caps_path = str(caps_path or config.CAPTIONS)
    caps = [json.loads(l) for l in open(caps_path)]
    if not caps:
        print("no captions in", caps_path)
        return {}

    by_seg = collections.defaultdict(list)
    for c in caps:
        by_seg[c["segment"]].append(c)
    refs = references if references is not None else load_references(
        episodes=list(by_seg))

    # segment-local intervals on both sides
    seg_t0 = {s["id"]: s["t0"] for s in json.load(open(config.SEGMENT_DEFS))}
    preds = {}
    for sid, group in by_seg.items():
        t0 = seg_t0.get(sid, 0.0)
        preds[sid] = sorted((c["start_ts"] - t0, c["end_ts"] - t0, c["text"])
                            for c in group)

    n_pred = sum(len(v) for v in preds.values())
    n_ref = sum(len(refs.get(k, [])) for k in preds)
    print("=" * 78)
    print("DENSE CAPTIONING vs shipped reference segments"
          + (f"  [{label}]" if label else ""))
    print(f"  {caps_path}")
    print(f"  {n_pred} predicted labels vs {n_ref} reference segments "
          f"across {len(preds)} clips")
    if n_ref:
        print(f"  granularity: predictions are {n_pred / n_ref:.1f}x denser "
              f"than the reference -- strict IoU penalises that, so read these "
              f"across arms, not as an absolute grade")

    results = {"n_pred": n_pred, "n_ref": n_ref,
               "granularity": n_pred / n_ref if n_ref else float("nan")}
    print(f"\n  {'IoU':>5s} {'TP':>4s} {'prec':>6s} {'rec':>6s} {'F1':>6s} "
          f"{'CIDEr':>7s} {'BLEU4':>7s} {'METEOR':>7s}")
    for thr in thresholds:
        hyps, gold, tp, tot_p, tot_r = [], [], 0, 0, 0
        for sid, plist in preds.items():
            rlist = refs.get(sid, [])
            if not rlist:
                continue
            pairs, np_, nr_ = match(plist, rlist, thr)
            tp += len(pairs)
            tot_p += np_
            tot_r += nr_
            for i, j in pairs:
                hyps.append(plist[i][2])
                gold.append(rlist[j][2])
        prec = tp / tot_p if tot_p else 0.0
        rec = tp / tot_r if tot_r else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        cap = _caption_scores(hyps, gold)
        results[thr] = dict(tp=tp, precision=prec, recall=rec, f1=f1, **cap)
        print(f"  {thr:5.2f} {tp:4d} {100 * prec:5.0f}% {100 * rec:5.0f}% "
              f"{100 * f1:5.0f}% {cap['CIDEr']:7.3f} {cap['BLEU4']:7.3f} "
              f"{cap['METEOR']:7.3f}")

    # SODA_c, over the whole sequence per clip
    from pycocoevalcap.meteor.meteor import Meteor
    meteor = Meteor()

    def clip_soda(plist, rlist):
        def pair_score(i, j):
            gts = {"0": [plist[i][2].lower()]}
            res = {"0": [rlist[j][2].lower()]}
            with _quiet():
                try:
                    return float(meteor.compute_score(res, gts)[0])
                except Exception:
                    return 0.0
        return soda_c(plist, rlist, pair_score)

    soda = [clip_soda(plist, refs.get(sid, []))
            for sid, plist in preds.items() if refs.get(sid)]
    if soda:
        p, r, f = (float(np.mean([x[k] for x in soda])) for k in range(3))
        results["soda_c"] = dict(precision=p, recall=r, f1=f)
        print(f"\n  SODA_c (sequence-level, METEOR x IoU): "
              f"precision {p:.3f}  recall {r:.3f}  F1 {f:.3f}")
    return results


def compare(paths, references=None):
    """Score several caption files against one reference set, side by side."""
    refs = references
    rows = {}
    for label, path in paths.items():
        if refs is None:
            caps = [json.loads(l) for l in open(path)]
            refs = load_references(episodes=list({c["segment"] for c in caps}))
        rows[label] = evaluate(path, references=refs, label=label)
        print()
    print("=" * 78)
    print("SUMMARY")
    print(f"  {'arm':<16s} {'labels':>7s} {'F1@0.3':>8s} {'F1@0.5':>8s} "
          f"{'CIDEr@0.3':>10s} {'METEOR@0.3':>11s} {'SODA_c':>8s}")
    for label, r in rows.items():
        if not r:
            continue
        print(f"  {label:<16s} {r['n_pred']:7d} {100 * r[0.3]['f1']:7.0f}% "
              f"{100 * r[0.5]['f1']:7.0f}% {r[0.3]['CIDEr']:10.3f} "
              f"{r[0.3]['METEOR']:11.3f} {r['soda_c']['f1']:8.3f}")
    return rows
