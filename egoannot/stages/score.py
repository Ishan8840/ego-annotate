"""
Score a caption set.

Three groups of metrics, and the distinction between them matters:

  FORMAT      does the caption obey the atomicity rules (the linter)
  DIVERSITY   uniqueness now, and a Heaps projection of uniqueness at scale
  GROUNDING   does the caption agree with what the pose actually measured

The grounding group is new. Format and diversity between them never check that
"white ceramic cup" is a white ceramic cup, so a caption set could score well
while being wrong about the world. Two pose channels give a partial check for
free, and both were already being computed and then discarded:

  * `ap_trend`  -- verbs implying closure (grasp, pinch, lift) should co-occur
    with fingers closing; release and place with fingers opening. The old
    `_ap_trend` docstring said it was "used to score whether the emitted verb
    agrees with what the fingers actually did", but no such scorer existed.
  * `rotation`  -- rule A9 already forbids a caption contradicting measured
    rotation, but the scorer never passed `rotation` into the label, so A9
    could not fire.

This is not a substitute for human review of caption correctness, which the
pipeline still lacks for every action class outside the gold set.
"""
from __future__ import annotations

import collections
import json
import math
import re

import numpy as np

from .. import config
from ..labels import atomicity as AL

# Verb -> the finger-aperture direction the verb implies. Verbs whose aperture
# behaviour is genuinely ambiguous (transport, steady, align, adjust, reach,
# withdraw, scrub, wipe, stir) are deliberately absent: scoring them would
# manufacture agreement or disagreement out of nothing.
CLOSING_VERBS = {"grasp", "pinch", "lift", "squeeze", "pull", "wrap", "press", "poke"}
OPENING_VERBS = {"release", "place", "drop", "pour", "open"}


def _norm(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower().strip())


def score(caps_path=None, band="measured"):
    caps_path = str(caps_path or config.CAPTIONS)
    caps = [json.loads(l) for l in open(caps_path)]
    if not caps:
        print("no captions")
        return {}

    print("=" * 74)
    print(f"SCORED {len(caps)} captions ({caps[0].get('backend', '?')})  {caps_path}")
    packs = collections.Counter(c.get("pack", "?") for c in caps)
    print("  domain packs:", dict(packs))

    # ---------------------------------------------------------- FORMAT
    fails = collections.Counter()
    n_ok = 0
    for c in caps:
        AL.use_domain(c.get("pack") or "retail_shelf")
        label = dict(
            start_ts=c["start_ts"], end_ts=c["end_ts"], text=c["text"],
            verb=c["verb"], noun=c["noun"],
            hand=c["hand"] if c["hand"] in AL.HANDS else "BOTH",
            visibility=c["visibility"], episode=c["episode"],
            uncertain=c.get("uncertain", False),
            # Passing the measured rotation is what activates rule A9.
            rotation=c.get("rotation"))
        errors = [e for e in AL.lint(label, band) if e[0] == "ERROR"]
        if errors:
            for _, code, _ in errors:
                fails[code] += 1
        else:
            n_ok += 1
    AL.use_domain("retail_shelf")
    print("\n  FORMAT")
    print(f"    atomicity: {n_ok}/{len(caps)} pass "
          f"({100 * n_ok / len(caps):.0f}%)")
    if fails:
        print("    failures by rule:", dict(fails.most_common()))

    durations = np.array([c["end_ts"] - c["start_ts"] for c in caps])
    lo, hi = AL.SPAN[band]
    out_of_band = int(((durations < lo) | (durations > hi)).sum())
    print(f"    span band [{lo}, {hi}]s: {len(caps) - out_of_band}/{len(caps)} "
          f"in band ({out_of_band} A1-eligible)")

    # ---------------------------------------------------------- DIVERSITY
    texts = [c["text"] for c in caps]
    unique = len({_norm(t) for t in texts})
    print("\n  DIVERSITY")
    print(f"    uniqueness: {unique} unique / {len(texts)} = "
          f"{100 * unique / len(texts):.1f}%")
    counts = collections.Counter(_norm(t) for t in texts)
    repeated = [(k, v) for k, v in counts.most_common(5) if v > 1]
    if repeated:
        print("    most repeated:")
        for text, n in repeated:
            print(f"       {n:2d}x  {text[:66]}")
    heaps = None
    if unique > 1 and len(texts) > 20:
        sizes = [max(10, len(texts) // 8), len(texts) // 2, len(texts)]
        rng = np.random.default_rng(0)
        points = []
        for n in sizes:
            v = np.mean([len({_norm(t) for t in rng.choice(texts, n, replace=False)})
                         for _ in range(60)])
            points.append((n, v))
        xs = np.log([p[0] for p in points])
        ys = np.log([p[1] for p in points])
        slope, intercept = np.polyfit(xs, ys, 1)
        K, b = math.exp(intercept), slope
        heaps = dict(K=K, b=b)
        print(f"    Heaps fit: types = {K:.3f}*n^{b:.3f} -> projected uniqueness")
        for n in (1000, 10000, 100000):
            print(f"       at n={format(n, ','):<7s} "
                  f"{100 * min(1, K * n ** (b - 1)):.0f}%")

    # ---------------------------------------------------------- GROUNDING
    print("\n  GROUNDING (caption vs what pose measured)")
    agree = conflict = untestable = 0
    conflicts = []
    for c in caps:
        trend, verb = c.get("ap_trend"), c["verb"]
        if trend not in ("closing", "opening"):
            untestable += 1
            continue
        if verb in CLOSING_VERBS:
            implied = "closing"
        elif verb in OPENING_VERBS:
            implied = "opening"
        else:
            untestable += 1
            continue
        if implied == trend:
            agree += 1
        else:
            conflict += 1
            conflicts.append((verb, trend, c["text"]))
    testable = agree + conflict
    if testable:
        print(f"    verb vs measured aperture trend: {agree}/{testable} agree "
              f"({100 * agree / testable:.0f}%), {conflict} conflict, "
              f"{untestable} not testable")
        for verb, trend, text in conflicts[:5]:
            print(f"       verb={verb:<9s} fingers={trend:<8s} {text[:52]}")
    else:
        print(f"    verb vs measured aperture trend: no testable captions "
              f"({untestable} lack a directional verb or a measured trend)")

    n_rot = sum(1 for c in caps if c.get("rotation"))
    rot_named = sum(1 for c in caps if c.get("rotation")
                    and "clockwise" in c["text"].lower())
    a9 = fails.get("A9", 0)
    print(f"    rotation measured on {n_rot}/{len(caps)} spans; "
          f"{rot_named} captions name a direction; {a9} contradict it (rule A9)")

    n_unc = sum(1 for c in caps if c.get("uncertain"))
    print(f"    marked uncertain: {n_unc}/{len(caps)} "
          f"({100 * n_unc / len(caps):.0f}%)")

    # ---------------------------------------------------------- SHAPE
    words = [len(t.split()) for t in texts]
    total = sum(c["end_ts"] - c["start_ts"] for c in caps)
    print("\n  SHAPE")
    print(f"    words: mean {np.mean(words):.1f}  median {np.median(words):.0f}  "
          f"range {min(words)}-{max(words)}")
    print(f"    density: {len(caps) / (total / 60):.1f} labels/min over "
          f"{total / 60:.1f} min captioned")
    print("    by class:", dict(collections.Counter(c["cls"] for c in caps)))
    print("\n  sample captions:")
    for c in caps[:8]:
        print(f"    [{c['cls'][:4]} {c['start_ts']:5.1f}-{c['end_ts']:5.1f}s "
              f"{c['hand']:<5s}] {c['text']}")

    return dict(n=len(caps), atomicity=n_ok / len(caps),
                uniqueness=unique / len(texts),
                grounding_agree=(agree / testable) if testable else None,
                out_of_band=out_of_band, heaps=heaps,
                density=len(caps) / (total / 60), fails=dict(fails))
