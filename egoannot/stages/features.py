"""
Pose-only per-frame features, and the univariate check on them.

Replaces a single thumb-index aperture with:

  * per-finger curl, all five digits, MCP->tip over hand scale
  * a 1-D closure score: PC1 of the full 21-joint configuration, canonicalised
    into a hand-local frame first so the PCA describes SHAPE, not orientation
  * wrist velocity / acceleration / jerk, with jerk as the primary contact
    observable (impulse is a consequence of contact; finger closure is a proxy
    for intent)
  * per-episode percentile normalisation of every channel
  * validity flags, propagated and never imputed

MediaPipe topology (verified from bone-length structure): 0 wrist; 1-4 thumb;
5-8 index; 9-12 middle; 13-16 ring; 17-20 pinky.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .. import config
from ..core.mcap_io import read_episode
from ..core.signal import gradient, pct_rank, speed, window

CFG = config.FEATURES

WRIST = 0
MCP = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}
TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
DIGITS = ["thumb", "index", "middle", "ring", "pinky"]

FEATURE_NAMES = (["curl_" + d for d in DIGITS]
                 + ["aperture", "closure_pc1", "closure_pc2",
                    "wrist_v", "wrist_a", "wrist_jerk"])


def canonical(J):
    """
    Express all 21 joints in a hand-local frame so shape is separated from pose.

    origin = wrist; x -> index MCP; z = palm normal; y = z x x. Scaled by
    wrist->middle-MCP so different operators are comparable. Returns
    (N, 21, 3) canonical coords and (N,) hand scale.
    """
    w = J[:, WRIST]
    x = J[:, MCP["index"]] - w
    u = J[:, MCP["pinky"]] - w
    scale = np.linalg.norm(J[:, MCP["middle"]] - w, axis=1)
    scale = np.where(scale < 1e-6, 1.0, scale)
    ex = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    z = np.cross(ex, u)
    ez = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-9)
    ey = np.cross(ez, ex)
    R = np.stack([ex, ey, ez], axis=1)                 # (N,3,3), rows = basis
    rel = J - w[:, None, :]
    can = np.einsum("nij,nkj->nki", R, rel) / scale[:, None, None]
    return can, scale


def curls(J, scale):
    """MCP->tip distance over hand scale, per digit."""
    return {"curl_" + d: np.linalg.norm(J[:, TIP[d]] - J[:, MCP[d]], axis=1) / scale
            for d in DIGITS}


def kinematics(t, P, fps, cfg=CFG):
    """Wrist speed / acceleration / |jerk|, interpolated back onto t."""
    win = window(cfg["smooth_s"], fps)
    tv, v = speed(t, P, win)
    if not len(tv):
        z = np.zeros(len(t))
        return z, z, z
    a = gradient(v, tv)
    j = np.abs(gradient(a, tv))
    return (np.interp(t, tv, v), np.interp(t, tv, a), np.interp(t, tv, j))


def episode_features(path, pca=None, cfg=CFG):
    ep = read_episode(path, want_video=False)
    fps = ep["src_fps"]
    bad = {round(b[0], 3): b[1] for b in ep["badf"]}
    out = {}
    for side in ("left", "right"):
        J = ep[f"/pose/{side}_hand_joints"]
        A = ep[f"/pose/{side}_hand"]
        if len(J) < 8:
            continue
        t = A[:, 0]
        can, scale = canonical(J)
        f = {"t": t}
        f.update(curls(J, scale))
        f["aperture"] = np.linalg.norm(J[:, TIP["thumb"]] - J[:, TIP["index"]], axis=1)
        f["hand_scale"] = scale
        v, a, jk = kinematics(t, A[:, 1:4], fps, cfg)
        f["wrist_v"], f["wrist_a"], f["wrist_jerk"] = v, a, jk
        # closure: PC1 of canonical joints (wrist row dropped -- always origin)
        X = can[:, 1:, :].reshape(len(can), -1)
        if pca is not None:
            Z = (X - pca["mean"]) @ pca["comp"].T
            f["closure_pc1"] = Z[:, 0] * pca["sign"]
            f["closure_pc2"] = Z[:, 1]
        # validity, propagated -- never imputed
        f["flag_nan"] = (~np.isfinite(X).all(axis=1)).astype(float)
        f["flag_vel"] = (v > cfg["max_wrist_speed"]).astype(float)
        f["flag_pose_bad"] = np.array(
            [1.0 if bad.get(round(x, 3), False) else 0.0 for x in t])
        for k in list(f):
            if k == "t" or k.startswith("flag_"):
                continue
            f["n_" + k] = pct_rank(f[k])
        out[side] = f
    meta = dict(episode=ep["name"], src_fps=fps, duration_s=ep["duration_s"],
                task=ep["meta"].get("task_name"),
                paradigm=ep["meta"].get("paradigm"))
    del ep
    return meta, out


def fit(paths, out=None, cfg=CFG):
    """Global closure PCA, so PC1 means the same thing in every episode."""
    out = str(out or config.CLOSURE_PCA)
    blocks = []
    for path in paths:
        ep = read_episode(path, want_video=False)
        for side in ("left", "right"):
            J = ep[f"/pose/{side}_hand_joints"]
            if len(J) < 8:
                continue
            can, _ = canonical(J)
            X = can[:, 1:, :].reshape(len(can), -1)
            X = X[np.isfinite(X).all(axis=1)]
            step = max(1, len(X) // cfg["fit_frames_per_hand"])
            blocks.append(X[::step])
        del ep
    X = np.concatenate(blocks, 0)
    mean = X.mean(0)
    _, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    var = S ** 2 / (len(X) - 1)
    ratio = var / var.sum()
    comp = Vt[:cfg["pca_components"]]
    # Orient PC1 so higher = more closed, against a mean-curl proxy
    # (mean MCP->tip distance in canonical space; smaller = more closed).
    proxy = np.mean([np.linalg.norm(
        X[:, 3 * (TIP[d] - 1):3 * (TIP[d] - 1) + 3]
        - X[:, 3 * (MCP[d] - 1):3 * (MCP[d] - 1) + 3], axis=1) for d in DIGITS], axis=0)
    z1 = (X - mean) @ comp[0]
    sign = -1.0 if np.corrcoef(z1, proxy)[0, 1] > 0 else 1.0
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(out, mean=mean, comp=comp, sign=sign,
             ratio=ratio[:cfg["pca_components"]])
    print(f"closure PCA fitted on {len(X)} frames x {X.shape[1]} dims "
          f"({len(paths)} episodes) -> {out}")
    print("  explained variance PC1..PC%d: " % cfg["pca_components"]
          + "  ".join(f"{r:.3f}" for r in ratio[:cfg["pca_components"]]))
    print(f"  PC1 sign chosen so higher = more closed (sign={sign:+.0f})")


def build(paths, outdir=None, cfg=CFG):
    outdir = str(outdir or config.FEATURES_DIR)
    os.makedirs(outdir, exist_ok=True)
    pca_path = str(config.CLOSURE_PCA)
    if not os.path.exists(pca_path):
        raise SystemExit(f"no closure PCA at {pca_path} - run `features fit` first")
    pca = dict(np.load(pca_path))
    metas = []
    for i, path in enumerate(paths, 1):
        meta, F = episode_features(path, pca, cfg)
        np.savez_compressed(
            os.path.join(outdir, meta["episode"] + ".npz"),
            **{f"{s}__{k}": v for s, f in F.items() for k, v in f.items()})
        n_bad = sum(int(f["flag_pose_bad"].sum()) for f in F.values())
        n_nan = sum(int(f["flag_nan"].sum()) for f in F.values())
        n_vel = sum(int(f["flag_vel"].sum()) for f in F.values())
        metas.append(dict(meta, hands=list(F),
                          frames=int(len(next(iter(F.values()))["t"])),
                          flag_pose_bad=n_bad, flag_nan=n_nan, flag_vel=n_vel))
        print(f"[{i:2d}/{len(paths)}] {meta['episode']:<16s} {metas[-1]['frames']:5d} "
              f"frames x {len(F)} hands  flags: pose_bad {n_bad}  nan {n_nan}  "
              f"vel {n_vel}", flush=True)
    json.dump(metas, open(os.path.join(outdir, "index.json"), "w"), indent=1)
    print(f"\nwrote {len(metas)} episodes -> {outdir}")


# ---------------------------------------------------------------- AUC check
def _auc(pos, neg):
    """Rank AUC with tie correction."""
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) < 5 or len(neg) < 5:
        return float("nan")
    a = np.concatenate([pos, neg])
    order = np.argsort(a)
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    # average ranks within tie groups
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def auc(outdir=None, gold_path=None, tol=0.15):
    """
    Univariate discriminability of each feature at gold event times.

    Not a classifier -- this is the sanity check that the jerk-as-primary
    hypothesis is supported before anything downstream spends effort on it.
    """
    outdir = str(outdir or config.FEATURES_DIR)
    gold = json.load(open(gold_path or config.GOLD))
    hand_map = {"L": "left", "R": "right"}
    rows = {}
    for g in gold:
        path = os.path.join(outdir, g["episode"] + ".npz")
        if not os.path.exists(path):
            print("missing features for", g["episode"])
            continue
        Z = np.load(path)
        for etype in ("contact", "release"):
            for code, side in hand_map.items():
                key = f"{side}__t"
                if key not in Z:
                    continue
                t = Z[key]
                in_seg = (t >= g["t0"]) & (t < g["t1"])
                times = [e["t"] for e in g["events"]
                         if e["type"] == etype and e["hand"] == code]
                if not times:
                    continue
                near = np.zeros(len(t), bool)
                for x in times:
                    near |= np.abs(t - x) <= tol
                pos_m, neg_m = near & in_seg, (~near) & in_seg
                for name in FEATURE_NAMES:
                    k = f"{side}__n_{name}"
                    if k not in Z:
                        continue
                    d = rows.setdefault((g["cls"], etype, name), dict(pos=[], neg=[]))
                    d["pos"].append(Z[k][pos_m])
                    d["neg"].append(Z[k][neg_m])

    print("=" * 78)
    print(f"UNIVARIATE AUC at gold event times (+/-{1000 * tol:.0f} ms), "
          f"percentile-normalised features")
    print("AUC 0.5 = no signal; distance from 0.5 is what matters (either direction)")
    for cls in sorted({k[0] for k in rows}):
        for etype in ("contact", "release"):
            sel = {k[2]: v for k, v in rows.items() if k[0] == cls and k[1] == etype}
            if not sel:
                continue
            scored = []
            for name, d in sel.items():
                a = _auc(np.concatenate(d["pos"]), np.concatenate(d["neg"]))
                scored.append((abs(a - 0.5), a, name, len(np.concatenate(d["pos"]))))
            scored.sort(reverse=True)
            print(f"\n  {cls:<15s} {etype:<8s} (n_pos={scored[0][3]} frames)")
            for dev, a, name, _ in scored:
                print(f"    {name:<14s} AUC {a:.3f}  |dev| {dev:.3f}  "
                      f"{'#' * int(round(dev * 60))}")
