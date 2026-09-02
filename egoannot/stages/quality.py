"""
Per-clip quality record. Four tiers, in order:

  T1  deterministic hard rejects   (blur, brightness, duration, modality, pose jumps)
  T2  head-motion score            (optical flow, 4 fps, 256x256)
  T3  hand-visibility rate         (wrist reprojection from shipped pose + calibration)
  T4  per-clip quality record      (the deliverable)

No learned filter at any tier: every decision is a threshold on a measured
quantity, so the accept/reject boundary is auditable and does not inherit a
model's preferences about what egocentric video should look like.

T1 is deliberately threshold-only. A learned quality filter at the hard-reject
tier would encode a model's opinion about what egocentric video should look
like and bake that bias into every dataset built downstream.
"""
from __future__ import annotations

import collections
import json
import math
import os

import numpy as np

from .. import config
from ..core.geometry import camera_convention, in_rect, project, quats_to_R
from ..core.imquality import piqe
from ..core.mcap_io import modality_audit, read_episode
from ..core.signal import angular_speed, speed
from ..core.video import decode_stream

CFG = config.QUALITY


# ---------------------------------------------------------------- T1 measures
def frame_sharpness(frame, cfg=CFG):
    """Tiled Laplacian variance, low-texture tiles excluded (covers de-ID blur)."""
    import cv2
    cols, rows = cfg["tile_grid"]
    h, w = frame.shape
    ys = np.linspace(0, h, rows + 1).astype(int)
    xs = np.linspace(0, w, cols + 1).astype(int)
    variances = np.array([
        cv2.Laplacian(frame[ys[i]:ys[i + 1], xs[j]:xs[j + 1]], cv2.CV_64F).var()
        for i in range(rows) for j in range(cols)])
    keep = variances >= cfg["lowtex_floor"]
    masked = 1.0 - keep.mean()
    if masked > cfg["lowtex_max_frac"]:
        # Too much of the frame is flat to be de-ID alone: it really is blurred.
        return float(np.percentile(variances, 75)), masked, True
    return float(np.percentile(variances[keep], 75)), masked, False


def exposure_bad(frame, cfg=CFG):
    """
    A frame is unusable on exposure if its mean is out of range OR too much of
    it is clipped. Mean alone misses blow-out: a frame can average 224 with
    most of its pixels pegged at 255, which is what the self-test caught.
    """
    mean = float(frame.mean())
    sat = float((frame >= 250).mean())
    dark = float((frame <= 5).mean())
    bad = (mean < cfg["bright_lo"] or mean > cfg["bright_hi"]
           or sat > cfg["max_sat_frac"] or dark > cfg["max_dark_frac"])
    return mean, sat, dark, bad


def pose_kinematics(ep, cfg=CFG):
    """Max linear speeds and head angular speed; per-time arrays too."""
    tl, vl = speed(ep["/pose/left_hand"][:, 0], ep["/pose/left_hand"][:, 1:4]) \
        if len(ep["/pose/left_hand"]) > 2 else (np.zeros(0), np.zeros(0))
    tr, vr = speed(ep["/pose/right_hand"][:, 0], ep["/pose/right_hand"][:, 1:4]) \
        if len(ep["/pose/right_hand"]) > 2 else (np.zeros(0), np.zeros(0))
    th, vh = speed(ep["/pose/head"][:, 0], ep["/pose/head"][:, 1:4]) \
        if len(ep["/pose/head"]) > 2 else (np.zeros(0), np.zeros(0))
    E = ep["extr"]
    te, w = angular_speed(E[:, 0], E[:, 4:8]) if len(E) > 2 else (np.zeros(0), np.zeros(0))
    return dict(tl=tl, vl=vl, tr=tr, vr=vr, th=th, vh=vh, te=te, w=w)


# ---------------------------------------------------------------- T3 hand FOV
def hand_visibility(ep, axis=None):
    """
    Reproject all 21 joints of both hands into head_left using the shipped
    per-frame extrinsic and intrinsic.

    Returns per-frame fractions of joints inside the full sensor rectangle and
    inside the central 50%. NOTE: this measures in-FOV, not unoccluded --
    occlusion by the body, the shelf or the held object cannot be determined
    from pose. On this corpus the full-frame rate saturates at 1.0, so
    `frac_c50` is the only pose-derived variant with any spread.
    """
    K, E = ep["K"], ep["extr"]
    if K is None or len(E) == 0:
        return None
    conv = camera_convention(E, ep["/pose/right_hand"])
    if axis is None:
        axis = conv["axis"] if conv else 1.0
    t_hand = ep["/pose/left_hand"][:, 0] if len(ep["/pose/left_hand"]) else np.zeros(0)
    JL, JR = ep["/pose/left_hand_joints"], ep["/pose/right_hand_joints"]
    n = min(len(JL), len(JR), len(t_hand))
    if n == 0:
        return None

    idx = np.clip(np.searchsorted(E[:, 0], t_hand[:n]), 0, len(E) - 1)
    R_all = quats_to_R(E[idx, 4:8])
    T_all = E[idx, 1:4]

    frac_full = np.zeros(n)
    frac_c50 = np.zeros(n)
    dist = np.zeros(n)
    offaxis = np.zeros(n)
    for k in range(n):
        R, t = R_all[k], T_all[k]
        joints = np.concatenate([JL[k], JR[k]], axis=0)
        u, v, _, ok = project(joints, R, t, K, axis)
        frac_full[k] = (ok & in_rect(u, v, K, 0.0)).mean()
        frac_c50[k] = (ok & in_rect(u, v, K, 0.25)).mean()
        d = JR[k][0] - t
        nd = float(np.linalg.norm(d))
        dist[k] = nd
        forward = R @ np.array([0.0, 0.0, axis])
        offaxis[k] = math.degrees(
            math.acos(np.clip(np.dot(forward, d / max(nd, 1e-9)), -1, 1)))
    return dict(axis=axis, t=t_hand[:n], frac_full=frac_full, frac_c50=frac_c50,
                dist=dist, offaxis=offaxis, rate=float(frac_full.mean()),
                rate_c50=float(frac_c50.mean()), conv=conv)


# ---------------------------------------------------------------- T2 / T1 series
def flow_series(ep, cfg=CFG):
    """Mean optical-flow magnitude per consecutive frame pair, 4 fps, 256x256."""
    import cv2
    s = cfg["flow_size"]
    frames = list(decode_stream(ep["vid"], ep["src_fps"], cfg["flow_fps"], size=(s, s)))
    mags = []
    for a, b in zip(frames, frames[1:]):
        f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mags.append(float(np.linalg.norm(f, axis=2).mean()))
    ts = (np.arange(len(mags)) + 0.5) / cfg["flow_fps"]     # pair midpoints
    return ts, np.array(mags), len(frames)


def blur_series(ep, cfg=CFG):
    """Sharpness, exposure and PIQE at native resolution, 2 fps."""
    sharp, bright, masked, glob, bad, sats, darks = [], [], [], [], [], [], []
    piqe_s, piqe_a = [], []
    for frame in decode_stream(ep["vid"], ep["src_fps"], cfg["blur_fps"], size=None):
        s, m, g = frame_sharpness(frame, cfg)
        sharp.append(s)
        masked.append(m)
        glob.append(g)
        mean, sat, dark, b = exposure_bad(frame, cfg)
        bright.append(mean)
        sats.append(sat)
        darks.append(dark)
        bad.append(b)
        p, a = piqe(frame, cfg.get("piqe_activity"))
        piqe_s.append(p)
        piqe_a.append(a)
    ts = np.arange(len(sharp)) / cfg["blur_fps"]
    return (ts, np.array(sharp), np.array(bright), np.array(masked),
            np.array(glob, bool), np.array(bad, bool), np.array(sats),
            np.array(darks), np.array(piqe_s), np.array(piqe_a))


# ---------------------------------------------------------------- T4 records
def in_window(t, a, b):
    return (t >= a) & (t < b)


def hard_rejects(m: dict, cfg=CFG) -> list[str]:
    """
    T1, as a pure function of measured quantities so it can be unit-tested by
    fault injection. Returns the list of reject reasons (empty = pass).
    """
    reasons = []
    if m["duration_s"] < cfg["min_clip_s"]:
        reasons.append("clip_too_short")
    if m.get("missing_topics"):
        reasons.append("missing_modality")
    if not m.get("n_frames"):
        reasons.append("no_video_frames")
    elif m["sharpness_p25"] < cfg["blur_floor"]:
        reasons.append("blurred")
    if m["brightness_out_frac"] > cfg["bright_bad_frac"]:
        reasons.append("brightness_out_of_range")
    if m["max_wrist_speed"] > cfg["max_wrist_speed"]:
        reasons.append("implausible_wrist_velocity")
    if m["max_head_speed"] > cfg["max_head_speed"]:
        reasons.append("implausible_head_velocity")
    if m["max_angular_speed"] > cfg["max_ang_speed"]:
        reasons.append("implausible_angular_velocity")
    # Full-frame in-FOV rate. Inert on this corpus (it is exactly 1.000 in every
    # episode, because the shipped hand pose is near-rigidly coupled to the head
    # camera) but it is a declared gate, so it is applied rather than left as
    # dead configuration.
    vis = m.get("hand_visible_rate")
    if vis is not None and vis < cfg["min_hand_vis"]:
        reasons.append("hands_out_of_view")
    c50 = m.get("hand_central50_rate")
    if c50 is not None and c50 < cfg["min_hand_c50"]:
        reasons.append("hands_poorly_framed")
    # PIQE catches compression artifacts and sensor noise, which the Laplacian
    # gate not only misses but inverts -- a noisy frame reads as very sharp.
    # Off until calibrated: `quality calibrate` prints the floor for a corpus.
    pq, pq_max = m.get("piqe"), cfg.get("piqe_max")
    if pq_max is not None and pq is not None and pq > pq_max:
        reasons.append("compression_artifacts")
    return reasons


def episode_records(path, cfg=CFG):
    """T1-T4 for one episode: (clip records, episode summary)."""
    ep = read_episode(path)
    name = ep["name"]
    duration = ep["duration_s"]

    missing, undelivered = modality_audit(ep)
    kin = pose_kinematics(ep, cfg)
    hv = hand_visibility(ep)
    f_ts, f_mag, n_flow = flow_series(ep, cfg)
    (b_ts, sharp, bright, mask, glob, badexp, sat, dark,
     piqe_score, piqe_active) = blur_series(ep, cfg)

    n_clips = max(1, int(math.ceil(duration / cfg["clip_len_s"])))
    records = []
    for i in range(n_clips):
        a = i * cfg["clip_len_s"]
        b = min(duration, a + cfg["clip_len_s"])
        mb, mf = in_window(b_ts, a, b), in_window(f_ts, a, b)
        sh, br, mk, gl = sharp[mb], bright[mb], mask[mb], glob[mb]
        bd, sa, dk, fl = badexp[mb], sat[mb], dark[mb], f_mag[mf]
        pq, pa = piqe_score[mb], piqe_active[mb]

        def peak(t, v):
            sel = in_window(t, a, b)
            return float(v[sel].max()) if sel.any() else 0.0

        wrist = max(peak(kin["tl"], kin["vl"]), peak(kin["tr"], kin["vr"]))
        head = peak(kin["th"], kin["vh"])
        angular = peak(kin["te"], kin["w"])

        vis = vis_c50 = hand_dist = hand_angle = None
        if hv is not None and len(hv["t"]):
            sel = in_window(hv["t"], a, b)
            if sel.any():
                vis = float(hv["frac_full"][sel].mean())
                vis_c50 = float(hv["frac_c50"][sel].mean())
                hand_dist = float(np.median(hv["dist"][sel]))
                hand_angle = float(np.median(hv["offaxis"][sel]))
            else:
                vis = vis_c50 = 0.0

        flagged = [x for x in ep["badf"] if a <= x[0] < b]
        bad_rate = (sum(1 for x in flagged if x[1]) / len(flagged)) if flagged else 0.0

        sharp_p25 = float(np.percentile(sh, 25)) if len(sh) else 0.0
        bright_bad = float(bd.mean()) if len(bd) else 1.0
        reasons = hard_rejects(dict(
            duration_s=b - a, missing_topics=missing, n_frames=len(sh),
            sharpness_p25=sharp_p25, brightness_out_frac=bright_bad,
            max_wrist_speed=wrist, max_head_speed=head, max_angular_speed=angular,
            hand_visible_rate=vis, hand_central50_rate=vis_c50,
            piqe=float(np.nanmedian(pq)) if len(pq) and np.isfinite(pq).any()
            else None), cfg)

        records.append(dict(
            clip_id=f"{name}#{i:04d}", episode=name, source_path=str(path),
            task=ep["meta"].get("task_name"), scene=ep["meta"].get("scene_id"),
            paradigm=ep["meta"].get("paradigm"),
            start_ts=round(a, 3), end_ts=round(b, 3), duration_s=round(b - a, 3),
            # T1
            sharpness_p25=round(sharp_p25, 2),
            sharpness_min=round(float(sh.min()), 2) if len(sh) else None,
            lowtex_masked_frac=round(float(mk.mean()), 4) if len(mk) else None,
            globally_blurred_frames=int(gl.sum()) if len(gl) else 0,
            piqe=round(float(np.nanmedian(pq)), 2)
            if len(pq) and np.isfinite(pq).any() else None,
            piqe_active_frac=round(float(np.mean(pa)), 3) if len(pa) else None,
            brightness_mean=round(float(br.mean()), 2) if len(br) else 0.0,
            brightness_out_frac=round(bright_bad, 4),
            saturated_pixel_frac=round(float(sa.max()), 4) if len(sa) else None,
            crushed_pixel_frac=round(float(dk.max()), 4) if len(dk) else None,
            max_wrist_speed=round(wrist, 3),
            max_head_speed=round(head, 3),
            max_angular_speed=round(angular, 3),
            missing_topics=missing, declared_but_undelivered=undelivered,
            src_fps=ep["src_fps"], fps_mismatch=ep["fps_mismatch"],
            shipped_bad_frame_rate=round(bad_rate, 4),
            # T2
            head_motion=round(float(fl.mean()) if len(fl) else 0.0, 4),
            # T3
            hand_visible_rate=None if vis is None else round(vis, 4),
            hand_central50_rate=None if vis_c50 is None else round(vis_c50, 4),
            hand_cam_dist_m=None if hand_dist is None else round(hand_dist, 3),
            hand_offaxis_deg=None if hand_angle is None else round(hand_angle, 1),
            hand_fov_axis=None if hv is None else hv["axis"],
            # T4 verdict (motion percentile filled in by apply_motion_filter)
            hard_reject=len(reasons) > 0, reject_reasons=reasons,
        ))

    summary = dict(
        episode=name, duration_s=round(duration, 2), n_clips=len(records),
        src_fps=ep["src_fps"], meta_fps=ep["meta_fps"], msg_fps=ep["msg_fps"],
        fps_mismatch=ep["fps_mismatch"], flow_frames=n_flow, blur_frames=len(sharp),
        hands_share_timebase=ep["hands_share_timebase"],
        hand_axis=None if hv is None else hv["axis"],
        hand_rate_episode=None if hv is None else round(hv["rate"], 4),
        hand_rate_c50=None if hv is None else round(hv["rate_c50"], 4),
        hand_convention=None if hv is None else hv["conv"],
        missing=missing, undelivered=undelivered, topics=sorted(ep["chans"]))
    return records, summary


def apply_motion_filter(records, cfg=CFG):
    """
    T2: drop the top fraction by head motion, among clips that survived T1.

    Ranking is per episode by default. Pooling the ranking across the corpus
    made this an episode selector rather than a clip filter: measured drop
    rates ran 0% (test, cart_wipes, cleaning_tools) to 75% (belts_a, which lost
    every clip it had), because optical-flow magnitude is scene-dependent and
    not comparable between episodes.
    """
    live = [r for r in records if not r["hard_reject"]]
    if live:
        groups = collections.defaultdict(list)
        if cfg.get("motion_scope", "episode") == "episode":
            for r in live:
                groups[r["episode"]].append(r)
        else:
            groups["__all__"] = live
        cut = 1.0 - cfg["drop_motion_frac"]
        for group in groups.values():
            vals = np.array([r["head_motion"] for r in group])
            pct = np.empty(len(vals))
            pct[vals.argsort()] = np.linspace(0, 1, len(vals), endpoint=False)
            for r, p in zip(group, pct):
                r["head_motion_pctile"] = round(float(p), 4)
                r["motion_reject"] = bool(p >= cut)
    for r in records:
        r.setdefault("head_motion_pctile", None)
        r.setdefault("motion_reject", False)
        r["accepted"] = (not r["hard_reject"]) and (not r["motion_reject"])
    return records


# ---------------------------------------------------------------- reporting
def report(records):
    n = len(records)
    if n == 0:
        print("no clip records")
        return
    hard = [r for r in records if r["hard_reject"]]
    motion = [r for r in records if not r["hard_reject"] and r.get("motion_reject")]
    accepted = [r for r in records if r.get("accepted")]
    print("=" * 74)
    print(f"{n} clips from {len({r['episode'] for r in records})} episodes "
          f"({sum(r['duration_s'] for r in records) / 60:.1f} min)")
    print(f"  T1 hard reject : {len(hard):4d} ({100 * len(hard) / n:5.1f}%)")
    print(f"  T2 motion drop : {len(motion):4d} ({100 * len(motion) / n:5.1f}%)")
    print(f"  accepted       : {len(accepted):4d} ({100 * len(accepted) / n:5.1f}%)  "
          f"{sum(r['duration_s'] for r in accepted) / 60:.1f} min")
    counts = collections.Counter(x for r in records for x in r["reject_reasons"])
    if counts:
        print("  reject reasons :", dict(counts.most_common()))
    per_ep = collections.defaultdict(lambda: [0, 0, 0])
    for r in records:
        cell = per_ep[r["episode"]]
        cell[0] += 1
        cell[1] += bool(r["hard_reject"])
        cell[2] += bool(r.get("motion_reject"))
    print(f"  {'episode':20s} {'clips':>5s} {'T1':>4s} {'T2':>4s} {'kept':>5s}")
    for name, (tot, h, m) in sorted(per_ep.items()):
        print(f"  {name:20s} {tot:5d} {h:4d} {m:4d} {tot - h - m:5d}")
    for key in ("sharpness_p25", "piqe", "piqe_active_frac",
                "brightness_mean", "saturated_pixel_frac",
                "crushed_pixel_frac", "head_motion", "hand_visible_rate",
                "hand_central50_rate", "hand_cam_dist_m", "hand_offaxis_deg",
                "max_wrist_speed", "max_angular_speed", "lowtex_masked_frac"):
        v = np.array([r[key] for r in records if r.get(key) is not None], float)
        if len(v):
            print(f"  {key:22s} p5 {np.percentile(v, 5):8.2f}  "
                  f"p50 {np.percentile(v, 50):8.2f}  "
                  f"p95 {np.percentile(v, 95):8.2f}  max {v.max():8.2f}")


def measure(paths, out=None, cfg=CFG):
    """Run T1-T4 over episodes and write the records."""
    out = out or config.QUALITY_RECORDS
    all_records, summaries = [], []
    for i, path in enumerate(paths, 1):
        try:
            recs, summary = episode_records(path, cfg)
            all_records += recs
            summaries.append(summary)
            print(f"[{i}/{len(paths)}] {summary['episode']:22s} "
                  f"{summary['duration_s']:7.1f}s {summary['src_fps']:.0f}fps "
                  f"{summary['n_clips']:3d} clips  "
                  f"hand={summary['hand_rate_episode']}  "
                  f"undelivered={summary['undelivered']}", flush=True)
        except Exception as e:
            print(f"[{i}/{len(paths)}] FAIL {path}: {type(e).__name__}: {e}",
                  flush=True)
    all_records = apply_motion_filter(all_records, cfg)
    out = str(out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        for r in all_records:
            fh.write(json.dumps(r) + "\n")
    with open(out.replace(".jsonl", ".episodes.json"), "w") as fh:
        json.dump(summaries, fh, indent=1)
    print(f"\nwrote {len(all_records)} clip records -> {out}")
    report(all_records)
    return all_records


# ---------------------------------------------------------------- calibration
def calibrate(paths, cfg=CFG, per_ep=14):
    """
    Data-driven thresholds with a synthetic positive control.

    Tile-variance floors cannot be guessed: they depend on resolution, codec
    and scene texture. We sample real frames, score them, then score
    deliberately blurred copies of the same frames. The floor goes between the
    two distributions -- so "blurred" means measurably worse than this
    footage's own sharp frames, not worse than some absolute constant.
    """
    import cv2
    tiles_all, sharp_ok, bright_ok = [], [], []
    piqe_ok, piqe_active, piqe_noise = [], [], []
    sharp_blur = {1.5: [], 3.0: [], 5.0: []}
    cols, rows = cfg["tile_grid"]
    for path in paths:
        ep = read_episode(path)
        frames = list(decode_stream(ep["vid"], ep["src_fps"], 0.5, size=None))
        if not frames:
            continue
        step = max(1, len(frames) // per_ep)
        for frame in frames[::step][:per_ep]:
            h, w = frame.shape
            ys = np.linspace(0, h, rows + 1).astype(int)
            xs = np.linspace(0, w, cols + 1).astype(int)

            def tile_vars(img):
                return [cv2.Laplacian(img[ys[i]:ys[i + 1], xs[j]:xs[j + 1]],
                                      cv2.CV_64F).var()
                        for i in range(rows) for j in range(cols)]

            vs = tile_vars(frame)
            tiles_all += vs
            sharp_ok.append(np.percentile(vs, 75))
            bright_ok.append(frame.mean())
            p_ok, a_ok = piqe(frame, cfg.get("piqe_activity"))
            piqe_ok.append(p_ok)
            piqe_active.append(a_ok)
            # synthetic positive control for the noise gate, matching how the
            # blur floor is derived from deliberately degraded copies
            noisy = np.clip(frame.astype(int)
                            + np.random.default_rng(0).normal(0, 15, frame.shape),
                            0, 255).astype(np.uint8)
            piqe_noise.append(piqe(noisy, cfg.get("piqe_activity"))[0])
            for sigma in sharp_blur:
                blurred = cv2.GaussianBlur(frame, (0, 0), sigma)
                sharp_blur[sigma].append(np.percentile(tile_vars(blurred), 75))
        del ep

    tiles = np.array(tiles_all)
    sharp = np.array(sharp_ok)
    bright = np.array(bright_ok)
    print("=" * 74)
    print(f"CALIBRATION  {len(paths)} episodes, {len(sharp)} frames, "
          f"{len(tiles)} tiles")
    print("  tile Laplacian variance:", "  ".join(
        f"p{q}={np.percentile(tiles, q):.1f}" for q in (1, 5, 10, 25, 50, 75, 95)))
    print("  frame sharpness (tile p75), sharp originals:", "  ".join(
        f"p{q}={np.percentile(sharp, q):.1f}" for q in (1, 5, 25, 50, 95)))
    for sigma in sorted(sharp_blur):
        sb = np.array(sharp_blur[sigma])
        sep = float((sharp > np.percentile(sb, 95)).mean())
        print(f"  synthetic blur sigma={sigma}: p50={np.median(sb):8.1f}  "
              f"p95={np.percentile(sb, 95):8.1f}   -> {100 * sep:5.1f}% of sharp "
              f"frames score above its p95")
    print("  brightness mean:", "  ".join(
        f"p{q}={np.percentile(bright, q):.1f}" for q in (1, 5, 50, 95, 99)))
    lo = float(np.percentile(sharp_blur[1.5], 95))
    hi = float(np.percentile(sharp, 5))
    rec_blur = round((lo + hi) / 2, 1) if lo < hi else round(hi * 0.6, 1)
    rec_lowtex = round(float(np.percentile(tiles, 10)), 1)
    pq = np.array([x for x in piqe_ok if np.isfinite(x)])
    pn = np.array([x for x in piqe_noise if np.isfinite(x)])
    rec_piqe = None
    if len(pq) and len(pn):
        print("\n  PIQE (lower is better), real frames: " + "  ".join(
            f"p{q}={np.percentile(pq, q):.2f}" for q in (5, 50, 95)))
        print(f"  PIQE with sigma=15 noise injected: p5={np.percentile(pn, 5):.2f}"
              f"  p50={np.percentile(pn, 50):.2f}")
        print(f"  judgeable blocks at activity={cfg.get('piqe_activity')}: "
              f"{100 * np.mean(piqe_active):.0f}%")
        real_hi = float(np.percentile(pq, 95))
        noise_lo = float(np.percentile(pn, 5))
        rec_piqe = (round((real_hi + noise_lo) / 2, 2)
                    if real_hi < noise_lo else None)
        if rec_piqe is None:
            print("  -> real and noisy frames are NOT separable by PIQE on this "
                  "footage; leave piqe_max unset rather than gate on it")
        else:
            print(f"  -> recommended piqe_max = {rec_piqe} "
                  f"(real p95={real_hi:.2f}, noisy p5={noise_lo:.2f})")

    print(f"\n  recommended  lowtex_floor = {rec_lowtex}   blur_floor = {rec_blur}   "
          f"(sigma1.5 p95={lo:.1f}, real p5={hi:.1f})")
    return dict(lowtex_floor=rec_lowtex, blur_floor=rec_blur, piqe_max=rec_piqe,
                bright_lo=round(float(np.percentile(bright, 1)) * 0.5, 1),
                bright_hi=round(min(250.0, float(np.percentile(bright, 99)) * 1.6), 1))


# ---------------------------------------------------------------- T1 self-test
def selftest(paths, cfg=CFG):
    """
    Fault injection against real footage. T1 rejects nothing on clean clips, so
    the only way to show it works is to break real clips in each way it claims
    to catch and confirm it fires.
    """
    import cv2
    passed = failed = 0

    def check(name, reasons, want):
        nonlocal passed, failed
        hit = want in reasons
        print(f"  {'PASS' if hit else 'FAIL'}  {name:38s} -> "
              f"{reasons or ['(accepted)']}")
        if hit:
            passed += 1
        else:
            failed += 1

    def clean(**kw):
        m = dict(duration_s=4.0, missing_topics=[], n_frames=8,
                 sharpness_p25=30.0, brightness_out_frac=0.0,
                 max_wrist_speed=1.0, max_head_speed=0.5, max_angular_speed=1.0,
                 hand_visible_rate=1.0, hand_central50_rate=0.95)
        m.update(kw)
        return m

    print("=" * 74)
    print("T1 SELF-TEST  (fault injection)")
    print("-- control")
    r = hard_rejects(clean(), cfg)
    print(f"  {'PASS' if not r else 'FAIL'}  {'clean clip accepted':38s} -> "
          f"{r or ['(accepted)']}")
    passed += (not r)
    failed += bool(r)

    print("-- synthetic faults on the decision function")
    check("duration 1.0s", hard_rejects(clean(duration_s=1.0), cfg), "clip_too_short")
    check("required topic absent",
          hard_rejects(clean(missing_topics=["/pose/left_hand"]), cfg),
          "missing_modality")
    check("no decoded frames", hard_rejects(clean(n_frames=0), cfg), "no_video_frames")
    check("40% frames out of exposure",
          hard_rejects(clean(brightness_out_frac=0.40), cfg),
          "brightness_out_of_range")
    check("wrist teleport 8 m/s",
          hard_rejects(clean(max_wrist_speed=8.0), cfg),
          "implausible_wrist_velocity")
    check("head teleport 5 m/s",
          hard_rejects(clean(max_head_speed=5.0), cfg), "implausible_head_velocity")
    check("head spin 20 rad/s",
          hard_rejects(clean(max_angular_speed=20.0), cfg),
          "implausible_angular_velocity")
    check("hands at frame edge",
          hard_rejects(clean(hand_central50_rate=0.20), cfg), "hands_poorly_framed")
    check("hands out of view entirely",
          hard_rejects(clean(hand_visible_rate=0.10), cfg), "hands_out_of_view")

    if not paths:
        print(f"\nSELF-TEST {'PASS' if failed == 0 else 'FAIL'}  "
              f"({passed} passed, {failed} failed)  [no footage given, "
              f"skipped injection into real frames]")
        return failed == 0

    print("-- blur + exposure faults injected into REAL frames")
    for path in paths[:2]:
        ep = read_episode(path)
        frames = list(decode_stream(ep["vid"], ep["src_fps"], 0.5, size=None))[:6]
        if not frames:
            continue
        name = os.path.basename(str(path))
        base = np.array([frame_sharpness(f, cfg)[0] for f in frames])
        print(f"  {name}: real sharpness p25={np.percentile(base, 25):.1f} "
              f"(floor {cfg['blur_floor']})")
        for sigma in (3.0, 5.0):
            v = np.array([frame_sharpness(cv2.GaussianBlur(f, (0, 0), sigma), cfg)[0]
                          for f in frames])
            m = clean(sharpness_p25=float(np.percentile(v, 25)))
            check(f"{name} gaussian blur sigma={sigma} "
                  f"(p25={m['sharpness_p25']:.1f})",
                  hard_rejects(m, cfg), "blurred")
        for label, fn in (("darkened x0.2", lambda f: (f * 0.2).astype(np.uint8)),
                          ("blown out +170",
                           lambda f: np.clip(f.astype(int) + 170, 0, 255).astype(np.uint8))):
            res = [exposure_bad(fn(f), cfg) for f in frames]
            mean = np.mean([x[0] for x in res])
            sat = np.mean([x[1] for x in res])
            bad = float(np.mean([x[3] for x in res]))
            check(f"{name} {label} (mean={mean:.0f}, clipped={sat:.2f})",
                  hard_rejects(clean(brightness_out_frac=bad), cfg),
                  "brightness_out_of_range")
        del ep

    print("-- pose-jump fault injected into a REAL pose stream")
    ep = read_episode(paths[0])
    arr = ep["/pose/right_hand"].copy()
    k0 = pose_kinematics(ep, cfg)
    print(f"  real max wrist speed = {k0['vr'].max():.2f} m/s "
          f"(threshold {cfg['max_wrist_speed']})")
    j = len(arr) // 2
    arr[j, 1:4] += np.array([1.2, 0.0, 0.0])          # 1.2 m teleport in one frame
    ep["/pose/right_hand"] = arr
    k1 = pose_kinematics(ep, cfg)
    check(f"1.2 m single-frame teleport ({k1['vr'].max():.1f} m/s)",
          hard_rejects(clean(max_wrist_speed=float(k1["vr"].max())), cfg),
          "implausible_wrist_velocity")
    del ep

    print(f"\nSELF-TEST {'PASS' if failed == 0 else 'FAIL'}  "
          f"({passed} passed, {failed} failed)")
    return failed == 0


def load_records(path=None):
    """Read clip records back, for downstream gating."""
    path = path or config.QUALITY_RECORDS
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path)]
