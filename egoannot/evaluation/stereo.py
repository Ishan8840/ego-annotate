"""
Stereo depth for EgoStandard, and the test that decides whether hand pose can
be used for hand-to-surface proximity at all.

The depth *modality* is declared in session metadata and never delivered, but a
calibrated stereo pair is: head_left and head_right both ship per-frame video,
intrinsics and extrinsics, on a perfectly rigid ~64 mm baseline. So depth is
recoverable, and with it the hand-to-object proximity that Phase 4's
aperture-only contact detector had to work around.

Before using it, one thing has to be checked. Phase 3 found the shipped hand
pose is near-rigidly coupled to the head camera (camera-frame spread 3-12 cm
against 12-65 cm in world). If the hand pose is head-anchored rather than
independently localised, its projection into the image does not land on the
real hand, and sampling scene depth there measures the wrong surface.

Stereo settles it in absolute units: the hand occludes whatever is behind it,
so measured depth at the projected hand pixel must equal the hand's own
predicted depth. If those disagree systematically, proximity-from-pose is
invalid and the finding matters more than the feature.

Driven by `python -m egoannot stereo <episode>... [--every 2.0] [--scale 0.5]`.
"""
from __future__ import annotations
import os, subprocess, tempfile
import numpy as np
import cv2


from ..core.geometry import quat_to_R as _quat_to_R
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

TIPS = [4, 8, 12, 16, 20]


def read_stereo(path):
    """both video streams, both calibrations, hand + head pose, in one pass"""
    r = make_reader(open(path, "rb"), decoder_factories=[DecoderFactory()])
    st = r.get_summary().statistics
    t0 = st.message_start_time
    vid = {"left": bytearray(), "right": bytearray()}
    K = {}
    E = {"left": [], "right": []}
    hands = {"left": [], "right": []}
    fps = None
    topics = ["/sensor/camera/head_left/video", "/sensor/camera/head_right/video",
              "/sensor/camera/head_left/intrinsic", "/sensor/camera/head_right/intrinsic",
              "/sensor/camera/head_left/extrinsic", "/sensor/camera/head_right/extrinsic",
              "/pose/left_hand", "/pose/right_hand", "/session/metadata"]
    for _, ch, msg, dec in r.iter_decoded_messages(topics=topics):
        tp, t = ch.topic, (msg.log_time - t0) / 1e9
        side = "left" if "head_left" in tp else "right"
        if tp.endswith("/video"):
            vid[side] += dec.data
        elif tp.endswith("/intrinsic"):
            if side not in K:
                K[side] = dict(fx=dec.K[0], fy=dec.K[4], cx=dec.K[2], cy=dec.K[5],
                               w=dec.width, h=dec.height, D=list(dec.D))
        elif tp.endswith("/extrinsic"):
            tr = dec.transforms[0]
            E[side].append((t, tr.translation.x, tr.translation.y, tr.translation.z,
                            tr.rotation.w, tr.rotation.x, tr.rotation.y, tr.rotation.z))
        elif tp == "/session/metadata":
            for d in dec.devices:
                if d.device_type == "head_left_camera":
                    try:
                        fps = float(d.fps)
                    except (TypeError, ValueError):
                        pass
        else:
            s = "left" if tp == "/pose/left_hand" else "right"
            hands[s].append((t, [[p.pos.x, p.pos.y, p.pos.z] for p in dec.transforms]))
    for k in E:
        E[k] = np.array(E[k])
    dur = (st.message_end_time - t0) / 1e9
    if fps is None:
        fps = len(E["left"]) / dur if dur else 30.0
    return dict(vid=vid, K=K, E=E, hands=hands, dur=dur, fps=fps)


def rectify_maps(S, scale=0.5):
    """
    stereoRectify from the shipped calibration. The rig is rigid (baseline std
    0.000000 m over 400 sampled frames), so one relative pose serves the episode.
    """
    KL, KR = S["K"]["left"], S["K"]["right"]
    w, h = int(KL["w"] * scale), int(KL["h"] * scale)

    def mat(k):
        return np.array([[k["fx"] * scale, 0, k["cx"] * scale],
                         [0, k["fy"] * scale, k["cy"] * scale],
                         [0, 0, 1.0]])
    M1, M2 = mat(KL), mat(KR)
    D1 = np.array(KL["D"][:5], float) if len(KL["D"]) >= 5 else np.zeros(5)
    D2 = np.array(KR["D"][:5], float) if len(KR["D"]) >= 5 else np.zeros(5)

    # p_right = R_cv @ p_left + T_cv, from the two camera-to-world extrinsics
    EL, ER = S["E"]["left"], S["E"]["right"]
    n = min(len(EL), len(ER))
    i = n // 2
    RL, RR = _quat_to_R(*EL[i, 4:8]), _quat_to_R(*ER[i, 4:8])
    R_cv = RR.T @ RL
    T_cv = RR.T @ (EL[i, 1:4] - ER[i, 1:4])

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        M1, D1, M2, D2, (w, h), R_cv, T_cv,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    m1 = cv2.initUndistortRectifyMap(M1, D1, R1, P1, (w, h), cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(M2, D2, R2, P2, (w, h), cv2.CV_32FC1)
    return dict(size=(w, h), R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, m1=m1, m2=m2,
                R_cv=R_cv, T_cv=T_cv, baseline=float(np.linalg.norm(T_cv)),
                D_nonzero=bool(np.any(D1) or np.any(D2)))


def decode_pairs(S, every_s, scale):
    """synchronised rectified stereo frames, sampled every `every_s` seconds"""
    rc = rectify_maps(S, scale)
    w, h = rc["size"]
    out_fps = 1.0 / every_s
    streams = {}
    for side in ("left", "right"):
        with tempfile.NamedTemporaryFile(suffix=".h264", delete=False) as f:
            f.write(bytes(S["vid"][side]))
            streams[side] = f.name
    procs = {}
    try:
        for side in ("left", "right"):
            procs[side] = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-f", "h264", "-r", str(S["fps"]),
                 "-i", streams[side], "-vf", f"fps={out_fps},scale={w}:{h}",
                 "-pix_fmt", "gray", "-f", "rawvideo", "-"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        n = w * h
        k = 0
        while True:
            bl = procs["left"].stdout.read(n)
            br = procs["right"].stdout.read(n)
            if not bl or not br or len(bl) < n or len(br) < n:
                break
            fl = np.frombuffer(bl, np.uint8).reshape(h, w)
            fr = np.frombuffer(br, np.uint8).reshape(h, w)
            gl = cv2.remap(fl, rc["m1"][0], rc["m1"][1], cv2.INTER_LINEAR)
            gr = cv2.remap(fr, rc["m2"][0], rc["m2"][1], cv2.INTER_LINEAR)
            yield k * every_s, gl, gr, rc
            k += 1
        for p in procs.values():
            p.stdout.close(); p.wait()
    finally:
        for f in streams.values():
            os.unlink(f)


def make_sgbm(w):
    nd = 16 * max(4, int(round(w / 96.0)))      # disparity range scales with width
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=nd, blockSize=5,
        P1=8 * 5 * 5, P2=32 * 5 * 5, disp12MaxDiff=1,
        uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
        preFilterCap=63, mode=cv2.StereoSGBM_MODE_SGBM_3WAY), nd


def validate(path, every_s=2.0, scale=0.5, max_frames=80):
    """
    Does the shipped hand pose land where the image says the hand is?

    For each sampled frame: project the hand joints into the rectified left
    camera, read stereo depth in a small window at that pixel, and compare with
    the hand's own predicted depth. A hand occludes what is behind it, so
    agreement is expected if the pose is real. Systematic disagreement means the
    pose is not registered to the imagery and proximity-from-pose is invalid.
    """
    S = read_stereo(path)
    name = os.path.basename(path)
    rows, cov = [], []
    sgbm = None
    EL = S["E"]["left"]
    HR, HL = S["hands"]["right"], S["hands"]["left"]
    if not len(EL) or not HR:
        print(f"{name}: missing calibration or hand pose"); return None

    for t, gl, gr, rc in decode_pairs(S, every_s, scale):
        if sgbm is None:
            sgbm, nd = make_sgbm(rc["size"][0])
            print(f"{name}: {rc['size'][0]}x{rc['size'][1]}  baseline "
                  f"{1000*rc['baseline']:.1f} mm  disparities {nd}  "
                  f"distortion_nonzero={rc['D_nonzero']}")
        if len(rows) >= max_frames:
            break
        disp = sgbm.compute(gl, gr).astype(np.float32) / 16.0
        xyz = cv2.reprojectImageTo3D(disp, rc["Q"])
        Z = xyz[:, :, 2]
        valid = (disp > 0.5) & np.isfinite(Z) & (Z > 0.05) & (Z < 6.0)
        cov.append(float(valid.mean()))

        i = int(np.clip(np.searchsorted(EL[:, 0], t), 0, len(EL) - 1))
        RLw = _quat_to_R(*EL[i, 4:8]); tLw = EL[i, 1:4]
        P1, R1 = rc["P1"], rc["R1"]
        h, w = Z.shape

        for side, HS in (("right", HR), ("left", HL)):
            if not HS:
                continue
            k = int(np.clip(np.searchsorted([x[0] for x in HS], t), 0, len(HS) - 1))
            J = np.array(HS[k][1])
            for jname, ji in (("wrist", 0), ("index_tip", 8), ("thumb_tip", 4)):
                pw = J[ji]
                pc = RLw.T @ (pw - tLw)          # left camera frame
                pr = R1 @ pc                      # rectified left frame
                if pr[2] <= 1e-6:
                    continue
                u = P1[0, 0] * pr[0] / pr[2] + P1[0, 2]
                v = P1[1, 1] * pr[1] / pr[2] + P1[1, 2]
                ui, vi = int(round(u)), int(round(v))
                if not (0 <= ui < w and 0 <= vi < h):
                    rows.append(dict(t=t, side=side, joint=jname, z_pose=float(pr[2]),
                                     z_stereo=None, in_frame=False))
                    continue
                r0, r1 = max(0, vi - 6), min(h, vi + 7)
                c0, c1 = max(0, ui - 6), min(w, ui + 7)
                win = Z[r0:r1, c0:c1][valid[r0:r1, c0:c1]]
                rows.append(dict(t=t, side=side, joint=jname, u=ui, v=vi,
                                 z_pose=float(pr[2]),
                                 z_stereo=float(np.median(win)) if win.size >= 8 else None,
                                 in_frame=True))
    ok = [r for r in rows if r.get("z_stereo")]
    print(f"  frames {len(cov)}  disparity coverage mean {100*np.mean(cov):.1f}%  "
          f"samples {len(rows)}  with depth {len(ok)}")
    if not ok:
        print("  no comparable samples"); return dict(name=name, rows=rows)
    zp = np.array([r["z_pose"] for r in ok])
    zs = np.array([r["z_stereo"] for r in ok])
    d = zs - zp
    print(f"  z_pose   p5 {np.percentile(zp,5):.3f}  p50 {np.percentile(zp,50):.3f}  p95 {np.percentile(zp,95):.3f} m")
    print(f"  z_stereo p5 {np.percentile(zs,5):.3f}  p50 {np.percentile(zs,50):.3f}  p95 {np.percentile(zs,95):.3f} m")
    print(f"  signed error (stereo - pose): median {np.median(d):+.3f} m  "
          f"MAE {np.abs(d).mean():.3f} m  p5 {np.percentile(d,5):+.3f}  p95 {np.percentile(d,95):+.3f}")
    within = [float((np.abs(d) <= x).mean()) for x in (0.02, 0.05, 0.10, 0.20)]
    print("  |error| within 2/5/10/20 cm: " + "  ".join(f"{100*x:.0f}%" for x in within))
    corr = float(np.corrcoef(zp, zs)[0, 1]) if len(zp) > 3 and zp.std() > 1e-6 else float("nan")
    print(f"  correlation(z_pose, z_stereo) = {corr:+.3f}")
    return dict(name=name, n=len(ok), median_err=float(np.median(d)),
                mae=float(np.abs(d).mean()), corr=corr,
                within_5cm=within[1], within_10cm=within[2],
                coverage=float(np.mean(cov)), rows=rows)
