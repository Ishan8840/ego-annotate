"""
Render an annotated demo clip: the segment video with the projected hand
skeleton, the caption that is live at each moment, and the pose-derived fields
the model was never asked for.

This is the visual QA view for the pipeline's output -- it makes span
boundaries, hand assignment and caption timing inspectable at a glance -- and
it doubles as the README demo.

    python -m egoannot demo --segments pp_shampoo np_storagebox --out demo.mp4
    python -m egoannot demo --segments pp_shampoo:1.9:10 --gif demo.gif

A segment may carry its own window as `id:start` or `id:start:duration`, in
clip-local seconds, so a demo can open on the moment that shows the pipeline
working rather than on whatever the clip happens to begin with.
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from .. import config

# MediaPipe 21-landmark hand topology
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),              # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),              # index
    (5, 9), (9, 10), (10, 11), (11, 12),         # middle
    (9, 13), (13, 14), (14, 15), (15, 16),       # ring
    (13, 17), (17, 18), (18, 19), (19, 20),      # pinky
    (0, 17),                                     # palm
]

# BGR, on a dark panel
INK = (238, 240, 240)
MUTED = (150, 162, 165)
LINE = (58, 70, 74)
PANEL = (22, 26, 28)
LEFT_HAND = (212, 160, 74)      # amber
RIGHT_HAND = (120, 200, 130)    # green
ACCENT = (196, 168, 86)

PANEL_H = 150
BAR_H = 26


def _put(img, text, org, scale=0.42, color=INK, thick=1):
    import cv2
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, color, thick,
                cv2.LINE_AA)


def _wrap(text, width):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if len(trial) > width and line:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


def _draw_hand(frame, flat, colour, w, h):
    """One hand's 21 joints from a flat [ap_mm, x0, y0, ...] overlay row."""
    import cv2
    if not flat:
        return
    pts = []
    for i in range(21):
        x, y = flat[1 + 2 * i], flat[2 + 2 * i]
        pts.append(None if (x == -999 or y == -999) else (int(x), int(y)))
    for a, b in CONNECTIONS:
        if pts[a] and pts[b]:
            cv2.line(frame, pts[a], pts[b], colour, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        if p:
            cv2.circle(frame, p, 4 if i in (0, 4, 8) else 2, colour, -1, cv2.LINE_AA)


def _timeline(panel, spans, t, x0, y0, width, active):
    """Span boundaries as ticks, with the current position marked."""
    import cv2
    if not spans:
        return
    lo = min(s["start_ts"] for s in spans)
    hi = max(s["end_ts"] for s in spans)
    span = max(hi - lo, 1e-6)
    cv2.line(panel, (x0, y0), (x0 + width, y0), LINE, 1, cv2.LINE_AA)
    for s in spans:
        for edge in (s["start_ts"], s["end_ts"]):
            x = int(x0 + width * (edge - lo) / span)
            cv2.line(panel, (x, y0 - 5), (x, y0 + 5), LINE, 1, cv2.LINE_AA)
        if s is active:
            a = int(x0 + width * (s["start_ts"] - lo) / span)
            b = int(x0 + width * (s["end_ts"] - lo) / span)
            cv2.rectangle(panel, (a, y0 - 4), (b, y0 + 4), ACCENT, -1)
    x = int(x0 + width * (min(max(t, lo), hi) - lo) / span)
    cv2.line(panel, (x, y0 - 9), (x, y0 + 9), INK, 2, cv2.LINE_AA)


def parse_spec(spec):
    """`id`, `id:start` or `id:start:duration` -> (id, start, duration)."""
    parts = str(spec).split(":")
    sid = parts[0]
    start = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
    duration = float(parts[2]) if len(parts) > 2 and parts[2] else None
    return sid, start, duration


def _frames_for_segment(sid, captions, seconds=None, start=0.0):
    """Yield composited BGR frames for one segment window."""
    import cv2
    meta = json.load(open(os.path.join(str(config.SEGMENTS_DIR), sid + ".json")))
    path = os.path.join(str(config.SEGMENTS_DIR), sid + ".mp4")
    spans = sorted([c for c in captions if c["segment"] == sid],
                   key=lambda c: c["start_ts"])
    t0, ov_fps = meta["t0"], meta["fps"]
    w, h = meta["w"], meta["h"]

    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_local = i / fps
        i += 1
        if t_local < start:
            continue
        if seconds and t_local - start > seconds:
            break
        t_abs = t0 + t_local
        k = int(round(t_local * ov_fps))

        for lane, colour in (("L", LEFT_HAND), ("R", RIGHT_HAND)):
            rows = meta.get(lane) or []
            if 0 <= k < len(rows):
                _draw_hand(frame, rows[k], colour, w, h)

        active = next((s for s in spans
                       if s["start_ts"] <= t_abs < s["end_ts"]), None)

        panel = np.full((PANEL_H, w, 3), PANEL, np.uint8)
        _put(panel, f"{sid}   {meta['cls']}", (12, 20), 0.40, MUTED)
        _put(panel, f"t={t_abs:6.2f}s", (w - 96, 20), 0.40, MUTED)

        if active:
            for n, line in enumerate(_wrap(active["text"], 52)):
                _put(panel, line, (12, 46 + 18 * n), 0.46, INK)
            ap = active.get("aperture_mm")
            facts = [f"hand={active['hand']}"]
            if ap:
                facts.append(f"aperture {ap[0]}-{ap[1]}mm")
            if active.get("ap_trend"):
                facts.append(active["ap_trend"])
            if active.get("fingers"):
                facts.append(active["fingers"])
            if active.get("rotation"):
                facts.append(active["rotation"])
            # Wrap rather than let a long fact list run off the panel edge.
            for n, line in enumerate(_wrap(" | ".join(facts), 66)[:2]):
                _put(panel, line, (12, 92 + 15 * n), 0.34, MUTED)
        else:
            _put(panel, "(no span)", (12, 46), 0.46, MUTED)

        _timeline(panel, spans, t_abs, 12, PANEL_H - 14, w - 24, active)

        bar = np.full((BAR_H, w, 3), PANEL, np.uint8)
        _put(bar, "hands: left", (12, 18), 0.34, LEFT_HAND)
        _put(bar, "right", (98, 18), 0.34, RIGHT_HAND)
        _put(bar, "text: VLM   spans + hand + aperture: pose",
             (166, 18), 0.34, MUTED)
        yield np.vstack([bar, frame, panel])
    cap.release()


def render(segments, out, seconds=None, gif=None, fps=None,
           gif_fps=8, gif_width=430, gif_colors=64):
    """Composite one or more segments into a single annotated clip."""
    captions = [json.loads(l) for l in open(config.CAPTIONS)]
    specs = [parse_spec(x) for x in segments]
    first = json.load(open(os.path.join(str(config.SEGMENTS_DIR),
                                        specs[0][0] + ".json")))
    import cv2
    probe = cv2.VideoCapture(os.path.join(str(config.SEGMENTS_DIR),
                                          specs[0][0] + ".mp4"))
    src_fps = fps or float(probe.get(cv2.CAP_PROP_FPS) or 30.0)
    probe.release()
    w = first["w"]
    h = first["h"] + PANEL_H + BAR_H

    out = str(out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", str(src_fps), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "26",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    for sid, start, duration in specs:
        for frame in _frames_for_segment(sid, captions,
                                         duration or seconds, start):
            proc.stdin.write(frame.tobytes())
            n += 1
    proc.stdin.close()
    proc.wait()
    print(f"wrote {out}  {n} frames  {n / src_fps:.1f}s  "
          f"{os.path.getsize(out) / 1e6:.2f} MB")

    if gif:
        # A README GIF has to stay small enough to load, so it trades frame
        # rate and palette rather than legibility of the caption panel.
        gif = str(gif)
        palette = out + ".palette.png"
        vf = f"fps={gif_fps},scale={gif_width}:-1:flags=lanczos"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", out, "-vf",
                        f"{vf},palettegen=max_colors={gif_colors}", palette],
                       check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", out, "-i", palette,
                        "-lavfi", f"{vf} [x]; [x][1:v] paletteuse="
                        "dither=bayer:bayer_scale=4", gif], check=True)
        os.unlink(palette)
        print(f"wrote {gif}  {os.path.getsize(gif) / 1e6:.2f} MB "
              f"({gif_fps} fps, {gif_width}px, {gif_colors} colors)")
    return out
