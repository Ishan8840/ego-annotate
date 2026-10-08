"""Small licensed hand detector + conservative temporal box association.

Coordinates are pixel xyxy in the source image. IDs are track IDs, not anatomical
left/right labels. A predicted box is never evidence that a hand is visible.
"""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

MODEL_SHA256 = '568d3ea97a5b142488366b67e036b6a5cb0a1fef9087a710cb8e66b6979fbac2'


def iou(a, b):
    lo, hi = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    intersection = float(np.prod(np.maximum(hi - lo, 0)))
    area_a = float(np.prod(np.maximum(np.asarray(a[2:]) - a[:2], 0)))
    area_b = float(np.prod(np.maximum(np.asarray(b[2:]) - b[:2], 0)))
    return intersection / max(area_a + area_b - intersection, 1e-9)


def suppress(detections, threshold=.5):
    selected = []
    for candidate in sorted(detections, key=lambda d: d['score'], reverse=True):
        if all(iou(candidate['box'], old['box']) < threshold for old in selected):
            selected.append(candidate)
    return selected


def suppress_contained(detections, threshold=.85):
    """Remove low-score enclosing duplicates around a stronger tight box.

    Ordinary IoU NMS misses these when an arm-sized box surrounds a hand box.
    This does not claim to identify wrists or establish anatomical box extent.
    """
    selected = []
    for d in sorted(detections, key=lambda d: d['score'], reverse=True):
        box = np.asarray(d['box'])
        area = float(np.prod(np.maximum(box[2:] - box[:2], 0)))
        duplicate = False
        for old in selected:
            other = np.asarray(old['box'])
            small = float(np.prod(np.maximum(other[2:] - other[:2], 0)))
            intersection = float(np.prod(np.maximum(np.minimum(box[2:], other[2:]) - np.maximum(box[:2], other[:2]), 0)))
            if area > small and intersection / max(small, 1e-9) >= threshold:
                duplicate = True
                break
        if not duplicate:
            selected.append(d)
    return selected


def letterbox(frame, size=320):
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError('Expected uint8 BGR image')
    h, w = frame.shape[:2]
    if min(h, w) < 1:
        raise ValueError('Empty detector image')
    ratio = min(size / w, size / h)
    resized = cv2.resize(frame, (max(1, int(w * ratio)), max(1, int(h * ratio))))
    padded = np.full((size, size, 3), 114, np.uint8)
    padded[:resized.shape[0], :resized.shape[1]] = resized
    # Upstream exported pipeline.json uses BGR normalization (to_rgb=false).
    normalized = (padded.astype(np.float32) - np.array([103.53, 116.28, 123.675], np.float32)) / np.array([57.375, 57.12, 58.395], np.float32)
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None]), ratio


class HandDetector:
    model_name = 'RTMDet-Nano hand ONNX (comparison only)'
    model_sha256 = MODEL_SHA256
    def __init__(self, path, threads=1):
        import onnxruntime as ort
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
            raise ValueError('Unexpected detector checkpoint; use the audited downloader')
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(path), sess_options=options,
                                            providers=['CPUExecutionProvider'])
        if self.session.get_inputs()[0].shape != [1, 3, 320, 320]:
            raise ValueError('Unexpected detector input geometry')
        self.name = self.session.get_inputs()[0].name
        self.calls, self.seconds = 0, 0.
        self.path = str(path)

    def __call__(self, frame, offset=(0, 0), source='full'):
        began = time.perf_counter()
        tensor, ratio = letterbox(frame)
        dets, labels = self.session.run(['dets', 'labels'], {self.name: tensor})
        if dets.ndim != 3 or dets.shape[-1] != 5 or labels.shape != dets.shape[:2]:
            raise ValueError('Unexpected detector output shape')
        h, w = frame.shape[:2]
        result = []
        for row, label in zip(dets[0], labels[0]):
            if label != 0 or row[4] < .10 or not np.isfinite(row).all():
                continue
            raw_box = row[:4] / ratio
            box = np.clip(raw_box, [0, 0, 0, 0], [w, h, w, h])
            if min(box[2:] - box[:2]) < 3:
                continue
            box += np.array([offset[0], offset[1], offset[0], offset[1]])
            result.append(dict(box=box.tolist(), score=float(row[4]), source=source))
        self.calls += 1
        self.seconds += time.perf_counter() - began
        return result


@dataclass
class Track:
    id: int
    box: np.ndarray
    velocity: np.ndarray
    last_time: float
    last_observed: float
    last_strong: float
    score: float
    observed_box: np.ndarray
    hits: int = 1

    def predict(self, t):
        # Bound velocity extrapolation: beyond 300 ms retain uncertainty instead
        # of accelerating a box across the image throughout an occlusion.
        return self.box + self.velocity * min(max(t - self.last_time, 0), .3)


class BoxTracker:
    def __init__(self, high=.45, low=.15, max_gap=.7, max_tracks=4, weak_gap=.3):
        if not 0 <= low < high <= 1 or max_gap <= 0:
            raise ValueError('Invalid tracking thresholds')
        self.high, self.low, self.max_gap, self.max_tracks = high, low, max_gap, max_tracks
        self.tracks, self.next_id, self.previous_time = [], 1, None
        self.weak_gap = weak_gap

    def update(self, detections, t):
        if not math.isfinite(t) or (self.previous_time is not None and t <= self.previous_time):
            raise ValueError('Tracking requires increasing finite timestamps')
        self.previous_time = t
        self.tracks = [track for track in self.tracks if t - track.last_strong <= self.max_gap]
        strong = [d for d in detections if d['score'] >= self.high]
        weak = [d for d in detections if self.low <= d['score'] < self.high]
        unmatched = set(range(len(self.tracks)))
        observed = {}
        for candidates, weak_stage in ((strong, False), (weak, True)):
            indices = sorted(unmatched)
            pairs = []
            if indices and candidates:
                cost = np.full((len(indices), len(candidates)), 1e6)
                for a, index in enumerate(indices):
                    tr = self.tracks[index]
                    pred = tr.predict(t)
                    for b, candidate in enumerate(candidates):
                        box = np.asarray(candidate['box'])
                        overlap = iou(pred, box)
                        scale = max(float(np.linalg.norm(pred[2:] - pred[:2])), 10.)
                        distance = float(np.linalg.norm((pred[:2] + pred[2:] - box[:2] - box[2:]) / 2) / scale)
                        if (overlap >= (.2 if weak_stage else .05) or (not weak_stage and distance < .6)) and (not weak_stage or t - tr.last_strong <= self.weak_gap):
                            cost[a, b] = 1 - overlap + .2 * distance
                aa, bb = linear_sum_assignment(cost)
                pairs = [(indices[a], int(b)) for a, b in zip(aa, bb) if cost[a, b] < 1e5]
            assigned = set()
            for index, b in pairs:
                tr, d = self.tracks[index], candidates[b]
                box = np.asarray(d['box'], dtype=float)
                dt = max(t - tr.last_observed, .001)
                velocity = (box - tr.observed_box) / dt
                # Prevent a single large detector box jump from producing a
                # strong extrapolated exit on the following frame.
                cap = max(float(np.linalg.norm(box[2:] - box[:2])) * 3, 30.)
                tr.velocity = .5 * tr.velocity + .5 * np.clip(velocity, -cap, cap)
                tr.box, tr.observed_box = box, box.copy()
                tr.last_time = tr.last_observed = t
                tr.score = d['score']
                tr.hits += 1
                if not weak_stage:
                    tr.last_strong = t
                observed[tr.id] = dict(d, status='observed_weak' if weak_stage else 'observed_strong')
                unmatched.discard(index)
                assigned.add(b)
            if not weak_stage:
                for b, d in enumerate(candidates):
                    if b in assigned or len(self.tracks) >= self.max_tracks:
                        continue
                    box = np.asarray(d['box'], dtype=float)
                    tr = Track(self.next_id, box, np.zeros(4), t, t, t, d['score'], box.copy())
                    self.next_id += 1
                    self.tracks.append(tr)
                    observed[tr.id] = dict(d, status='observed_strong')
        return observed

    def snapshot(self, t, width, height, observed=None, detection_ran=False):
        observed = observed or {}
        rows = []
        for tr in self.tracks:
            if t - tr.last_strong > self.max_gap:
                continue
            d = observed.get(tr.id)
            box = np.asarray(d['box']) if d else tr.predict(t)
            box = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
            if min(box[2:] - box[:2]) < 1:
                continue
            margin = .025 * min(width, height)
            edge = box[0] < margin or box[1] < margin or box[2] > width - margin or box[3] > height - margin
            status = d['status'] if d else ('predicted_gap' if detection_ran else 'propagated_between_detections')
            rows.append(dict(track_id=tr.id, box=box.tolist(), score=d['score'] if d else tr.score,
                status=status, observed=bool(d), crop_candidate=bool(edge) if d else False,
                detection_source=d['source'] if d else None,
                seconds_since_observation=float(t - tr.last_observed)))
        return rows


class SparseFlow:
    """Forward/backward-checked KLT translation; never a hand observation.

    Features are redetected inside each prior box. Background features can still
    drift; bounded lifetime and fresh full-image detections limit that risk.
    """
    def __init__(self):
        self.previous = None

    def update(self, frame, tracks, t):
        h, w = frame.shape[:2]
        scale = min(1., 640 / max(h, w))
        small = cv2.resize(frame, (round(w * scale), round(h * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sx, sy = gray.shape[1] / w, gray.shape[0] / h
        scale4 = np.array([sx, sy, sx, sy])
        supported = set()
        if self.previous is not None and self.previous.shape == gray.shape:
            groups = []
            for tr in tracks:
                mask = np.zeros_like(gray)
                x1, y1, x2, y2 = np.round(tr.box * scale4).astype(int)
                x1, x2 = np.clip([x1, x2], 0, gray.shape[1])
                y1, y2 = np.clip([y1, y2], 0, gray.shape[0])
                mask[y1:y2, x1:x2] = 255
                pts = cv2.goodFeaturesToTrack(self.previous, maxCorners=30, qualityLevel=.03, minDistance=4, mask=mask)
                if pts is not None and len(pts) >= 4:
                    groups.append((tr, pts))
            if groups:
                # One pyramid build and one forward/backward pair for all hands.
                pts = np.concatenate([p for _, p in groups])
                nxt, ok, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, pts, None, winSize=(21, 21), maxLevel=2)
                if nxt is not None:
                    back, okback, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, nxt, None, winSize=(21, 21), maxLevel=2)
                    if back is not None:
                        valid = (ok.ravel() > 0) & (okback.ravel() > 0) & (np.linalg.norm(back[:, 0] - pts[:, 0], axis=1) < 1.5)
                        start = 0
                        for tr, original in groups:
                            end = start + len(original)
                            good = valid[start:end]
                            delta = nxt[start:end, 0][good] - original[:, 0][good]
                            start = end
                            if len(delta) < 4:
                                continue
                            median = np.median(delta, axis=0)
                            agreeing = np.linalg.norm(delta - median, axis=1) < 3
                            if agreeing.sum() < 4 or agreeing.mean() < .5:
                                continue
                            shift = np.median(delta[agreeing], axis=0) / [sx, sy]
                            if np.linalg.norm(shift) > .5 * np.linalg.norm(tr.box[2:] - tr.box[:2]):
                                continue
                            tr.box = tr.box + np.tile(shift, 2)
                            tr.last_time = t
                            tr.velocity[:] = 0
                            supported.add(tr.id)
        self.previous = gray
        return supported


class HandVisibility:
    def __init__(self, detector, sample_hz=10., profile='adaptive', flow=False):
        if not math.isfinite(sample_hz) or sample_hz <= 0 or profile not in ('baseline', 'adaptive'):
            raise ValueError('Invalid detector cadence or profile')
        self.detector, self.sample_hz, self.profile = detector, sample_hz, profile
        if hasattr(detector, 'reset_sequence'):
            detector.reset_sequence()
        self.tracker = BoxTracker(high=getattr(detector, 'tracker_high', .45),
                                  low=getattr(detector, 'tracker_low', .15),
                                  max_gap=max(.7, 1.5 / sample_hz), weak_gap=max(.3, 1 / sample_hz + 1e-5))
        self.flow = SparseFlow() if flow else None
        self.last_detection = -math.inf
        self.next_detection = None
        self.last_frame, self.last_time = -1, -math.inf
        self.rows, self.seconds = [], 0.
        self.start_calls, self.start_seconds = detector.calls, detector.seconds

    def __call__(self, index, t, frame):
        began = time.perf_counter()
        if index != self.last_frame + 1 or not math.isfinite(t) or t <= self.last_time:
            raise ValueError('Hand tracking requires sequential frames with increasing PTS')
        self.last_frame, self.last_time = index, t
        h, w = frame.shape[:2]
        self.tracker.tracks = [tr for tr in self.tracker.tracks if t - tr.last_strong <= self.tracker.max_gap]
        flow_ids = self.flow.update(frame, self.tracker.tracks, t) if self.flow else set()
        ran = self.next_detection is None or t >= self.next_detection - 1e-5
        observed, detections, crop_calls = {}, [], 0
        if ran:
            detections = self.detector(frame)
            if self.profile == 'adaptive':
                # Full-frame search is never disabled. Extra crops only recover
                # recent tracks with weak/no full-frame support, bounded to two.
                for tr in sorted(self.tracker.tracks, key=lambda tr: tr.last_strong, reverse=True):
                    if crop_calls >= 2 or t - tr.last_strong > .5:
                        continue
                    pred = tr.predict(t)
                    if any(d['score'] >= self.tracker.high and iou(d['box'], pred) > .25 for d in detections):
                        continue
                    center = (pred[:2] + pred[2:]) / 2
                    side = max(float(np.max(pred[2:] - pred[:2])) * 2.2, 80.)
                    lo = np.maximum(np.floor(center - side / 2), [0, 0]).astype(int)
                    hi = np.minimum(np.ceil(center + side / 2), [w, h]).astype(int)
                    if min(hi - lo) < 16 or np.prod(hi - lo) > w * h * .85:
                        continue
                    candidates = self.detector(frame[lo[1]:hi[1], lo[0]:hi[0]], tuple(lo), 'recovery_crop')
                    # Crop detections must overlap the known track; unrelated
                    # objects seen only in a tiny crop cannot create new IDs.
                    detections.extend(d for d in candidates if iou(d['box'], pred) >= .15)
                    crop_calls += 1
            detections = suppress(detections)
            observed = self.tracker.update(detections, t)
            self.last_detection = t
            # Keep deadlines on the requested time grid. Resetting the clock to
            # each sampled frame would make 4 Hz become 3.75 Hz on 30 fps input.
            if self.next_detection is None:
                self.next_detection = t
            steps = math.floor(max(0., t - self.next_detection) * self.sample_hz + 1e-5) + 1
            self.next_detection += steps / self.sample_hz
        tracks = self.tracker.snapshot(t, w, h, observed, ran)
        for tr in tracks:
            tr['flow_supported'] = tr['track_id'] in flow_ids and not tr['observed']
        row = dict(frame_id=index, time_s=float(t), detection_ran=ran, recovery_calls=crop_calls,
                   strong_count=sum(x['status'] == 'observed_strong' for x in tracks),
                   weak_count=sum(x['status'] == 'observed_weak' for x in tracks),
                   crop_candidate=any(x['crop_candidate'] for x in tracks), tracks=tracks)
        self.rows.append(row)
        self.seconds += time.perf_counter() - began
        return row

    def finish(self, duration):
        if not self.rows or duration <= self.rows[-1]['time_s']:
            raise ValueError('Invalid hand tracking duration')
        samples = [r for r in self.rows if r['detection_ran']]
        def intervals(predicate):
            ranges = []
            for i, row in enumerate(samples):
                if not predicate(row):
                    continue
                end = samples[i + 1]['time_s'] if i + 1 < len(samples) else duration
                # No evidence is extrapolated across an unexpectedly large PTS gap.
                end = min(end, row['time_s'] + 1.5 / self.sample_hz)
                if ranges and abs(ranges[-1][1] - row['time_s']) < 1e-5:
                    ranges[-1][1] = end
                else:
                    ranges.append([row['time_s'], end])
            return ranges
        gaps = intervals(lambda r: not r['strong_count'] and not r['weak_count'])
        strong_gaps = intervals(lambda r: not r['strong_count'])
        crop_ranges = intervals(lambda r: r['crop_candidate'])
        denominator = len(samples)
        return dict(model=getattr(self.detector, 'model_name', 'test detector'),
            model_sha256=getattr(self.detector, 'model_sha256', None), mediapipe=False,
            optical_flow=self.flow is not None, detector_size=getattr(self.detector, 'size', 320),
            precision=getattr(self.detector, 'precision', 'fp32'),
            high_threshold=self.tracker.high, low_threshold=self.tracker.low,
            recovery=getattr(self.detector, 'recovery_config', None),
            detection_prompts=getattr(self.detector, 'prompts', None),
            profile=self.profile, sample_hz=self.sample_hz, detector_samples=denominator,
            observed_any_sample_pct=100 * sum(r['strong_count'] + r['weak_count'] > 0 for r in samples) / denominator,
            observed_strong_sample_pct=100 * sum(r['strong_count'] > 0 for r in samples) / denominator,
            two_strong_hands_sample_pct=100 * sum(r['strong_count'] >= 2 for r in samples) / denominator,
            crop_candidate_sample_pct=100 * sum(r['crop_candidate'] for r in samples) / denominator,
            no_detection_intervals_s=gaps, no_strong_detection_intervals_s=strong_gaps,
            crop_candidate_intervals_s=crop_ranges,
            longest_no_detection_gap_s=max((b - a for a, b in gaps), default=0.),
            inference_calls=self.detector.calls - self.start_calls,
            inference_seconds=self.detector.seconds - self.start_seconds,
            tracking_stage_seconds=self.seconds,
            frames=self.rows,
            limitations=['Detection coverage is not true visibility recall; absence, occlusion and missed detection remain ambiguous.',
                'Predicted/interpolated boxes never contribute to observed coverage.',
                'Border proximity is a cropping candidate, not verified missing fingertips.',
                'No anatomical handedness, keypoints, 3D pose, task relevance or calibrated utility score.'])
