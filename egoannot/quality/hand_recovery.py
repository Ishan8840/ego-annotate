"""Multiscale hand discovery; uses the same licensed detector checkpoint.

Unlike temporal recovery, image-space discovery can initialize an unseen hand.
All returned boxes are source-image pixel xyxy, with explicit detection sources.
"""
import numpy as np

from .hand_tracking import iou, suppress, suppress_contained


def discovery_regions(width, height):
    """Two overlapping views along the long image axis; no assumed handedness."""
    if width >= height:
        span = int(np.ceil(width * .65))
        return [(0, 0, span, height), (width - span, 0, width, height)]
    span = int(np.ceil(height * .65))
    return [(0, 0, width, span), (0, height - span, width, height)]


def valid_crop_box(box, region, width, height, margin=3.):
    """Reject boxes truncated by an artificial crop edge, but allow image edges."""
    x1, y1, x2, y2 = region
    if x1 > 0 and box[0] <= x1 + margin:
        return False
    if y1 > 0 and box[1] <= y1 + margin:
        return False
    if x2 < width and box[2] >= x2 - margin:
        return False
    if y2 < height and box[3] >= y2 - margin:
        return False
    return True


class RecoveryHandDetector:
    def __init__(self, detector, discovery=True, verify_weak=True, discovery_every=1):
        if discovery_every < 1:
            raise ValueError('discovery_every must be positive')
        self.detector, self.discovery, self.verify_weak = detector, discovery, verify_weak
        self.discovery_every = discovery_every
        self.frames = 0
        self.recovery_config = dict(discovery=discovery, verify_weak=verify_weak,
                                    discovery_every=discovery_every,
                                    discovery_threshold=.20, max_weak_crops=2)
        self.model_name = detector.model_name + ' + multiscale discovery'
        for name in ['model_sha256', 'size', 'precision', 'path', 'tracker_high', 'tracker_low']:
            setattr(self, name, getattr(detector, name))
        self.prompts = getattr(detector, 'prompts', None)

    @property
    def calls(self):
        return self.detector.calls

    @property
    def seconds(self):
        return self.detector.seconds

    def reset_sequence(self):
        self.frames = 0

    def __call__(self, frame, offset=(0, 0), source='full'):
        if source != 'full' or offset != (0, 0):
            return self.detector(frame, offset, source)
        h, w = frame.shape[:2]
        full = self.detector(frame)
        accepted = list(full)
        if self.discovery and self.frames % self.discovery_every == 0:
            for region in discovery_regions(w, h):
                x1, y1, x2, y2 = region
                candidates = self.detector(frame[y1:y2, x1:x2], (x1, y1), 'discovery_tile')
                for d in candidates:
                    # Discovery-only births use a stricter threshold than full
                    # frame detections; no fixed "two hands must exist" rule.
                    if d['score'] >= .20 and valid_crop_box(d['box'], region, w, h):
                        accepted.append(d)
        if self.verify_weak:
            weak = [d for d in full if self.tracker_low <= d['score'] < self.tracker_high]
            weak.sort(key=lambda d: d['score'], reverse=True)
            calls = 0
            for seed in weak:
                if calls >= 2:
                    break
                if any(d['score'] >= self.tracker_high and iou(d['box'], seed['box']) >= .25 for d in accepted):
                    continue
                box = np.asarray(seed['box'])
                center = (box[:2] + box[2:]) / 2
                side = max(float(max(box[2:] - box[:2])) * 2.5, 96.)
                lo = np.maximum(np.floor(center - side / 2), [0, 0]).astype(int)
                hi = np.minimum(np.ceil(center + side / 2), [w, h]).astype(int)
                if min(hi - lo) < 32 or np.prod(hi - lo) > w * h * .8:
                    continue
                region = (*lo, *hi)
                candidates = self.detector(frame[lo[1]:hi[1], lo[0]:hi[0]], tuple(lo), 'verified_weak_crop')
                calls += 1
                for d in candidates:
                    if d['score'] >= self.tracker_high and iou(d['box'], seed['box']) >= .25 and valid_crop_box(d['box'], region, w, h):
                        accepted.append(d)
        self.frames += 1
        return suppress_contained(suppress(accepted))


def make_hand_detector(path, precision='fp16', accuracy='balanced'):
    """No new weights. Competing queries are background classes, not hands."""
    from .owl_hand import OwlHandDetector
    if accuracy not in ('fast', 'balanced', 'high'):
        raise ValueError('Unknown hand accuracy profile')
    prompts = None if accuracy == 'fast' else [
        'a photo of a human hand', 'a photo of an empty rubber glove',
        'a photo of a towel', 'a photo of a door handle']
    detector = OwlHandDetector(path, precision=precision, prompts=prompts)
    if accuracy == 'fast':
        return detector
    return RecoveryHandDetector(detector, discovery_every=2 if accuracy == 'balanced' else 1)
