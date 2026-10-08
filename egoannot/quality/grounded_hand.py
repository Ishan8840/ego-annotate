"""Apache-2.0 Grounding DINO detector; no generated captions or hand pose.

Run sparsely and track in source-image pixel coordinates. Scores are model
similarities, not calibrated probabilities of correct hand localization.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .hand_tracking import suppress

REVISION = 'a2bb814dd30d776dcf7e30523b00659f4f141c71'


class GroundedHandDetector:
    model_name = 'Grounding DINO Tiny / hand prompt'

    def __init__(self, path, size=640, device='cuda', precision='fp32'):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        path = Path(path)
        manifest = json.loads((path / 'manifest.json').read_text())
        if manifest['revision'] != REVISION or manifest['weight_license'] != 'Apache-2.0':
            raise ValueError('Use the audited pinned checkpoint downloader')
        for name, expected in manifest['files'].items():
            if Path(name).name != name:
                raise ValueError('Unsafe manifest path')
            with (path / name).open('rb') as handle:
                if hashlib.file_digest(handle, 'sha256').hexdigest() != expected:
                    raise ValueError('Checkpoint changed: ' + name)
        if size < 128 or precision not in ('fp32', 'fp16'):
            raise ValueError('Invalid detector configuration')
        if precision == 'fp16' and device != 'cuda':
            raise ValueError('FP16 profile requires CUDA')
        torch.set_num_threads(4)
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.processor.image_processor.size = dict(shortest_edge=size, longest_edge=round(size * 1333 / 800))
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            path, local_files_only=True, use_safetensors=True, disable_custom_kernels=True).to(device).eval()
        self.device, self.precision, self.size = device, precision, size
        self.calls, self.seconds, self.path = 0, 0., str(path)
        self.model_sha256 = manifest['files']['model.safetensors']
        self.tracker_high, self.tracker_low = .30, .20

    def __call__(self, frame, offset=(0, 0), source='full'):
        import torch
        from PIL import Image
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1:
            raise ValueError('Expected nonempty uint8 BGR source image')
        began = time.perf_counter()
        h, w = frame.shape[:2]
        inputs = self.processor(images=Image.fromarray(frame[:, :, ::-1]), text='hand.', return_tensors='pt').to(self.device)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=self.precision == 'fp16'):
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=self.tracker_low, text_threshold=.20,
            target_sizes=[(h, w)])[0]
        detections = []
        # CPU transfer synchronizes CUDA, so measured time includes completed GPU work.
        for box, score in zip(result['boxes'].float().cpu().numpy(), result['scores'].float().cpu().numpy()):
            if not np.isfinite(box).all() or not np.isfinite(score):
                continue
            box = np.clip(box, [0, 0, 0, 0], [w, h, w, h])
            if min(box[2:] - box[:2]) < 3:
                continue
            box += np.array([*offset, *offset])
            detections.append(dict(box=box.tolist(), score=float(score), source=source))
        self.calls += 1
        self.seconds += time.perf_counter() - began
        return suppress(detections)
