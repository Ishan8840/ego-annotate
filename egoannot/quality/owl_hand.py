"""Sparse OWLv2 hand boxes using the Apache-2.0 official checkpoint."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .hand_tracking import suppress, suppress_contained


class OwlHandDetector:
    model_name = 'OWLv2 base / human hand prompt'
    def __init__(self, path, size=960, device='cuda', precision='fp16', fast=True, prompts=None):
        import torch
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        path = Path(path)
        manifest = json.loads((path / 'manifest.json').read_text())
        if (manifest['revision'] != 'cfd3195ba4ea9592eec887ded089f4c08eff231d'
                or manifest['weight_license'] != 'Apache-2.0'
                or manifest['files'].get('model.safetensors') != 'e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7'):
            raise ValueError('Use the audited pinned checkpoint downloader')
        for name, expected in manifest['files'].items():
            if Path(name).name != name:
                raise ValueError('Unsafe manifest path')
            with (path / name).open('rb') as handle:
                if hashlib.file_digest(handle, 'sha256').hexdigest() != expected:
                    raise ValueError('Checkpoint changed: ' + name)
        if size != 960 or precision not in ('fp32', 'fp16'):
            raise ValueError('OWLv2 requires its native 960px profile')
        if precision == 'fp16' and device != 'cuda':
            raise ValueError('FP16 profile requires CUDA')
        torch.set_num_threads(4)
        self.processor = Owlv2Processor.from_pretrained(path, local_files_only=True, use_fast=fast)
        self.model = Owlv2ForObjectDetection.from_pretrained(path, local_files_only=True, use_safetensors=True).to(device).eval()
        self.device, self.precision, self.size, self.fast = device, precision, size, fast
        self.calls, self.seconds, self.path = 0, 0., str(path)
        self.model_sha256 = manifest['files']['model.safetensors']
        self.tracker_high, self.tracker_low = .15, .08
        self.prompts = prompts or ['a photo of a human hand']

    def __call__(self, frame, offset=(0, 0), source='full'):
        import torch
        from PIL import Image
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1:
            raise ValueError('Expected nonempty uint8 BGR source image')
        began = time.perf_counter()
        h, w = frame.shape[:2]
        pixels = torch.from_numpy(np.ascontiguousarray(frame[:, :, ::-1])).permute(2, 0, 1).to(self.device) if self.fast else Image.fromarray(frame[:, :, ::-1])
        inputs = self.processor(images=pixels, text=[self.prompts], return_tensors='pt').to(self.device)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=self.precision == 'fp16'):
            outputs = self.model(**inputs)
        result = self.processor.post_process_object_detection(outputs=outputs, threshold=self.tracker_low, target_sizes=[(h, w)])[0]
        detections = []
        for box, score, label in zip(result['boxes'].float().cpu().numpy(), result['scores'].float().cpu().numpy(), result['labels'].cpu().numpy()):
            # Optional competing background queries suppress lookalikes. Only
            # the first prompt is the target; other query labels are not hands.
            if label != 0:
                continue
            if not np.isfinite(box).all() or not np.isfinite(score):
                continue
            box = np.clip(box, [0, 0, 0, 0], [w, h, w, h])
            if min(box[2:] - box[:2]) < 3:
                continue
            box += np.array([*offset, *offset])
            detections.append(dict(box=box.tolist(), score=float(score), source=source))
        self.calls += 1
        self.seconds += time.perf_counter() - began
        return suppress_contained(suppress(detections))
