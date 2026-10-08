"""Shared OWLv2 executor for independent episode workers.

Each episode owns its RecoveryHandDetector and tracking state. Only this worker
accesses the GPU model. Outstanding image requests are bounded by the number of
synchronous episode callers. Batch size 1 uses the incumbent inference method.
"""
from concurrent.futures import Future
import queue
import threading
import time


class SharedHandExecutor:
    def __init__(self, detector, max_batch=1, wait_ms=3):
        if max_batch < 1 or wait_ms < 0:
            raise ValueError('Invalid batching limits')
        self.detector, self.max_batch = detector, max_batch
        self.wait_s = wait_ms / 1000
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.closed = False
        self.stats = []
        self.thread = threading.Thread(target=self._work, name='shared-hand-gpu', daemon=True)
        self.thread.start()

    def client(self):
        return HandClient(self)

    def submit(self, frame, offset, source):
        with self.lock:
            if self.closed:
                raise RuntimeError('Shared detector is closed')
            future = Future()
            self.queue.put((frame, offset, source, future))
        return future.result()

    def _work(self):
        while True:
            first = self.queue.get()
            if first is None:
                return
            batch = [first]
            deadline = time.perf_counter() + self.wait_s
            stop = False
            while len(batch) < self.max_batch:
                try:
                    item = self.queue.get(timeout=max(0., deadline - time.perf_counter()))
                except queue.Empty:
                    break
                if item is None:
                    stop = True
                    break
                batch.append(item)
            started = time.perf_counter()
            try:
                if len(batch) == 1:
                    outputs = [self.detector(*batch[0][:3])]
                else:
                    outputs = owl_batch(self.detector, [item[:3] for item in batch])
                elapsed = time.perf_counter() - started
                if len(outputs) != len(batch):
                    raise RuntimeError('Batch output count mismatch')
                self.stats.append(dict(size=len(batch), seconds=elapsed))
                for item, output in zip(batch, outputs):
                    item[3].set_result((output, elapsed / len(batch)))
            except Exception as exc:
                for item in batch:
                    item[3].set_exception(exc)
            if stop:
                return

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                self.queue.put(None)
        self.thread.join()


class HandClient:
    """Per-episode counters; never share recovery cadence or tracking state."""
    def __init__(self, executor):
        self.executor = executor
        self.calls, self.seconds = 0, 0.
        for name in ['model_name', 'model_sha256', 'size', 'precision', 'path',
                     'tracker_high', 'tracker_low', 'prompts']:
            setattr(self, name, getattr(executor.detector, name, None))

    def __call__(self, frame, offset=(0, 0), source='full'):
        output, seconds = self.executor.submit(frame, offset, source)
        self.calls += 1
        self.seconds += seconds
        return output


def owl_batch(detector, requests):
    """Keep individual preprocessing, concatenate tensors only at model forward.

Batch numerics can differ from singleton inference. This path is an experiment,
not a claim of equivalent detections; compare complete reports before adoption.
"""
    import numpy as np
    import torch
    from .hand_tracking import suppress, suppress_contained
    if not detector.fast or detector.device != 'cuda':
        raise ValueError('Batch experiment requires the existing fast CUDA profile')
    prepared, sizes = [], []
    for frame, _, _ in requests:
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1:
            raise ValueError('Expected nonempty uint8 BGR source image')
        sizes.append(frame.shape[:2])
        pixels = torch.from_numpy(np.ascontiguousarray(frame[:, :, ::-1])).permute(2, 0, 1).to(detector.device)
        prepared.append(detector.processor(images=pixels, text=[detector.prompts], return_tensors='pt').to(detector.device))
    inputs = {key: torch.cat([item[key] for item in prepared], dim=0) for key in prepared[0]}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=detector.precision == 'fp16'):
        outputs = detector.model(**inputs)
    results = detector.processor.post_process_object_detection(outputs=outputs, threshold=detector.tracker_low,
                                                               target_sizes=sizes)
    all_detections = []
    for result, (h, w), (_, offset, source) in zip(results, sizes, requests):
        detections = []
        for box, score, label in zip(result['boxes'].float().cpu().numpy(), result['scores'].float().cpu().numpy(), result['labels'].cpu().numpy()):
            if label != 0 or not np.isfinite(box).all() or not np.isfinite(score):
                continue
            box = np.clip(box, [0, 0, 0, 0], [w, h, w, h])
            if min(box[2:] - box[:2]) < 3:
                continue
            box += np.array([*offset, *offset])
            detections.append(dict(box=box.tolist(), score=float(score), source=source))
        all_detections.append(suppress_contained(suppress(detections)))
    return all_detections
