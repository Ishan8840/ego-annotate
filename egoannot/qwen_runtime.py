"""Persistent Qwen inference with an optional independent-request microbatcher.

The conservative path keeps the incumbent BF16 weights, processor, prompts,
pixels, and output cap. Batching is experimental because floating-point changes
can alter greedy decoding. There is no response cache or shared conversation.
"""
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import queue
import threading
import time


@dataclass
class Request:
    system: str
    parts: list
    batch: list
    future: Future


class QwenRuntime:
    name = 'qwen-local'

    def __init__(self, model_id, max_batch_size=1, cpu_threads=8,
                 batch_wait_ms=5, image_cache_bytes=128 * 1024**2,
                 trace_dir=None):
        self._validate_limits(max_batch_size, cpu_threads, batch_wait_ms, image_cache_bytes)
        import torch
        from .stages.caption import QwenLocal
        from .config import CAPTION

        if max_batch_size > 1 and not CAPTION.get('greedy'):
            raise ValueError('Microbatching requires CAPTION_GREEDY=1')
        torch.set_num_threads(cpu_threads)
        started = time.perf_counter()
        self.engine = QwenLocal(model_id)
        self.load_seconds = time.perf_counter() - started
        self.torch = torch
        self._start_worker(max_batch_size, batch_wait_ms, image_cache_bytes, trace_dir)

    @staticmethod
    def _validate_limits(batch_size, threads, wait_ms, cache_bytes):
        if batch_size < 1 or threads < 1 or wait_ms < 0 or cache_bytes < 0:
            raise ValueError('Invalid runtime limits')

    def _start_worker(self, max_batch_size, batch_wait_ms, cache_bytes, trace_dir):
        self.max_batch_size = max_batch_size
        self.batch_wait_s = batch_wait_ms / 1000
        self.cache_limit = cache_bytes
        self.images = OrderedDict()
        self.cache_bytes = 0
        self.queue = queue.Queue()
        self.local = threading.local()
        self.stats = []
        self.closed = False
        self.state_lock = threading.Lock()
        self.trace_dir = Path(trace_dir) if trace_dir else None
        if self.trace_dir:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.worker = threading.Thread(target=self._work, daemon=True, name='qwen-gpu')
        self.worker.start()

    @property
    def last_raw(self):
        # Existing stages read last_raw immediately after calling the engine.
        # A single shared string here would silently mix different stages' replies.
        return getattr(self.local, 'last_raw', '')

    def __call__(self, system, parts, batch):
        from .stages.caption import parse_objects

        with self.state_lock:
            if self.closed:
                raise RuntimeError('Runtime has been closed')
            future = Future()
            self.queue.put(Request(system, parts, batch, future))
        raw, usage = future.result()
        self.local.last_raw = raw
        return parse_objects(raw, batch), usage

    def _image(self, blob):
        """Cache only decoded RGB pixels, with content identity and an LRU bound."""
        from PIL import Image

        key = hashlib.sha256(blob).digest()
        if key in self.images:
            self.images.move_to_end(key)
            return self.images[key]
        with Image.open(io.BytesIO(blob)) as source:
            image = source.convert('RGB')
        size = image.width * image.height * 3
        if size <= self.cache_limit:
            while self.images and self.cache_bytes + size > self.cache_limit:
                _, old = self.images.popitem(last=False)
                self.cache_bytes -= old.width * old.height * 3
            self.images[key] = image
            self.cache_bytes += size
        return image

    def _infer(self, requests):
        from .config import CAPTION as cfg

        proc, model = self.engine.proc, self.engine.model
        texts, images, counts = [], [], []
        for request in requests:
            content, count = [], 0
            for kind, value in request.parts:
                if kind == 'text':
                    content.append({'type': 'text', 'text': value})
                else:
                    content.append({'type': 'image'})
                    images.append(self._image(value))
                    count += 1
            messages = [
                {'role': 'system', 'content': [{'type': 'text', 'text': request.system}]},
                {'role': 'user', 'content': content},
            ]
            texts.append(proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            counts.append(count)
        # Size one retains the incumbent processor kwargs exactly.
        kwargs = {} if len(requests) == 1 else dict(padding=True, padding_side='left')
        inputs = proc(text=texts, images=images, return_tensors='pt', **kwargs).to(model.device)
        generation = dict(max_new_tokens=1024, do_sample=not cfg.get('greedy'))
        if getattr(self, 'prompt_lookup_tokens', 0):
            generation['prompt_lookup_num_tokens'] = self.prompt_lookup_tokens
        if not cfg.get('greedy'):
            generation.update(temperature=cfg['temperature'], top_p=.9)
        with self.torch.inference_mode():
            generated = model.generate(**inputs, **generation)
        trimmed = generated[:, inputs.input_ids.shape[1]:]
        results = []
        for index in range(len(requests)):
            tokens = trimmed[index].tolist()
            eos = model.generation_config.eos_token_id
            eos = [eos] if isinstance(eos, int) else (eos or [])
            stop = next((pos + 1 for pos, token in enumerate(tokens) if token in eos), len(tokens))
            tokens = tokens[:stop]
            raw = proc.decode(tokens, skip_special_tokens=True)
            results.append((raw, dict(
                input_tokens=int(inputs.attention_mask[index].sum()),
                output_tokens=len(tokens), n_images=counts[index],
                batch_size=len(requests), output_token_ids=tokens,
            )))
        return results

    def _trace(self, request, raw, usage, index):
        import json

        if self.trace_dir is None:
            return
        parts = []
        for kind, value in request.parts:
            if kind == 'image':
                name = hashlib.sha256(value).hexdigest() + '.jpg'
                path = self.trace_dir / name
                if not path.exists():
                    path.write_bytes(value)
                parts.append([kind, name])
            else:
                parts.append([kind, value])
        row = dict(index=index, system=request.system, parts=parts,
                   batch=request.batch, raw=raw, usage=usage)
        (self.trace_dir / f'{index:06d}.json').write_text(json.dumps(row, indent=2) + '\n')

    def _work(self):
        index = 0
        while True:
            first = self.queue.get()
            if first is None:
                return
            requests = [first]
            deadline = time.monotonic() + self.batch_wait_s
            while len(requests) < self.max_batch_size:
                try:
                    item = self.queue.get(timeout=max(0, deadline - time.monotonic()))
                except queue.Empty:
                    break
                if item is None:
                    self.queue.put(None)
                    break
                requests.append(item)
            started = time.perf_counter()
            try:
                results = self._infer(requests)
                if len(results) != len(requests):
                    raise RuntimeError('Incomplete inference batch')
                self.torch.cuda.synchronize()
                self.stats.append(dict(
                    batch_size=len(requests), seconds=time.perf_counter() - started,
                    input_tokens=sum(usage['input_tokens'] for _, usage in results),
                    output_tokens=sum(usage['output_tokens'] for _, usage in results),
                ))
                for request, (raw, usage) in zip(requests, results):
                    self._trace(request, raw, usage, index)
                    index += 1
                    request.future.set_result((raw, usage))
            except Exception as error:
                # Every waiting caller must be released, including trace failures
                # and malformed backend results. Never substitute an empty reply.
                for request in requests:
                    if not request.future.done():
                        request.future.set_exception(error)

    def close(self):
        with self.state_lock:
            if self.closed:
                return
            self.closed = True
            self.queue.put(None)
        self.worker.join()


class VLLMRuntime(QwenRuntime):
    """Experimental backend; use a separate environment and compare outputs."""

    def __init__(self, model_id, max_batch_size=4, cpu_threads=8,
                 batch_wait_ms=20, image_cache_bytes=128 * 1024**2,
                 trace_dir=None):
        self._validate_limits(max_batch_size, cpu_threads, batch_wait_ms, image_cache_bytes)
        import torch
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        from .config import CAPTION

        if not CAPTION.get('greedy'):
            raise ValueError('vLLM comparison requires CAPTION_GREEDY=1')
        torch.set_num_threads(cpu_threads)
        self.torch = torch
        started = time.perf_counter()
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.llm = LLM(
            model=model_id, dtype='bfloat16', max_model_len=32768,
            max_num_seqs=max_batch_size, gpu_memory_utilization=.85,
            limit_mm_per_prompt={'image': 20, 'video': 0}, enable_prefix_caching=True,
            mm_processor_cache_gb=4, seed=0,
        )
        self.sampling = SamplingParams(temperature=0, max_tokens=1024)
        self.load_seconds = time.perf_counter() - started
        self._start_worker(max_batch_size, batch_wait_ms, image_cache_bytes, trace_dir)

    def _infer(self, requests):
        prompts, counts = [], []
        for request in requests:
            content, images = [], []
            for kind, value in request.parts:
                if kind == 'text':
                    content.append({'type': 'text', 'text': value})
                else:
                    content.append({'type': 'image'})
                    images.append(self._image(value))
            messages = [
                {'role': 'system', 'content': [{'type': 'text', 'text': request.system}]},
                {'role': 'user', 'content': content},
            ]
            text = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append({'prompt': text, 'multi_modal_data': {'image': images}})
            counts.append(len(images))
        outputs = self.llm.generate(prompts, self.sampling, use_tqdm=False)
        return [(output.outputs[0].text, dict(
            input_tokens=len(output.prompt_token_ids),
            output_tokens=len(output.outputs[0].token_ids),
            output_token_ids=list(output.outputs[0].token_ids),
            n_images=count, batch_size=len(requests),
        )) for output, count in zip(outputs, counts)]
