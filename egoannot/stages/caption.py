"""
Model-agnostic span captioner.

Spans come from the activity-trough segmenter, NOT from contact detection: the
contact detector sits at F1 0.12-0.16 against gold and would cap density at
15/min with 8-14% recall. Troughs give ~30 spans/min uniformly across all
action classes, including tool-mediated work where contact detection is blind.

Fields the VLM is NOT asked for, because pose gives them free and more
reliably: `start_ts`, `end_ts`, `hand`, plus aperture / rotation / finger
state as context. The VLM produces only `text`, `verb`, `noun`, `visibility`
and an `uncertain` flag.

Backends are swappable. `openai` targets any OpenAI-compatible server (vLLM,
SGLang, llama.cpp, LM Studio), which is how a local Qwen-VL is reached.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time

import numpy as np

from .. import config
from ..core.video import SegmentFrames
from ..labels import domains as DM

CFG = config.CAPTION


# ---------------------------------------------------------------- prompt
def system_prompt(pack_key=None, frames_per_span=None, free_text=None):
    """
    Universal rules + the core verbs + the verified exemplars.

    Intentionally domain-agnostic: putting domain vocabulary and object lists
    in the prompt was measured to cost 10 points of uniqueness (88.7% -> 79.0%)
    for zero atomicity gain, because the model reaches for the listed words
    instead of describing what it sees. Vocabulary belongs in the LINTER.
    """
    n = frames_per_span if frames_per_span is not None else CFG["frames_per_span"]
    free = CFG["free_text"] if free_text is None else free_text
    rules = DM.CORE_RULES
    if free:
        rules = rules.replace(
            rules[rules.index("R2."):rules.index("R3.")], DM.FREE_TEXT_R2 + "\n")
    return (
        "You caption short atomic actions in egocentric (head-camera) video of "
        "manipulation work.\n\n"
        "You are given a batch of consecutive spans from one continuous episode. "
        "Each span has already been cut at a hand-motion boundary; you do NOT "
        f"decide the boundaries. For each span you see {n} frames sampled evenly "
        "across it.\n\n"
        "For each span output one JSON object:\n"
        '  {"span_id": "<given>", "text": "...", "verb": "...", "noun": "...", '
        '"visibility": "...", "uncertain": false}\n\n'
        f"{rules}\n\n{DM.DETAIL_RULES}\n\n"
        f"The verb must be one of: {', '.join(DM.CORE_VERBS)}\n\n"
        "Worked examples - each is 10-15 words and passes validation:\n"
        + "\n".join("  " + e for e in DM.PROMPT_EXEMPLARS) + "\n\n"
        "Do not output `hand`, `start_ts` or `end_ts` - those come from motion "
        "capture.\nOutput one JSON object per line, in the order given, nothing "
        "else.")


def build_user(batch, context, episode_task, frames, cfg=CFG):
    """The user turn: task, recent captions, then per-span facts and frames."""
    parts = []
    head = f"Episode task: {episode_task or 'unknown'}\n"
    if context:
        head += ("\nPrevious captions in this episode (most recent last), in the "
                 "same JSON form you must produce - do not repeat their wording:\n")
        if cfg["context_mode"] == "shuffled":
            import random
            sel = random.Random(1234).sample(context, min(cfg["context_n"], len(context)))
        else:
            sel = context[-cfg["context_n"]:]
        head += "\n".join(
            "  " + json.dumps({k: c[k] for k in ("text", "verb", "noun") if k in c})
            for c in sel)
    head += f"\n\nSpans to caption ({len(batch)}):"
    parts.append(("text", head))

    for span in batch:
        meta = (f"\n[{span['span_id']}] {span['duration']:.2f}s, "
                f"hand={span['hand']}, wrist_speed={span['wrist_speed']:.2f} m/s")
        ap = span.get("aperture_mm")
        if ap:
            meta += f", grasp aperture {ap[0]}-{ap[1]} mm"
        if span.get("aperture_end_mm") is not None:
            meta += f" (ending {span['aperture_end_mm']} mm)"
        if span.get("ap_trend") in ("closing", "opening"):
            meta += f", fingers {span['ap_trend']}"
        if span.get("rotation"):
            meta += f", hand rotating {span['rotation']} (measured)"
        if span.get("fingers"):
            meta += f", finger state: {span['fingers']}"
        # Contact hints are deliberately NOT shown: the detector behind them
        # measures F1 0.12-0.16 against gold, so conditioning the caption on
        # them feeds noise into the one part of the pipeline that is working.
        parts.append(("text", meta))
        for jpg in frames.get(span, cfg["frames_per_span"]):
            parts.append(("image", jpg))
    return parts


# ---------------------------------------------------------------- backends
class Stub:
    """No model; exercises the whole pipeline so everything else is verified."""
    name = "stub"

    def __call__(self, system, parts, batch):
        out = []
        for span in batch:
            out.append(dict(
                span_id=span["span_id"],
                text="Grasp the cardboard box on the shelf with the right hand",
                verb="grasp", noun="cardboard box", visibility="FULL"))
        return out, dict(stub=True)


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, model=None):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model or CFG["anthropic_model"]

    def __call__(self, system, parts, batch):
        content = []
        for kind, v in parts:
            if kind == "text":
                content.append({"type": "text", "text": v})
            else:
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(v).decode()}})
        r = self.client.messages.create(
            model=self.model, max_tokens=16000,
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            thinking={"type": "adaptive"},
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": content}])
        txt = "".join(b.text for b in r.content if b.type == "text")
        self.last_raw = txt
        return parse_objects(txt, batch), dict(
            input_tokens=r.usage.input_tokens,
            output_tokens=r.usage.output_tokens,
            cache_read=getattr(r.usage, "cache_read_input_tokens", 0))


class OpenAICompat:
    """Any OpenAI-compatible server: vLLM, SGLang, llama.cpp, LM Studio."""
    name = "openai"

    def __init__(self, base=None, model=None):
        self.base = (base or CFG["openai_base"]).rstrip("/")
        self.model = model or CFG["openai_model"]

    def __call__(self, system, parts, batch):
        import urllib.request
        content = []
        for kind, v in parts:
            if kind == "text":
                content.append({"type": "text", "text": v})
            else:
                content.append({"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64,"
                           + base64.standard_b64encode(v).decode()}})
        body = json.dumps(dict(
            model=self.model, max_tokens=2048, temperature=CFG["temperature"],
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": content}])).encode()
        req = urllib.request.Request(self.base + "/chat/completions", body,
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            j = json.loads(r.read())
        txt = j["choices"][0]["message"]["content"]
        self.last_raw = txt
        return parse_objects(txt, batch), j.get("usage", {})


class QwenLocal:
    """In-process transformers, for a single box without a serving stack."""
    name = "qwen-local"

    def __init__(self, model_id=None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        mid = model_id or CFG["qwen_model"]
        self.proc = AutoProcessor.from_pretrained(mid)
        self.model = AutoModelForImageTextToText.from_pretrained(
            mid, dtype=torch.bfloat16, device_map="auto")
        self.model.eval()
        self.torch = torch
        print("loaded", mid, "on", next(self.model.parameters()).device)

    def __call__(self, system, parts, batch):
        from PIL import Image
        content, images = [], []
        for kind, v in parts:
            if kind == "text":
                content.append({"type": "text", "text": v})
            else:
                images.append(Image.open(io.BytesIO(v)).convert("RGB"))
                content.append({"type": "image"})
        msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": content}]
        text = self.proc.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        inputs = self.proc(text=[text], images=images,
                           return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=1024, do_sample=True,
                                      temperature=CFG["temperature"], top_p=0.9)
        trimmed = gen[0][inputs.input_ids.shape[1]:]
        txt = self.proc.decode(trimmed, skip_special_tokens=True)
        self.last_raw = txt
        return parse_objects(txt, batch), dict(
            input_tokens=int(inputs.input_ids.shape[1]),
            output_tokens=int(trimmed.shape[0]), n_images=len(images))


BACKENDS = {"stub": Stub, "anthropic": AnthropicBackend,
            "openai": OpenAICompat, "qwen-local": QwenLocal}


# ---------------------------------------------------------------- parsing
def parse_objects(txt, batch):
    """
    Pull JSON objects out of whatever the model wrapped them in, and bind each
    to a span ONLY when the binding is unambiguous.

    The original fell back to positional mapping whenever an id did not match,
    including when the model returned fewer objects than the batch. That
    silently attached captions to the wrong time spans: in one measured run a
    10-span batch came back with 8 objects and they were bound to spans 0-7
    regardless of which spans they described. A mislabelled timestamp is worse
    than a missing one, so positional binding now applies only when the counts
    match exactly.
    """
    found = []
    for m in re.finditer(r"\{[^{}]*\}", txt):
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            continue
        if "text" in obj:
            found.append(obj)

    ids = [s["span_id"] for s in batch]
    id_set = set(ids)
    bound, unbound = [], []
    for obj in found:
        if obj.get("span_id") in id_set:
            bound.append(obj)
        else:
            unbound.append(obj)
    if unbound and len(found) == len(ids):
        # Full-length reply with missing or mangled ids: order is trustworthy.
        for obj, span_id in zip(found, ids):
            obj["span_id"] = span_id
        return found
    return bound


# ---------------------------------------------------------------- run
def run(spans_path=None, backend="stub", out=None, limit=None, cfg=CFG):
    spans_path = str(spans_path or config.SPANS)
    out = str(out or config.CAPTIONS)
    spans = [json.loads(l) for l in open(spans_path)]
    if limit:
        spans = spans[:limit]
    if not spans:
        raise SystemExit(f"no spans in {spans_path}")

    engine = BACKENDS[backend]()

    tasks_path = str(config.EVENTS_RECORDS).replace(".jsonl", ".episodes.json")
    task_of = {}
    if os.path.exists(tasks_path):
        task_of = {t["episode"]: t.get("task") for t in json.load(open(tasks_path))}

    frames = SegmentFrames(config.SEGMENTS_DIR, cfg["jpeg_quality"])
    planned = frames.plan(spans, cfg["frames_per_span"])
    print(f"frame plan: {sum(planned.values())} distinct frames across "
          f"{len(planned)} segments "
          f"(vs decoding every frame of each segment, which held 8.2 GB)")

    by_segment: dict[str, list] = {}
    for s in spans:
        by_segment.setdefault(s["segment"], []).append(s)

    captions, usage, dropped, t_start = [], [], 0, time.time()
    for segment, group in by_segment.items():
        group.sort(key=lambda x: x["start_ts"])
        context = []
        for i in range(0, len(group), cfg["spans_per_call"]):
            batch = group[i:i + cfg["spans_per_call"]]
            task = task_of.get(batch[0]["episode"])
            pack = DM.pack_for(task, batch[0]["segment"], batch[0]["cls"])
            parts = build_user(batch, context, task, frames, cfg)
            try:
                got, u = engine(system_prompt(pack, cfg["frames_per_span"]),
                                parts, batch)
            except Exception as e:
                print(f"  CALL FAILED {segment}: {type(e).__name__}: {e}", flush=True)
                dropped += len(batch)
                continue
            usage.append(u)
            by_id = {s["span_id"]: s for s in batch}
            if len(got) != len(batch):
                dropped += len(batch) - len(got)
                if os.environ.get("CAP_DEBUG"):
                    print(f"    DEBUG got={len(got)} want={len(batch)} "
                          f"raw={(getattr(engine, 'last_raw', '') or '')[:260]!r}",
                          flush=True)
            for obj in got:
                span = by_id.get(obj.get("span_id"))
                if not span:
                    continue
                captions.append(dict(
                    span_id=span["span_id"], segment=span["segment"],
                    episode=span["episode"], cls=span["cls"], pack=pack,
                    # pose-derived, authoritative
                    start_ts=span["start_ts"], end_ts=span["end_ts"],
                    hand=span["hand"], rotation=span.get("rotation"),
                    fingers=span.get("fingers"), ap_trend=span.get("ap_trend"),
                    aperture_mm=span.get("aperture_mm"),
                    wrist_speed=span.get("wrist_speed"),
                    # model-produced
                    text=obj.get("text", ""),
                    verb=(obj.get("verb") or "").lower(),
                    noun=(obj.get("noun") or "").lower(),
                    visibility=(obj.get("visibility") or "FULL").upper(),
                    uncertain=bool(obj.get("uncertain")),
                    backend=engine.name))
                if obj.get("text"):
                    context.append(obj)
            print(f"  {segment:<14s} [{pack}] spans {i:3d}-{i + len(batch) - 1:3d} "
                  f"-> {len(got):2d} captions  "
                  f"({', '.join(f'{k}={v}' for k, v in list(u.items())[:3])})",
                  flush=True)
        frames.release(segment)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        for c in captions:
            fh.write(json.dumps(c) + "\n")

    elapsed = time.time() - t_start
    video = sum(s["duration"] for s in spans)
    print(f"\n{len(captions)} captions from {len(spans)} spans in {elapsed:.0f}s"
          + (f"  ({dropped} spans unparsed)" if dropped else ""))
    print(f"  throughput: {len(captions) / max(elapsed, 1e-9):.2f} spans/s  |  "
          f"{video / max(elapsed, 1e-9):.1f}x realtime "
          f"({video:.0f}s video in {elapsed:.0f}s)")
    print(f"  frame store peak {frames.bytes_peak / 1e6:.1f} MB "
          f"({sum(planned.values())} JPEG frames; the old raw-frame cache held 8.2 GB)")
    if usage and "input_tokens" in usage[0]:
        print(f"  mean tokens/call: in "
              f"{np.mean([u.get('input_tokens', 0) for u in usage]):.0f}  out "
              f"{np.mean([u.get('output_tokens', 0) for u in usage]):.0f}")
    print("  wrote", out)
    return captions
