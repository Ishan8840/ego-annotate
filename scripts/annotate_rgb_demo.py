"""Run the pipeline's RGB boundaries, caption backend, binding and validation.

RGB-only inputs carry no measured hand poses: handedness is explicitly a model
prediction, and no 3D motion values are synthesized. Preserve all raw replies.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egoannot import config
from egoannot.rgb_caption import prepare_sources, system_prompt, caption_batch
from egoannot.core.video import SegmentFrames
from egoannot.labels import domains as DM
from egoannot.stages import caption, rgb_boundaries, spans


def main(argv=None, engine=None):
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('videos', nargs='+', type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--model')
    p.add_argument('--domain', choices=sorted(DM.PACKS), default='food_preparation')
    p.add_argument('--camera-modality', choices=['rgb','monochrome'], default='rgb')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--max-retries', type=int, default=2)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.batch_size < 1 or a.max_retries < 0:
        p.error('batch-size must be positive and max-retries nonnegative')
    if len({v.parent.resolve() for v in a.videos}) != 1:
        raise ValueError('Input videos must share a directory for the frame sampler.')
    pack = a.domain
    rows, sources = prepare_sources(a.videos, pack, a.camera_modality)
    (a.out/'spans.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    frames = SegmentFrames(a.videos[0].parent, 88)
    frames.plan(rows, 5)
    engine = engine or caption.QwenLocal(a.model)
    system = system_prompt(pack, a.camera_modality)
    calls, captions = [], []
    started = time.time()
    for source in sources:
        group = [r for r in rows if r['segment']==source['id']]
        for i in range(0, len(group), a.batch_size):
            batch = group[i:i+a.batch_size]
            labels, batch_calls = caption_batch(batch, source, frames, engine, system, pack, a.max_retries)
            captions.extend(labels)
            calls.extend(batch_calls)
            (a.out/'captions.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in captions))
            (a.out/'run.json').write_text(json.dumps(dict(system_prompt=system,sources=sources,calls=calls),indent=2)+'\n')
            print(source['id'], f'{min(i+a.batch_size,len(group))}/{len(group)} spans', flush=True)
        frames.release(source['id'])
    seconds = sum(s['duration_s'] for s in sources)
    result = dict(sources=sources, spans=len(rows), captions=len(captions), seconds=seconds,
                  annotations_per_minute=len(captions)/seconds*60,
                  median_span_s=float(np.median([c['end_ts']-c['start_ts'] for c in captions])),
                  time_coverage=sum(c['end_ts']-c['start_ts'] for c in captions)/seconds,
                  format_valid=sum(not c['validation_errors'] for c in captions),
                  uncertain=sum(c['uncertain'] for c in captions), elapsed_s=time.time()-started,
                  model=a.model, boundary_source='RGB optical-flow troughs with 1.3–2.5s band',
                  notes='Raw RGB input; no measured poses. Captions are model outputs, not source dataset labels. Density is not semantic accuracy.')
    (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2), flush=True)


if __name__ == '__main__':
    main()
