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
from egoannot.core.video import SegmentFrames
from egoannot.labels import domains as DM
from egoannot.stages import caption, rgb_boundaries, spans


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('videos', nargs='+', type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--model')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--max-retries', type=int, default=2)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.batch_size < 1 or a.max_retries < 0:
        p.error('batch-size must be positive and max-retries nonnegative')
    if len({v.parent.resolve() for v in a.videos}) != 1:
        raise ValueError('Input videos must share a directory for the frame sampler.')
    pack = 'food_preparation'
    cfg = dict(config.SPANS_CFG, band=(1.3, 2.5), quality_gate=False)
    rows, sources = [], []
    for path in a.videos:
        probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                         '-show_streams','-show_format','-of','json',str(path)]))
        stream = probe['streams'][0]
        n, d = map(float, stream['avg_frame_rate'].split('/'))
        duration = float(probe['format']['duration'])
        h264 = subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-map','0:v:0',
                                      '-c:v','copy','-bsf:v','h264_mp4toannexb','-f','h264','-'])
        ep = dict(vid=h264, src_fps=n/d)
        ts, activity = rgb_boundaries.signal(ep, 'rgb_flow')
        intervals, stats = spans.enforce_band([(0., duration)], ts, activity, cfg)
        intervals, _, refinement = spans.refine_boundaries(intervals, ts, activity, cfg)
        for start, end in intervals:
            rows.append(dict(span_id=f'{path.stem}#{start:07.3f}', segment=path.stem,
                             episode=path.stem, cls='food preparation', camera_modality='rgb',
                             pose_source=None, hand_source='vlm_prediction',
                             boundary_signal='rgb_flow', start_ts=round(start, 3), end_ts=round(end, 3),
                             v_start=round(start, 3), v_end=round(end, 3), duration=round(end-start, 3)))
        sources.append(dict(id=path.stem, path=str(path.resolve()), duration_s=duration,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            fps=n/d, width=stream['width'], height=stream['height'],
                            boundary_statistics=stats, refinement=refinement))
    (a.out/'spans.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    frames = SegmentFrames(a.videos[0].parent, 88)
    frames.plan(rows, 5)
    engine = caption.QwenLocal(a.model)
    system = ('Annotate short egocentric RGB video spans. Boundaries are supplied by an RGB motion '
              'segmenter. There is no motion capture or measured hand pose. For each given span, '
              'describe the single main visible manipulation action. Use only visible evidence. '
              'Do not guess a brand, material, hidden action or task success. '
              'If the action is unclear, set uncertain=true. Hand is a visual prediction. '
              'Return one flat JSON object per span, with exactly these keys: '
              'span_id, text, verb, noun, hand, visibility, uncertain. '
              'hand is LEFT, RIGHT, BOTH or NEITHER; visibility is FULL, PARTIAL or OCCLUDED. '
              'Text must be an imperative sentence of 10–15 words describing ONE main action. '
              'Favor 11–13 words. Use modifiers rather than linking different actions. '
              'Do not reuse phrases from unrelated spans. Preserve the supplied IDs. '
              'Allowed verbs: '+', '.join(DM.verbs_for(pack))+'.\n'+DM.CORE_RULES)
    calls, captions = [], []
    started = time.time()
    for source in sources:
        group = [r for r in rows if r['segment']==source['id']]
        for i in range(0, len(group), a.batch_size):
            batch = group[i:i+a.batch_size]
            pending, best = batch, {}
            for attempt in range(a.max_retries + 1):
                if not pending:
                    break
                parts = [('text', 'Consecutive source-aligned RGB spans. Caption the images independently of any source dataset annotations.')]
                for row in pending:
                    parts.append(('text',f'{row["span_id"]}: {row["start_ts"]:.3f}–{row["end_ts"]:.3f} seconds.'))
                    for jpg in frames.get(row, 5):
                        parts.append(('image', jpg))
                    if row['span_id'] in best:
                        old = best[row['span_id']]
                        parts.append(('text','Fix these format errors using the SAME images: '+
                                      json.dumps(old['validation_errors'])+' Previous reply: '+json.dumps(old)))
                parts.append(('text', 'Return exactly '+str(len(pending))+' JSON object(s), one for EACH of these IDs: '+', '.join(r['span_id'] for r in pending)+'. Count words before replying; aim for 11 to 13 words unless uncertain.'))
                _, usage = engine(system, parts, pending)
                parsed = caption.parse_objects(engine.last_raw, pending)
                by_id = {o['span_id']:o for o in parsed}
                call = dict(requested_ids=[r['span_id'] for r in pending], attempt=attempt,
                            raw=engine.last_raw, usage=usage)
                for row in pending:
                    obj = by_id.get(row['span_id'])
                    if not obj:
                        continue
                    try:
                        hand = obj.get('hand', 'UNKNOWN')
                        if not isinstance(hand,str):
                            raise ValueError('hand must be a string')
                        label = caption._label(obj, dict(row, hand=hand.upper()), pack, engine.name)
                        label.update(hand_source='vlm_prediction', boundary_signal='rgb_flow', caption_attempt=attempt)
                        label['validation_errors'] = caption._errors(label)
                        old = best.get(row['span_id'])
                        if old is None or len(label['validation_errors']) < len(old['validation_errors']):
                            best[row['span_id']] = label
                    except (ValueError, TypeError, KeyError) as exc:
                        call.setdefault('parse_errors',[]).append(str(exc))
                calls.append(call)
                pending = [r for r in pending if r['span_id'] not in best or best[r['span_id']]['validation_errors']]
            captions.extend(best[r['span_id']] for r in batch if r['span_id'] in best)
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
