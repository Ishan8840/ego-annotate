"""Shared RGB caption preparation and independently bindable span requests."""
import hashlib
import json
from pathlib import Path
import subprocess
from egoannot import config
from egoannot.labels import domains as DM
from egoannot.stages import caption, rgb_boundaries, spans


def prepare_sources(videos, pack, modality):
    cfg = dict(config.SPANS_CFG, band=(1.3, 2.5), quality_gate=False)
    rows, sources = [], []
    for path in videos:
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
                             episode=path.stem, cls=pack.replace('_',' '), camera_modality=modality,
                             pose_source=None, hand_source='vlm_prediction',
                             boundary_signal='rgb_flow', start_ts=round(start, 3), end_ts=round(end, 3),
                             v_start=round(start, 3), v_end=round(end, 3), duration=round(end-start, 3)))
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        sources.append(dict(id=path.stem, path=str(path.resolve()), duration_s=duration,
                            sha256=digest,
                            fps=n/d, width=stream['width'], height=stream['height'],
                            boundary_statistics=stats, refinement=refinement))
    return rows, sources


def system_prompt(pack, modality):
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
    if modality=='monochrome':
        system+=' The images are monochrome: describe light/dark appearance, never infer color hues.'
    return system


def caption_batch(batch, source, frames, engine, system, pack, max_retries=2):
    calls = []
    pending, best = batch, {}
    for attempt in range(max_retries + 1):
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
                label.update(hand_source='vlm_prediction', boundary_signal='rgb_flow', caption_attempt=attempt,
                             source_video_sha256=source['sha256'])
                label['validation_errors'] = caption._errors(label)
                old = best.get(row['span_id'])
                if old is None or len(label['validation_errors']) < len(old['validation_errors']):
                    best[row['span_id']] = label
            except (ValueError, TypeError, KeyError) as exc:
                call.setdefault('parse_errors',[]).append(str(exc))
        calls.append(call)
        pending = [r for r in pending if r['span_id'] not in best or best[r['span_id']]['validation_errors']]
    return [best[row['span_id']] for row in batch if row['span_id'] in best], calls
