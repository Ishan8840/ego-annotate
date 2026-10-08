"""Create explicitly synthetic degradation controls from research footage."""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser(__doc__)
p.add_argument('source',type=Path)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
if a.out.exists():p.error('Use a new output path')
a.out.parent.mkdir(parents=True,exist_ok=True)
filters="trim=duration=12,setpts=PTS-STARTPTS,gblur=sigma=12:enable='gte(t,4)*lt(t,8)',eq=brightness=-0.65:enable='gte(t,8)',tpad=stop_mode=clone:stop_duration=4"
subprocess.run(['ffmpeg','-v','error','-threads','4','-i',str(a.source),'-an','-vf',filters,'-filter_threads','2','-c:v','libx264','-threads','4','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(a.out)],check=True)
a.out.with_suffix('.controls.json').write_text(json.dumps(dict(source=str(a.source),synthetic=True,intervals=[dict(start_s=0,end_s=4,intervention='source excerpt, re-encoded'),dict(start_s=4,end_s=8,intervention='Gaussian blur sigma=12'),dict(start_s=8,end_s=12,intervention='brightness offset -0.65'),dict(start_s=12,end_s=16,intervention='freeze last darkened frame')],limitation='Synthetic interventions test sensitivity, not accuracy on naturally occurring defects.'),indent=2)+'\n')
