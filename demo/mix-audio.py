"""Mix the ambient score at a quiet, fixed level, preserving its natural swells."""
from pathlib import Path
import json
import subprocess

root = Path(__file__).resolve().parents[1] / 'artifacts/demo'
video, score = root/'ego-demo-silent.mp4', root/'ego-score.wav'
duration = float(subprocess.check_output(['ffprobe','-v','error','-show_entries',
    'format=duration','-of','default=nw=1:nk=1',str(video)]))
assert abs(duration-30) < .05, 'Render the complete 30-second silent film first.'
probe = subprocess.run(['ffmpeg','-hide_banner','-nostats','-i',str(score),'-af',
    'loudnorm=I=-20:TP=-3:LRA=8:print_format=json','-f','null','-'],
    capture_output=True,text=True,check=True)
start=probe.stderr.rfind('{')
levels=json.loads(probe.stderr[start:probe.stderr.find('}',start)+1])
# A constant gain avoids automatic gain changes over the long fades.
gain=min(-20-float(levels['input_i']),-3-float(levels['input_tp']))
subprocess.run(['ffmpeg','-y','-v','error','-i',str(video),'-i',str(score),
    '-map','0:v:0','-map','1:a:0','-c:v','copy','-af',f'volume={gain:.3f}dB',
    '-c:a','aac','-b:a','192k','-ar','48000','-t','30','-movflags','+faststart',
    str(root/'ego-demo.mp4')],check=True)
path=root/'music-verification.json'
checks=json.loads(path.read_text())
checks.update(source_integrated_lufs=float(levels['input_i']),source_true_peak_dbfs=float(levels['input_tp']),
              static_mix_gain_db=gain,target_integrated_lufs=-20,automatic_gain_changes=False)
path.write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
