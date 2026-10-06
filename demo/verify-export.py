"""Verify the complete deliverable, including decoder health and audio level."""
from pathlib import Path
import subprocess
import json
import hashlib

root = Path(__file__).resolve().parents[1] / 'artifacts/demo'
path = root / 'ego-demo.mp4'
p = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
                                      '-show_streams', '-show_format', '-of', 'json', str(path)]))
v = next(s for s in p['streams'] if s['codec_type'] == 'video')
a = next(s for s in p['streams'] if s['codec_type'] == 'audio')
assert (v['width'], v['height'], v['codec_name'], v['pix_fmt'], v['r_frame_rate'],
        int(v['nb_read_frames'])) == (1920, 1080, 'h264', 'yuv420p', '30/1', 900)
assert v['color_range'] == 'tv' and v['color_space'] == 'bt709'
assert a['codec_name'] == 'aac' and a['sample_rate'] == '48000' and a['channels'] == 2
assert abs(float(p['format']['duration']) - 30) < .05
check = subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path), '-f', 'null', '-'],
                       capture_output=True, text=True)
assert check.returncode == 0 and not check.stderr.strip(), check.stderr
loud = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-i', str(path), '-af',
                       'loudnorm=I=-20:TP=-3:LRA=8:print_format=json', '-f', 'null', '-'],
                      capture_output=True, text=True)
start = loud.stderr.rfind('{')
end = loud.stderr.find('}', start) + 1
levels = json.loads(loud.stderr[start:end])
assert float(levels['input_tp']) < -3
assert abs(float(levels['input_i']) + 20) < .5
result = dict(duration_s=float(p['format']['duration']), frames=int(v['nb_read_frames']),
              resolution=[v['width'], v['height']], fps=v['r_frame_rate'], video_codec=v['codec_name'],
              pixel_format=v['pix_fmt'], color_space=v['color_space'], color_range=v['color_range'],
              audio_codec=a['codec_name'], audio_sample_rate_hz=int(a['sample_rate']),
              audio_channels=a['channels'], size_bytes=int(p['format']['size']),
              bitrate=int(p['format']['bit_rate']), full_decode_errors=[],
              audio_integrated_lufs=float(levels['input_i']), audio_true_peak_dbfs=float(levels['input_tp']),
              sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(root / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
