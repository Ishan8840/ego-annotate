"""Paired numerical-equivalence and wall-time check; no model needed."""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ['EGO_VIDEO_THREADS']='4'
os.environ['OPENCV_FFMPEG_THREADS']='4'
import cv2
cv2.setNumThreads(1)
from egoannot.quality.analysis import measure_video
p=argparse.ArgumentParser(__doc__)
p.add_argument('videos',type=Path,nargs='+')
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
rows=[]
for path in a.videos:
    reports={};times={}
    for fast in (False,True):
        start=time.perf_counter()
        reports[str(fast)]=measure_video(path,fast=fast)
        times['fast' if fast else 'original']=time.perf_counter()-start
    row=dict(source=str(path),sha256=reports['True']['sha256'],video_seconds=reports['True']['duration_s'],times=times,identical=reports['True']==reports['False'])
    rows.append(row)
    print(json.dumps(row),flush=True)
a.out.write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
if not all(r['identical'] for r in rows):raise SystemExit('Measurement parity failed')
