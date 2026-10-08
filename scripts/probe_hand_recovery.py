import os,sys,json,time
from pathlib import Path
os.environ['OPENCV_FFMPEG_THREADS']='4'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
from egoannot.quality.owl_hand import OwlHandDetector
from egoannot.quality.hand_recovery import RecoveryHandDetector
cv2.setNumThreads(1)
det=OwlHandDetector('models/owlv2-hand')
det(np.zeros((720,1280,3),np.uint8))
out=Path('artifacts/hand-accuracy/probe');out.mkdir(parents=True,exist_ok=True)
rows=[]
for name,ts in [('epic-P02-60',[2,6,10,14,18,22]),('epic-P03-60',[2,6,10,14,18,22]),('epic-P03-120-60s',[5,15,25,35,45,55]),('hot3d-quest-000100',[.2,1,2,3,4,4.8]),('hot3d-quest-000050',[.2,1,2,3,4,4.8])]:
 cap=cv2.VideoCapture('artifacts/speed/public/'+name+'.mp4');tiles=[]
 for t in ts:
  cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,frame=cap.read();assert ok
  cv2.imwrite(str(out/f'{name}-{t}-raw.jpg'),frame)
  results={}
  for mode in ['baseline','zoom','tiles','both']:
   model=det if mode=='baseline' else RecoveryHandDetector(det,discovery=mode in ['tiles','both'],verify_weak=mode in ['zoom','both'])
   start=time.perf_counter();preds=model(frame);elapsed=time.perf_counter()-start
   results[mode]=dict(predictions=preds,seconds=elapsed)
  h,w=frame.shape[:2];scale=min(480/w,300/h);thumb=cv2.resize(frame,(round(w*scale),round(h*scale)));panels=[]
  for mode in ['baseline','both']:
   tile=np.full((340,480,3),18,np.uint8);tile[:thumb.shape[0],:thumb.shape[1]]=thumb
   for d in results[mode]['predictions']:
    if d['score']<.15:continue
    x1,y1,x2,y2=[round(v*scale) for v in d['box']];cv2.rectangle(tile,(x1,y1),(x2,y2),(70,240,120),2);cv2.putText(tile,f"{d['score']:.2f}",(max(0,x1),max(12,y1)),0,.45,(255,255,255),1)
   cv2.putText(tile,f'{t:g}s {mode}',(4,325),0,.55,(200,240,210),1);panels.append(tile)
  tiles.append(np.concatenate(panels,axis=1));rows.append(dict(clip=name,time=t,width=w,height=h,modes=results))
 cap.release();cv2.imwrite(str(out/(name+'.jpg')),np.concatenate(tiles,axis=0));print(name,flush=True)
(out/'results.json').write_text(json.dumps(rows,indent=2))
print('done',flush=True)
