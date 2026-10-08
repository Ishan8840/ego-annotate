"""Compare observed detections at matching frame IDs and render paired demos.

These are detector-agreement/coverage statistics, not ground-truth accuracy.
"""
import argparse,json,sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np


def draw(frame,row,title):
    canvas=frame.copy()
    for tr in row['tracks']:
        x1,y1,x2,y2=map(round,tr['box']);color=(100,240,145) if tr['observed'] else (60,190,255)
        if tr['observed']:cv2.rectangle(canvas,(x1,y1),(x2,y2),color,3)
        else:
            for x in range(x1,x2,20):
                cv2.line(canvas,(x,y1),(min(x+10,x2),y1),color,2);cv2.line(canvas,(x,y2),(min(x+10,x2),y2),color,2)
            for y in range(y1,y2,20):
                cv2.line(canvas,(x1,y),(x1,min(y+10,y2)),color,2);cv2.line(canvas,(x2,y),(x2,min(y+10,y2)),color,2)
    h,w=canvas.shape[:2];target_h=640 if h>w else 480;canvas=cv2.resize(canvas,(2*round(w*target_h/h/2),target_h))
    cv2.rectangle(canvas,(0,0),(canvas.shape[1],65),(14,38,23),-1)
    cv2.putText(canvas,title,(12,24),0,.6,(160,245,188),1,cv2.LINE_AA)
    cv2.putText(canvas,f"{row['time_s']:.2f}s | solid: detected; dashed: unverified",(12,49),0,.45,(220,235,224),1,cv2.LINE_AA)
    return canvas


def render(a,b,path,start,end):
    assert a['source_sha256']==b['source_sha256']
    cap=cv2.VideoCapture(a['source']);first=round(start*a['fps']);cap.set(cv2.CAP_PROP_POS_FRAMES,first)
    lookup={r['frame_id']:r for r in b['frames']};writer=None
    try:
        for row in a['frames'][first:]:
            if row['time_s']>=end:break
            ok,frame=cap.read();assert ok
            other=lookup[row['frame_id']];assert abs(row['time_s']-other['time_s'])<1e-5
            panel=np.concatenate([draw(frame,row,'BEFORE | full-image detector'),draw(frame,other,'AFTER | discovery + verified crops')],axis=1)
            if writer is None:
                h,w=panel.shape[:2];writer=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r',str(a['fps']),'-i','-','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
            writer.stdin.write(panel.tobytes())
    finally:
        cap.release()
        if writer:
            writer.stdin.close()
            if writer.wait():raise RuntimeError('Encoder failed')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('before',type=Path);p.add_argument('after',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1)
    result=[]
    for file in sorted(a.before.glob('*/hands.json')):
        old=json.loads(file.read_text());new=json.loads((a.after/file.parent.name/'hands.json').read_text());assert old['source_sha256']==new['source_sha256']
        oldrows={r['frame_id']:r for r in old['frames'] if r['detection_ran']};newrows={r['frame_id']:r for r in new['frames'] if r['detection_ran']}
        common=sorted(set(oldrows)&set(newrows));n=len(common)
        entry=dict(clip=file.parent.name,source_sha256=old['source_sha256'],matched_samples=n,
          before_two_observed_pct=100*sum(oldrows[i]['strong_count']+oldrows[i]['weak_count']>=2 for i in common)/n,
          after_two_observed_pct=100*sum(newrows[i]['strong_count']+newrows[i]['weak_count']>=2 for i in common)/n,
          before_two_strong_pct=100*sum(oldrows[i]['strong_count']>=2 for i in common)/n,
          after_two_strong_pct=100*sum(newrows[i]['strong_count']>=2 for i in common)/n,
          before_wall_seconds=old['wall_seconds'],after_wall_seconds=new['wall_seconds'],
          gained_second_detection_frames=[i for i in common if oldrows[i]['strong_count']+oldrows[i]['weak_count']<2<=newrows[i]['strong_count']+newrows[i]['weak_count']],
          lost_second_detection_frames=[i for i in common if newrows[i]['strong_count']+newrows[i]['weak_count']<2<=oldrows[i]['strong_count']+oldrows[i]['weak_count']])
        result.append(entry)
        if file.parent.name=='hot3d-quest-000100':render(old,new,a.out/'quest-before-after.mp4',0,5)
        if file.parent.name=='epic-P03-120-60s':render(old,new,a.out/'epic-before-after-15s.mp4',35,50)
    (a.out/'comparison.json').write_text(json.dumps(dict(clips=result,caveat='Two-hand detection rate is not recall: hands can be absent, missed or falsely detected. No forced two-hand output.'),indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
