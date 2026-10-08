"""Build an offline hand-overlay review with blind frame-count labels."""
import argparse,base64,html,json
from pathlib import Path
import cv2


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    clips=[]
    for path in sorted(a.root.glob('*/hands.json')):
        r=json.loads(path.read_text());samples=[f for f in r['frames'] if f['detection_ran']]
        indices=sorted(set(round((len(samples)-1)*i/5) for i in range(6)))
        cap=cv2.VideoCapture(r['source']);frames=[]
        for index in indices:
            row=samples[index];cap.set(cv2.CAP_PROP_POS_FRAMES,row['frame_id']);ok,frame=cap.read()
            if not ok:raise ValueError('Cannot read review source')
            h,w=frame.shape[:2];scale=min(640/w,440/h);frame=cv2.resize(frame,(round(w*scale),round(h*scale)))
            ok,data=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,85]);assert ok
            frames.append(dict(**row,image='data:image/jpeg;base64,'+base64.b64encode(data).decode(),width=w,height=h))
        cap.release()
        clips.append(dict(name=path.parent.name,video=path.parent.name+'/overlay.mp4',summary={k:v for k,v in r.items() if k!='frames'},samples=frames))
    template=Path(__file__).resolve().parents[1]/'demo/hand-review.html'
    payload=json.dumps(clips).replace('<', chr(92)+'u003c')
    (a.root/'index.html').write_text(template.read_text().replace('__HAND_DATA__',payload))
    print(a.root/'index.html')

if __name__=='__main__':main()
