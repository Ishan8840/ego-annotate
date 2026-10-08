"""Fetch a fixed, previously unused public-ego evaluation sample (no training)."""
from pathlib import Path
import hashlib,json,subprocess,tarfile


def main():
    root=Path('artifacts/speed/public');root.mkdir(parents=True,exist_ok=True)
    rows=[]
    def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    def record(row,p):
     row.update(path=str(p),sha256=digest(p));rows.append(row)
     (root/'manifest.json').write_text(json.dumps({'purpose':'separate research evaluation only; no training or commercial dataset redistribution','clips':rows},indent=2)+'\n')
    # Fixed times chosen before examining outputs; preserve full source credits.
    # Freeze the exact recordings selected by the official split list for this run.
    for pid,video_id in [('P02','P02_01'),('P03','P03_02')]:
     url=f'https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/{pid}/{video_id}.MP4'
     p=root/f'epic-{pid}-60.mp4'
     if not p.exists():
      subprocess.run(['ffmpeg','-nostdin','-v','error','-rw_timeout','60000000','-ss','60','-i',url,'-t','24','-an','-vf','scale=1280:720,fps=30','-c:v','libx264','-preset','fast','-crf','18','-threads','4',str(p)],check=True,timeout=600)
     record({'dataset':'EPIC-KITCHENS','participant':pid,'source':url,'source_start_s':60,'duration_s':24,'license':'CC-BY-NC-4.0','license_url':'https://epic-kitchens.github.io/2025.html','split':'heldout' if pid=='P03' else 'development'},p)
     print('READY',p,flush=True)
    rev='30fe9674782f32e1e5edba98476b6ff4300132c5'
    for number in [50,100]:
     from huggingface_hub import hf_hub_download
     import cv2,numpy as np
     archive=Path(hf_hub_download('bop-benchmark/hot3d',f'train_quest3/clip-{number:06d}.tar',repo_type='dataset',revision=rev))
     p=root/f'hot3d-quest-{number:06d}.mp4'
     with tarfile.open(archive) as tar:
      ids=sorted(int(n.split('.')[0]) for n in tar.getnames() if n.endswith('.info.json'))
      proc=None
      for i in ids:
       cams=json.load(tar.extractfile(f'{i:06d}.cameras.json'));sid=next(k for k,v in cams.items() if v['calibration']['label'].endswith('left'))
       im=cv2.imdecode(np.frombuffer(tar.extractfile(f'{i:06d}.image_{sid}.jpg').read(),np.uint8),cv2.IMREAD_COLOR)
       im=cv2.rotate(im,cv2.ROTATE_90_CLOCKWISE)
       if proc is None:
        h,w=im.shape[:2];proc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-threads','4',str(p)],stdin=subprocess.PIPE)
       proc.stdin.write(im.tobytes())
      proc.stdin.close();assert proc.wait()==0
     record({'dataset':'HOT3D Quest3','clip':number,'revision':rev,'source_archive_sha256':digest(archive),'source':f'https://huggingface.co/datasets/bop-benchmark/hot3d/resolve/{rev}/train_quest3/clip-{number:06d}.tar','duration_s':len(ids)/30,'license':'Recordings CC-BY-SA-4.0; hand annotations excluded','gt_loaded':False,'split':'development' if number==50 else 'heldout'},p)
     print('READY',p,flush=True)


if __name__ == '__main__':
    main()
