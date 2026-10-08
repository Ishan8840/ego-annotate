import hashlib,json,urllib.request
from pathlib import Path
from huggingface_hub import snapshot_download
repo='google/owlv2-base-patch16-ensemble';revision='cfd3195ba4ea9592eec887ded089f4c08eff231d'
out=Path('models/owlv2-hand');out.mkdir(parents=True,exist_ok=True)
card=urllib.request.urlopen(f'https://huggingface.co/{repo}/raw/{revision}/README.md',timeout=30).read()
if b'license: apache-2.0' not in card.lower():
    raise ValueError('Published checkpoint license changed; refusing download')
(out/'MODEL_CARD.md').write_bytes(card)
(out/'LICENSE.txt').write_bytes(urllib.request.urlopen('https://www.apache.org/licenses/LICENSE-2.0.txt').read())
snapshot_download(repo,revision=revision,local_dir=out,allow_patterns=['*.json','*.txt','model.safetensors','README.md'],max_workers=4)
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'}
(out/'manifest.json').write_text(json.dumps(dict(repository=repo,revision=revision,weight_license='Apache-2.0',code_license='Apache-2.0 (Transformers/Scenic)',files=files),indent=2)+'\n')
print('Verified license and recorded checkpoint hashes',flush=True)
