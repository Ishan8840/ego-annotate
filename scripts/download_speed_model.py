"""Download the existing Apache-2.0 Qwen checkpoint at its baseline revision."""
from pathlib import Path
import json
from huggingface_hub import HfApi,snapshot_download


def main():
    repo='Qwen/Qwen3-VL-8B-Instruct';revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
    info=HfApi().model_info(repo,revision=revision)
    assert info.card_data.license=='apache-2.0', info.card_data.license
    root=Path('models/qwen3-vl-8b')
    Path('artifacts/speed').mkdir(parents=True,exist_ok=True)
    path=snapshot_download(repo,revision=revision,local_dir=root,allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','README.md','LICENSE'],max_workers=4)
    Path('artifacts/speed/model.json').write_text(json.dumps({'repo':repo,'revision':info.sha,'license':info.card_data.license,'path':path},indent=2))
    print(path,flush=True)


if __name__ == '__main__':
    main()
