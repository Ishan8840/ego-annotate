"""Download one explicitly Apache-2.0-declared hand checkpoint, with pinned hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REPOSITORY = 'bukuroo/RTMDet-ONNX'
REVISION = 'aa91dbccc283db36b2c250a1d91ba9518db553da'
ARTIFACT = 'rtmdet-n-hand.onnx'
SHA256 = '568d3ea97a5b142488366b67e036b6a5cb0a1fef9087a710cb8e66b6979fbac2'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, default=Path('models/rtmdet-hand'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    base = f'https://huggingface.co/{REPOSITORY}'
    card = urllib.request.urlopen(f'{base}/raw/{REVISION}/README.md', timeout=60).read()
    if b'license: apache-2.0' not in card.lower():
        raise ValueError('Expected explicit Apache-2.0 checkpoint license')
    (args.out / 'MODEL_CARD.md').write_bytes(card)
    license_text = urllib.request.urlopen('https://www.apache.org/licenses/LICENSE-2.0.txt', timeout=60).read()
    (args.out / 'LICENSE.txt').write_bytes(license_text)
    target = args.out / ARTIFACT
    if not target.exists():
        data = urllib.request.urlopen(f'{base}/resolve/{REVISION}/{ARTIFACT}', timeout=120).read()
        if hashlib.sha256(data).hexdigest() != SHA256:
            raise ValueError('Checkpoint download hash mismatch')
        temporary = target.with_suffix('.part')
        temporary.write_bytes(data)
        temporary.replace(target)
    if hashlib.sha256(target.read_bytes()).hexdigest() != SHA256:
        raise ValueError('Existing checkpoint hash mismatch')
    manifest = dict(repository=REPOSITORY, revision=REVISION, artifact=ARTIFACT,
                    sha256=SHA256, weight_license='Apache-2.0',
                    license_basis='Explicit distributor model-card declaration',
                    code_license='Apache-2.0 upstream MMDetection/MMPose; MIT ONNX Runtime',
                    source_url=f'{base}/tree/{REVISION}', mediapipe=False)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(target)


if __name__ == '__main__':
    main()
