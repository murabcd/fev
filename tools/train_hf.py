# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch==2.14.0", "torchvision==0.29.0", "transformers==5.17.0",
#   "accelerate==1.10.1", "safetensors==0.8.0", "huggingface-hub==1.31.0",
#   "pillow==12.3.0", "flash-linear-attention==0.5.2", "einops==0.8.1", "peft==0.21.2",
# ]
# ///
"""One shared-adapter experiment; private mounted artifacts, no model publishing."""
import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--repo', default='murabcd/fev-4b')
    parser.add_argument('--steps', type=int, default=6)
    args = parser.parse_args()
    if len(args.revision) != 40 or any(c not in '0123456789abcdef' for c in args.revision):
        raise ValueError('Pin an immutable 40-character Hub commit')
    args.output.mkdir(parents=True, exist_ok=False)
    names = ('source.zip', 'train.jsonl', 'development.jsonl', 'calibration.jsonl', 'test.jsonl')
    intent = {'repo': args.repo, 'revision': args.revision, 'steps': args.steps,
              'inputHashes': {n: hashlib.sha256((args.inputs/n).read_bytes()).hexdigest() for n in names},
              'published': False, 'automaticAdoption': False}
    (args.output/'intent.json').write_text(json.dumps(intent, indent=2)+'\n')
    code = Path('/tmp/fev-training-source')
    with zipfile.ZipFile(args.inputs/'source.zip') as archive:
        if any(Path(n).is_absolute() or '..' in Path(n).parts or
               not n.startswith(('fev/', 'fev4b/')) or not n.endswith('.py') for n in archive.namelist()):
            raise ValueError('Source ZIP must contain only fev/ and fev4b/ Python files')
        archive.extractall(code)
    from huggingface_hub import snapshot_download
    model = snapshot_download(args.repo, revision=args.revision, local_dir='/tmp/fev-training-base')
    def run(module, *parameters):
        subprocess.run([sys.executable, '-m', module, '--model-dir', model, *map(str, parameters)],
                       cwd=code, check=True)
    run('fev4b.train', '--train', args.inputs/'train.jsonl', '--development', args.inputs/'development.jsonl',
        '--steps', args.steps, '--out', args.output/'adapter')
    run('fev4b.calibrate', '--adaptation-dir', args.output/'adapter', '--data', args.inputs/'calibration.jsonl',
        '--out', args.output/'calibrated')
    run('fev4b.benchmark', '--adaptation-dir', args.output/'calibrated', '--data', args.inputs/'test.jsonl',
        '--out', args.output/'test')
    report = json.loads((args.output/'test/report.json').read_text())
    if report['overall']['errors']:
        raise RuntimeError('Incomplete test predictions; do not adopt')
    (args.output/'complete.json').write_text(json.dumps({'complete': True, 'adopted': False})+'\n')


if __name__ == '__main__':
    main()
