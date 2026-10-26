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
import math
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

EXPERIMENT_FORMAT = 'fev-paired-training-v1'


def run_experiment(code, model, inputs, output, *, steps, learning_rate, seed):
    """One fixed final comparison, including raw and calibrated adapter effects."""
    def run(module, *parameters):
        subprocess.run([sys.executable, '-m', module, '--model-dir', str(model), *map(str, parameters)],
                       cwd=code, check=True)
    run('fev4b.train', '--train', inputs/'train.jsonl', '--development', inputs/'development.jsonl',
        '--steps', steps, '--learning-rate', learning_rate, '--seed', seed, '--out', output/'adapter')
    run('fev4b.calibrate', '--adaptation-dir', output/'adapter', '--data', inputs/'calibration.jsonl',
        '--out', output/'calibrated')
    for name, overlay in (('baseline-test', None), ('raw-test', output/'adapter'),
                          ('test', output/'calibrated')):
        command = [sys.executable, '-m', 'fev4b.benchmark', '--model-dir', str(model),
                   '--data', str(inputs/'test.jsonl'), '--out', str(output/name)]
        if overlay:
            command += ['--adaptation-dir', str(overlay)]
        completed = subprocess.run(command, cwd=code, check=False)
        report_path = output/name/'report.json'
        # The benchmark exits 1 for saved failed predictions. Retain and compare
        # those failures; an infrastructure crash with no report stops the run.
        if not report_path.is_file():
            raise RuntimeError(f'{name} did not produce a complete report')
        report = json.loads(report_path.read_text())
        expected = 1 if report['overall']['errors'] else 0
        if completed.returncode != expected:
            raise RuntimeError(f'{name} exited unexpectedly')
    for name, candidate in (('comparison-raw', 'raw-test'), ('comparison', 'test')):
        subprocess.run([sys.executable, str(inputs/'compare.py'), '--checkout', str(code),
            '--data', str(inputs/'test.jsonl'), '--baseline', str(output/'baseline-test'),
            '--candidate', str(output/candidate), '--out', str(output/name)], check=True)
    errors = {name: json.loads((output/name/'report.json').read_text())['overall']['errors']
              for name in ('baseline-test', 'raw-test', 'test')}
    (output/'complete.json').write_text(json.dumps({
        'complete': True, 'predictionErrors': errors, 'reviewRequired': True, 'adopted': False})+'\n')
    if any(errors.values()):
        raise RuntimeError('Predictions include failures; inspect saved comparisons and do not adopt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--repo', default='murabcd/fev-4b')
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.00002)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if len(args.revision) != 40 or any(c not in '0123456789abcdef' for c in args.revision):
        raise ValueError('Pin an immutable 40-character Hub commit')
    args.output.mkdir(parents=True, exist_ok=False)
    args.inputs = args.inputs.resolve()
    args.output = args.output.resolve()
    names = ('source.zip', 'train.jsonl', 'development.jsonl', 'calibration.jsonl', 'test.jsonl',
             'train_hf.py', 'compare.py', 'common.py')
    package = json.loads((args.inputs/'package.json').read_text())
    if package.get('format') != EXPERIMENT_FORMAT or set(package['files']) != set(names):
        raise ValueError('Use package_job.py to prepare the complete paired experiment')
    for name in names:
        if hashlib.sha256((args.inputs/name).read_bytes()).hexdigest() != package['files'][name]:
            raise ValueError(f'Packaged input hash mismatch: {name}')
    intent = {'format': EXPERIMENT_FORMAT, 'repo': args.repo, 'revision': args.revision, 'steps': args.steps,
              'learningRate': args.learning_rate, 'seed': args.seed, 'source': package,
              'inputHashes': {n: hashlib.sha256((args.inputs/n).read_bytes()).hexdigest() for n in names},
              'published': False, 'automaticAdoption': False}
    (args.output/'intent.json').write_text(json.dumps(intent, indent=2)+'\n')
    code = args.output/'source'
    with zipfile.ZipFile(args.inputs/'source.zip') as archive:
        if any(Path(n).is_absolute() or '..' in Path(n).parts or
               not n.startswith(('fev/', 'fev4b/')) or not n.endswith('.py') for n in archive.namelist()):
            raise ValueError('Source ZIP must contain only fev/ and fev4b/ Python files')
        archive.extractall(code)
    sys.path.insert(0, str(args.inputs))
    from common import load_core, load_splits
    data, _ = load_core(code)
    load_splits(args.inputs, data)
    if args.steps < 1 or not math.isfinite(args.learning_rate) or not 0 < args.learning_rate <= .001:
        raise ValueError('Steps must be positive; learning rate must be finite and in (0, .001]')
    from huggingface_hub import snapshot_download
    with tempfile.TemporaryDirectory(prefix='fev-training-base-') as directory:
        model = snapshot_download(args.repo, revision=args.revision, local_dir=directory)
        run_experiment(code, model, args.inputs, args.output, steps=args.steps,
                       learning_rate=args.learning_rate, seed=args.seed)


if __name__ == '__main__':
    main()
