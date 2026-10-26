"""Build minimal private HF Job inputs; never upload data or allocate compute."""
import argparse
import ast
import subprocess
import zipfile
from pathlib import Path

from common import PARTITIONS, load_core, load_splits, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.checkout.resolve()
    data, _ = load_core(root)
    load_splits(args.split, data)
    worker = root / 'tools/train_hf.py'
    tree = ast.parse(worker.read_text())
    supported = any(isinstance(node, ast.Assign) and
                    any(isinstance(t, ast.Name) and t.id == 'EXPERIMENT_FORMAT' for t in node.targets) and
                    isinstance(node.value, ast.Constant) and node.value.value == 'fev-paired-training-v1'
                    for node in tree.body)
    if not supported:
        raise ValueError('Checkout lacks the paired experiment worker; select a newer reviewed revision')
    names = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z', 'fev', 'fev4b'], text=True).split('\0')
    names = [name for name in names if name.endswith('.py')]
    if not names or any((root / name).is_symlink() for name in names):
        raise ValueError('Expected tracked Python source without symlinks')
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--',
                                    'fev', 'fev4b', 'tools/train_hf.py'], text=True).strip()
    args.out.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(args.out / 'source.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):
            archive.write(root / name, name)
    for name in [p + '.jsonl' for p in PARTITIONS]:
        (args.out / name).write_bytes((args.split / name).read_bytes())
    (args.out / worker.name).write_bytes(worker.read_bytes())
    for name in ('compare.py', 'common.py'):
        (args.out / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    write_json(args.out / 'package.json', {
        'format': 'fev-paired-training-v1', 'sourceCommit': commit, 'sourceDirty': bool(dirty),
        'files': {p.name: sha256(p) for p in sorted(args.out.iterdir())},
        'uploaded': False, 'computeAllocated': False})
    print(f'Prepared {args.out.resolve()}; inspect package.json before uploading')


if __name__ == '__main__':
    main()
