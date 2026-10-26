"""Clone a reviewed immutable source revision without installing GPU dependencies."""
import argparse
import re
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', default='https://github.com/murabcd/fev.git')
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.revision):
        raise ValueError('Use a reviewed 40-character Git commit, not a mutable branch')
    if args.out.exists():
        raise ValueError('Output already exists; do not reset an existing checkout')
    subprocess.run(['git', 'clone', '--no-checkout', '--', args.source, str(args.out)], check=True)
    subprocess.run(['git', '-C', str(args.out), 'checkout', '--detach', args.revision], check=True)
    actual = subprocess.check_output(['git', '-C', str(args.out), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != args.revision or not (args.out / 'fev4b/train.py').is_file():
        raise ValueError('Checkout does not match the requested training source')
    print(f'Ready: {args.out.resolve()} @ {actual}; dependencies and model weights not downloaded')


if __name__ == '__main__':
    main()
