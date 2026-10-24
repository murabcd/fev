"""Fev-4B CLI. Loading and inference use local files only."""
import argparse
import json
import sys
from pathlib import Path

from fev4b import Fev4B, verify_package


def main():
    parser = argparse.ArgumentParser(description='Run the packaged Fev-4B decision model')
    parser.add_argument('--model-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--input', type=Path, help='JSON containing request and unknownKey; default: stdin')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--execution', choices=('reference', 'compiled'), default='reference')
    parser.add_argument('--adaptation-dir', type=Path)
    parser.add_argument('--verify-only', action='store_true', help='Verify all hashes without importing PyTorch')
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_package(args.model_dir), ensure_ascii=False))
        return
    payload = json.loads(args.input.read_text() if args.input else sys.stdin.read())
    if not isinstance(payload, dict) or set(payload) != {'request', 'unknownKey'}:
        raise ValueError('Expected exactly request and unknownKey')
    model = Fev4B(args.model_dir, device=args.device, execution=args.execution,
                  adaptation_dir=args.adaptation_dir)
    result = model.route(payload['request'], unknown_key=payload['unknownKey'])
    # Raw native evidence is available through the Python interface for debugging.
    result.pop('native')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
