"""Start the resident Fev text-decision reference server."""
import argparse
import os
from pathlib import Path

from fev4b import Fev4B
from fev4b.server import create_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--execution', choices=('reference', 'compiled'), default='reference')
    parser.add_argument('--adaptation-dir', type=Path, help='Explicit, verified research adapter overlay')
    args = parser.parse_args()
    api_key = os.environ.get('FEV_API_KEY')
    if args.host not in ('127.0.0.1', 'localhost') and not api_key:
        parser.error('Set FEV_API_KEY before binding outside localhost')
    model = Fev4B(args.model_dir, device=args.device, execution=args.execution,
                  adaptation_dir=args.adaptation_dir)
    with create_server(model, (args.host, args.port), api_key=api_key) as server:
        print(f'Fev ready at http://{args.host}:{args.port}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
