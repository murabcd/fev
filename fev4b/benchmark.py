"""Score local Fev or a System One endpoint on labelled JSONL. No automatic retries."""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .data import load_records, request_of
from .metrics import report_for, rows_for


def evaluate(records, predict, out, *, provenance):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rows, latencies, served = [], [], set()
    with (out / 'predictions.jsonl').open('x') as stream:
        for record in records:
            began = time.perf_counter()
            try:
                response = predict(request_of(record))
                scored = rows_for(record, response)
                served.add(response.get('model', 'unspecified'))
                saved = {'id': record['id'], 'response': response}
            except (ValueError, RuntimeError, KeyError, TypeError, urllib.error.URLError, TimeoutError) as error:
                # Do not serialize exception text: remote errors can contain sensitive headers.
                saved = {'id': record['id'], 'error': type(error).__name__}
                scored = rows_for(record, error=saved['error'])
            latency = (time.perf_counter() - began)*1000
            saved['wallMs'] = latency
            latencies.append(latency)
            rows.extend(scored)
            stream.write(json.dumps(saved, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
    result = {**report_for(rows), 'provenance': provenance, 'servedModels': sorted(served),
              'records': len(records), 'startedUtc': provenance.get('startedUtc'),
              'wallLatencyMs': {'median': sorted(latencies)[len(latencies)//2],
                                'p95': sorted(latencies)[min(len(latencies)-1, int(.95*len(latencies)))]},
              'latencyMeaning': 'Sequential wall time including client overhead/cold calls; not model-only time or throughput'}
    (out / 'rows.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    (out / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def endpoint_predictor(endpoint, model, api_key=None):
    if not endpoint.startswith(('http://127.0.0.1:', 'http://localhost:', 'https://')):
        raise ValueError('Use localhost HTTP or an HTTPS endpoint')
    def predict(request):
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = 'Bearer ' + api_key
        req = urllib.request.Request(endpoint.rstrip('/') + '/v1/systemone',
            data=json.dumps({'model': model, **request}, ensure_ascii=False).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.load(response)
    return predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--model-dir', type=Path)
    source.add_argument('--endpoint')
    parser.add_argument('--model', default='fev-4b')
    parser.add_argument('--api-key-env', default='FEV_API_KEY')
    parser.add_argument('--execution', choices=('reference', 'compiled'), default='reference')
    parser.add_argument('--adaptation-dir', type=Path)
    args = parser.parse_args()
    records = load_records(args.data)
    provenance = {'dataSha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
                  'startedUtc': datetime.now(timezone.utc).isoformat(), 'execution': args.execution,
                  'requestedModel': args.model}
    if args.endpoint:
        predict = endpoint_predictor(args.endpoint, args.model, os.environ.get(args.api_key_env))
        provenance['endpoint'] = args.endpoint
    else:
        from .model import Fev4B
        model = Fev4B(args.model_dir, execution=args.execution, adaptation_dir=args.adaptation_dir)
        predict = model.decide
        provenance['integrity'] = model.integrity
        provenance['modelManifestSha256'] = hashlib.sha256((args.model_dir/'manifest.json').read_bytes()).hexdigest()
        provenance['temperature'] = model.model.temperature
        provenance['adaptation'] = model.adaptation
    result = evaluate(records, predict, args.out, provenance=provenance)
    print(json.dumps(result['overall'], indent=2))
    if result['overall']['errors']:
        sys.exit(1)


if __name__ == '__main__':
    main()
