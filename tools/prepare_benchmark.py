"""Fetch the pinned public development suite and append authored Russian probes."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

REVISION = '2074cc44256200bb37540103fa4572901faa10f3'
ORIGINAL_SHA256 = 'ff374c49c6c9f15f8a56fb274b4a4857d20497eb8dd1ac07ce01560e682a5f2e'
COMPARISON_SHA256 = '9266d6ea1ce83c9755a60cf48c7469b0fa3264c229e96f4142d683bada8fef91'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Output must be a new directory')
    url = f'https://raw.githubusercontent.com/jaredpalmer/kev/{REVISION}/evals/v4/transfer-v4/development.jsonl'
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != ORIGINAL_SHA256:
        raise ValueError('Pinned upstream dataset hash mismatch')
    converted = []
    for i, line in enumerate(data.decode().splitlines()):
        row = json.loads(line)
        converted.append({'id': f'transfer-v4-{i:04}', 'source': row['_meta']['source'],
            'state': row['state'], 'questions': {k: {n:v for n,v in q.items() if n != 'src'}
                                               for k,q in row['questions'].items()}})
    text = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in converted)
    russian = Path(__file__).resolve().parents[1]/'evaluation/benchmarks/russian-typed.jsonl'
    combined = (text+russian.read_text()).encode()
    if len(converted) != 764 or hashlib.sha256(combined).hexdigest() != COMPARISON_SHA256:
        raise ValueError('Converted comparison hash mismatch')
    args.out.mkdir(parents=True)
    (args.out/'transfer-v4.jsonl').write_text(text)
    (args.out/'comparison.jsonl').write_bytes(combined)
    (args.out/'manifest.json').write_text(json.dumps({'url':url, 'upstreamSha256':ORIGINAL_SHA256,
        'comparisonSha256':COMPARISON_SHA256, 'records':812, 'questions':908,
        'usage':'Development comparison, not training or fresh customer gold',
        'rights':'Upstream data retains its original source licenses; review Kev transfer-v4 source manifest'}, indent=2)+'\n')
    print(COMPARISON_SHA256)


if __name__ == '__main__':
    main()
