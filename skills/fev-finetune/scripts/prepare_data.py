"""Validate native JSONL or map CSV labels to typed requests, then freeze grouped splits."""
import argparse
import copy
import csv
import json
import tempfile
from collections import Counter
from pathlib import Path

from common import load_core, load_splits, sha256, write_json


def convert_csv(path, spec, mappings, state_field):
    if set(mappings) != set(spec):
        raise ValueError('Map exactly one CSV label column for every question in the spec')
    rows = []
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if len(reader.fieldnames or []) != len(set(reader.fieldnames or [])):
            raise ValueError('CSV headers must be unique')
        required = {'id', 'domain', 'group', state_field, *mappings.values()}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f'CSV requires columns: {sorted(required)}')
        for item in reader:
            questions = copy.deepcopy(spec)
            for key, column in mappings.items():
                value = item[column]
                kind = questions[key]['type']
                if kind == 'noul':
                    if value not in ('true', 'false'):
                        raise ValueError('CSV Noul labels must be literal true or false')
                    value = value == 'true'
                elif kind == 'score':
                    value = int(value)
                questions[key]['label'] = value
            rows.append({'id': item['id'], 'domain': item['domain'], 'group': item['group'],
                         'state': item[state_field], 'questions': questions})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--spec', type=Path, help='JSON question definitions for CSV input')
    parser.add_argument('--label', action='append', default=[], help='QUESTION=CSV_COLUMN')
    parser.add_argument('--state-field', default='text')
    parser.add_argument('--holdout-domain', action='append', default=[])
    args = parser.parse_args()
    data, _ = load_core(args.checkout)
    if args.out.exists():
        raise ValueError('Output already exists; keep frozen partitions immutable')
    if args.spec:
        mappings = dict(item.split('=', 1) for item in args.label)
        if len(mappings) != len(args.label):
            raise ValueError('Duplicate question mapping')
        rows = convert_csv(args.data, json.loads(args.spec.read_text()), mappings, args.state_field)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'converted.jsonl'
            path.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows))
            records = data.load_records(path)
    else:
        if args.label:
            raise ValueError('--label requires --spec')
        records = data.load_records(args.data)
    if any(not row.get('domain') or not row.get('group') for row in records):
        raise ValueError('Supply domain and family group on every record before splitting')
    held = set(args.holdout_domain)
    if held - {row['domain'] for row in records}:
        raise ValueError('Requested holdout domain is absent')
    parts = data.split_records([r for r in records if r['domain'] not in held], args.seed)
    parts['test'].extend(r for r in records if r['domain'] in held)
    data.require_disjoint(*parts.values())
    if len({row['domain'] for row in parts['train']}) < 2:
        raise ValueError('Split leaves fewer than two training domains; revise the corpus, not test labels')
    # Validate before creating the final directory. Preserve criterion order: it
    # determines label indices and breaks ties during inference.
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        for name, rows in parts.items():
            (temporary / (name + '.jsonl')).write_text(''.join(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows))
        load_splits(temporary, data)
        args.out.mkdir(parents=True)
        for path in temporary.iterdir():
            (args.out / path.name).write_bytes(path.read_bytes())
    manifest = {'sourceSha256': sha256(args.data), 'seed': args.seed, 'heldOutDomains': sorted(held),
                'specSha256': sha256(args.spec) if args.spec else None,
                'labelMapping': dict(item.split('=', 1) for item in args.label),
                'grouping': 'connected exact states and supplied families; no fuzzy deduplication',
                'files': {name + '.jsonl': {'sha256': sha256(args.out / (name + '.jsonl')),
                         'records': len(rows), 'questions': sum(len(r['questions']) for r in rows),
                         'domains': dict(Counter(r['domain'] for r in rows))} for name, rows in parts.items()}}
    write_json(args.out / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
