"""Labelled typed requests and leakage-aware deterministic data splitting."""
import argparse
import hashlib
import json
import random
from pathlib import Path

from .decisions import validate_request


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def state_hash(row):
    return hashlib.sha256(canonical(row['state']).encode()).hexdigest()


def request_of(row):
    return validate_request({'state': row['state'], 'questions': {
        key: {k: v for k, v in q.items() if k != 'label'} for key, q in row['questions'].items()}})


def option_keys(question):
    if question['type'] == 'choice':
        return list(question['criteria'])
    if question['type'] == 'noul':
        return ['false', 'true']
    return [str(i) for i in range(len(question['criteria']))]


def target_index(question):
    label = question['label']
    if question['type'] == 'noul':
        if type(label) is not bool:
            raise ValueError('Noul label must be boolean')
        return int(label)
    if question['type'] == 'score':
        if type(label) is not int or not 0 <= label < len(question['criteria']):
            raise ValueError('Score label must be a zero-based level index')
        return label
    if not isinstance(label, str) or label not in question['criteria']:
        raise ValueError('Choice label must name an option')
    return list(question['criteria']).index(label)


def load_records(path):
    records = []
    seen = set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (not isinstance(row, dict) or set(row) - {'id', 'group', 'domain', 'source', 'state', 'questions'}
                or not isinstance(row.get('id'), str) or not row['id'] or row['id'] in seen):
            raise ValueError('Records require unique nonempty IDs and supported fields')
        for name in ('group', 'domain', 'source'):
            if name in row and (not isinstance(row[name], str) or not row[name]):
                raise ValueError('Record metadata must be nonempty strings')
        request_of(row)
        for question in row['questions'].values():
            if set(question) - {'type', 'instructions', 'criteria', 'label'} or 'label' not in question:
                raise ValueError('Every question needs a label')
            target_index(question)
        seen.add(row['id'])
        records.append(row)
    if not records:
        raise ValueError('Dataset is empty')
    return records


def require_disjoint(*partitions):
    prior_ids, prior_states, prior_groups = set(), set(), set()
    for records in partitions:
        ids = {r['id'] for r in records}
        states = {state_hash(r) for r in records}
        groups = {r['group'] for r in records if 'group' in r}
        if ids & prior_ids or states & prior_states or groups & prior_groups:
            raise ValueError('Partitions overlap by ID, exact state, or supplied group')
        prior_ids |= ids
        prior_states |= states
        prior_groups |= groups


def split_records(records, seed=42):
    # Connected components prevent a repeated state from bridging two groups.
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owners = {}
    for i, row in enumerate(records):
        keys = [('state', state_hash(row))]
        if 'group' in row:
            keys.append(('group', row['group']))
        for key in keys:
            if key in owners:
                parent[find(i)] = find(owners[key])
            owners[key] = i
    groups = {}
    for i, row in enumerate(records):
        groups.setdefault(find(i), []).append(row)
    values = list(groups.values())
    if len(values) < 4:
        raise ValueError('Need at least four independent groups for train/dev/calibration/test')
    random.Random(seed).shuffle(values)
    n = max(1, len(values) // 10)
    boundaries = (len(values) - 3*n, len(values) - 2*n, len(values) - n, len(values))
    result = {}
    start = 0
    for name, stop in zip(('train', 'development', 'calibration', 'test'), boundaries):
        result[name] = [row for group in values[start:stop] for row in group]
        start = stop
    require_disjoint(*result.values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    records = load_records(args.data)
    partitions = split_records(records, args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {'seed': args.seed, 'sourceSha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
                'grouping': 'connected exact states and supplied family groups; not fuzzy decontamination', 'files': {}}
    for name, rows in partitions.items():
        path = args.out / (name + '.jsonl')
        path.write_text(''.join(canonical(row) + '\n' for row in rows))
        manifest['files'][path.name] = {'records': len(rows), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
