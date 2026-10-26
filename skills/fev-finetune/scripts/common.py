"""CPU-only utilities shared by the skill's command-line helpers."""
import hashlib
import json
import sys
from pathlib import Path

PARTITIONS = ('train', 'development', 'calibration', 'test')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def load_core(checkout):
    root = Path(checkout).resolve()
    if not (root / 'fev4b/train.py').is_file():
        raise ValueError('Checkout must contain the Fev training tools')
    sys.path.insert(0, str(root))
    from fev4b import data, metrics
    return data, metrics


def load_splits(directory, data):
    parts = {name: data.load_records(Path(directory) / (name + '.jsonl')) for name in PARTITIONS}
    for records in parts.values():
        if any(not row.get('domain') or not row.get('group') for row in records):
            raise ValueError('Every record requires domain and family group metadata')
    data.require_disjoint(*parts.values())
    if len({row['domain'] for row in parts['train']}) < 2:
        raise ValueError('Shared training requires at least two training domains')
    return parts
