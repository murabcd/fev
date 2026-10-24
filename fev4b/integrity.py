"""Streaming integrity checks; no model imports, weights loading, or network calls."""
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_package(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['model'] != 'Fev-4B' or manifest['version'] not in ('0.1.0', '0.2.0', '0.2.1'):
        raise ValueError('Unsupported package manifest')
    files = manifest.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('No files in package manifest')
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe path in manifest')
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Missing or external package file: ' + name)
        if path.stat().st_size != expected['bytes'] or file_sha256(path) != expected['sha256']:
            raise ValueError('Package checksum mismatch: ' + name)
    return {'model': manifest['model'], 'version': manifest['version'],
            'verifiedFiles': len(files), 'totalBytes': sum(s['bytes'] for s in files.values())}
