"""Explicit adapter overlays; the downloaded package is never overwritten."""
import hashlib
import json
import math
from pathlib import Path

from .integrity import file_sha256


def base_identity(root):
    manifest = json.loads((Path(root) / 'manifest.json').read_text())
    weights = {k: v for k, v in manifest['files'].items()
               if k.startswith(('checkpoint/', 'adapter/'))}
    return hashlib.sha256(json.dumps(weights, sort_keys=True).encode()).hexdigest()


def verify_adaptation(root, directory):
    directory = Path(directory).resolve()
    meta = json.loads((directory / 'adaptation.json').read_text())
    if meta.get('format') != 'fev-adaptation-v1' or meta.get('baseIdentity') != base_identity(root):
        raise ValueError('Adaptation does not match the packaged base and original adapter')
    if set(meta.get('files', {})) != {'adapter_config.json', 'adapter_model.safetensors'}:
        raise ValueError('Unexpected adaptation payload')
    for name, digest in meta['files'].items():
        path = directory / name
        if path.is_symlink() or file_sha256(path) != digest:
            raise ValueError('Adaptation checksum mismatch')
    if json.loads((directory / 'adapter_config.json').read_text()) != json.loads((Path(root) / 'adapter/adapter_config.json').read_text()):
        raise ValueError('Adapter architecture differs from the released checkpoint')
    temperature = meta.get('temperature')
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Invalid adaptation temperature')
    return meta


def load_adaptation(wrapper, directory):
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    meta = verify_adaptation(wrapper.root, directory)
    state = load_file(str(Path(directory) / 'adapter_model.safetensors'))
    text = wrapper.model.backbone.language_model
    existing = get_peft_model_state_dict(text)
    if set(state) != set(existing) or any(state[k].shape != existing[k].shape for k in state):
        raise ValueError('Adaptation tensor keys or shapes differ')
    if any(v.dtype != existing[k].dtype or not v.isfinite().all() for k, v in state.items()):
        raise ValueError('Adaptation tensor dtype or finite-value check failed')
    restored = set_peft_model_state_dict(text, state)
    if restored.unexpected_keys or any('lora_' in k for k in restored.missing_keys):
        raise ValueError('Adaptation was not restored completely')
    wrapper.model.temperature = meta['temperature']
    return meta
