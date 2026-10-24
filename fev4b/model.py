"""Offline Fev loader and the unchanged benchmarked routing composition.

The imported fev primitives are byte-identical to the frozen benchmark runtime.
No customer route definitions or default customer rules are embedded here.
"""
import copy
import json
import os
import sys
import threading
import time
from pathlib import Path

from fev.intent_membership import request_for
from fev.ranked_requests import prepare
from fev.route_applicability import build_request
from fev.ranked_membership import resolve
from fev.research_controls import compare_responses
from .integrity import verify_package
from .decisions import decide_with_predict, validate_request


def route_with_predict(request, *, unknown_key, predict):
    """Compose the frozen primary and independent applicability calls.

    predict must accept a complete native request and return a native response.
    Clarification is a decision signal; this function does not generate text.
    """
    request_for(request, unknown_key)  # Strict validation before any inference.
    primary = predict(request)
    compare_responses(primary, primary)
    if set(primary['answers']) != {'route'}:
        raise ValueError('Unexpected primary answer scope')
    choice = primary['answers']['route']['probabilities']
    config = prepare(request, choice, unknown_key=unknown_key)
    secondary_request = None
    secondary = None
    evidence = dict.fromkeys(k for k in choice if k != unknown_key)
    if config['request'] is not None:
        secondary_request = build_request(request, unknown_key=unknown_key, keys=config['queriedKeys'])
        secondary = predict(secondary_request)
        compare_responses(secondary, secondary)
        if set(secondary['answers']) != set(config['queriedKeys']):
            raise ValueError('Unexpected applicability answer scope')
        evidence.update({key: value['noul'] for key, value in secondary['answers'].items()})
    decision = resolve(evidence, choice, config['precedence'], unknown_key=unknown_key, yes=.8, no=.2)
    responses = [primary] + ([secondary] if secondary is not None else [])
    return {'model': 'Fev-4B', 'version': '0.2.0', 'decision': decision,
            'route': decision.get('key'), 'probabilities': choice, 'applicability': evidence,
            'queriedKeys': config['queriedKeys'], 'precedence': config['precedence'],
            'usage': {key: sum(part['usage'][key] for part in responses)
                      for key in ['input_tokens', 'output_tokens', 'sequences']},
            'native': {'primary': primary, 'secondaryRequest': secondary_request, 'secondary': secondary}}


class Fev4B:
    """Load one complete local package; CUDA BF16 reference execution only.

    One instance keeps weights resident. Calls are serialized because concurrency
    and concurrent GPU-memory use have not been qualified for this release.
    """
    def __init__(self, directory, *, device='cuda:0'):
        self.root = Path(directory).resolve()
        self.integrity = verify_package(self.root)
        # All files are bundled. Runtime inference must not resolve Hub resources.
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict, get_peft_model_state_dict
        from safetensors.torch import load_file
        if torch.device(device).type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('This release requires a BF16-capable CUDA GPU; CPU/MPS inference is not validated.')
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError('CUDA BF16 support is required')
        torch.set_num_threads(4)
        self.config = json.loads((self.root / 'fev_config.json').read_text())
        expected = {'maxLength': 16384, 'batchTokens': 16384, 'seed': 20261030,
                    'nativeTemperature': 1.5539092221462472, 'thresholds': {'yes': .8, 'no': .2}}
        if self.config['runtime'] != expected:
            raise ValueError('Runtime configuration differs from the benchmarked release')
        torch.manual_seed(expected['seed'])
        sdk = self.root / 'checkpoint' / 'source' / 'src'
        sys.path.insert(0, str(sdk))
        from pplx_decider import Decider
        module_path = Path(sys.modules['pplx_decider'].__file__).resolve()
        if not module_path.is_relative_to(sdk):
            raise RuntimeError('A conflicting native SDK is already imported; use an isolated Python process')
        self.model = Decider.from_pretrained(self.root / 'checkpoint', device=device, fast=False,
            max_length=expected['maxLength'], batch_tokens=expected['batchTokens'], cpu_threads=4)
        if self.model.temperature != expected['nativeTemperature']:
            raise ValueError('Native temperature changed')
        adapter = self.root / 'adapter'
        inject_adapter_in_model(LoraConfig.from_pretrained(adapter), self.model.backbone.language_model)
        for name, param in self.model.backbone.named_parameters():
            if 'lora_' in name: param.data = param.data.float()
            param.requires_grad_(False)
        for param in self.model.readout.parameters(): param.requires_grad_(False)
        state = load_file(str(adapter / 'adapter_model.safetensors'))
        if set(state) != set(get_peft_model_state_dict(self.model.backbone.language_model)):
            raise ValueError('Adapter key set changed')
        restored = set_peft_model_state_dict(self.model.backbone.language_model, state)
        if restored.unexpected_keys or any('lora_' in k for k in restored.missing_keys):
            raise ValueError('Adapter was not restored completely')
        loras = [p for n,p in self.model.backbone.named_parameters() if 'lora_' in n]
        if len(loras) != 240 or sum(p.numel() for p in loras) != 10387456 or not all(p.dtype == torch.float32 for p in loras):
            raise ValueError('Adapter shape or dtype differs from the benchmark')
        self.model.backbone.eval()
        self.parameters = list(self.model.backbone.parameters()) + list(self.model.readout.parameters())
        if any(p.requires_grad for p in self.parameters):
            raise ValueError('Inference parameters must be frozen')
        self._versions = [(p,p._version) for p in self.parameters]
        self._lock = threading.Lock()
        self.device = device

    def _predict(self, request):
        import torch
        chunks = self.model.encoder.plan(request, 16384)
        if len(chunks) != 1 or chunks[0].length > 16384:
            raise ValueError('Request exceeds the 16,384-token single-sequence limit; rules are never truncated')
        with torch.inference_mode(): response = self.model.decide(request)
        if response['usage'] != {'input_tokens': chunks[0].length, 'output_tokens': 0, 'sequences': 1}:
            raise ValueError('Unexpected native token usage')
        if set(response['answers']) != set(request['questions']):
            raise ValueError('Incomplete native answer')
        return response

    def route(self, request, *, unknown_key):
        import torch
        request = copy.deepcopy(request)
        with self._lock:
            torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            result = route_with_predict(request, unknown_key=unknown_key, predict=self._predict)
            torch.cuda.synchronize(self.device)
            if any(p._version != version or p.requires_grad for p,version in self._versions):
                raise RuntimeError('Model parameters changed during inference')
            result['latencyMs'] = 1000 * (time.perf_counter() - started)
        return result

    def decide(self, request):
        """Answer Choice/Noul/Score questions in one joint forward pass.

        This exposes the native decision head, without the two-stage routing
        policy. Question quality and calibration need workload-specific checks.
        """
        request = validate_request(request)
        with self._lock:
            result = decide_with_predict(request, predict=self._predict)
            if any(p._version != version or p.requires_grad for p, version in self._versions):
                raise RuntimeError('Model parameters changed during inference')
        return result
