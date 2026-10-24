"""Continue the released shared LoRA on generic labelled typed decisions (CUDA)."""
import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

from .adaptation import base_identity
from .data import load_records, request_of, require_disjoint, target_index, state_hash
from .integrity import file_sha256


def differentiable_logits(wrapper, request):
    """Same encoder, full attention and readout as reference inference, with gradients."""
    from pplx_decider.protocol import left_pad
    model = wrapper.model
    chunks = model.encoder.plan(request, 16384)
    if len(chunks) != 1 or chunks[0].length > 16384:
        raise ValueError('Training request exceeds the native single-sequence context')
    chunk = chunks[0]
    inputs = {k: v.to(model.device) for k, v in left_pad(chunks, model.encoder.pad_id).items()}
    hidden = model.backbone(**inputs, use_cache=False).last_hidden_state
    offset = inputs['input_ids'].shape[1] - chunk.length
    logits = model.readout(hidden[0, chunk.markers.to(model.device) + offset]).float()
    return {key: logits[i, :count] for i, (key, count) in enumerate(zip(chunk.keys, chunk.counts, strict=True))}


def train(model_dir, records, development, out, *, steps=100, learning_rate=2e-5, seed=42):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    from .model import Fev4B
    from .benchmark import evaluate
    if steps < 1 or not math.isfinite(learning_rate) or not 0 < learning_rate <= .001:
        raise ValueError('Invalid training steps or learning rate')
    require_disjoint(records, development)
    domains = {row.get('domain') for row in records}
    if None in domains or len(domains) < 2:
        raise ValueError('Shared training requires domain metadata and at least two domains')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    model = Fev4B(model_dir)
    # The inference loader sets its own seed while constructing/restoring the
    # released adapter. Training randomness belongs to this experiment.
    torch.manual_seed(seed)
    # Planning is checked before allocating optimizer state or taking a step.
    for row in records + development:
        chunks = model.model.encoder.plan(request_of(row), 16384)
        if len(chunks) != 1 or chunks[0].length > 16384:
            raise ValueError('Training/evaluation request exceeds context; nothing is truncated')
    baseline = evaluate(development, model.decide, out/'baseline', provenance={'purpose': 'training development baseline'})
    if baseline['overall']['errors']:
        raise RuntimeError('Baseline evaluation failed; training was not started')
    native = model.model
    native.backbone.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    trainable = []
    frozen = []
    for name, parameter in native.backbone.named_parameters():
        if 'lora_' in name:
            parameter.requires_grad_(True)
            trainable.append(parameter)
        else:
            frozen.append((parameter, parameter._version))
    frozen += [(p, p._version) for p in native.readout.parameters()]
    if len(trainable) != 240 or sum(p.numel() for p in trainable) != 10387456:
        raise ValueError('Only the existing shared rank-8 adapter may be trained')
    before = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(native.backbone.language_model).items()}
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0)
    native.backbone.train()
    order = list(range(len(records)))
    rng = random.Random(seed)
    history = []
    started = time.perf_counter()
    for step in range(steps):
        if step % len(order) == 0:
            rng.shuffle(order)
        record = records[order[step % len(order)]]
        optimizer.zero_grad(set_to_none=True)
        values = differentiable_logits(model, request_of(record))
        losses = [torch.nn.functional.cross_entropy(
            (values[key]/native.temperature)[None, :],
            torch.tensor([target_index(question)], device=native.device))
            for key, question in record['questions'].items()]
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        loss.backward()
        gradients = [p.grad for p in trainable if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError('Missing or non-finite adapter gradients')
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        history.append({'step': step+1, 'loss': float(loss.detach()), 'gradientNorm': float(norm)})
        print(json.dumps(history[-1]), flush=True)
    if any(p._version != version or p.grad is not None for p, version in frozen):
        raise RuntimeError('Frozen backbone/readout changed')
    state = {k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(native.backbone.language_model).items()}
    changed = sum(not torch.equal(v, before[k]) for k, v in state.items())
    if not changed:
        raise RuntimeError('Training did not update the adapter')
    save_file(state, str(out/'adapter_model.safetensors'))
    shutil.copyfile(Path(model_dir)/'adapter/adapter_config.json', out/'adapter_config.json')
    meta = {'format': 'fev-adaptation-v1', 'baseIdentity': base_identity(model_dir),
            'temperature': native.temperature, 'temperatureStatus': 'inherited; not fitted to these examples',
            'files': {name: file_sha256(out/name) for name in ('adapter_config.json', 'adapter_model.safetensors')},
            'purpose': 'shared customer-agnostic research adapter; not automatically adopted',
            'steps': steps, 'seed': seed, 'learningRate': learning_rate,
            'trainRecords': len(records), 'trainDomains': sorted(domains), 'changedAdapterTensors': changed,
            'consumed': {'ids': sorted({r['id'] for r in records + development}),
                         'states': sorted({state_hash(r) for r in records + development}),
                         'groups': sorted({r['group'] for r in records + development if 'group' in r})},
            'backboneAndReadoutFrozen': True, 'seconds': time.perf_counter()-started}
    (out/'adaptation.json').write_text(json.dumps(meta, indent=2)+'\n')
    (out/'steps.json').write_text(json.dumps(history, indent=2)+'\n')
    native.backbone.language_model.gradient_checkpointing_disable()
    native.backbone.eval()
    for p in trainable:
        p.requires_grad_(False)
    model._versions = [(p, p._version) for p in model.parameters]
    adapted = evaluate(development, model.decide, out/'adapted', provenance={'purpose': 'training development result'})
    if adapted['overall']['errors']:
        raise RuntimeError('Adapted evaluation failed; overlay is not qualified')
    summary = {'baseline': baseline['overall'], 'adapted': adapted['overall'],
               'training': meta, 'adopted': False, 'testPartitionRead': False}
    (out/'training-report.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--development', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    result = train(args.model_dir, load_records(args.train), load_records(args.development), args.out,
                   steps=args.steps, learning_rate=args.learning_rate, seed=args.seed)
    result['dataHashes'] = {name: file_sha256(path) for name, path in
                           [('train', args.train), ('development', args.development)]}
    (args.out/'training-report.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
