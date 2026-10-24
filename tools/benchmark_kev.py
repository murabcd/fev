# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "kev @ git+https://github.com/jaredpalmer/kev.git@2074cc44256200bb37540103fa4572901faa10f3",
#   "torch==2.8.0", "transformers==5.17.0", "peft==0.21.2",
#   "accelerate==1.15.0", "numpy==2.5.3", "datasets==5.0.1",
#   "scikit-learn==1.9.1", "pydantic==2.13.5", "huggingface-hub==1.31.0",
#   "safetensors==0.8.0", "flash-linear-attention==0.5.2", "einops==0.8.1",
# ]
# ///
"""Run the pinned official Kev BF16 reference path on Fev's common JSONL."""
import argparse
import hashlib
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('This measured path needs a CUDA GPU; no CPU/MPS comparison is qualified')
    from huggingface_hub import snapshot_download
    from kev.api import SystemOneRequest, to_record
    from kev.checkpoint import Checkpoint, LoadOptions
    from fev4b.data import load_records
    from fev4b.benchmark import evaluate
    revision = '6cfce5c2fa4b4bd64026336ab649c5ca78857d52'
    torch.set_num_threads(4)
    path = snapshot_download('jaredpalmer/kev-4b', revision=revision)
    checkpoint = Checkpoint(path)
    tok, model = checkpoint.load('cuda', LoadOptions(dtype=torch.bfloat16, merge=True,
        attn='sdpa', backend='torch', fused=False, cuda_graphs=False))
    def predict(request):
        rec, meta = to_record(SystemOneRequest(**request))
        enc = model.encode(tok, rec, max_state=16384, max_branch=16384, strict=True)
        if len(enc['ids']) > 16384:
            raise ValueError('Comparison single-request token limit exceeded')
        with torch.inference_mode():
            logits = model.forward(enc)
        answers = {}
        for key, q, z in zip(request['questions'], meta, logits, strict=True):
            probabilities = z.float().softmax(-1).cpu().tolist()
            if q['type'] == 'noul':
                answers[key] = {'type': 'noul', 'noul': probabilities[1]}
            else:
                answers[key] = {'type': q['type'], 'probabilities': dict(zip(q['keys'], probabilities, strict=True))}
        return {'model': 'kev-4b@'+revision, 'answers': answers}
    report = evaluate(load_records(args.data), predict, args.out, provenance={
        'revision': revision, 'codeRevision': '2074cc44256200bb37540103fa4572901faa10f3',
        'dataSha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'precision': 'BF16 merged, official eager SDPA; no graphs/fused',
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'temperature': model.head.temperature})
    if report['overall']['errors']:
        raise RuntimeError('Comparison has failed questions; inspect saved results')


if __name__ == '__main__':
    main()
