# Fev

A customer-agnostic 4B decision model by [Murad](https://github.com/murabcd). Classify, route, triage and check supplied rules without generating text. Categories, instructions, examples and history are runtime inputs.

**Weights and model card:** [murabcd/fev-4b](https://huggingface.co/murabcd/fev-4b).

## Quick start

Linux, Python 3.12 and a BF16-capable NVIDIA GPU. Tested on A10G 24 GB.

```bash
git clone https://github.com/murabcd/fev.git
cd fev
uvx --from huggingface-hub==1.31.0 hf download murabcd/fev-4b \
  --revision v0.2.0 --local-dir models/fev-4b
uv venv --python 3.12
uv pip install -r requirements.txt -r requirements-serve.txt
.venv/bin/python run.py --model-dir models/fev-4b --verify-only
.venv/bin/python serve.py --model-dir models/fev-4b --port 8000
```

One model stays resident and serves requests sequentially. Cold inference can take about 50 seconds; allow a 120-second client timeout. After download and dependency installation, inference uses local files.

## TypeSafe SDK

Install the optional client with `pip install 'typesafe-sdk[zstd]==0.7.5'`:

```python
from typesafe_sdk import Choice, Noul, TypeSafeClient

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8000",
                        model="fev-4b", timeout=120)
response = client.system_one(
    state="Мне дважды списали оплату. Верните лишнее списание.",
    questions={
        "team": Choice(criteria={"billing": "Оплата и возврат", "delivery": "Доставка"}),
        "refund": Noul(instructions="Клиент просит вернуть деньги?"),
    },
)
print(response.choices["team"].choice, response.nouls["refund"].noul)
```

Choice, Noul and Score use the text-only `POST /v1/systemone` endpoint. See [API.md](API.md) for typed questions, raw HTTP, authentication, compression and limits. The conversation router adds applicability checks and can return clarification; see [RUNNING.md](RUNNING.md). The full [model card](https://huggingface.co/murabcd/fev-4b) describes training ancestry, results and limitations.

## Tests without a GPU

```bash
uv venv --python 3.12
uv pip install -r requirements-test.txt
.venv/bin/python -m unittest discover -s tests -v
```

Synthetic protocol tests check routing composition, validation and real TypeSafe SDK requests against a fake model. They do not measure learned-model accuracy. No weights or customer records are needed.

## Benchmarks

| Frozen native comparison | Fev-4B | Kev-4B | Jev-1.14.0 |
|---|---:|---:|---:|
| Public development, 764 questions | 83.6% | 80.2% | 84.6% |
| Authored Russian, 144 questions | 100.0% | 99.3% | 100.0% |

Same inputs, zero failed responses. The public suite is already published development data; Russian probes repeat templates. These are not customer accuracy or production resolution rates. No benchmark inputs were used by the new trainer.

See [BENCHMARKS.md](BENCHMARKS.md) for the reproducible Fev/Kev/Jev comparison, source/type breakdowns, probability metrics and measured runtime trade-offs. Development agreement and authored probes are reported separately from customer accuracy.

## Optional compiled execution

```bash
.venv/bin/python serve.py --model-dir models/fev-4b --execution compiled --port 8000
```

Reference execution remains the default. The optional path compiles decoder layers with CUDA graphs, FP8, readout pruning and prefetching disabled. It uses the same released checkpoint and shared adapter. Compilation has a cold-start cost; warm speed depends on workload and GPU. Probability values and internal route shortlists can differ even when final routes agree. See [BENCHMARKS.md](BENCHMARKS.md) before selecting it.

## Training

Use runtime configuration to change categories and rules. For a demonstrated weakness across domains, the native shared-adapter trainer, grouped data split, calibration tool and evaluator are documented in [TRAINING.md](TRAINING.md). The trainer writes a separate overlay; the downloaded base remains unchanged. No per-customer adapter is required.

After this revision is published, install the companion agent skill:

```bash
npx skills add murabcd/fev@fev-finetune
```

The included toy corpus checks the training machinery. Its outputs are not a new released checkpoint or evidence of general improvement.

## Distribution

Weights are distributed on Hugging Face, not GitHub. The full download includes checkpoint, tokenizer, native source, adapter and SHA256 manifest. Exact ancestry and hashes are in [provenance.json](provenance.json); release history is in [CHANGES.md](CHANGES.md). Preserve [LICENSE](LICENSE), [NOTICE](NOTICE) and bundled notices. Apache-2.0. Report issues through this repository's Issues tab.

CPU, Apple Silicon, quantization, concurrent throughput and new-workload calibration have not been qualified. Use your managed reverse proxy and operational controls for deployment.
