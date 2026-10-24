# Fev

A small decision model you can configure, train and run on your own server.

<p>
  <a href="https://github.com/murabcd/fev/actions/workflows/test.yml"><img alt="Contract tests" src="https://img.shields.io/github/actions/workflow/status/murabcd/fev/test.yml?branch=main&amp;style=for-the-badge&amp;labelColor=000000" height="28"></a>
  <a href="https://huggingface.co/murabcd/fev-4b"><img alt="Weights: Fev-4B" src="https://img.shields.io/badge/WEIGHTS-4B-0a0a0a.svg?style=for-the-badge&amp;labelColor=000000" height="28"></a>
  <a href="skills/fev-finetune/SKILL.md"><img alt="Agent skill: fev-finetune" src="https://img.shields.io/badge/SKILL-fev--finetune-0a0a0a.svg?style=for-the-badge&amp;labelColor=000000" height="28"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/LICENSE-Apache--2.0-0a0a0a.svg?style=for-the-badge&amp;labelColor=000000" height="28"></a>
</p>

Fev-4B is a customer-agnostic decision model by [Murad](https://github.com/murabcd), built on the Qwen3.5-4B architecture with a native decision readout and a shared LoRA adapter. Supply a state, questions and answer options; Fev returns probabilities without generating text. Its text API supports the TypeSafe SDK through a custom base URL.

[Models](#models) · [Quick Start](#quick-start) · [Fine-Tuning](#fine-tuning) · [Agent Skills](#agent-skills) · [Benchmarks](#benchmarks) · [Serving Performance](#serving-performance)

## Highlights

- Named choices, yes/no checks and ordered scores in one request.
- Categories, business rules, examples and conversation history supplied at runtime; no per-customer adapter required.
- Text-only `POST /v1/systemone`, real TypeSafe SDK support, optional bearer authentication and zstd requests.
- One resident 4B model, with reference execution by default and an optional compiled CUDA path.
- Shared-adapter training, grouped data splitting, separate temperature calibration and evaluation tools.
- Complete versioned package and SHA256 manifest on Hugging Face; inference stays local after setup.

## Models

| Model | Architecture | Reference hardware | Encoded request limit | Card |
|---|---|---|---|---|
| [Fev-4B](https://huggingface.co/murabcd/fev-4b) | Qwen3.5-4B; shared rank-8 LoRA | NVIDIA A10G, 24 GB; CUDA BF16 | 16,384 tokens, including state and questions | [Details](MODEL_CARD.md) |

The context limit is an execution bound, not a measured long-document accuracy guarantee. CPU, Apple Silicon, quantization and concurrent throughput have not been qualified. Adapter weights remain FP32 and unmerged.

### Comparison

| Model | Public dev. | Russian probes | Combined | Brier ↓ |
|---|---:|---:|---:|---:|
| Kev‑4B | 80.2% | 99.3% | 83.3% | 0.2543 |
| **Fev‑4B** | **83.6%** | **100.0%** | **86.2%** | **0.2330** |
| Jev‑1.14.0 | 84.6% | 100.0% | 87.0% | 0.2250 |

Public development: 764 questions. Authored Russian probes: 144 questions. Combined: 908 questions. Brier is measured on the public split; lower is better.

Same 812 requests, zero failed responses. Open models use their BF16 serving paths on the same GPU class; Jev uses its pinned hosted API. This is not Kev's published FP32 evaluation. The public suite is already published development data; Russian probes repeat authored templates. Results are engineering comparisons, not customer accuracy or production resolution. No comparison inputs entered the new trainer. [Methods, revisions and raw results](BENCHMARKS.md).

## Quick Start

### Run on a GPU Server

Linux, Python 3.12 and a BF16-capable NVIDIA GPU. The reference machine is an A10G 24 GB.

```bash
git clone https://github.com/murabcd/fev.git
cd fev
uvx --from huggingface-hub==1.31.0 hf download murabcd/fev-4b \
  --revision v0.2.1 --local-dir models/fev-4b
uv venv --python 3.12
uv pip install -r requirements.txt -r requirements-serve.txt
.venv/bin/python run.py --model-dir models/fev-4b --verify-only
.venv/bin/python serve.py --model-dir models/fev-4b --port 8000
```

One model stays resident and serves requests sequentially. Cold inference can take about 50 seconds; allow a 120-second client timeout. After download and dependency installation, inference uses local files.

### Use From Python

Install the optional client with `pip install 'typesafe-sdk[zstd]==0.7.5'`:

```python
from typesafe_sdk import Choice, Noul, TypeSafeClient

client = TypeSafeClient(
    api_key="local", base_url="http://127.0.0.1:8000",
    model="fev-4b", timeout=120,
)
response = client.system_one(
    state="Мне дважды списали оплату. Верните лишнее списание.",
    questions={
        "team": Choice(criteria={"billing": "Оплата и возврат", "delivery": "Доставка"}),
        "refund": Noul(instructions="Клиент просит вернуть деньги?"),
    },
)
print(response.choices["team"].choice, response.nouls["refund"].noul)
```

| Question | Input | Output |
|---|---|---|
| `choice` | 1–255 named options and optional descriptions | Selected option, probabilities and confidence |
| `noul` | Yes/no question and optional descriptions | Probability of yes |
| `score` | 2–255 ordered level descriptions | Distribution, expected level index and confidence |

Questions share one sequence and can attend to each other. Oversized requests are rejected rather than truncated. See [API.md](API.md) for schemas, raw HTTP, authentication, compression and limits. The separate conversation router adds applicability checks and can return clarification; see [RUNNING.md](RUNNING.md).

## Fine-Tuning

Use runtime configuration to change categories and rules. For a demonstrated weakness across domains, continue the existing shared rank-8 adapter with native decision cross-entropy. The backbone and readout stay frozen; new weights are written to an explicit overlay. This is a shared-model research workflow, not a requirement for onboarding a customer.

### With a Coding Agent

```bash
npx skills add murabcd/fev@fev-finetune
```

Then ask your agent: “Evaluate a shared Fev-4B adapter on separate domains, with grouped splits, a bounded GPU job, calibration and a comparison against the starting checkpoint.” The [skill](skills/fev-finetune/SKILL.md) and its [training guide](skills/fev-finetune/references/training.md) contain the workflow. It does not automatically publish, deploy or replace released weights. Use a GPU server or Hugging Face Job; a local GPU is not required for preparation.

### By Hand

Add a `label` to each typed question in a JSONL request, plus a unique `id`, a `domain` and a `group` for related records. Choice labels are option keys, Noul labels are booleans, and Score labels are zero-based rubric indices. Keep paraphrase/source families together and freeze train, development, calibration and test partitions before training.

```bash
.venv/bin/python -m fev4b.data --data examples/training.jsonl --out work/split --seed 42
.venv/bin/python -m fev4b.train --model-dir models/fev-4b \
  --train work/split/train.jsonl --development work/split/development.jsonl \
  --steps 6 --learning-rate 0.00002 --seed 42 --out work/adapter-smoke
.venv/bin/python -m fev4b.calibrate --model-dir models/fev-4b \
  --adaptation-dir work/adapter-smoke --data work/split/calibration.jsonl \
  --out work/calibrated-smoke
.venv/bin/python -m fev4b.benchmark --model-dir models/fev-4b \
  --adaptation-dir work/calibrated-smoke --data work/split/test.jsonl \
  --out work/test-once
```

Install `requirements-test.txt` alongside `requirements.txt` for the tools. Six steps on the included toy corpus check mechanics; they are not a recommended training duration or evidence of general improvement. The completed smoke run changed all 240 adapter tensors, verified frozen backbone/readout, restored tensors bit-for-bit and completed separate calibration/test. No toy adapter or fitted temperature is included in the released weights. [Full recipe and bounded HF Job instructions](TRAINING.md) · [Validation evidence](evaluation/benchmarks/training-validation.json).

## Agent Skills

| Skill | Purpose | Install |
|---|---|---|
| [fev-finetune](skills/fev-finetune/SKILL.md) | Prepare labelled decisions, train a shared adapter, calibrate and compare on separate data | `npx skills add murabcd/fev@fev-finetune` |

The skill is distributed from this repository in standard `SKILL.md` format. Its instructions keep the downloaded base immutable and require evidence before adopting an overlay.

## Serve Your Own Endpoint

```bash
export FEV_API_KEY='your-local-server-key'
.venv/bin/python serve.py --model-dir models/fev-4b --host 0.0.0.0 --port 8000
# Opt in to layer compilation on the same weights:
.venv/bin/python serve.py --model-dir models/fev-4b --execution compiled --port 8000
```

Binding outside localhost requires `FEV_API_KEY`; the server checks bearer authentication. Put TLS, rate limits and operational monitoring in your reverse proxy. No external inference provider is needed after setup. For an explicitly selected research overlay, add `--adaptation-dir PATH`; overlays are never discovered or adopted automatically.

## Benchmarks

| Suite | Questions | Purpose |
|---|---:|---|
| Public `transfer-v4` development | 764 | Public knowledge, routing, entailment, sentiment and ordinal decisions |
| Authored Russian probes | 144 | Controlled fictional facts across six domains; repetitive templates |
| Earlier conversation-routing development | 303 cases | Final routing/reference agreement; distinct from native typed accuracy |

Inputs, model revisions and source hashes are frozen in the [benchmark documentation](BENCHMARKS.md). Saved predictions, failed-response counts, Brier/NLL/ECE, fixed-threshold coverage and paired outcomes are included. These development measurements do not establish unseen-task calibration or completed-conversation coverage.

```bash
.venv/bin/python tools/prepare_benchmark.py --out work/common-inputs
.venv/bin/python -m fev4b.benchmark --model-dir models/fev-4b \
  --data work/common-inputs/comparison.jsonl --out work/fev-common
```

[BENCHMARKS.md](BENCHMARKS.md#reproduce) also reproduces pinned Kev and hosted Jev, and compares saved runs without requerying. Review confident mistakes and source/type breakdowns before choosing a threshold.

## Serving Performance

Measured on the same A10G, three selected routing workloads with five warm repeats each:

| Execution | Warm median | Warm p95 | Peak allocated GPU memory |
|---|---:|---:|---:|
| Reference | 2,451.97 ms | 2,539.22 ms | 9.997 GB |
| Compiled | 1,690.33 ms | 1,733.43 ms | 10.086 GB |

Compiled execution reduced warm median time by **31.1%**, with **0.9% more allocated memory**, while preserving **303/303 final routing decisions**. Internal candidate sets changed on 33 cases; probabilities are not numerically identical. Loading, compilation, HTTP, queueing and concurrent throughput are excluded. This is not a production cost estimate. [Runtime evidence](evaluation/benchmarks/runtime.json).

Reference remains the default. CUDA graphs, FP8, pruning and prefetching are disabled in the optional compiled path.

## Development

```bash
uv venv --python 3.12
uv pip install -r requirements-test.txt
.venv/bin/python -m unittest discover -s tests -v
```

Contract tests cover routing composition, validation, training-data safeguards, metrics and real TypeSafe SDK requests against a fake model. They need no GPU, weights or customer records. Learned-model measurements are separate.

## Limitations

Fev scores supplied options and evidence; it does not generate explanations, retrieve missing facts or execute business actions. Ambiguous definitions and missing history can cause mistakes or clarification. Top-option probabilities and routing cutoffs are not correctness guarantees. CPU, Apple Silicon, quantization, concurrent capacity and new-workload calibration remain unqualified.

## Author and License

Developed by [Murad](https://github.com/murabcd). Weights and the [model card](https://huggingface.co/murabcd/fev-4b) are on Hugging Face; code, training tools and skills are here. [Release history](CHANGES.md) · [Exact ancestry and hashes](provenance.json).

Apache-2.0. Preserve [LICENSE](LICENSE), [NOTICE](NOTICE) and bundled notices. Report issues through this repository's Issues tab.
