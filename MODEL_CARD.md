---
language:
  - ru
license: apache-2.0
pipeline_tag: text-classification
tags:
  - fev
  - decision-model
  - routing
  - qwen3.5
  - lora
  - custom-code
---

# Fev-4B

## Model summary

Fev-4B is a decision model for classification, routing, triage and checks against supplied rules. Give it a document or conversation (the *state*) and named, typed questions. It returns option probabilities and structured decisions without generating text. Its text-decision server implements `POST /v1/systemone`, for use with the TypeSafe SDK through a custom base URL.

Built on the Qwen3.5-4B architecture, Fev combines a native decision readout with a shared LoRA adapter. Categories, business rules, examples and conversation history are supplied at runtime: customer-specific retraining is not required to change them. The package also includes a two-stage conversation router that can return a clarification signal when no route is sufficiently supported. Reference execution uses one 24 GB NVIDIA GPU.

This card describes Fev v0.2.0. The weights and routing policy are unchanged from v0.1.0; this version adds the public typed-decision interface and HTTP server.

## Model details

| | |
|---|---|
| Developer | Flomni ([github.com/murabcd/fev](https://github.com/murabcd/fev)) |
| Model type | Decision model: shared state and questions scored by a native readout; no text generation |
| Backbone | Qwen3.5-4B architecture; noncausal full attention |
| Adapter | Shared LoRA, rank 8, alpha 16; 10,387,456 parameters |
| Precision | BF16 checkpoint and readout; FP32 adapter, unmerged |
| Head | Native readout over question option codes; softmax probabilities |
| Context | 16,384 encoded tokens per native call, including state, questions and template; oversized requests rejected |
| Temperature | Fixed native temperature T = 1.5539092221462472; new-workload calibration has not been measured |
| Languages | Russian |
| Reference hardware | NVIDIA A10G, 24 GB; CUDA BF16 |
| License | Apache-2.0; bundled notices retained |
| Version | Fev v0.2.0; [versioned package](https://huggingface.co/murabcd/fev-4b/tree/v0.2.0) |

**Input.** Text, a JSON object or an array as the state, plus 1–64 named questions:

| Type | Options | Output |
|---|---|---|
| `choice` | 1–255 named options with optional descriptions | Probability per option, selected option and confidence |
| `score` | 2–255 ordered level descriptions | Probability per level, expected zero-based level index, legend and confidence |
| `noul` | Yes / no, with optional descriptions | Probability of yes |

A typed request is processed as one joint sequence in one forward pass. Questions share attention; they are not isolated from one another. The context limit is an execution bound, not a validated accuracy guarantee for long documents.

## Intended uses

- Classify or route messages using supplied category definitions and exclusions.
- Make yes/no checks against supplied requirements.
- Rate a state against an explicit ordered rubric.
- Keep business configuration outside model weights.
- Run conversation routing with explicit uncertainty handling.

The model returns decisions. The application owns follow-up questions, scenario execution and business actions. A clarification signal can continue automation; it does not require operator handoff.

## Out-of-scope uses

- Chat, summarisation, free-form answers or other text generation.
- Image assets, box questions and multipart requests through this API.
- Claims of production accuracy, calibrated probabilities on arbitrary workloads, or autonomous conversation resolution based only on this development benchmark.
- CPU, Apple Silicon, quantized or merged-adapter serving: these paths are not validated in this release.

## How to use

Download the complete versioned model package. On a Linux server with a compatible NVIDIA GPU and driver:

```bash
uvx --from huggingface-hub==1.31.0 hf download murabcd/fev-4b \
  --revision v0.2.0 --local-dir fev-4b
cd fev-4b
uv venv --python 3.12
uv pip install -r requirements.txt -r requirements-serve.txt
.venv/bin/python run.py --verify-only
.venv/bin/python serve.py --port 8000
```

The server keeps one model resident. Its default address is localhost. Code, examples and integration instructions are also on [GitHub](https://github.com/murabcd/fev).

```python
from typesafe_sdk import Choice, Noul, TypeSafeClient

client = TypeSafeClient(
    api_key="local", base_url="http://127.0.0.1:8000",
    model="fev-4b", timeout=120,
)
response = client.system_one(
    state="Мне дважды списали оплату. Верните лишнее списание.",
    questions={
        "team": Choice(
            instructions="Кто должен обработать обращение?",
            criteria={"billing": "Оплата и возврат денег", "delivery": "Доставка"},
        ),
        "refund": Noul(instructions="Клиент просит вернуть деньги?"),
    },
)
print(response.choices["team"].choice, response.nouls["refund"].noul)
```

Install the client with `pip install 'typesafe-sdk[zstd]==0.7.5'`. Text requests support plain JSON and zstd compression. The server exposes `GET /v1/models` and `POST /v1/systemone`. See [API.md](API.md) for limits, authentication, raw HTTP and Python usage.

For the conversation-routing policy, use `Fev4B.route(...)` or `run.py --input examples/request.json`; see [RUNNING.md](RUNNING.md). This policy makes a primary Choice call and, when applicable, a second call containing up to three route-applicability checks. It is distinct from a single native typed request.

## Training and release procedure

This release distributes a complete decision checkpoint, one shared adapter and inference code. The shared adapter comes from customer-agnostic synthetic decision research. No customer-specific adapter is selected at runtime, and no training was performed for the v0.2.0 interface release.

Checkpoint and adapter files retain their original precision and hashes. Exact ancestry, adapter hashes and changed-file boundaries are recorded in [provenance.json](provenance.json) and [CHANGES.md](CHANGES.md). This is an inference release, not a complete reproduction of historical training data or an independent audit of all upstream sources.

## Evaluation

The conversation-routing composition was measured on **303 development cases** with the full supplied business rules. Cases include synthetic and source-derived examples already consumed during development; this is not a fresh, independently human-labelled test set.

| Outcome | Cases |
|---|---:|
| Correct automatic route | 258 |
| Automatic route disagreement | 4 |
| Correct clarification | 35 |
| Unnecessary clarification | 6 |
| **Reference agreement** | **293 / 303 (96.7%)** |

Five references have unresolved business mappings, including all four automatic disagreements. Excluding those five gives 293 / 298 (98.3%). Both denominators are disclosed; the headline keeps the disputed cases.

| Development group | Agreement |
|---|---:|
| Earlier routing checks | 62 / 64 |
| Taxonomy checks | 54 / 55 |
| Boundary and transfer checks | 47 / 52 |
| Generic decision checks | 98 / 100 |
| Runtime instruction switches | 32 / 32 |

There were 48 / 48 consistent route-order pairs and 16 / 16 instruction-switch pairs with both answers correct. Eight same-job native repeat checks had no numerical change. [Aggregate results](evaluation/summary.json) are included; customer records and configurations are excluded.

These results measure the routing composition. They do not establish general typed-task accuracy, unseen-customer generalization, resolved-conversation coverage or production readiness. API contract tests are separate from model-quality evaluation.

**Release verification.** The complete package passed a CUDA A10G smoke check through the real TypeSafe Python SDK. Mixed Choice/Noul/Score requests used one backbone forward and generated zero tokens; plain JSON and zstd produced identical answers. Routing before and after typed calls passed the strict same-job comparison. Historical probability differences are reported separately. All 303 saved routing decisions also replayed unchanged locally. See [API verification](evaluation/api-verification.json). This is interface and packaging evidence, not a new accuracy benchmark.

## Compute and serving

The sequential 303-case run on one A10G had a median model time of 613 ms and p95 of 2,442 ms, excluding download, model loading, network and queueing. The first case took 50.6 seconds including cold kernel initialization. Peak allocated GPU memory was 10.01 GB and reserved memory was 10.45 GB (decimal units).

The reference server serializes requests. Concurrent throughput, production cost and smaller hardware have not been qualified. After downloading the package and installing dependencies, inference uses local files and requires no Hub access or external inference provider. Use a reverse proxy for TLS, rate limits and production observability.

## Limitations

- Ambiguous or contradictory definitions can produce disagreement or excess clarification.
- Missing conversation history is not reconstructed. Runtime examples are not actual dialogue turns.
- The routing resolver supports a documented subset of explicit priority syntax, not arbitrary workflow DSL execution.
- Routing cutoffs of 0.8 / 0.2 are policy settings, not correctness guarantees. Probability calibration must be checked on the intended workload.
- Ordered-score quality and long-context accuracy have not been independently benchmarked.

## License and reproducibility

Apache-2.0. Preserve [LICENSE](https://huggingface.co/murabcd/fev-4b/blob/main/LICENSE), [NOTICE](https://huggingface.co/murabcd/fev-4b/blob/main/NOTICE) and bundled upstream notices. [manifest.json](https://huggingface.co/murabcd/fev-4b/blob/main/manifest.json) records package hashes. Pin the `v0.2.0` tag or its commit when downloading. Weights are hosted on Hugging Face; source and usage examples are maintained on [GitHub](https://github.com/murabcd/fev).
