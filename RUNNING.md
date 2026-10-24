# Running Fev-4B

## Requirements

Linux, Python 3.12 and a BF16-capable NVIDIA CUDA GPU. The reference run used an A10G with 24 GB VRAM, PyTorch 2.14.0+cu130 and Transformers 5.17.0. Install a driver compatible with the PyTorch CUDA build. Provision at least 20 GB host RAM and sufficient disk for about 9.2 GB of model files plus Python/CUDA dependencies. The RAM recommendation is a provisioning allowance, not a measured minimum.

CPU and MPS are deliberately rejected by this reference loader because they have not been validated. An 8 GB Mac is suitable for downloading files and running integrity/replay checks, not for loading this reference model.

## Install and verify

```bash
uvx --from huggingface-hub==1.31.0 hf download murabcd/fev-4b --revision v0.2.0 --local-dir Fev-4B
cd Fev-4B
python3 run.py --verify-only
uv venv --python 3.12
uv pip install -r requirements.txt -r requirements-test.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run.py --input examples/request.json
```

The integrity-only command needs the Python standard library and streams files without loading tensors. `Fev4B` verifies the package before loading. Keep the full directory intact, including `checkpoint/source/src`, the adapter and runtime modules. No token generation, adapter merging or quantization is involved.

Download the package and Python wheels before moving to an isolated server. Runtime sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` and loads only local files. The public Hub distributes files; it is not a production inference dependency. Host and execute within your required data region.

## Input contract

The CLI expects a JSON object containing exactly `request` and `unknownKey`. See [examples/request.json](examples/request.json).

`request.state` contains:

- `message`: nonempty current message, up to 4,000 characters.
- `history`: optional actual previous turns, at most 30. Each has exactly `role` (`user` or `assistant`) and `content` (1–4,000 characters).
- `examples`: optional archive examples, each with exactly `message` and `category` strings. Do not supply synthetic history as real dialogue.

`request.questions.route` contains exactly `type: "choice"`, `criteria` (2–100 nonempty route keys and descriptions) and nonempty `instructions` containing the complete business rules. `unknownKey` must name an explicit unknown category in `criteria`.

Rules, routes, examples and history are runtime configuration. No customer-specific training or embedded client configuration is required. Keep definitions distinct and encode exceptions explicitly. The model cannot supply missing business decisions for you.

The encoded native primary and applicability calls must each fit one 16,384-token sequence, including all protocol overhead. The loader checks the actual encoder planner and rejects overflow; it never truncates rules. Passing the primary call's bound does not guarantee the secondary call fits. Callers must handle validation/runtime exceptions as errors, without inventing a route.

## Outputs

`decision` is either `{"kind":"route","key":"..."}` or `{"kind":"clarification","reason":"..."}`. `route` is the chosen key or null. `probabilities` contains the native competing Choice distribution. `applicability` holds independent yes probabilities for queried routes and null for unqueried routes. It is not a distribution over routes. `usage` totals encoded tokens and native sequences across both stages; `output_tokens` is zero. Python additionally returns `native` evidence for diagnostics; the CLI omits it. `latencyMs` excludes initial loading and checksum verification.

A clarification is a workflow signal: ask a suitable follow-up, retain actual history and classify again. It is not an automatic operator transfer and does not include a generated follow-up question. Scenario execution and answering are separate application responsibilities.

## Explicit pair precedence

The deterministic resolver recognizes only this exact authored Russian syntax, where the quoted values must equal complete configured route **descriptions**, not keys:

```text
если одновременно выражены цели «<description A>» и «<description B>», основной является цель «<preferred description>»
```

The preferred description must be A or B and each description must identify one route. Ambiguous references or cycles are errors. Other rules remain model inputs; arbitrary business prose is not compiled into deterministic edges. This preserves the measured runtime rather than introducing a new parser during packaging.

## Integration

Run one resident `Fev4B` instance in an isolated Python process and call `route` from your service. Calls within an instance are serialized. Multiple workers replicate the weights; budget VRAM accordingly. Concurrency and capacity have not been measured. Put authentication, TLS, request-size controls, timeouts and operational monitoring in your service. The loader does not start an endpoint automatically. Run `serve.py` explicitly for the text-only typed-decision API; see API.md.

Use a separate test application for integration. This release does not change any client workflow. After deploying a service in your own environment, connect its endpoint through the provider/API mechanism supported by your application; merely selecting a Hugging Face model ID does not start this custom runtime.

## Reproducibility and updates

Keep `v0.2.0` or the corresponding Hub commit pinned. The exact checkpoint, shared adapter and routing composition are recorded in `provenance.json`; every distributed file except `manifest.json` itself is hashed in the manifest. Hub download-cache metadata and Python bytecode are not release files.

Do not run with Python `-O`; native research invariants include assertions. This reference uses BF16 base/readout and FP32 unmerged adapter. Different precision, kernels, context, temperature, thresholds, wrapper logic or model merging require separate parity and quality verification. No training is run by the loader.

## Typed decisions and HTTP

`Fev4B.decide` exposes Choice, Noul and Score without the routing policy. Use `serve.py` for the text-only System One endpoint; see [API.md](API.md). The typed interface scores the supplied questions without applying the conversation-routing policy.

## Execution mode and explicit adapters

`--execution reference` is the default. `--execution compiled` enables only the qualified layer-compilation options; kernels are compiled lazily. It retains the BF16 backbone and FP32 unmerged adapter. Warm speed measurements do not include compilation and are not throughput guarantees. See [BENCHMARKS.md](BENCHMARKS.md) for the exact tested workloads and probability/parity limits.

The Python constructor accepts the same selection:

```python
model = Fev4B("models/fev-4b", execution="compiled")
```

A research adapter is loaded only with explicit `adaptation_dir=...` or `--adaptation-dir ...`. It is checked against the base identity and payload hashes before use. Reference weights are never overwritten. Newly trained or calibrated overlays need their own routing-quality and compiled-parity checks; the released checkpoint's results do not transfer automatically. See [TRAINING.md](TRAINING.md).
