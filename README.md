# Fev

A customer-agnostic 4B decision model from Flomni. Classify, route, triage and check supplied rules without generating text. Categories, instructions, examples and history are runtime inputs.

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

## Release

v0.2.0 exposes the existing typed decision head. Weights and routing policy are unchanged from v0.1.0. Routing had 293/303 reference agreement on already-consumed synthetic/source-derived development cases; this is not fresh human-gold accuracy or resolved-conversation coverage. The complete denominator and caveats are in [MODEL_CARD.md](MODEL_CARD.md).

The model uses a shared rank-8 LoRA adapter on a Qwen3.5-4B decision checkpoint. No per-customer training is performed. CPU, Apple Silicon, quantization, concurrent throughput and new-workload calibration have not been validated. Use your managed reverse proxy and operational controls for deployment.

Weights are distributed on Hugging Face, not GitHub. The full download includes checkpoint, tokenizer, native source, adapter and SHA256 manifest. Exact ancestry and hashes are in [provenance.json](provenance.json). Preserve [LICENSE](LICENSE), [NOTICE](NOTICE) and bundled notices. Apache-2.0. Report issues through this repository's Issues tab.
