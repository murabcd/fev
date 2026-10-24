# Fev typed-decision API

The reference server exposes the text-only subset of the System One API. Choice, Noul and Score requests work with `typesafe-sdk[zstd]==0.7.5` using `base_url="http://127.0.0.1:8000"` and `model="fev-4b"`. No SDK patch is required. This is not an OpenAI chat-completions endpoint.

## Start

Download the full HF package as shown in README.md. Install `requirements.txt` and `requirements-serve.txt`, then run:

```bash
.venv/bin/python serve.py --model-dir . --port 8000
```

One resident CUDA BF16 model serves requests sequentially. Startup verification hashes all bundled files. The first inference may take about 50 seconds for cold kernels; use a client timeout of at least 120 seconds. CPU and Apple Silicon are not supported by this reference loader.

For access beyond localhost, set `FEV_API_KEY` to a secret and pass `--host 0.0.0.0`. The server requires `Authorization: Bearer <secret>` on both endpoints when the variable is set. Put it behind your managed reverse proxy for TLS, admission control, rate limits and monitoring. It logs no request contents. This small server is an integration reference, not a qualified production serving stack.

## Request and response

`POST /v1/systemone`, `Content-Type: application/json`:

```json
{
  "model": "fev-4b",
  "state": "Мне дважды списали оплату.",
  "questions": {
    "billing": {"type": "noul", "instructions": "Это вопрос об оплате?"}
  }
}
```

Response shape (probabilities shown here are illustrative):

```json
{
  "model": "fev-4b",
  "answers": {"billing": {"type": "noul", "noul": 0.95}},
  "usage": {"input_tokens": 100, "output_tokens": 0, "sequences": 1}
}
```

The state can be a string, JSON object or array. Instructions and option descriptions can also be strings, objects or arrays. All rules and category definitions come from the caller. Named answers preserve question IDs.

- `choice`: `criteria` maps 1–255 option names to descriptions (or null to use the name). Returns `choice`, `probabilities` and `confidence`.
- `noul`: optional `criteria` maps `true` and/or `false` to descriptions. Returns `noul`, the probability of yes.
- `score`: `criteria` lists 2–255 ordered level descriptions. Returns the expected zero-based level index in `score`, plus `legend`, `probabilities` and `confidence`. A score may lie between levels; it is not the most likely integer level.

See `examples/typed-request.json` for all three types in one request:

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @examples/typed-request.json
```

Python without HTTP:

```python
from fev4b import Fev4B
model = Fev4B("/path/to/downloaded/fev-4b")
result = model.decide({"state": "Нужен возврат денег", "questions": {
    "refund": {"type": "noul", "instructions": "Клиент просит вернуть деньги?"}
}})
```

`GET /v1/models` returns the available model ID. Responses include `x-typesafe-request-id`.

## Limits and errors

1–64 questions, 16,384 encoded tokens for the complete joint sequence, 1 MiB JSON body before and after decompression, JSON nesting at most 32 levels. Oversized token requests are rejected before inference; no state or rules are truncated. Requests use one joint forward pass. Questions share attention and can influence one another.

Plain JSON and `Content-Encoding: zstd` are accepted. Images, assets, box questions, multipart bodies, transfer-encoded/chunked bodies and unknown fields are rejected. The Python SDK selects compatible JSON automatically for text-only questions. No outbound image fetching or external inference occurs.

Errors use an HTTP status and JSON `detail`: 401 for authentication, 404 for endpoints, 411 for missing length, 413 for body size, 415 for media/encoding, 422 for invalid schema or token length, and 500 for inference failure. No partial answers are returned.

The `confidence` field is a native distribution-derived statistic, not an independently measured probability of correctness. Set decision thresholds using evidence appropriate to your workload. API compatibility does not imply quality equivalence to other decision models.
