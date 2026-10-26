# Running a Fev-4B training experiment

## Model and environment

This workflow continues the released rank-8, alpha-16 adapter: 10,387,456 FP32 trainable parameters. The BF16 backbone and native decision readout stay frozen. Training minimizes typed decision cross-entropy, not next-token prediction. Use labelled Choice, Noul and Score requests as described in [data-and-evaluation.md](data-and-evaluation.md).

Use Linux, Python 3.12 and a BF16-capable NVIDIA GPU for training. The repository's recorded short training smoke check used an A10G 24 GB; it does not establish memory requirements at the 16,384-token limit. The bundled data/report helpers and contract tests do not require CUDA or model weights. Do not install the GPU stack on a machine used only for data preparation.

In the reviewed source checkout, install pinned dependencies and download the separately pinned model package:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt -r requirements-test.txt
uvx --from huggingface-hub==1.31.0 hf download murabcd/fev-4b \
  --revision "$MODEL_REVISION" --local-dir models/fev-4b
.venv/bin/python run.py --model-dir models/fev-4b --verify-only
```

`MODEL_REVISION` must be the full 40-character Hub commit selected for the baseline. It is distinct from the Git source revision. Verification checks the complete package's hashes; it does not authenticate an unknown publisher.

## Local GPU execution

Freeze the partitions before starting. Use at least two application domains in training. Set the run directory to a new location and keep all previous outputs. `STEPS`, `LEARNING_RATE` and `SEED` must be selected from development evidence before reading the final test. Six steps on the included toy corpus check mechanics only.

Run these commands from the source checkout:

```bash
.venv/bin/python -m fev4b.train --model-dir models/fev-4b \
  --train work/split/train.jsonl --development work/split/development.jsonl \
  --steps "$STEPS" --learning-rate "$LEARNING_RATE" --seed "$SEED" \
  --out work/run/adapter

.venv/bin/python -m fev4b.calibrate --model-dir models/fev-4b \
  --adaptation-dir work/run/adapter --data work/split/calibration.jsonl \
  --out work/run/calibrated
```

The trainer records baseline/adapted **development** predictions, finite losses/gradients, changed adapter tensors and frozen parameter versions. It writes `adaptation.json`, adapter weights/config and `training-report.json`. The loader verifies the overlay's base identity, tensor keys/shapes/dtypes, finite payload and file hashes. An overlay is loaded only when explicitly selected; the original model directory is unchanged.

The encoder rejects requests beyond its single-sequence context rather than truncating rules. An out-of-memory error calls for inspecting actual planned token lengths and resource use; do not silently shorten requests or change the model size. A failed run is evidence to retain, not a reason to retry unchanged.

For the predeclared final comparison, run all three on exactly the same test file:

```bash
.venv/bin/python -m fev4b.benchmark --model-dir models/fev-4b \
  --data work/split/test.jsonl --out work/run/baseline-test
.venv/bin/python -m fev4b.benchmark --model-dir models/fev-4b \
  --adaptation-dir work/run/adapter --data work/split/test.jsonl --out work/run/raw-test
.venv/bin/python -m fev4b.benchmark --model-dir models/fev-4b \
  --adaptation-dir work/run/calibrated --data work/split/test.jsonl --out work/run/test
```

Run `scripts/compare.py` from the installed skill against baseline/raw and baseline/calibrated separately. Failed predictions remain in the denominator. Calibration fits a positive temperature on calibration NLL only. It preserves option ranking mathematically, but can change threshold-based routing and applicability. Validate the complete routing composition before adoption. Comparing raw and calibrated adapters separately avoids attributing a calibration gain to weight training.

Once final predictions are opened, use them for the adoption decision. Further tuning requires a new independent final test. Keep any additional regression corpus frozen and separate from training; a transfer improvement does not excuse unacceptable regressions on previously supported behavior.

## Bounded Hugging Face Job

The skill's `package_job.py` prepares `work/cloud-inputs` from a reviewed checkout and four validated partitions. `package.json` records source commit, source dirty status and each input SHA256. Use clean reviewed source for a real run. The worker checks those hashes and partition isolation before downloading weights.

`train_hf.py` has pinned UV dependencies. It runs training, calibration, unchanged baseline, raw candidate and calibrated candidate, then writes both paired reports. It never publishes model weights or automatically adopts an overlay.

Before submission, check `hf auth whoami`, `hf jobs ps` and `hf jobs hardware`. Confirm the data destination and a private output bucket, save the planned resource/time/compute cap and create a unique run ID. The following is a **submission template**, not a command to execute merely because the skill is installed:

```bash
hf jobs uv run --flavor a10g-small --timeout 20m --detach \
  --volume ./work/cloud-inputs:/inputs \
  --volume hf://buckets/YOUR_USER/YOUR_PRIVATE_BUCKET:/outputs:rw \
  work/cloud-inputs/train_hf.py --inputs /inputs --output /outputs/UNIQUE_RUN \
  --revision "$MODEL_REVISION" --steps "$STEPS" \
  --learning-rate "$LEARNING_RATE" --seed "$SEED"
```

Local directory mounts upload their contents. Inspect the package and destination permissions first. Never include credentials in the source ZIP, dataset or command. A timeout bounds job duration, not all account charges; choose the cap from current pricing and expected workload, including model loading and all three final evaluations. See the official [Jobs guide](https://huggingface.co/docs/huggingface_hub/guides/jobs) for current CLI behavior and billing.

Save the returned job ID immediately. Inspect `hf jobs inspect JOB_ID` and `hf jobs logs JOB_ID`; do not blindly resubmit after a connection error. Cancel an unwanted job with `hf jobs cancel JOB_ID` and verify its final status. Only stop jobs belonging to this experiment.

## Outputs and failure handling

| Path within the run | Contents |
|---|---|
| `intent.json`, `source/` | Pinned model revision, configuration, input hashes and executed source |
| `adapter/`, `calibrated/` | Explicit raw and temperature-calibrated overlays |
| `baseline-test/`, `raw-test/`, `test/` | Raw predictions, question rows, metrics and provenance for each variant |
| `comparison-raw/`, `comparison/` | Paired report, domain/type/source breakdowns and candidate error list |
| `complete.json` | Pipeline completion, prediction error counts and `adopted: false` |

`complete.json` means the sequence finished, not that quality passed. Saved prediction failures still cause a nonzero worker exit after reporting. Infrastructure or training failures may leave only partial outputs. Missing reports, changed hashes, leakage, nonfinite loss, changed frozen parameters or reload mismatch all block adoption. Preserve logs and report the failed stage; do not overwrite it with a clean-looking retry.

After downloading private artifacts, verify hashes and read both paired reports. Additional serving qualification requires separate latency, memory, concurrency, reliability and hardware measurements. Training scores alone do not establish deployment economics. To inspect a reviewed overlay locally, explicitly pass `--adaptation-dir work/run/calibrated` to `serve.py`; publication or deployment is a separate decision.
