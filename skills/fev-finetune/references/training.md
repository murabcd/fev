# Training Fev-4B

Training is optional. Customer categories, business rules, examples and actual conversation history are runtime inputs. The default Fev checkpoint remains shared across customers. This workflow continues the released rank-8, alpha-16 adapter (10,387,456 FP32 parameters); the BF16 backbone and native readout stay frozen. It uses decision cross-entropy, not next-token/chat SFT.

## Environment and baseline

Use Linux, Python 3.12 and a BF16-capable NVIDIA GPU. An A10G 24 GB is the research reference. The short smoke workload does not establish memory requirements at the 16,384-token limit. An 8 GB Mac can prepare data and run contract tests; use a GPU server or a bounded Hugging Face Job for training.

In a checkout containing the training tools:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt -r requirements-test.txt
uvx --from huggingface-hub==1.31.0 hf download murabcd/fev-4b \
  --revision v0.2.0 --local-dir models/fev-4b
.venv/bin/python run.py --model-dir models/fev-4b --verify-only
```

The unchanged v0.2.0 package is an explicit baseline. Pin its immutable commit for a recorded experiment. Do not train on published benchmark inputs. Do not interpret multiple-domain metadata alone as evidence of generalization.

## Labelled data

Use one JSON object per line. `id` must be unique. `domain` identifies the application domain; `group` joins a document, paraphrase family or generation template so it cannot cross splits. Each question includes its normal typed request and a `label`:

```json
{"id":"library-001","group":"library-reservation-family-a","domain":"library","state":{"request":"Отмените бронь книги","reserved":2,"limit":4},"questions":{"action":{"type":"choice","instructions":"Выберите текущую просьбу.","criteria":{"cancel":"Отменить бронь","change":"Изменить бронь","info":"Запросить информацию"},"label":"cancel"},"capacity":{"type":"noul","instructions":"Количество reserved меньше limit?","label":true},"load":{"type":"score","instructions":"Оцените reserved.","criteria":["Меньше 2","От 2 до 4 включительно","Больше 4"],"label":1}}}
```

Choice labels are option keys; Noul labels are JSON booleans; Score labels are zero-based integer indices into the rubric. `label` and dataset metadata are removed before inference. Include `domain` on every training record and at least two training domains. Use group IDs across domains when one template/source produced both. Review ambiguous definitions and labels instead of letting a model's prediction become its own gold answer.

`examples/training.jsonl` contains controlled toy examples for mechanics only. It is not a production training corpus or a quality benchmark.

```bash
.venv/bin/python -m fev4b.data --data examples/training.jsonl --out work/split --seed 42
```

The deterministic split uses connected exact states and supplied groups: about 70% training and 10% each development, calibration and test by group count. It needs at least four independent groups. It does not detect fuzzy duplicates; choose meaningful family groups and audit the result. For cross-domain evidence, also hold out entire domains explicitly rather than relying on a random group split.

## Train, calibrate, test

Run a baseline on the frozen development partition, then take one predeclared bounded run. The trainer records its own development baseline and rejects overlap, oversized requests and a changed adapter architecture. It never truncates rules and never writes into the original model directory.

```bash
.venv/bin/python -m fev4b.train \
  --model-dir models/fev-4b \
  --train work/split/train.jsonl --development work/split/development.jsonl \
  --steps 6 --learning-rate 0.00002 --seed 42 --out work/adapter-smoke

.venv/bin/python -m fev4b.calibrate \
  --model-dir models/fev-4b --adaptation-dir work/adapter-smoke \
  --data work/split/calibration.jsonl --out work/calibrated-smoke

.venv/bin/python -m fev4b.benchmark \
  --model-dir models/fev-4b --adaptation-dir work/calibrated-smoke \
  --data work/split/test.jsonl --out work/test-once
```

Six steps on the toy file test the machinery; they are not a recommended training duration or an improved released model. Select step count and learning rate using development data before opening a final test. Output directories must be new. Preserve failed-run evidence instead of overwriting it.

The trainer verifies adapter-only gradients, frozen parameter versions, finite losses and an actual adapter update. Its outputs include weights/config, `adaptation.json`, baseline/adapted predictions and a training report. The loader checks the base identity, adapter configuration, tensor keys/shapes/dtypes, finite values and payload SHA256. Checksums detect accidental changes; they do not authenticate an untrusted publisher.

Calibration fits one positive temperature by calibration NLL. It preserves per-question option ranking mathematically, but can change routing cutoffs, coverage and applicability decisions. Evaluate the complete routing composition before adopting a calibrated overlay. The base loader's fixed thresholds are not correctness guarantees.

The benchmark reports all questions in the accuracy denominator, including failures; Brier/NLL/ECE use answered questions and disclose their count. It also reports fixed-threshold coverage and errors. For Score, probabilities and expected ordinal position matter as well as top-level accuracy. ECE on a small or synthetic sample is not proof of general calibration.

Compare unchanged baseline and candidate on exactly the same untouched test data. For a predeclared final comparison, also save the unchanged baseline:

```bash
.venv/bin/python -m fev4b.benchmark --model-dir models/fev-4b \
  --data work/split/test.jsonl --out work/baseline-test
```

The baseline and candidate may both read the frozen test in that final comparison; do not use the result to tune and retest the same candidate family. Retain original labels and raw responses, inspect confident mistakes, and report denominators separately by source/type/domain. Do not select among repeated candidates on the final test. A failed generalization check is a reason to retain the baseline.

## Load an explicit overlay

```bash
.venv/bin/python serve.py --model-dir models/fev-4b \
  --adaptation-dir work/calibrated-smoke --port 8000
```

No overlay is discovered or adopted automatically. It must match the exact packaged checkpoint/original adapter. Do not distribute mechanics-only toy overlays as a new Fev release.

## Hugging Face GPU Job

The repository provides `tools/train_hf.py`, a UV script with pinned dependencies. It downloads an immutable model revision and uses a separate source ZIP and four input partitions. Outputs go only to the mounted output directory. It does not upload to a model repository.

Prepare a source ZIP containing `fev/` and `fev4b/` Python files from the same reviewed checkout; exclude credentials, environments and weights. Put it at `cloud-inputs/source.zip` beside the four JSONL files. Record its SHA256 and the data hashes before allocation. Run `hf auth whoami`, inspect `hf jobs hardware` pricing, and create/use a **private** output bucket. Keep a timeout and record the returned job ID:

```bash
hf jobs uv run --flavor a10g-small --timeout 20m --detach \
  --volume ./cloud-inputs:/inputs \
  --volume hf://buckets/YOUR_USER/YOUR_PRIVATE_BUCKET:/outputs:rw \
  tools/train_hf.py --inputs /inputs --output /outputs/UNIQUE_RUN \
  --revision 18ab9ad9f0000345eade9f001f16085ef4a34370 --steps 6
```

Local-directory mounts upload their contents to HF. Use only data approved for that destination. The example performs one training/calibration/test sequence; it does not run a tuning loop. The output path must be new. A job timeout limits runtime, not all account costs; check current pricing first. Inspect logs/results and confirm the job finished before allocating a follow-up. Production inference can remain wholly inside the required region after downloading files and dependencies.
