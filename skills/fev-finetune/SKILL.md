---
name: fev-finetune
description: Prepare data, train and evaluate a shared Fev-4B decision adapter with native Choice/Noul/Score labels, grouped splits, calibration, paired comparisons and bounded GPU jobs. Use for evidence-led Fev adaptation, not runtime category configuration.
license: Apache-2.0
metadata:
  author: murabcd
  version: "2.0"
  repository: https://github.com/murabcd/fev
---

# Fev fine-tuning

Improve one shared **Fev-4B** model across domains. Category definitions, business rules, examples and conversation history belong in runtime requests. A new category alone is not a reason to train. Fev scores typed questions; it does not generate chat completions. Continue its existing rank-8 adapter with native decision cross-entropy, keeping the backbone and decision readout frozen. A generic causal-LM SFT recipe is incompatible.

## Bundled tools

All helpers run with Python 3.12 and the standard library. They do not install CUDA, call a model, upload data or allocate compute. Resolve `SKILL` to this installed skill's directory; run commands from a separate working directory.

| Helper | Purpose |
|---|---|
| `scripts/bootstrap.py` | Clone an explicit, reviewed Git commit without downloading model weights |
| `scripts/prepare_data.py` | Validate JSONL or convert CSV, preserve option order, freeze grouped splits and hashes |
| `scripts/package_job.py` | Check all four splits and build minimal, inspectable HF Job inputs |
| `scripts/compare.py` | Re-score saved paired predictions, report domain regressions and family-cluster uncertainty |

Read [training.md](references/training.md) when executing GPU training, calibration or HF Jobs. Read [data-and-evaluation.md](references/data-and-evaluation.md) for CSV mapping, synthetic data, held-out domains and report interpretation.

## 1. Establish the experiment

Inspect existing failures before running anything. Write one hypothesis, the excluded diagnostic examples, the proposed shared change, and the pass/fail rule. State the 4B deployment constraint, hardware, maximum runtime and compute budget. Choose steps/learning rate using development data, not final test results.

If the weakness is ambiguous labels, missing rules or a routing integration bug, repair that first. Do not train an adapter to memorize one organization's routes. If there is no reproducible model weakness, report that training is unwarranted.

## 2. Pin source and model

Review the selected source commit and retain its full 40-character SHA. Do not silently follow `main` during a run.

```bash
python3 "$SKILL/scripts/bootstrap.py" --revision "$SOURCE_REVISION" --out work/fev
```

The checkout must include the paired experiment worker. Pin a separate immutable `murabcd/fev-4b` Hub revision; source and weights have different SHAs. Verify the complete model package before use. An 8 GB Mac can run the helpers and contract checks; GPU training needs a separate Linux environment.

## 3. Prepare shared data

Use native labelled JSONL, or the CSV mapping described in the data reference. Every record needs unique `id`, `domain`, and meaningful `group` identifying its source/template/paraphrase family. Choice labels are option keys, Noul labels are booleans, Score labels are zero-based rubric indices.

```bash
python3 "$SKILL/scripts/prepare_data.py" --checkout work/fev \
  --data labelled.jsonl --out work/split --seed 42
```

Inspect `manifest.json` and representative records in every partition. The helper joins exact states and supplied family groups; it cannot detect semantic duplicates. Training must retain at least two domains. For transfer evidence, add `--holdout-domain DOMAIN` to exclude an entire domain from training, development and calibration. Never split variants of the same generated family across folds.

Synthetic examples can establish mechanics and expose a narrow weakness. They do not become authoritative labels merely because another model generated them. Keep authored, generated and observed sources distinct, audit ambiguous labels, and keep published benchmark inputs out of training.

## 4. Package before allocating

```bash
python3 "$SKILL/scripts/package_job.py" --checkout work/fev \
  --split work/split --out work/cloud-inputs
```

Review `package.json`: exact source/data hashes, source commit and dirty-source flag. Use a clean reviewed checkout for an experiment. The bundle contains Python source, four data files, the worker and comparison helpers; it excludes environments, credentials and model weights.

Confirm authorization for the data destination and paid run. Check current hardware prices and existing jobs. Save allocation intent, cap, stopping rule and the returned job ID; reconcile an uncertain submission before retrying. Installation of this skill authorizes none of these actions.

## 5. Train, calibrate and compare once

Follow the GPU commands in the training reference. The worker preserves the downloaded base, trains an explicit overlay, then fits temperature on calibration data only. It scores the unchanged baseline, raw adapter and calibrated adapter on the same frozen final test. It writes both paired comparisons automatically. Six default steps are a mechanics smoke check, not a training recommendation.

For existing saved runs, compare without another model call:

```bash
python3 "$SKILL/scripts/compare.py" --checkout work/fev \
  --data work/split/test.jsonl --baseline work/run/baseline-test \
  --candidate work/run/test --out work/comparison
```

Read `summary.md`, `comparison.json` and `candidate-errors.jsonl`. Check all-question accuracy, failed responses, Brier/NLL, confident mistakes, domain regressions and the paired interval together. Separate raw weight changes from calibration effects. Temperature can change threshold routing even when per-question ranking is unchanged. Final-test findings are for the adoption decision, not another tuning loop on the same test.

## 6. Decide and close the run

Report the baseline, raw and calibrated candidates; sample and independent-family counts; known label limitations; regressions; measured resources; and estimated versus billed cost. Retain the released baseline if the predeclared improvement or regression gates fail. A small interval or a synthetic perfect score does not establish unseen-domain quality.

Keep failed logs and artifacts. Inspect the recorded job's final status, stop only resources owned by this run, and verify billing has stopped. Do not delete shared volumes or unrelated jobs. Leave outputs private until the user chooses a release. Never automatically adopt an overlay, publish weights, deploy a server or alter a live workflow.
