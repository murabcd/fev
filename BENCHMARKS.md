# Fev benchmarks

## Native typed-decision comparison

This comparison sends exactly the same 812 requests (908 questions) to frozen Fev-4B, Kev-4B and Jev-1.14.0. It measures the native typed interfaces, without Fev's two-stage conversation-routing policy. It does not compare generated answers or completed conversations.

- 764 public records from Kev's `transfer-v4` **development** split: 444 Choice, 280 Noul and 40 Score questions. States, instructions, criteria, order and labels are preserved. The suite spans public knowledge, routing, entailment, sentiment and ordinal tasks. The original source metadata and licenses remain with their respective datasets.
- 48 authored Russian records with 144 questions: Choice, Noul and Score on explicit fictional facts across six domains. They repeat eight templates; this is a diagnostic, not independent customer accuracy. These inputs were not used by the new trainer.
- No private customer data appears in the published benchmark artifacts. No upstream-pretraining disjointness claim is made. Results from an already public development suite can inform engineering comparisons but cannot establish an unbiased unseen-task leaderboard.

Inputs SHA256: `9266d6ea1ce83c9755a60cf48c7469b0fa3264c229e96f4142d683bada8fef91`.

| System | Revision and execution |
|---|---|
| Fev-4B | Hub `3ef077cd7cc3c97d696c18150358ab4de3a6884e`; BF16 backbone, FP32 shared adapter unmerged; reference native readout |
| Kev-4B | Hub `6cfce5c2fa4b4bd64026336ab649c5ca78857d52`; official code `2074cc44256200bb37540103fa4572901faa10f3`; BF16 merged, eager SDPA, no fused path/graphs |
| Jev | Hosted API pinned to `jev-1.14.0`; served model recorded per response; server hardware and precision undisclosed |

The two open models use the same A10G GPU class in separate jobs and their own supported runtime/dependency versions. This is a deployment-precision comparison, not Kev's published FP32 evaluation. No temperature or threshold is refitted on these inputs. Fev processes questions jointly; Kev isolates question rows. Those architectural differences remain part of the measured behavior.

Top-option accuracy counts failures as incorrect. Noul is thresholded at 0.5; Score uses the most probable rubric level, with ordinal error reported separately. Brier/NLL/ECE and fixed-threshold coverage use top-option probability, not the API's normalized `confidence` field. Distributions are normalized only when their mass is within 0.02 of one, allowing hosted probability rounding; raw responses remain preserved. A small-sample ECE does not establish calibrated probabilities on new workloads.

### Results (2026-10-24)

| Measure | Fev-4B | Kev-4B | Jev-1.14.0 |
|---|---:|---:|---:|
| Public development (764 questions) | 639/764 (83.6%) | 613/764 (80.2%) | 646/764 (84.6%) |
| Authored Russian (144 questions) | 144/144 (100.0%) | 143/144 (99.3%) | 144/144 (100.0%) |
| Combined (908 questions) | 783/908 (86.2%) | 756/908 (83.3%) | 790/908 (87.0%) |
| Brier, combined (lower is better) | 0.1962 | 0.2305 | 0.1893 |
| NLL, combined (lower is better) | 0.3740 | 0.4291 | 0.8839 |
| ECE10, combined (lower is better) | 0.0421 | 0.0516 | 0.0466 |
| Confident wrong, probability ≥0.9 | 26 | 6 | 36 |
| Failed responses | 0 | 0 | 0 |

Fev is 7/908 answers behind Jev and 27/908 ahead of Kev on this particular combined set. The combined score weights each question equally and includes the easier repetitive Russian probes; the public-only row is the more useful broad comparison. This is not evidence of a statistically significant ranking or an unseen-customer success rate. NLL clips zero probabilities at 1e-12; the hosted API’s rounded zeros affect that metric.

| Public source | Questions | Fev | Kev | Jev |
|---|---:|---:|---:|---:|
| composition_holdout | 96 | 87.5% | 97.9% | 89.6% |
| emotion | 116 | 59.5% | 56.0% | 60.3% |
| legacy_holdout | 80 | 96.2% | 81.2% | 98.8% |
| mmlu | 116 | 74.1% | 68.1% | 83.6% |
| paws | 80 | 85.0% | 80.0% | 78.8% |
| qnli | 80 | 92.5% | 91.2% | 91.2% |
| sciq | 116 | 97.4% | 98.3% | 97.4% |
| tweet_offensive | 80 | 85.0% | 73.8% | 81.2% |

| Typed question, combined | Questions | Fev | Kev | Jev |
|---|---:|---:|---:|---:|
| choice | 492 | 81.3% | 81.3% | 84.1% |
| noul | 328 | 90.9% | 86.6% | 88.1% |
| score | 88 | 96.6% | 81.8% | 98.9% |

Expected-score mean absolute error on the 40 public ordinal questions: Fev 0.1062, Kev 0.3423, Jev 0.0708. This complements the Score top-option metric.

| Fixed threshold ≥0.95, combined | Fev | Kev | Jev |
|---|---:|---:|---:|
| Observed coverage | 62.4% | 30.4% | 69.3% |
| Observed error among accepted questions | 1.2% | 0.7% | 4.3% |

This fixed threshold was not chosen using these results. Coverage is descriptive; it is not a validated future error budget or a recommendation to change routing policy. See [machine-readable results](evaluation/benchmarks/results.json), including all six fixed thresholds and paired outcomes. Raw public/authored predictions are included beside it.

### Reproduce

```bash
.venv/bin/python tools/prepare_benchmark.py --out work/common-inputs
.venv/bin/python -m fev4b.benchmark --model-dir models/fev-4b \
  --data work/common-inputs/comparison.jsonl --out work/fev-common
# Pinned Kev code/dependencies and BF16 model, on a CUDA server:
uv run tools/benchmark_kev.py --data work/common-inputs/comparison.jsonl --out work/kev-common
# Pinned hosted Jev, public/authored inputs only; set TYPESAFE_API_KEY in your shell:
.venv/bin/python -m fev4b.benchmark --endpoint https://api.typesafe.ai \
  --model jev-1.14.0 --api-key-env TYPESAFE_API_KEY \
  --data work/common-inputs/comparison.jsonl --out work/jev-common
# Against a System One server:
.venv/bin/python -m fev4b.benchmark --endpoint http://127.0.0.1:8000 \
  --model fev-4b --data work/common-inputs/comparison.jsonl --out work/http-common
```

Use `--api-key-env NAME` to read credentials from an environment variable; do not put keys in datasets or committed commands. Remote use sends the selected dataset to that endpoint. The data-preparation script downloads only the pinned development file and checks its original and converted hashes. No test split is opened. Preserve predictions and failures rather than rerunning until a favorable result appears. Summarize saved runs without requerying:

```bash
.venv/bin/python -m tools.compare_runs --run Fev=work/fev-common \
  --run Kev=work/kev-common --run Jev=work/jev-common --out work/comparison.json
```

## Conversation routing development evidence

The earlier routing composition had 293/303 reference agreement (96.7%) on consumed synthetic/source-derived development cases: 258 correct automatic routes, four automatic disagreements, 35 correct clarifications and six unnecessary clarifications. Five references had unresolved business mappings. The complete denominator is retained. These numbers are not native typed-task accuracy, production resolution, an independently labelled customer test or comparable to dashboard effectiveness percentages.

## Compiled runtime

The completed earlier regression preserved 303/303 final routing decisions. Internal candidate sets changed on 33 cases, so this is not numerical parity. Three selected routing workloads, five warm repeats each, measured on the same A10G and unchanged BF16/FP32 weights:

| Metric | Reference | Compiled |
|---|---:|---:|
| Warm median | 2,451.97 ms | 1,690.33 ms |
| Warm p95 | 2,539.22 ms | 1,733.43 ms |
| Peak allocated VRAM (decimal GB) | 9.997 | 10.086 |

This is a 31.1% warm median reduction and a 0.9% allocated-memory increase on those workloads. Compilation/load time, HTTP, queueing and concurrent throughput are excluded. It does not predict an H100 benchmark or general production speed. Reference remains the default; CUDA graphs, FP8, pruning and prefetching are disabled.

The packaged implementation additionally passed 6/6 routing controls, all 12 typed winners across four Russian requests, and authenticated TypeSafe SDK calls with plain JSON and zstd. The maximum typed probability change was 0.0016202. See [runtime evidence](evaluation/benchmarks/runtime.json).

## Training validation

The new trainer is checked on a separate, authored four-domain toy corpus. Its purpose is to establish training mechanics and adapter serialization, not improved accuracy. Train/development/calibration/test partitions are frozen before training; the public comparison and Russian diagnostic inputs never enter this trainer. No resulting toy adapter is adopted into Fev's released weights.

The completed smoke run took six steps on 28 records spanning four fictional domains. All 240 shared-adapter tensors changed (10,387,456 FP32 parameters), with finite gradients and the backbone/readout frozen. Development stayed 12/12 correct before and after; no accuracy improvement is established.

The first post-training reload assertion failed: maximum probability difference 0.00048164 against a 0.00001 tolerance, after compiled and HTTP work in that process. Its cause was not isolated. A fresh-process follow-up then verified all 240 saved tensors bit-for-bit, zero probability difference across repeated calls and against the saved post-training predictions, and all 12 option winners unchanged. Both outcomes are retained.

Separate calibration and final toy evaluation completed with 12/12 valid, correct test predictions. The temperature fit selected its lower search bound on this tiny all-correct sample; it is a machinery check, not useful evidence of production calibration. Toy templates recur across partitions, so this is not cross-domain generalization evidence. The toy adapter and fitted temperature are not adopted or distributed. See [training validation evidence](evaluation/benchmarks/training-validation.json).
