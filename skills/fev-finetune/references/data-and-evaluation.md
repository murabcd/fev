# Data and evaluation details

## Native JSONL

Each line is one state plus its typed questions. Metadata and labels are removed before inference. Option insertion order is preserved by the skill's writer; do not sort criteria while serializing a frozen experiment.

```json
{"id":"library-001","domain":"library","group":"library-cancel-family-a","source":"authored","state":"Please cancel my reservation.","questions":{"action":{"type":"choice","criteria":{"cancel":"Cancel a reservation","change":"Change a reservation","info":"Ask for information"},"label":"cancel"},"cancel_requested":{"type":"noul","instructions":"Is cancellation explicitly requested?","label":true},"urgency":{"type":"score","criteria":["No deadline","Deadline stated","Immediate action requested"],"label":0}}}
```

`domain` is an application domain, not a train/test marker. `source` distinguishes observed, authored or generated data; use names specific enough to audit provenance. `group` must join all records derived from a document, template or paraphrase family, including across domains. Exact repeated states also join groups transitively. IDs alone do not prevent leakage.

## CSV conversion

Use CSV columns `id,domain,group,text` and one label column for each question. Copy [questions.example.json](../assets/questions.example.json), adjust the runtime definitions, then map columns explicitly:

```bash
python3 "$SKILL/scripts/prepare_data.py" --checkout work/fev \
  --data labelled.csv --spec questions.json \
  --label action=action_label --label cancel_requested=cancel_label \
  --label urgency=urgency_label --state-field text \
  --holdout-domain travel --out work/split
```

CSV Noul values must be literal `true` or `false`. Score values are integer indices, starting at zero. Choice values must exactly match an option key. All questions in the spec require mappings. For nested state objects, different questions per record, or source metadata, use native JSONL instead of CSV.

The splitter uses approximately 70/10/10/10 percent **by independent family count**, with at least one family in each fold. An explicitly held-out domain is appended to test; it is absent from all other folds. A family bridge into another fold is rejected. Counts and hashes appear in `manifest.json`. Do not shop seeds to improve the final score or repair a sparse corpus by duplicating rows. Add genuinely independent families when there are too few groups or training domains.

## Creating synthetic examples

Start from the specific weakness. Vary domain, rule phrasing, state length, distractors, option ordering and negation independently. Include minimal counterfactual pairs where the relevant fact changes, out-of-scope requests, ambiguous states and conflicting evidence. Keep each family in one fold. Distinct surface wording does not imply independent examples.

Generate from fictional facts and a written rule with a checkable expected answer. Save generator/source identity in separate provenance and keep generated labels explicitly marked. Review a diverse sample manually; reject definitions with no unambiguous answer. Reserve entire templates and domains before generation. Do not ask the evaluated model to label its own test, copy published benchmarks into training, or call generator agreement real-world accuracy.

This skill deliberately does not select an external generator, send data, or promise a sample size from a short pilot. Those choices require the task's data permissions, resources and observed failure frequency.

## Reading the paired comparison

The helper validates frozen test hashes, exact prediction IDs, matching base-model manifests, execution mode and requested model. It re-scores raw predictions rather than trusting cached metrics. Failed predictions remain in the question denominator. Brier/NLL/ECE use answered questions; report the answered count. The error file contains IDs and probability-level diagnostics, not input text, but can still be sensitive.

Accuracy uncertainty uses a seeded paired bootstrap over connected exact-state/family clusters. All questions and variants in a sampled cluster move together. With fewer than two clusters the interval is absent; small or repetitive sets remain weak evidence even when a numeric interval exists. This is not a significance certificate. Inspect per-domain and per-type regressions, ordinal error for Score questions, applicability/routing behavior and confident errors against predeclared gates. An uncertain result is a reason to retain the baseline, not run an automatic search.

For cost reporting, supply all three inputs from a current provider quote and the actual billed allocation duration:

```bash
python3 "$SKILL/scripts/compare.py" --checkout work/fev \
  --data work/split/test.jsonl --baseline work/run/baseline-test \
  --candidate work/run/test --out work/comparison-with-cost \
  --hourly-rate-usd "$HOURLY_RATE" --billed-seconds "$BILLED_SECONDS" \
  --rate-source "$QUOTE_URL_AND_DATE"
```

The result is an explicitly labelled compute estimate. Download time, minimum billing, storage and other charges may change the invoice. Sequential benchmark latency does not establish throughput, monthly serving cost or CPU feasibility.
