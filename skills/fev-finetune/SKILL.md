---
name: fev-finetune
description: Train and evaluate a shared Fev-4B decision adapter, including labelled Choice/Noul/Score data, leakage-aware splits, temperature calibration, and bounded GPU jobs. Use for Fev training or adaptation, rather than configuring customer routing categories.
---

# Fev fine-tuning

Fev scores typed questions; it does not generate chat completions. Train its existing shared rank-8 adapter with native decision cross-entropy. Do not use a generic causal-LM SFT recipe or replace the decision head.

Read [the training guide](references/training.md) for the data contract and exact commands. Work from a pinned checkout of `https://github.com/murabcd/fev` and a complete, verified `murabcd/fev-4b` package. The checkout must contain `fev4b/train.py`; older releases expose inference only.

Before allocating compute:

1. Identify a reproducible weakness on examples excluded from training. Changing a customer's categories or rules normally requires runtime configuration, not training.
2. Agree on one bounded experiment, hardware/time cap and a pass/fail rule. Check current provider pricing and pending jobs; persist allocation intent and job ID. Reconcile an uncertain submission before retrying.
3. Use shared data spanning multiple domains. Group paraphrases, source documents and generated families together; keep train, development, calibration and test separate. The splitter catches exact-state/group overlap, not semantic duplicates. Inspect synthetic labels and describe their limitations.
4. Run the released baseline with the same questions, precision and routing policy. Keep test data out of training, tuning and calibration. Never send customer data to an external service merely because this skill is installed.

Training writes an explicit overlay directory; keep the downloaded base immutable. Verify finite gradients, adapter changes, frozen backbone/readout, save/reload agreement and complete predictions. Fit temperature on calibration data only. Inspect accuracy, Brier/NLL, confident mistakes and coverage together. Temperature can change threshold-based routing even when option argmax stays the same.

Use an untouched cross-domain test for adoption. A toy training smoke check establishes mechanics, not generalization or improved accuracy. Do not automatically publish, deploy, modify customer workflows or replace released weights. Report the evidence and obtain the user's requested release decision.
