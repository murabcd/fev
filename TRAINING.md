# Fine-tuning Fev

The complete [training guide](skills/fev-finetune/references/training.md) covers labelled typed decisions, grouped splits, the shared adapter trainer, separate temperature calibration, evaluation and bounded Hugging Face GPU jobs.

Install the companion agent skill after this revision is published:

```bash
npx skills add murabcd/fev@fev-finetune
```

It teaches agents to use Fev's native decision training workflow, preserve the base package, separate evaluation data and avoid per-customer retraining. Installing a skill does not train or publish a model.
