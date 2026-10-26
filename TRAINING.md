# Fine-tuning Fev

The complete [training guide](skills/fev-finetune/references/training.md) covers labelled typed decisions, grouped splits, the shared adapter trainer, separate temperature calibration, evaluation and bounded Hugging Face GPU jobs.

Install the companion agent skill:

```bash
npx skills add murabcd/fev@fev-finetune
```

It teaches agents to use Fev's native decision training workflow, preserve the base package, separate evaluation data and keep application rules in runtime inputs. Installing a skill does not train or publish a model.

The skill bundles CPU-only helpers for pinned setup, CSV/JSONL preparation, private job packaging and paired reports. Start with its [workflow](skills/fev-finetune/SKILL.md); see [data and evaluation details](skills/fev-finetune/references/data-and-evaluation.md) for the schema and report interpretation.
