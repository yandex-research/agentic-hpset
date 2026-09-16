---
name: autoresearch-inspect
description: Inspect and internalize the modular LightGBM pipeline — its stages, registries, and existing module variants — before generating or implementing any ideas. Use this first, at the start of an autoresearch loop.
---

# Autoresearch: Inspect the pipeline

Build an accurate mental model of the modular LightGBM pipeline before proposing
or changing anything. Everything runs in the current repository — no worktrees,
and no real dataset paths, names, or feature descriptions anywhere.

The base is a strong, per-dataset-tuned LightGBM (gradient-boosted decision
trees). The three module stages wrap that learner; index 0 of every stage is the
**identity** transform, so the all-v0 pipeline is plain tuned LightGBM.

## Read

- `bin/lightgbm/pipeline.py` — how a run is assembled: it applies `feature` →
  `data_aug` → fit LightGBM → `postproc`. Note `REGISTRY_KEY_MAP`,
  `get_registries(task_type)`, `sample_implementation_indices`, and the
  `feature_mode` (append vs replace) flag.
- `bin/lightgbm/model.py` — the LightGBM fit/predict, `LIGHTGBM_FIXED_PARAMS`,
  and `suggest_lightgbm_params` (the boosting hyperparameter search).
- `bin/lightgbm/hparams.py` — the single entry point: `sample_hps`, `resolve`,
  the sampler, and the trial budget.
- The index-0 (identity) file and docstring for each stage under
  `bin/lightgbm/modules/<stage>/` — the interface a new variant must mirror.

## List the live registries

```bash
uv run python agentic/lightgbm/skills/autoresearch-inspect/references/inspect_pipeline.py
```

This prints the `--<stage>-idx` flags and, per task type
(`regression` / `binclass` / `multiclass`), the registered indices already
available for each stage.

## The stages

Each stage is a registry `{index: builder}`; index 0 is the identity transform.

- `feature` (`feature_idx`) — feature engineering applied to the design matrix
  before fitting (idx 0 = identity). A `feature_mode` flag controls whether new
  columns are appended or replace the originals.
- `data_aug` (`data_aug_idx`) — training-set augmentation (idx 0 = identity):
  resampling, noise, mixup-style row synthesis, etc. Applied to train only.
- `postproc` (`postproc_idx`, task-dispatched: regression / classification) —
  post-processing of predictions (idx 0 = identity): clipping, calibration,
  rounding, prior correction, etc.

Boosting hyperparameters (learning rate, num_leaves, feature/bagging fractions,
L2, ...) are searched separately by `suggest_lightgbm_params`; they are not a
module stage.

## Output

A short internalized map: for each stage, what the identity default does, which indices already exist, and where a new variant would
plug in. This grounds the idea-generation step (`autoresearch-hypothesis`).
