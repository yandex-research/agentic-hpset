---
name: autoresearch-inspect
description: Inspect and internalize the modular TabICL (TabICL) evaluation recipe — its axes, registries, baskets, and existing variants — before generating or implementing any ideas. Use this first, at the start of an autoresearch loop.
---

# Autoresearch: Inspect the pipeline

Build an accurate mental model of the modular TabICL evaluation recipe before
proposing or changing anything. Everything runs in the current repository — no
worktrees, and no real dataset paths, names, or feature descriptions anywhere.

The base is TabICL, a **pretrained** in-context-learning tabular transformer:
the frozen model predicts a query set from a context set, with no gradient
training. The modular pipeline is the *evaluation recipe* wrapped around it. Index
0 of every axis is the default recipe component; while inspecting, work out why
each default is set the way it is — the recipe was tuned to make the pretrained
model work out of the box, and a change that ignores that usually rediscovers a
rejected option.

## Read

- `bin/tabicl/pipeline.py` — how a run is assembled: `implementation_choices`
  (the recipe axes and their choices), `resolve_evaluation_spec` (basket indices
  → the concrete `EvaluationSpec`), `build_evaluator_from_indices`, and the
  `TabICLEvaluator`. Note which axes are **baskets** (a basket index expands to a
  tuple of concrete member indices).
- `bin/tabicl/hparams.py` — `sample_hps`, the sampler, and `total_grid_size`.
- `bin/tabicl/modules/TabICL/model.py` — the frozen model (`model_idx` is
  always 0).
- The index-0 file and docstring for each recipe axis under
  `bin/tabicl/modules/<axis>/` — the interface a new variant must mirror.

## List the live recipe axes

```bash
uv run python agentic/tabicl/skills/autoresearch-inspect/references/inspect_pipeline.py
```

This prints, per task type (`regression` / `binclass` / `multiclass`), the recipe
axes and the indices already available for each, plus the total grid size.

## The recipe axes

Index 0 of every axis is the default recipe component.

- `model_idx` — the frozen model (always `0` = TabICL).
- `feature_preproc_basket_idx` — a basket of per-member feature preprocessings
  (identity, power, quantile-normal, ...); different members can use different
  preprocessings.
- `feature_permutation_idx` — how feature order is permuted across members
  (none / random / shift / latin-square) — TabICL is not permutation-invariant,
  so this is a real ensembling axis.
- `augmentation_basket_idx` — a basket of input augmentations applied per member
  (identity, feature noise, ...).
- `ensemble_idx` — how member predictions are aggregated (mean, median,
  geometric mean, uncertainty-weighted, ...); task-dependent.
- `postproc_basket_idx` — a basket of output post-processings (task-dispatched:
  regression standardize/log1p, classification proba handling).

## Output

A short internalized map: for each axis, what the default component does and why,
which indices/baskets already exist, and where a new variant (and, if needed, a
basket that uses it) would plug in. This grounds `autoresearch-hypothesis`.
