---
name: autoresearch-inspect
description: Inspect and internalize the modular TabM pipeline — its stages, registries, and existing module variants — before generating or implementing any ideas. Use this first, at the start of an autoresearch loop.
---

# Autoresearch: Inspect the pipeline

Build an accurate mental model of the modular TabM pipeline before proposing or
changing anything. Everything runs in the current repository — no worktrees, and
no real dataset paths, names, or feature descriptions anywhere.

The base is TabM: a parameter-efficient deep ensemble of `k=32` MLPs that share
most weights (BatchEnsemble-style), with piecewise-linear (PLE) numeric
embeddings and learned categorical embeddings, trained jointly and averaged at
inference. Its idx-0 choices reflect competitive tabular-DL defaults; while
inspecting, work out for each stage *why* v0 does what it does. A proposal that
ignores that reasoning usually rediscovers something strong defaults already
handle. Note the shared-weight ensemble: modules must respect the `k`-member
batched structure.

## Read

- `bin/tabm/pipeline.py` — the module docstring and `train_and_eval` show how the
  stages are assembled into one train+eval run and how the sampled
  `trial_params` (hyperparameters, incl. the ensemble size `k`) and
  `implementation_indices` flow into every stage.
- `bin/tabm/hparams.py` — the registries: `IMPLEMENTATION_REGISTRIES`,
  `INDEX_KEYS`, `resolve()`, and how the task-dispatched stages split into
  `_reg` / `_clf`. It also holds the search space (`sample_hps`) and per-dataset
  defaults.
- The index-0 implementation and docstring for each stage under
  `bin/tabm/modules/<stage>/` — index 0 and the interface a new variant must
  mirror. Each stage's index-0 file states the exact builder signature and
  returned-object contract. `bin/tabm/modules/_shared.py` holds ensemble helpers.

## List the live registries

```bash
uv run python agentic/tabm/skills/autoresearch-inspect/references/inspect_pipeline.py
```

This prints the `--<stage>-idx` flags and, per task type, the registered indices
already available for each stage.

## The stages (`INDEX_KEYS`)

Each stage is a registry `{index: builder}`; index 0 is the base reference
implementation for that stage.

- `numerical_preprocess` — numeric feature preprocessing (scaling / transforms).
- `categorical_preprocess` — categorical feature preprocessing.
- `target_preprocess` (task-dispatched: `_clf` / `_reg`) — target transform.
- `num_embedding` — numeric feature embedding (idx 0 = piecewise-linear PLE).
- `cat_embedding` — categorical feature embedding.
- `model` — the TabM backbone (idx 0 = shared-weight `k`-member ensemble MLP).
- `train` — the training loop (idx 0 = jointly batched per-member loop, early
  stopping).
- `optimizer` — base update rule + weight-decay treatment (idx 0 = AdamW).
- `inference` (task-dispatched: `_clf` / `_reg`) — ensemble aggregation across
  the `k` members at eval time.
- `loss` (task-dispatched: `_clf` / `_reg`) — training loss (idx 0 reg = MSE;
  idx 0 clf = cross-entropy).

## Output

A short internalized map: for each stage, what `v0` does and why it is set up
that way, which indices already exist, and where a new variant would plug in.
This grounds the idea-generation step (`autoresearch-hypothesis`).
