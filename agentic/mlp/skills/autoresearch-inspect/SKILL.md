---
name: autoresearch-inspect
description: Inspect and internalize the modular MLP pipeline — its stages, registries, and existing module variants — before generating or implementing any ideas. Use this first, at the start of an autoresearch loop.
---

# Autoresearch: Inspect the pipeline

Build an accurate mental model of the modular MLP pipeline before proposing or
changing anything. Everything runs in the current repository — no worktrees, and
no real dataset paths, names, or feature descriptions anywhere.

The base is a strong, per-dataset-tuned tabular MLP: piecewise-linear (PLE)
numeric feature embeddings and learned categorical embeddings feeding a plain
multi-layer perceptron backbone with dropout, trained with AdamW. Its idx-0
choices reflect competitive tabular-DL defaults; while inspecting, work out for
each stage *why* v0 does what it does — a proposal that ignores that reasoning
usually rediscovers something strong defaults already handle.

## Read

- `bin/mlp/pipeline.py` — the module docstring and `train_and_eval` show how the
  stages are assembled into one train+eval run and how the sampled
  `trial_params` (hyperparameters) and `implementation_indices` flow into every
  stage.
- `bin/mlp/hparams.py` — the registries: `IMPLEMENTATION_REGISTRIES`,
  `INDEX_KEYS`, `resolve()`, and how the task-dispatched stages split into
  `_reg` / `_clf`. It also holds the search space (`sample_hps`) and per-dataset
  defaults.
- The index-0 implementation and docstring for each stage under
  `bin/mlp/modules/<stage>/` — index 0 and the interface a new variant must
  mirror. Each stage's index-0 file states the exact builder signature and
  returned-object contract.

## List the live registries

```bash
uv run python agentic/mlp/skills/autoresearch-inspect/references/inspect_pipeline.py
```

This prints the `--<stage>-idx` flags and, per task type, the registered indices
already available for each stage.

## The stages (`INDEX_KEYS`)

Each stage is a registry `{index: builder}`; index 0 is the base reference
implementation for that stage.

- `numerical_preprocess` — numeric feature preprocessing (scaling / transforms
  applied before embedding).
- `categorical_preprocess` — categorical feature preprocessing (encoding, rare-
  category handling).
- `target_preprocess` (task-dispatched: `_clf` / `_reg`) — target
  transformation (reg: standardization / warping; clf: label encoding).
- `num_embedding` — numeric feature embedding (idx 0 = piecewise-linear PLE).
- `cat_embedding` — categorical feature embedding (idx 0 = learned embedding
  tables / one-hot).
- `model` — the MLP backbone (idx 0 = plain MLP blocks with dropout).
- `train` — the training loop (idx 0 = standard mini-batch loop with early
  stopping).
- `optimizer` — base update rule + weight-decay treatment (idx 0 = AdamW).
- `inference` (task-dispatched: `_clf` / `_reg`) — prediction / output
  post-processing at eval time.
- `loss` (task-dispatched: `_clf` / `_reg`) — training loss (idx 0 reg = MSE;
  idx 0 clf = cross-entropy).

## Output

A short internalized map: for each stage, what `v0` does and why it is set up
that way, which indices already exist, and where a new variant would plug in.
This grounds the idea-generation step (`autoresearch-hypothesis`).
