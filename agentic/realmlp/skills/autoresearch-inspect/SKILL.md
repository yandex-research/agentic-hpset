---
name: autoresearch-inspect
description: Inspect and internalize the modular RealMLP pipeline — its stages, registries, and existing module variants — before generating or implementing any ideas. Use this first, at the start of an autoresearch loop.
---

# Autoresearch: Inspect the pipeline

Build an accurate mental model of the modular RealMLP pipeline before proposing
or changing anything. Everything runs in the current repository — no worktrees,
and no real dataset paths, names, or feature descriptions anywhere.

The base is RealMLP-TD, a **meta-tuned** model: robust scaling with smooth clipping, PLR numeric embeddings, parametric Mish,
per-parameter lr/wd factor groups, the `coslog4` schedule, scheduled label
smoothing, output clamping — were jointly optimized across a large benchmark
suite. While inspecting, work out for each stage *why* v0 does what it does.

## Read

- `bin/realmlp/assemble.py` — the module docstring and `main`
  show how the stages are assembled into one train+eval run and how the shared
  `cfg` (model config) dict flows into every stage.
- `bin/realmlp/modules/__init__.py` — the registry:
  `STAGE_KEYS`, `IMPLEMENTATION_REGISTRIES`, `resolve()`, and how the three
  task-dispatched stages split into `_clf` / `_reg`.
- The `v0` implementation and docstring for each stage under
  `bin/realmlp/modules/<stage>/realmlp.py` — index 0 and the
  interface a new variant must mirror. Each `v0` file states the exact builder
  signature and returned-object contract for that stage.

## List the live registries

```bash
uv run python agentic/realmlp/skills/autoresearch-inspect/references/inspect_pipeline.py
```

This prints the `--<stage>-idx` flags and, per task type, the registered indices
already available for each stage.

## The stages (`STAGE_KEYS`)

- `numerical_preprocess` — numeric feature preprocessing (idx 0 = median
  centering + robust IQR scaling + smooth clipping)
- `categorical_preprocess` — categorical feature preprocessing (idx 0 =
  ordinal encoding with an unknown bucket; small-cardinality features become a
  one-hot block, large-cardinality features stay as int indices for learned
  embeddings)
- `target_preprocess` (task-dispatched: `_clf` / `_reg`) — target
  preprocessing (idx 0 reg = per-member mean/std standardize, with min/max
  recorded for inference-time clamping; idx 0 clf = ordinal label encoding)
- `num_embedding` — numeric feature embedding (idx 0 = PBLD-style PLR with a
  densenet skip)
- `cat_embedding` — large-cardinality categorical embedding (idx 0 =
  per-member learned embedding tables)
- `model` — backbone (idx 0 = front scale -> (BatchedLinear -> ParametricMish ->
  SmoothDropout) × L -> BatchedLinear, with NTK-style data-dependent init)
- `train` — training loop (idx 0 = jointly batched per-member loop with
  per-member shuffling, per-step schedule updates via the optimizer's
  `update_fn`, early stopping)
- `optimizer` — base update rule + weight-decay treatment (idx 0 = Adam with
  per-parameter lr/wd factor groups and manual lr-coupled decoupled weight
  decay applied before `optimizer.step()`)
- `inference` (task-dispatched: `_clf` / `_reg`) — ensemble aggregation across
  members (idx 0 reg = denormalize + optional clamp -> mean; idx 0 clf =
  softmax->mean->log for multiclass, sigmoid->mean->logit for binclass)
- `loss` (task-dispatched: `_clf` / `_reg`) — training loss (idx 0 reg =
  per-member MSE; idx 0 clf = CE with optional scheduled label smoothing)

## Output

A short internalized map: for each stage, what `v0` does and why it is set up
that way, which indices already exist, and where a new variant would plug in.
This grounds the idea-generation step (`autoresearch-hypothesis`).
