---
name: autoresearch-hypothesis
description: Generate a set of evidence-grounded module ideas across the pipeline stages — every idea must cite established support and a specific gap in the strong tuned MLP defaults. Use after inspecting the pipeline and before implementing.
---

# Autoresearch: Generate ideas (evidence-first)

Propose a set of module-variant ideas for the modular MLP pipeline.
Assumes you have already internalized the pipeline with `autoresearch-inspect`.

## Start from the right prior

The base is a strong, per-dataset-tuned tabular MLP (PLE numeric embeddings +
plain MLP backbone). "Sounds reasonable" is not a bar; treat every candidate idea as wrong until it earns its place with
evidence.

## Evidence bar

Every idea must state all three of the following. If any one cannot be filled in
honestly, the idea is rejected. Then generate a different one; never lower the bar to hit the requested count.

1. **Mechanism** — what the module changes, in one line.
2. **Support** — named, established backing: a published tabular-DL method or
   ablation (e.g. in the RTDL / FT-Transformer / TabR / RealMLP / TabM line of
   work), or a practice that is standard and replicated across strong tabular
   pipelines. Vague appeals ("often helps in deep learning") do not count. You
   may appeal to the CV and NLP literature if you argue it transfers.
3. **Gap** — why the tuned v0 does not already deliver the effect: a specific
   data regime (heavy-tailed targets, high-cardinality categoricals, tiny n,
   label noise, class imbalance, ...) where v0's fixed choice is suboptimal *and*
   the per-dataset tuner can select the module exactly there.

## Spread ideas across the stages

Aim for coverage across the pipeline stages rather than many ideas for one stage, but a stage where nothing clears the bar gets no ideas. The stages (index 0 = base reference):

- `numerical_preprocess` — numeric feature preprocessing.
- `categorical_preprocess` — categorical feature preprocessing.
- `target_preprocess` (task-dispatched: `_clf` / `_reg`) — target transform.
- `num_embedding` — numeric feature embedding (idx 0 = piecewise-linear PLE).
- `cat_embedding` — categorical feature embedding.
- `model` — the MLP backbone (keep it a single feed-forward net).
- `train` — the training loop.
- `optimizer` — base update rule + weight-decay treatment (idx 0 = AdamW).
- `inference` (task-dispatched: `_clf` / `_reg`) — eval-time output handling.
- `loss` (task-dispatched: `_clf` / `_reg`).

## Rules

- One idea maps to one module in one stage.
- Stay within the model family. Keep every backbone
  variant a single FC net.
- Prefer ideas with bounded parameter, memory, and runtime cost. Avoid designs
  likely to be 10x larger or slower than the all-v0 baseline unless the tradeoff
  is explicit and important.
- Each idea must fit the existing stage interface (mirror the index-0 builder for
  that stage) and tolerate the full `trial_params` config, using only the keys
  it needs.
- A module reads any hyperparameter it needs from the config with an in-code
  default.

## Output

A numbered list of ideas. Each entry names its target stage, the next free
integer index in that stage's registry, and the three evidence-bar fields
(mechanism, support, gap). If fewer than the requested count cleared
the bar, state how many were rejected and why. This list drives the
implement-and-test loop, one idea at a time (`autoresearch-implement` ->
`autoresearch-testing`).
