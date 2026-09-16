---
name: autoresearch-hypothesis
description: Generate a set of evidence-grounded module ideas across the pipeline stages — every idea must cite established support and a specific gap in the meta-tuned RealMLP-TD defaults. Use after inspecting the pipeline and before implementing.
---

# Autoresearch: Generate ideas (evidence-first)

Propose a set of module-variant ideas for the modular RealMLP pipeline.
Assumes you have already internalized the pipeline with `autoresearch-inspect`.

## Start from the right prior

RealMLP-TD is a meta-tuned reference: its defaults were jointly optimized
across a large benchmark suite of tabular datasets. For a model like this, the
correct prior on any plausible-sounding change is that it **may hurt**. The
meta-tuning has already explored and rejected most of the obvious moves. 
"Sounds reasonable" is not a bar; treat every candidate idea
as wrong until it earns its place with evidence.

Naive modules such as optimizer-family swaps (SGD, RMSProp), robust regression
losses, piewise-linear embeddings, are unlikely to work. Even though they sound 
plausible, they override a tuned mechanism (e.g. the per-parameter lr/wd factor 
structure, the per-member target standardization) or fix a problem the benchmarks 
do not have (robust losses on clean data). Ideas from this catalog are not banned, 
but their **Support** and **Gap** fields must engage with that history and explain 
why this time is different.

## Evidence bar

Every idea must state all three of the following. If any one cannot be filled
in honestly, the idea is rejected — generate a different one; never lower the
bar to hit the requested count.

1. **Mechanism** — what the module changes, in one line.
2. **Support** — named, established backing: a published tabular-DL method or
   ablation (e.g., in the RealMLP / TabM / FT-Transformer / TabR line of
   work), or a practice that is standard and replicated across strong tabular
   pipelines. Vague appeals ("often helps in deep learning") do not count. You 
   may, however, appeal to the CV and NLP literature if you think it's applicable.
3. **Gap** — why the meta-tuned v0 does not already deliver the effect: a
   specific data regime (heavy-tailed targets, high-cardinality categoricals,
   tiny n, label noise, class imbalance, ...) where v0's fixed choice is
   suboptimal *and* the per-dataset tuner can select the module exactly there.

## Spread ideas across the stages

Aim for coverage across the pipeline stages rather than many ideas for one
stage — but a stage where nothing clears the bar gets no ideas. A shorter,
stronger list beats a padded one; if the requested count cannot be met, deliver
fewer ideas and say so explicitly.

- `numerical_preprocess` — numeric feature preprocessing (idx 0 = median
  centering + robust IQR scaling + smooth clipping)
- `categorical_preprocess` — categorical feature preprocessing (idx 0 =
  ordinal encoding with an unknown bucket; small-cardinality features become a
  one-hot block, large-cardinality features stay as indices for embeddings)
- `target_preprocess` (task-dispatched: `_clf` / `_reg`) — target
  preprocessing (idx 0 reg = per-member mean/std standardize, min/max recorded
  for clamping; idx 0 clf = ordinal label encoding)
- `num_embedding` — numeric feature embedding (idx 0 = PBLD-style PLR with a
  densenet skip)
- `cat_embedding` — large-cardinality categorical embedding (idx 0 =
  per-member learned embedding tables)
- `model` — backbone (idx 0 = front scale → (BatchedLinear → ParametricMish →
  SmoothDropout) × L → BatchedLinear, NTK-style data-dependent init)
- `train` — training loop (idx 0 = jointly batched per-member loop with
  per-member shuffling, per-step schedule updates, early stopping)
- `optimizer` — base update rule + weight-decay treatment (idx 0 = Adam with
  per-parameter lr/wd factor groups and manual lr-coupled decoupled decay)
- `inference` (task-dispatched: `_clf` / `_reg`) — ensemble aggregation across
  members (idx 0 reg = denormalize + optional clamp → mean; idx 0 clf =
  softmax→mean→log for multiclass, sigmoid→mean→logit for binclass)
- `loss` (task-dispatched: `_clf` / `_reg`) — training loss (idx 0 reg =
  per-member MSE; idx 0 clf = CE with optional scheduled label smoothing)

## Rules

- One idea maps to one module in one stage.
- Stay within the model family: the base is an MLP ensemble with `n_ens` members.
- Prefer ideas with bounded parameter, memory, and runtime cost. Avoid designs
  likely to be 10x larger or slower than the all-v0 baseline unless the
  tradeoff is explicit and important.
- Each idea must fit the existing stage interface (mirror the `v0` builder for
  that stage) and tolerate the full `cfg` (model config) dict, using only the
  keys it needs.
- A module reads any hyperparameter it needs from `cfg` with an in-code
  default.

## Output

A numbered list of ideas. Each entry names its target stage, the next free
integer index in that stage's registry, and the four evidence-bar fields
(mechanism, support, gap). If fewer than the requested count
cleared the bar, state how many were rejected and why. This list drives the
implement-and-test loop, one idea at a time (`autoresearch-implement` ->
`autoresearch-testing`).
