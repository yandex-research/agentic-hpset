---
name: autoresearch-hypothesis
description: Generate a set of evidence-grounded module ideas across the LightGBM pipeline stages — every idea must cite established support and a specific gap that a tuned GBDT cannot cover. Use after inspecting the pipeline and before implementing.
---

# Autoresearch: Generate ideas (evidence-first)

Propose a set of module-variant ideas for the modular LightGBM pipeline.
Assumes you have already internalized the pipeline with `autoresearch-inspect`.

## Start from the right prior

A tuned LightGBM is a tabular baseline. GBDT's
already handles internally much of what a wrapper might try to add: monotone
feature transforms (trees are invariant to them), missing values, mixed feature
types, and feature interactions. "Sounds
reasonable" is not a bar; treat every candidate idea as wrong until it earns its place with evidence.

## Evidence bar

Every idea must state all three of the following. If any one cannot be filled in
honestly, the idea is rejected — generate a different one; never lower the bar to
hit the requested count.

1. **Mechanism** — what the module changes, in one line.
2. **Support** — named, established backing: a published GBDT/tabular result or
   ablation, a Kaggle-standard technique with a known rationale, or a practice
   replicated across strong tree pipelines. Vague appeals ("feature engineering
   usually helps") do not count.
3. **Gap** — why a tuned GBDT does not already deliver the effect. Name the
   specific regime (high-cardinality categoricals, rare classes, cross-feature
   structure trees split poorly, targets needing a scale/clip the model cannot
   express) where the module helps *and* the tuner can select it there.

## Spread ideas across the stages

Aim for coverage across the three stages rather than many ideas for one — but a
stage where nothing clears the bar gets no ideas. A shorter, stronger list beats
a padded one; if the requested count cannot be met, deliver fewer ideas and say
so. The stages (index 0 = identity):

- `feature` — feature engineering on the design matrix (encodings the trees
  cannot derive, cross-feature constructions, cluster/PCA summaries). Respect the
  `feature_mode` append/replace flag.
- `data_aug` — training-set augmentation (resampling for imbalance, structured
  noise). Applied to train only; bound the row multiplier.
- `postproc` (task-dispatched: regression / classification) — prediction
  post-processing (calibration, clipping to a train range, rounding, prior
  correction).

## Rules

- One idea maps to one module in one stage.
- Stay within the model family: LightGBM stays the learner; modules only reshape
  its inputs, training data, or outputs.
- Prefer bounded column-count, row-count, memory, and runtime cost. Avoid feature
  modules that explode dimensionality or augmentations that multiply rows beyond
  ~10x.
- Each idea must fit the existing stage interface (mirror the identity builder for
  that stage) and read any hyperparameter it needs from its config with an
  in-code default.

## Output

A numbered list of ideas. Each entry names its target stage, the next free
integer index in that stage's registry, and the four evidence-bar fields
(mechanism, support, gap). If fewer than the requested count cleared
the bar, state how many were rejected and why. This list drives the
implement-and-test loop, one idea at a time (`autoresearch-implement` ->
`autoresearch-testing`).
