---
name: autoresearch-hypothesis
description: Generate a set of evidence-grounded recipe-module ideas across the TabICL evaluation axes — every idea must cite established support and a specific gap the default recipe cannot cover. Use after inspecting the pipeline and before implementing.
---

# Autoresearch: Generate ideas (evidence-first)

Propose a set of recipe-module ideas for the modular TabICL pipeline.
Assumes you have already internalized the recipe with `autoresearch-inspect`.

## Start from the right prior

The model is **frozen and pretrained**. You are only changing the evaluation
recipe around it. The default recipe was chosen to make that pretrained model
work well out of the box. "Sounds reasonable" is not a bar; treat every
candidate idea as wrong until it earns its place with evidence.

## Evidence bar

Every idea must state all three of the following. If any one cannot be filled in
honestly, the idea is rejected — generate a different one; never lower the bar to
hit the requested count.

1. **Mechanism** — what the recipe module changes, in one line.
2. **Support** — named, established backing: a published in-context-learning /
   TabPFN / TabICL result or ablation, or a standard, replicated ensembling /
   preprocessing practice with a known rationale. Vague appeals do not count.
3. **Gap** — why the default recipe does not already deliver the effect, given a
   frozen model pretrained on a particular input distribution. Name the specific
   regime (many features and permutation sensitivity, heavy-tailed numerics,
   context/query distribution shift, ...) where the module helps *and* the tuner
   can select it there.

## Spread ideas across the axes

Aim for coverage across the recipe axes rather than many ideas for one — but an
axis where nothing clears the bar gets no ideas. A shorter, stronger list beats a
padded one. The axes (index 0 = default; some are baskets of per-member choices):

- `feature_preproc` (basket) — per-member numeric feature preprocessing.
- `feature_permutation` — feature-order permutation across members (TabICL is
  not permutation-invariant, so this drives ensemble diversity).
- `augmentation` (basket) — per-member input augmentations.
- `ensemble` — how member predictions are aggregated (task-dependent).
- `postproc` (basket, task-dispatched) — output post-processing.

A new member-level component may also need a **basket** that includes it; note
that in the idea.

## Rules

- One idea maps to one recipe module (and, if needed, one basket that uses it).
- Stay within the model family: the TabICL model stays frozen; modules only
  change the recipe around it — never train or fine-tune weights.
- Avoid recipes >10x the default cost.
- Each idea must fit the existing axis interface (mirror the index-0 component)
  and read any hyperparameter it needs from its config with an in-code default;
  the search over recipe indices lives in `bin/tabicl/hparams.py`.

## Output

A numbered list of ideas. Each entry names its target axis, the next free integer
index (and any new basket), and the four evidence-bar fields (mechanism, support,
gap, harm case). If fewer than the requested count cleared the bar, state how
many were rejected and why. This list drives the implement-and-test loop, one
idea at a time (`autoresearch-implement` -> `autoresearch-testing`).
