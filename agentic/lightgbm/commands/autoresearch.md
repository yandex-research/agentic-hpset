---
name: autoresearch
description: Orchestrate the autoresearch loop — inspect the pipeline, generate evidence-grounded ideas, implement and test each module, then run the exhaustive check.
argument-hint: "--num-hypotheses=<n>"
---

Run all work in the current repository. Do not use isolated worktrees. Do not use
real dataset paths, real dataset names, or real feature descriptions in commands,
logs, docs, ideas, or summaries.

## Required input

- `--num-hypotheses` — the maximum number of candidate hypotheses to generate and
  implement. Treat it as a ceiling, not a quota: an idea only enters the set if it
  clears the evidence bar in `autoresearch-hypothesis`, and delivering fewer,
  stronger modules is the correct outcome when the bar cannot be met.

If the field is missing, use `AskUserQuestion` before editing code:

```text
AskUserQuestion: How many hypotheses should I generate and implement?
```

## Constraints

- Every module variant keeps LightGBM as the learner; do not swap in a different model. The modular pipeline wraps that model in three stages: feature engineering, data augmentation, and prediction post-processing. 
- Efficiency guardrail: avoid modules that are 10x larger, slower, or more memory hungry than the all-v0 baseline unless the tradeoff is explicit and justified. Tree models are cheap — a feature-engineering module that explodes column count or an augmentation that multiplies rows can blow that budget.
- Evidence bar: GBDTs already handle monotone transforms, missing values, mixed feature types internally. Do not implement ideas without named, established support and a specific gap that a tuned GBDT cannot cover (see `autoresearch-hypothesis`).

## Research loop

Drive the loop by invoking these skills in order:

1. **Inspect** — invoke `autoresearch-inspect` to read the pipeline and list the live registries, internalizing the stages, the existing modules, and *why* the tuned v0 (identity) choices look the way they do.
2. **Generate ideas** — invoke `autoresearch-hypothesis` to produce up to `--num-hypotheses` ideas spread across the pipeline stages, each clearing the evidence bar (mechanism, support, gap). Ideas that cannot clear the bar are not added.
3. **Implement and test, one idea at a time** — for each idea in the set:
   1. invoke `autoresearch-implement` to add and register the module;
   2. invoke `autoresearch-testing` to run the targeted test for that module's index;
   3. apply the keep/drop policy: if the module fails to run, is non-finite, or is ~10x worse than baseline, debug it (up to 3 attempts); if it still does not satisfy the requirements, drop it and move on to the next idea.
4. **Exhaustive check** — once all kept modules are implemented, invoke `autoresearch-testing` for the exhaustive interaction sweep (`uv run bin/lightgbm/test.py --exhaustive`) and review the combined efficiency output.
