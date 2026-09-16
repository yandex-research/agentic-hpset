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

- Stay within the model family: the base learner is TabICL, a pretrained in-context-learning tabular transformer. There is **no gradient training**: the frozen model predicts a query set given a context set. A "run" is an *evaluation recipe* around the frozen model: how features are preprocessed and permuted, how ensemble members are formed, and how outputs are aggregated and post-processed. Every module variant keeps the frozen TabICL model as the learner. The weights are never trained.
- Efficiency guardrail: avoid recipes that are 10x more members, passes, or memory than the all-v0 recipe unless the tradeoff is explicit and justified.
- Evidence bar: no idea is implemented without named, established support and a specific gap that the default recipe cannot cover (see `autoresearch-hypothesis`).

## Research loop

Drive the loop by invoking these skills in order:

1. **Inspect** — invoke `autoresearch-inspect` to read the pipeline and list the live recipe axes, internalizing the stages, the existing modules and baskets, and *why* the v0 recipe choices look the way they do.
2. **Generate ideas** — invoke `autoresearch-hypothesis` to produce up to `--num-hypotheses` ideas spread across the recipe axes, each clearing the evidence bar (mechanism, support, gap). Ideas that cannot clear the bar are not padded in.
3. **Implement and test, one idea at a time** — for each idea in the set:
   1. invoke `autoresearch-implement` to add and register the module (and, where relevant, a basket that uses it);
   2. invoke `autoresearch-testing` to run the targeted test for that recipe;
   3. apply the keep/drop policy: if the module fails to run, is non-finite, or is ~10x worse than baseline, debug it (up to 3 attempts); if it still does not satisfy the requirements, drop it and move on to the next idea.
4. **Exhaustive check** — once all kept modules are implemented, invoke `autoresearch-testing` for the exhaustive recipe sweep (`uv run bin/tabicl/test.py --exhaustive`) and review the combined efficiency output.
