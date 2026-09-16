---
name: autoresearch-testing
description: Test registered TabICL recipe modules on generated synthetic scenarios and compare inference efficiency to the default recipe. Use per module right after implementing it, and once more for the final exhaustive recipe check.
---

# Autoresearch: Testing

The canonical runner is `bin/tabicl/test.py`, driven only on generated synthetic
scenarios (never real data). Use it in two places in the loop: on each new module
right after `autoresearch-implement`, and for the final recipe check. There is no
gradient training — the model is frozen — so a test exercises the *recipe* around
it: build the evaluator from the recipe indices, run it on a synthetic
context/query split, and check the predictions and cost.

## Targeted test (per module)

Run a recipe that selects the new module (and, if it is a per-member component,
the basket that includes it):

```bash
uv run bin/tabicl/test.py --<axis>-idx <idx>
```

Examples:

```bash
uv run bin/tabicl/test.py --feature-permutation-idx 1
uv run bin/tabicl/test.py --ensemble-idx 1
uv run bin/tabicl/test.py --feature-preproc-basket-idx 1
```

Cover regression and a classification scenario, since `ensemble` and `postproc`
are task-dispatched. The runner prints default-recipe baseline numbers first,
then the candidate recipe's metrics, wall time, number of members / forward
passes, and peak memory.

## Exhaustive recipe check (final)

After all kept modules are implemented, sweep the recipe grid:

```bash
uv run bin/tabicl/test.py --exhaustive
```

`--exhaustive` runs the Cartesian product of the recipe axes. This is only
tractable while the grid is small; once it grows, check interactions with a
tractable proxy instead — pin most axes to their default and sweep one or two at
a time, or sample random valid recipes and confirm each runs, stays finite, and
is within ~10x the default cost. Report any coverage you skipped rather than
implying the whole grid was swept.

## Keep / drop criteria

Review the efficiency output against the default recipe. A recipe module that
fails to run, produces non-finite or invalid predictions (e.g. probabilities that
do not normalize), leaks the query into the context, or is at least ~10x more
members / passes / memory than the default needs an explicit justification before
it is kept — otherwise debug it (a few attempts) and, failing that, drop it.

A passing smoke test means the recipe runs correctly and is not wildly
inefficient — it is **not** evidence it improves real-task quality. Do not upgrade
a pass into a claim of improvement; quality was decided at the hypothesis-vetting
stage, and the real verdict comes from the per-dataset tuner.
