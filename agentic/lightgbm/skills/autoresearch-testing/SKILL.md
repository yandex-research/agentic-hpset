---
name: autoresearch-testing
description: Test registered LightGBM module variants on generated synthetic scenarios and compare efficiency to the all-identity baseline. Use per module right after implementing it, and once more for the final exhaustive interaction check.
---

# Autoresearch: Testing

The canonical runner is `bin/lightgbm/test.py`, driven only on generated
synthetic scenarios (never real data). Use it in two places in the loop: on each
new module right after `autoresearch-implement`, and for the final interaction
check. The runner builds every run from a fixed base config, so a module's knobs
are exercised at their in-code defaults.

## Targeted test (per module)

Run the new module index directly:

```bash
uv run bin/lightgbm/test.py --<stage>-idx <idx>
```

Examples:

```bash
uv run bin/lightgbm/test.py --feature-idx 1
uv run bin/lightgbm/test.py --data-aug-idx 1
uv run bin/lightgbm/test.py --postproc-idx 1
```

The `--<stage>-idx` flags cover the three stages (`--feature-idx`,
`--data-aug-idx`, `--postproc-idx`). The runner prints all-identity baseline
numbers first, then candidate metrics, wall/fit time, and process memory. Cover
regression and a classification scenario, since `postproc` is task-dispatched.

## Exhaustive interaction check (final)

After all kept modules are implemented, sweep the registered indices:

```bash
uv run bin/lightgbm/test.py --exhaustive
```

`--exhaustive` runs the full Cartesian product of every stage's registered
indices. This is only tractable when few modules exist: once each stage has many
variants the product explodes and a full sweep is infeasible. When that is the
case, check interactions with a tractable proxy instead — pin most stages to `0`
and sweep one or two at a time, or sample random full-pipeline configs and
confirm each runs, stays finite, and is within ~10x baseline cost. Report any
coverage you skipped rather than implying the whole product was swept.

## Keep / drop criteria

Review the efficiency output against the all-identity baseline. A module that
fails to run, produces non-finite predictions, changes row counts where it must
not, leaks the target, or is at least ~10x slower / larger than baseline needs an
explicit justification before it is kept — otherwise debug it (a few attempts)
and, failing that, drop it.

A passing smoke test means the module runs correctly and is not wildly
inefficient — it is **not** evidence the module improves real-task quality. Do
not upgrade a pass into a claim of improvement; quality was decided at the
hypothesis-vetting stage, and the real verdict comes from the per-dataset tuner.
