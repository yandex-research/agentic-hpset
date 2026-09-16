---
name: autoresearch-testing
description: Test registered module variants on generated synthetic scenarios and compare efficiency to the all-v0 baseline. Use per module right after implementing it, and once more for the final exhaustive interaction check.
---

# Autoresearch: Testing

The canonical runner is `bin/tabm/test.py`, driven only on
generated synthetic scenarios (never real data). Use it in two places in the
loop: on each new module right after `autoresearch-implement`, and for the
final interaction check.

`bin/tabm/test.py` builds every run from a fixed `BASE_MODEL`
config, so a module's knobs are exercised at their in-code defaults.

## Targeted test (per module)

Run the new module index directly:

```bash
uv run bin/tabm/test.py --<stage>-idx <idx>
```

Examples:

```bash
uv run bin/tabm/test.py --model-idx 1
uv run bin/tabm/test.py --optimizer-idx 1
uv run bin/tabm/test.py --numerical-preprocess-idx 1
uv run bin/tabm/test.py --loss-idx 1
```

The `--<stage>-idx` flags cover every stage in `STAGE_KEYS`
(`--numerical-preprocess-idx`, `--categorical-preprocess-idx`,
`--target-preprocess-idx`, `--num-embedding-idx`, `--cat-embedding-idx`,
`--model-idx`, `--train-idx`, `--optimizer-idx`, `--inference-idx`,
`--loss-idx`). The runner prints all-v0 baseline numbers first, then candidate
metrics, wall/fit time, trainable parameter count, process memory, and GPU
peak memory when CUDA is used.

## Long-epoch coverage

Every invocation already includes one long scenario at `--long-epochs`
(default 12) alongside the quick passes. For modules that only activate after
warmup or many epochs (momentum build-up, schedule tails, EMA-style
averaging), raise it:

```bash
uv run bin/tabm/test.py --<stage>-idx <idx> --long-epochs 20
```

## Exhaustive interaction check (final)

After all kept modules are implemented, sweep the registered indices:

```bash
uv run bin/tabm/test.py --exhaustive
```

`--exhaustive` runs the full Cartesian product of every stage's registered
indices. This is only tractable when few modules exist: once each stage has
many variants the product explodes and a full sweep is infeasible. When that
is the case, check interactions with a tractable proxy instead — either pin
most stages to `0` and sweep one or two at a time
(`--exhaustive --<stage>-idx 0 ...` for the others), or sample random
full-pipeline configs (a random valid index per stage) and confirm each runs,
stays finite, and is within ~10x baseline. Report any coverage you skipped
rather than implying the whole product was swept.

## Keep / drop criteria

Review the efficiency output against the all-v0 baseline. A module that fails
to run, produces non-finite metrics, or is at least ~10x slower, larger, or
more memory hungry than baseline needs an explicit justification before it is
kept — otherwise debug it (a few attempts) and, failing that, drop it.

A passing smoke test means the module runs correctly and is not wildly
inefficient — it is **not** evidence the module improves real-task quality.
Do not upgrade a pass into a claim of improvement; quality was decided at the
hypothesis-vetting stage, and the real verdict comes from the per-dataset
tuner.
