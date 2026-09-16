---
name: autoresearch-implement
description: Implement and register one module variant for a pipeline stage under bin/tabm/modules, mirroring the v0 interface and keeping v0 and the base model schema intact. Use once per idea, before testing it.
---

# Autoresearch: Implement a module

Turn one vetted idea (from `autoresearch-hypothesis`) into a registered module.
Implement exactly one variant per invocation, then hand it to
`autoresearch-testing`.

## Steps

1. Add a new file under the matching
   `bin/tabm/modules/<stage>/` package. For a
   task-dispatched stage (`target_preprocess`, `inference`, `loss`), add it to
   the correct `_clf` or `_reg` package.
2. Mirror the index-0 interface for that stage — the index-0 file in the stage
   package states, in its docstring/signature, the exact builder signature and
   returned-object contract — and export a uniquely named builder, function, or
   class for the new variant.
3. Write a clear, concise docstring explaining what the module does and the
   evidence-backed hypothesis it implements.
4. Register it in that package's `__init__.py` `*_MAP` under the next integer
   index.

## Rules

- Keep `v0` (index 0) untouched.
- Do not change the wrapper `bin/tabm/pipeline.py`, the registries in
  `bin/tabm/hparams.py`, the shared ensemble helpers (`modules/_shared.py`), or
  the base model config schema just because a new module does not use some keys.
  Variants plug into a stage registry only.
- Implement the idea exactly as vetted — do not grow its scope or bolt on
  extra mechanisms while coding; an addition that was not vetted is an
  unvetted hypothesis.
- The module must tolerate the full `cfg` (model config) dict supplied by the
  pipeline, even if some values are unused; any hyperparameter it needs is
  read from `cfg` with an in-code default. Keep those defaults gentle — the
  harm case argued at vetting time assumed conservative defaults.
- Prefer bounded parameter, memory, and runtime cost — the module will be
  compared against the all-v0 baseline.

## Output

The registered `<stage>` and its new integer index, ready to test with
`autoresearch-testing` (e.g. `--<stage>-idx <index>`).
