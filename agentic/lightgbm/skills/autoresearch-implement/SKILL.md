---
name: autoresearch-implement
description: Implement and register one module variant for a LightGBM pipeline stage under bin/lightgbm/modules, mirroring the identity interface and keeping the identity default and model intact. Use once per idea, before testing it.
---

# Autoresearch: Implement a module

Turn one vetted idea (from `autoresearch-hypothesis`) into a registered module.
Implement exactly one variant per invocation, then hand it to
`autoresearch-testing`.

## Steps

1. Add a new file under the matching `bin/lightgbm/modules/<stage>/` package
   (`feature`, `data_aug`, or `postproc`). For `postproc`, add it to the correct
   `classification` or `regression` package.
2. Mirror the identity interface for that stage — the identity file in the stage
   package states, in its docstring/signature, the exact function signature and
   returned-object contract — and export a uniquely named function for the new
   variant.
3. Write a clear, concise docstring explaining what the module does and the
   evidence-backed hypothesis it implements.
4. Register it in that package's `__init__.py` `*_MAP` under the next integer
   index.

## Rules

- Keep the identity variant (index 0) untouched.
- Do not change `bin/lightgbm/pipeline.py`, `bin/lightgbm/model.py`, or the
  boosting-parameter search just because a new module does not use some keys.
  Variants plug into a stage registry only.
- Implement the idea exactly as vetted — do not grow its scope or bolt on extra
  mechanisms while coding; an addition that was not vetted is an unvetted
  hypothesis.
- A `feature` module must be fit on train and applied consistently to val/test
  (no target leakage — fit any target-based encoding on train folds only), and
  must not change row counts. A `data_aug` module touches train only. A
  `postproc` module maps predictions to predictions.
- Read any hyperparameter the module needs from its config with an in-code
  default; keep those defaults gentle — the harm case argued at vetting time
  assumed conservative defaults.
- Prefer bounded column-count, row-count, memory, and runtime cost — the module
  is compared against the all-identity baseline.

## Output

The registered `<stage>` and its new integer index, ready to test with
`autoresearch-testing` (e.g. `--<stage>-idx <index>`).
