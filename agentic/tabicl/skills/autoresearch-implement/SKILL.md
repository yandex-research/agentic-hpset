---
name: autoresearch-implement
description: Implement and register one recipe-module variant for a TabICL axis under bin/tabicl/modules, mirroring the index-0 interface and keeping the frozen model and default recipe intact. Use once per idea, before testing it.
---

# Autoresearch: Implement a module

Turn one vetted idea (from `autoresearch-hypothesis`) into a registered recipe
module. Implement exactly one variant per invocation, then hand it to
`autoresearch-testing`.

## Steps

1. Add a new file under the matching `bin/tabicl/modules/<axis>/` package
   (`feature_preproc`, `feature_permutation`, `augmentation`, `ensemble`,
   `postproc_reg` / `postproc_clf`). Mirror the index-0 file's interface — its
   docstring/signature states the exact contract — and export a uniquely named
   component.
2. Register it in that package's registry (`*_MAP` / `*_NAMES`) under the next
   integer index.
3. If the idea is a per-member component (feature preproc, augmentation,
   postproc), add or extend a **basket** in `bin/tabicl/pipeline.py` so the tuner
   can select a recipe that uses it. Keep existing baskets unchanged.
4. Write a clear, concise docstring explaining what the module does and the
   evidence-backed hypothesis it implements.

## Rules

- Keep the index-0 (default) component and all existing baskets untouched.
- Never modify the frozen model or train/fine-tune weights. `model_idx` stays 0.
- Do not change the `TabICLEvaluator` orchestration or the base
  `EvaluationSpec` schema just because a new module does not use some fields.
- Implement the idea exactly as vetted — do not grow its scope while coding.
- A `feature_preproc` module must be fit on the context and applied consistently
  to the query (no leakage); an `augmentation` affects only the member that uses
  it; a `postproc` maps predictions to predictions. Read any hyperparameter from
  config with a gentle in-code default.
- The cost driver is inference — keep added members/passes bounded.

## Output

The registered `<axis>` component and its new integer index (and any new basket
index), ready to test with `autoresearch-testing`.
