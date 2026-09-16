# `bin/realmlp` — modular RealMLP family

A modular [RealMLP](https://github.com/dholzmueller/pytabkit) (the strong
meta-tuned tabular MLP from *Better by Default*), assembled from per-stage module
implementations layered on top of `pytabkit`'s RealMLP-TD. Backs two methods in `exp/`:

- **`realmlp`** — non-agentic baseline: plain tuned `pytabkit` RealMLP-TD
  (`bin/realmlp/base.py`), equivalently every stage at index `0`.
- **`agentic-realmlp`** — the agentic search over the 13 evidence-vetted module
  variants below.

Unlike the other families, RealMLP samples **no hyperparameters in Python**: each
stage reads its settings from a shared `cfg` dict (defaults live in the modules),
and the search space is declared in `search_space.toml`.

## Files

| File | Role |
|------|------|
| `modules/` | per-stage candidate modules + registry (`resolve`, `STAGE_KEYS`, `IMPLEMENTATION_REGISTRIES`) |
| `pipeline.py` | `run(config, ...)` dispatches baseline/agentic saved configs and returns metrics/predictions in memory; `main(config, exp)` retains the file-based driver interface |
| `base.py` | the index-0 reference: a thin `pytabkit` RealMLP-TD wrapper |
| `hparams.py` | loads `search_space.toml`, exposes `resolve`, the trial budget, and the TPE warmup |
| `search_space.toml` | the exact `[space]` searched (TOML DSL documented inline) |
| `_lib/` | vendored dataset/metric/experiment helpers from the source repo (non-driver only) |

Requires `pytabkit`, `torch`, `delu`, `rtdl_num_embeddings` (see the top-level deps).

## Reconstruct a model from a report

Each `exp/tuned/agentic-realmlp/<dataset>/report_<budget>.json` stores, per seed,
`experiments[i].config.model` including `implementation_indices` (a 10-stage dict).
Pass the full experiment config to the same `pipeline.run` interface as the other families:

```python
import json
from bin.realmlp.pipeline import run

report = json.load(open("exp/tuned/agentic-realmlp/california/report_200.json"))
cfg = report["experiments"][0]["config"]
result = run(cfg, dataset_root="data/california", device="cpu")
print(result.report["metrics"]["test"])
# result.predictions contains NumPy arrays; result.model is the fitted model.
```

The wrapper dispatches configs without `model.implementation_indices` to the
baseline in `base.py`; configs with indices use the modular implementation. It
also accepts the resolved `trial_params` in baseline ensemble member files.
Use `device="cuda"` to train on GPU. The return value's `model` is the fitted
pytabkit estimator for the baseline or a `FittedMember` (model and preprocessing
state) for the modular implementation.

`search_space.toml` is included in source and wheel distributions. Its model
space matches hpset's `exp/realmlp/agentic_v8/0/california/tuning/config.toml`
(and the Churn reference); dataset paths/batch sizes are supplied per run.
The `is_large` conditional selects a sampled model-size/training branch, rather
than being determined by the dataset name. The original distributed tuner is
not required to replay the concrete configs stored in reports.

## Index schema (base vs agentic)

`agentic-realmlp` reports carry the full 10-stage `implementation_indices`. The
base `realmlp` reports carry none — they are plain RealMLP-TD, i.e. every stage at
index `0`. The task-dispatched stages (`target_preprocess`, `inference`, `loss`)
resolve to their `_reg` / `_clf` registry by task type.

## Module registry

Index `0` is the meta-tuned RealMLP-TD default in every stage. Paths relative to
`bin/realmlp/`.

| stage | idx | builder | file |
|---|---|---|---|
| numerical_preprocess | 0 | `build_numerical_v0` | `modules/preprocess_numerical/realmlp.py` |
| numerical_preprocess | 1 | `build_numerical_quantile_normal` | `modules/preprocess_numerical/quantile_normal.py` |
| numerical_preprocess | 2 | `build_numerical_yeo_johnson` | `modules/preprocess_numerical/yeo_johnson.py` |
| categorical_preprocess | 0 | `build_categorical_v0` | `modules/preprocess_categorical/realmlp.py` |
| categorical_preprocess | 1 | `build_categorical_rare_merge` | `modules/preprocess_categorical/rare_merge.py` |
| categorical_preprocess | 2 | `build_categorical_large_cat_frequency` | `modules/preprocess_categorical/large_cat_frequency.py` |
| target_preprocess (reg/clf) | 0 | `target_preprocess_*_v0` | `modules/preprocess_target_{reg,clf}/realmlp.py` |
| num_embedding | 0 | `build_num_embedding_v0` | `modules/embedding_num/realmlp.py` |
| num_embedding | 1 | `build_num_embedding_ple` | `modules/embedding_num/piecewise_linear.py` |
| cat_embedding | 0 | `build_cat_embedding_v0` | `modules/embedding_cat/realmlp.py` |
| cat_embedding | 1 | `build_cat_embedding_shared_adapter` | `modules/embedding_cat/shared_adapter.py` |
| model | 0 | `build_model_v0` | `modules/model/realmlp.py` |
| model | 1 | `build_model_batch_ensemble` | `modules/model/batch_ensemble.py` |
| model | 2 | `build_model_residual` | `modules/model/residual.py` |
| train | 0 | `train_member_v0` | `modules/train/realmlp.py` |
| train | 1 | `train_member_ema` | `modules/train/ema.py` |
| train | 2 | `train_member_mixup` | `modules/train/mixup.py` |
| optimizer | 0 | `build_optimizer_v0` | `modules/optimizer/realmlp.py` |
| optimizer | 1 | `build_optimizer_grad_clip` | `modules/optimizer/grad_clip.py` |
| inference (reg/clf) | 0 | `predict_*_v0` | `modules/inference_{reg,clf}/realmlp.py` |
| loss_reg | 0 | `regression_loss_step_v0` | `modules/loss_reg/realmlp.py` |
| loss_reg | 1 | `regression_loss_step_huber` | `modules/loss_reg/huber.py` |
| loss_clf | 0 | `classification_loss_step_v0` | `modules/loss_clf/realmlp.py` |
| loss_clf | 1 | `classification_loss_step_focal` | `modules/loss_clf/focal.py` |
