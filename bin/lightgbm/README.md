# `bin/lightgbm` — modular LightGBM family

A [LightGBM](https://github.com/microsoft/LightGBM) gradient-boosting pipeline
wrapped in three interchangeable module stages — **feature engineering**, **data
augmentation**, and **prediction post-processing** — plus the boosting
hyperparameters. Backs two methods in `exp/`:

- **`lightgbm`** — non-agentic baseline: every stage pinned to index `0` (identity).
- **`agentic-lightgbm`** — agentic search over the module space below.

## Files

| File | Role |
|------|------|
| `modules/feature/`, `modules/data_aug/`, `modules/postproc/{regression,classification}/` | candidate transforms + `*_MAP` registries |
| `model.py` | LightGBM estimator construction, fixed params, and `suggest_lightgbm_params` (the boosting HP search) |
| `pipeline.py` | `run(config, ...)` loads data and replays a saved config; `run_full_pipeline(...)` assembles feature → data_aug → normalized-target fit → postproc from arrays |
| `hparams.py` | one entry point: `sample_hps`, `resolve`, TPE sampler, trial budget |
| `core.py` | dataset loading + metric helpers (pandas frames; no torch) |

Task types use LightGBM's vocabulary: `regression`, `binclass`, `multiclass`.

## Reconstruct a model from a report

```python
import json
from bin.lightgbm.pipeline import run

report = json.load(open("exp/tuned/agentic-lightgbm/california/report_200.json"))
cfg = report["experiments"][0]["config"]
result = run(cfg, dataset_root="data/california", device="cpu")
print(result.report["metrics"]["test"])
# result.predictions contains NumPy arrays; result.model is the fitted model.
```

`run` accepts both single-model `trial_params` and recovered ensemble
`lightgbm_params`/`fit_seed` configs. It preserves `implementation_config.feature_mode`,
standardizes augmented training targets for regression, applies validation early
stopping, and returns predictions in original target units (probabilities for
classification). This recipe runs on CPU. Dataset and fit/thread settings can be
supplied in the config; the research cluster driver is not required.

## Index schema (base vs agentic)

`agentic-lightgbm` reports carry `feature_idx`, `data_aug_idx`, `postproc_idx` and a
`feature_mode` (`append`/`replace`). The base `lightgbm` reports omit these — all
stages are index `0` (identity), i.e. plain tuned LightGBM.

## Module registry

Paths relative to `bin/lightgbm/`. Index `0` is the identity transform in every stage.

### `feature_idx`
| idx | file |
|---|---|
| 0 | `modules/feature/identity.py` |
| 1 | `modules/feature/quantile_bin.py` |
| 2 | `modules/feature/pairwise_product.py` |
| 3 | `modules/feature/row_stats.py` |
| 4 | `modules/feature/nan_indicator.py` |
| 5 | `modules/feature/frequency_encoding.py` |
| 6 | `modules/feature/target_mean_encoding.py` |
| 7 | `modules/feature/pca_components.py` |
| 8 | `modules/feature/kmeans_cluster.py` |
| 9 | `modules/feature/cat_pair_concat.py` |
| 10 | `modules/feature/groupby_mean.py` |
| 11 | `modules/feature/log1p_skew.py` |
| 12 | `modules/feature/rare_category_collapse.py` |
| 13 | `modules/feature/pairwise_difference.py` |
| 14 | `modules/feature/hash_high_card_cat.py` |
| 15 | `modules/feature/rank_percentile.py` |
| 16 | `modules/feature/cat_target_std.py` |

### `data_aug_idx`
| idx | file |
|---|---|
| 0 | `modules/data_aug/identity.py` |
| 1 | `modules/data_aug/bootstrap.py` |
| 2 | `modules/data_aug/gaussian_noise.py` |
| 3 | `modules/data_aug/feature_nan_dropout.py` |
| 4 | `modules/data_aug/column_swap_noise.py` |
| 5 | `modules/data_aug/stratified_bootstrap.py` |
| 6 | `modules/data_aug/oversample_minority.py` |
| 7 | `modules/data_aug/mixup_same_target.py` |
| 8 | `modules/data_aug/cat_dropout.py` |
| 9 | `modules/data_aug/concat_perturbed_copy.py` |
| 10 | `modules/data_aug/intra_target_swap.py` |
| 11 | `modules/data_aug/mean_imputation_dropout.py` |
| 12 | `modules/data_aug/random_duplicate.py` |
| 13 | `modules/data_aug/uniform_jitter.py` |
| 14 | `modules/data_aug/target_quantile_bootstrap.py` |
| 15 | `modules/data_aug/swap_low_variance.py` |
| 16 | `modules/data_aug/rank_jitter.py` |

### `postproc_idx` — regression / classification (task-dispatched)
| idx | regression | classification |
|---|---|---|
| 0 | `postproc/regression/identity.py` | `postproc/classification/identity_clf.py` |
| 1 | `clip_train_range.py` | `clip_renormalize.py` |
| 2 | `clip_train_quantile.py` | `label_smoothing.py` |
| 3 | `quantile_match.py` | `temperature_scaling.py` |
| 4 | `integer_round.py` | `prior_correction.py` |
| 5 | `softclip_tanh.py` | `train_prior_blend.py` |
| 6 | `shrink_to_train_mean.py` | `power_sharpen.py` |
| 7 | `nonneg_clip.py` | `min_proba_floor.py` |
| 8 | `iqr_winsorize.py` | `power_smooth.py` |
