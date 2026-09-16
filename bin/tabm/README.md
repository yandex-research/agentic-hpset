# `bin/tabm` — modular TabM family

A [TabM](https://github.com/yandex-research/tabm) (parameter-efficient deep
ensemble) assembled from interchangeable per-stage modules, in the same layout as
[`bin/mlp`](../mlp/README.md). Backs two methods in `exp/`:

- **`tabm`** — non-agentic baseline: every stage pinned to index `0`.
- **`agentic-tabm`** — agentic search over the module space below.

## Files

| File | Role |
|------|------|
| `modules/<stage>/` | candidate implementations + a `*_MAP` registry per `__init__.py` |
| `hparams.py` | search space, per-dataset defaults, TPE sampler, trial budget. The TabM ensemble size `k` is fixed at 32. |
| `pipeline.py` | `run(config, dataset_root=..., device=...)` replays report configs; `train_and_eval` exposes the lower-level API (bf16 autocast on supported CUDA GPUs) |
| `core.py` | dataset loading + tensor scaffolding |
| `_util.py` | vendored GPU OOM-retry helper |

## Reconstruct a model from a report

```python
import json
from bin.tabm.pipeline import run

report = json.load(open("exp/tuned/agentic-tabm/california/report_200.json"))
cfg = report["experiments"][0]["config"]
result = run(cfg, dataset_root="data/california", device="cpu")
print(result.report["metrics"]["test"])
# result.predictions contains NumPy arrays; result.model is the fitted model.
```

## Index schema (base vs agentic)

`agentic-tabm` reports carry the full 10-axis index set (`hparams.INDEX_KEYS`). The
base `tabm` reports predate the split-out `optimizer`/`loss` axes and use the older
`evaluate_idx` (≡ `inference_idx`); all their indices are `0` — the reference TabM. Both `run` and `train_and_eval` translate the legacy key and supply the missing index-0 optimizer/loss choices.

## Module registry

Each row is `index | builder | file` (paths relative to `bin/tabm/`). Some files
host multiple builders (e.g. `embedding_num/advanced_plr.py`).

### `numerical_preprocess`
| idx | builder | file |
|---|---|---|
| 0 | `numerical_preprocess_v0` | `modules/preprocess_numerical/numerical.py` |
| 1 | `numerical_preprocess_v1` | `modules/preprocess_numerical/numerical_standardize.py` |
| 2 | `numerical_preprocess_v2` | `modules/preprocess_numerical/numerical_log1p.py` |
| 3 | `numerical_preprocess_v3` | `modules/preprocess_numerical/numerical_robust.py` |
| 4 | `numerical_preprocess_v4` | `modules/preprocess_numerical/missing_indicators.py` |
| 5 | `numerical_preprocess_v5` | `modules/preprocess_numerical/yeo_johnson.py` |
| 6 | `numerical_preprocess_v6` | `modules/preprocess_numerical/clipped_quantile.py` |
| 7 | `numerical_preprocess_v7` | `modules/preprocess_numerical/rank_uniform.py` |
| 8 | `numerical_preprocess_v8` | `modules/preprocess_numerical/quantile_kmeans.py` |

### `categorical_preprocess`
| idx | builder | file |
|---|---|---|
| 0 | `categorical_preprocess_v0` | `modules/preprocess_categorical/categorical.py` |
| 1 | `categorical_preprocess_v1` | `modules/preprocess_categorical/categorical_frequency.py` |
| 2 | `categorical_preprocess_v2` | `modules/preprocess_categorical/categorical_hashing.py` |

### `target_preprocess_reg` / `target_preprocess_clf`
| idx | builder | file |
|---|---|---|
| 0 | `target_preprocess_v0` | `modules/preprocess_target_reg/target.py` |
| 1 | `target_preprocess_v1` | `modules/preprocess_target_reg/target_log1p.py` |
| 2 | `target_preprocess_v2` | `modules/preprocess_target_reg/target_quantile.py` |
| 3 | `target_preprocess_v3` | `modules/preprocess_target_reg/target_robust.py` |
| clf 0 | `target_preprocess_clf_v0` | `modules/preprocess_target_clf/target.py` |

### `num_embedding`
| idx | builder | file |
|---|---|---|
| 0 | `build_num_embedding_v0` | `modules/embedding_num/piecewise_linear.py` |
| 1 | `build_num_embedding_v1` | `modules/embedding_num/gated_plr.py` |
| 2 | `build_num_embedding_v2` | `modules/embedding_num/polynomial.py` |
| 3 | `build_num_embedding_v3` | `modules/embedding_num/tokenized.py` |
| 4–8 | `build_num_embedding_v4`…`v8` | `modules/embedding_num/advanced_plr.py` (PLR variants) |

### `cat_embedding`
| idx | builder | file |
|---|---|---|
| 0 | `build_cat_embedding_v0` | `modules/embedding_cat/one_hot.py` |
| 1 | `build_cat_embedding_v1` | `modules/embedding_cat/one_hot_unknown.py` |
| 2 | `build_cat_embedding_v2` | `modules/embedding_cat/learned.py` |
| 3 | `build_cat_embedding_v3` | `modules/embedding_cat/target_mean.py` |
| 4 | `build_cat_embedding_v4` | `modules/embedding_cat/onehot_plus_learned.py` |

### `model`
| idx | builder | file |
|---|---|---|
| 0 | `build_tabm_v0` | `modules/model/tabm.py` |
| 1 | `build_tabm_v1` | `modules/model/geglu_tabm.py` |
| 2 | `build_tabm_v2` | `modules/model/swiglu_rms_tabm.py` |
| 3 | `build_tabm_v3` | `modules/model/feature_dropout_tabm.py` |
| 4 | `build_tabm_v4` | `modules/model/soft_moe_tabm.py` |
| 5 | `build_tabm_v5` | `modules/model/film_tabm.py` |
| 6 | `build_tabm_v6` | `modules/model/wide_shallow_tabm.py` |
| 7 | `build_tabm_v7` | `modules/model/linear_residual_tabm.py` |
| 8 | `build_tabm_v8` | `modules/model/input_gates_tabm.py` |
| 9 | `build_tabm_v9` | `modules/model/sparse_hidden_tabm.py` |
| 10 | `build_tabm_v10` | `modules/model/se_tabm.py` |
| 11 | `build_tabm_v11` | `modules/model/column_dropout_tabm.py` |
| 12 | `build_tabm_v12` | `modules/model/bilinear_tabm.py` |
| 13 | `build_tabm_v13` | `modules/model/lora_tabm.py` |
| 14 | `build_tabm_v14` | `modules/model/prenorm_tabm.py` |

### `train`
| idx | builder | file |
|---|---|---|
| 0 | `train_v0` | `modules/train/train.py` |
| 1 | `train_v1` | `modules/train/cosine_lr.py` |
| 2 | `train_v2` | `modules/train/swa.py` |
| 3 | `train_v3` | `modules/train/mixup.py` |
| 4 | `train_v4` | `modules/train/ema.py` |
| 5 | `train_v5` | `modules/train/sam.py` |
| 6 | `train_v6` | `modules/train/negative_correlation.py` |
| 7 | `train_v7` | `modules/train/weight_decorrelation.py` |
| 8 | `train_v8` | `modules/train/cutmix.py` |

### `inference_reg` / `inference_clf`
| idx | reg | clf |
|---|---|---|
| 0 | `inference.py` | `inference.py` |
| 1 | `median.py` | `temperature.py` |
| 2 | `trimmed_mean.py` | `tta_noise.py` |
| 3 | `geometric.py` | — |
| 4 | `tta_noise.py` | — |
| 5 | `quantile_avg.py` | — |
| 6 | `weighted_variance.py` | — |
| 7 | `clipped.py` | — |

### `loss_reg` / `loss_clf` / `optimizer`
| stage | idx 0 | idx 1 |
|---|---|---|
| loss_reg | `loss_reg/loss_reg.py` | `loss_reg/huber.py` |
| loss_clf | cross-entropy (`loss_clf/__init__.py`) | label smoothing (`loss_clf/__init__.py`) |
| optimizer | `build_adamw_v0` (`optimizer/__init__.py`) | `build_lion_v1` (`optimizer/lion.py`) |
