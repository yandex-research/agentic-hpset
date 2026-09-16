# `bin/tabicl` — modular NanoTabICL (TabICL) family

An in-context-learning tabular model (a small pretrained TabICL transformer,
"NanoTabICLv2") wrapped in a modular **evaluation recipe**: there is no gradient
training, so a "run" is a choice of feature preprocessing, feature permutation,
data augmentation, ensemble reducer, and post-processing applied around the frozen
model. Backs two methods in `exp/`:

- **`tabicl`** — fixed default NanoTabICL ensemble recipe (no search).
- **`agentic-tabicl`** — agentic search over the recipe space below.

Task types use `regression`, `binclass`, `multiclass`.

## Files

| File | Role |
|------|------|
| `modules/nanotabicl/model.py` | the NanoTabICL transformer (`MODEL_REGISTRY`, `model_idx` 0) |
| `modules/feature_preproc/`, `feature_permutation/`, `augmentation/`, `ensemble/`, `postproc_{reg,clf}/` | recipe stages + name registries |
| `pipeline.py` | `run(config, ...)` loads data/checkpoint and evaluates the recorded recipe; also exposes recipe resolution and `TabICLEvaluator` |
| `hparams.py` | `sample_hps`, the multivariate-TPE sampler, and `total_grid_size` (the recipe grid) |
| `core.py` | dataset loading + task helpers |

## Recipe indices → concrete spec

The agentic search samples **basket** indices (each basket expands to a tuple of
concrete module indices). Every `exp/tuned/agentic-tabicl` report records both the
sampled `implementation_indices` and the resolved `evaluation_spec`. Rebuild it with:

```python
import json
from bin.tabicl.pipeline import run

report = json.load(open("exp/tuned/agentic-tabicl/california/report_grid.json"))
cfg = report["experiments"][0]["config"]
result = run(cfg, dataset_root="data/california", device="cpu")
print(result.report["metrics"]["test"])
# result.predictions contains NumPy arrays; result.model is the fitted model.
```

`run` reads the recipe from `cfg["run"]`, including `refit_context` and the
requested evaluation parts. Published refit reports use train+validation as
context; their validation scores are consequently in-context scores. Baseline
reports without implementation indices use the version-0 recipe. The first run
loads the pretrained weights (and downloads them if not cached); `model_path`
and `allow_auto_download` can be set in the config for local weights. Use
`device="cuda"` for GPU evaluation.

## Registries

**`model_idx`**: `{0: NanoTabICLv2}` (`modules/nanotabicl/model.py`) — always 0.

**`feature_preproc`** (`modules/feature_preproc/`): 0 identity, 1 power, 2
quantile_normal, 3 quantile_uniform, 4 raw_quantile_normal.
`feature_preproc_basket_idx` → `{0: (0,1), 1: (0,1,2,3), 2: (0,1,4)}`.

**`feature_permutation_idx`** (`modules/feature_permutation/`): 0 latin, 1 none,
2 random, 3 shift.

**`ensemble_idx`** (`modules/ensemble/`): 0 mean, 1 median, 2 geometric_mean,
5 uncertainty_weighted (reg + clf differ in which are offered).

**`augmentation`** (`modules/augmentation/`): 0 identity, 1 feature_noise.
`augmentation_basket_idx` → `{0: (0,), 1: (0,1)}`.

**`postproc`** (task-dispatched): regression `modules/postproc_reg/`
{0 standardize, 1 log1p}; classification `modules/postproc_clf/` {0 proba}.
`postproc_basket_idx` → reg `{0: (0,), 1: (0,1)}`, clf `{0: (0,)}`.

Full grid size: **144** (regression), **96** (classification).
