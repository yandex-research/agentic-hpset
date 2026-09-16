# Replaying ensemble members

Each `<method>/<dataset>/report.json` records the selected members and weights.
Their evaluated configurations are in `cfg_<index>.json`; indices are zero-padded
to four digits in filenames.

| Method | Datasets | Candidate configs | Selected configs |
| --- | --- | --- | --- |
| `mlp` | 45 | 3,780 | 618 |
| `agentic-mlp` | 45 | 3,780 | 530 |
| `tabm` | 45 | 3,780 | 384 |
| `agentic-tabm` | 45 | 3,780 | 367 |
| `realmlp` | 45 | 3,780 | 580 |
| `lightgbm` | 45 | 3,480 | 433 |
| `agentic-lightgbm` | 45 | 2,667 | 326 |
| `agentic-realmlp` | 45 | 3,780 | 518 |


## Run a member

The same `pipeline.run` entry point accepts a single-model report config or an
ensemble member config:

```python
import json
from bin.lightgbm.pipeline import run

cfg = json.load(open("exp/ensembled/agentic-lightgbm/california/cfg_0000.json"))
result = run(cfg, dataset_root="data/california", device="cpu")
print(result.report["metrics"]["test"])
```

For MLP, TabM, and RealMLP, change the import to the corresponding family. Use
`device="cuda"` for neural training on a GPU. The dataset override replaces the
saved research path.

To reconstruct a selected ensemble, run each selected config and average its
predictions with `selected_member_weights`. For regression, predictions are
already in the original target units. For classification, normalize each member
to probabilities before blending: the main MLP/TabM pipelines return logits,
LightGBM returns probability matrices, and binary RealMLP returns positive-class
probabilities. For MLP/TabM, explicitly convert logits using the family's
`core.logits_to_probs` before calling `bin.analysis.ensemble_select.normalize_pred`
to obtain a two-dimensional probability matrix.
