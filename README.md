# agentic-tabular

We take **MLP**, **TabM**, **LightGBM**, **TabICL**, and **RealMLP**, split them into a pipeline of modules (e.g. preprocessing, embeddings, model, training, optimizer, loss, inference, post-processing), and then task the agent to implement additional candidates for each module. In this repo, we expose two artifacts:

- Our agentic pipeline search space (model code under `bin/`) and results (raw exp under `exp/` and processed artifacts under `artifacts/`)
- Agentic skills that we used to generate the modules for each model (under `agentic/`)

## Code structure

### Layout

```
agentic/                                          # agent instructions
bin/
  mlp/  tabm/  lightgbm/  tabicl/  realmlp/ 
    modules/                                      # candidate implementations
    hparams.py                                    # HP search space
    pipeline.py                                   # assembles + trains + evaluates a model from a config
    README.md                                     # maps every `*_idx` to the module file it selects
  mlp/<agent>-<idx>/                              # agent-ablation for MLP
  analysis/                                       # regenerate all paper tables/figures from exp/
exp/
  tuned/<method>/<dataset>/report_<budget>.json   # single-model results
  ensembled/<method>/<dataset>/                  # report.json + cfg_<member>.json; candidate pools
  results_manifest.json
artifacts/                                        # committed CSVs + PDF figures (incl. agent_ablation/)
results_analysis.ipynb
```

The `agentic/` provides **claude** and **codex** skills that were used to generate the modules; see `agentic/README.md`.

### The core idea: indices = modules

Suppose that for an MLP, the agent has generated `N` candidate modules.
```
mlp/
  modules/
    preprocessing/
      __init__.py
      modules_1.py  # idx = 0 base module, e.g., no preprocessing
      modules_2.py  # idx = 1 Yeo-Johnson preprocessing
      ...
      modules_N.py  # idx = N-1 some other preprocessing
```
At tuning time, module indices are categorical choices from the registered
indices (including index 0). Each family's `pipeline.py` assembles the chosen
modules and evaluates a configuration.

## Train and evaluate

Every family exposes the same entry point: `bin.<family>.pipeline.run(config,
dataset_root=..., device=...)`. Pass one `experiments[i]["config"]` from a
single-model report, or a selected ensemble member config. The result has
`report` (including metrics), `predictions` (NumPy arrays), and `model`.

```python
import json
from bin.mlp.pipeline import run

report = json.load(open("exp/tuned/agentic-mlp/california/report_200.json"))
cfg = report["experiments"][0]["config"]
result = run(cfg, dataset_root="data/california", device="cpu")
print(result.report["metrics"]["test"])
```

Use `device="cuda"` for the neural families on a GPU; LightGBM uses CPU.

## Datasets

The 45 datasets span three sources: 6 standard benchmarks, 31 from
[TabArena](https://tabarena.ai), and 8 from [TabReD](https://github.com/yandex-research/tabred); `exp/results_manifest.json` lists all datasets. The raw data (~7 GB) is not committed, but is available on request. Point each family's data loader at a local `data/<dataset>/` tree (NumPy arrays + `info.json`) to run training. The analysis pipeline does not need the datasets.

### Tuned (individual) model results

The main tuned comparisons select configurations by validation performance at
**200** trials (**100** for `tabred/*` and `microsoft`), then evaluate the selected
configuration across seeds. 

Search spaces and sampler settings are exposed in
`bin/<family>/hparams.py`; RealMLP's conditional space is in `search_space.toml`.

Metric direction must follow the family's objective (e.g. raw AUC is maximized,
error is minimized).

### Greedy ensembles

`exp/ensembled` holds greedy (Caruana) ensemble-selection results. The selected member indices/weights and the resulting metrics. The selection algorithm is in `bin/analysis/ensemble_select.py`.

Selected member configs are stored as `cfg_<index>.json`. Complete evaluated
candidate pools for all eight methods are in `candidate_configs.json`.
See [the ensemble replay guide](exp/ensembled/README.md)
for member IDs, coverage, and a replay example.

### Agent ablation

To measure how much the results depend on which coding agent generated the
modules, the MLP instruction set (`agentic/mlp/`) was re-run end-to-end by two different coding agents: **claude** and **codex** for five independent runs each. The resulting modules are available under `bin/mlp/<agent>-<idx>/`. The corresponding results are in `exp/tuned/mlp-ablation-<agent>-<idx>/<dataset>/report_<budget>.json`.

## Reproduce the paper tables and figures

The analysis reads only the committed `exp/` reports and `artifacts/` (no dataset tree required):

```bash
pip install -e .
python -m bin.analysis.gen_analysis           # regenerates the main artifacts/
python -m bin.analysis.gen_agent_ablation     # regenerates artifacts/agent_ablation/
```

The first command rebuilds the grouped comparison tables, the per-dataset tables, the budget curves, and the main summary figure; the second (standalone, so the main tables are unaffected) rebuilds the agent-ablation tables. Together they match the committed `artifacts/`.

## Reuse a family's modules

```bash
pip install -e ".[torch]"      # or .[lightgbm] / .[tabicl] / .[realmlp] / .[all]
```

```python
from bin.mlp.hparams import IMPLEMENTATION_REGISTRIES, resolve
# resolve("model_idx", 7, "regression") -> the builder for that model variant
```

Each family is importable and its modules are composable; running an actual
training additionally needs the datasets (below) and the family's extra deps.

## Requirements

Python >= 3.12. Core analysis: numpy, pandas, scipy, scikit-learn, optuna,
matplotlib. Per-family extras (torch, lightgbm, huggingface-hub, pytabkit, delu) are declared as optional-dependency groups in `pyproject.toml`.

## License

MIT — see [LICENSE](LICENSE).
