"""Exercise saved report schemas through the public API on tiny CPU datasets.

Run with pytest and the relevant family extras. TabICL's orchestration test
substitutes inference to avoid network access or pretrained weight downloads.
"""

from copy import deepcopy
from importlib import import_module
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def saved_config(method, dataset="california", report="report_200.json"):
    return json.loads((ROOT / "exp/tuned" / method / dataset / report).read_text())["experiments"][0]["config"]


def dataset_at(root, classification=False):
    root.mkdir()
    rng = np.random.default_rng(81)
    x = rng.normal(size=(112, 3)).astype(np.float32)
    y = (np.arange(len(x)) % 2).astype(np.int64) if classification else (100 + 12 * x[:, 0]).astype(np.float32)
    np.save(root / "x_num.npy", x)
    np.save(root / "y.npy", y)
    split = root / "splits/default"
    split.mkdir(parents=True)
    for part, indices in {"train": np.arange(64), "val": np.arange(64, 88), "test": np.arange(88, 112)}.items():
        np.save(split / f"{part}.npy", indices.astype(np.int32))
    (root / "info.json").write_text(json.dumps({"task": {
        "type": "binclass" if classification else "regression",
        "score": "accuracy" if classification else "rmse",
    }}))
    return root


def small_neural(params):
    params.update(max_epochs=1, patience=2, batch_size=32, eval_batch_size=64,
                  d_block=32, n_blocks=1, n_bins=4, d_embedding=4, dropout=0.0)


@pytest.mark.parametrize("family", ["mlp", "tabm"])
@pytest.mark.parametrize("classification", [False, True])
def test_legacy_baseline_report_runs(family, classification, tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    cfg = saved_config(family, "churn" if classification else "california")
    assert "evaluate_idx" in cfg["implementation_indices"]
    small_neural(cfg["trial_params"])
    if family == "tabm":
        cfg["trial_params"]["k"] = 2
    original = deepcopy(cfg)
    result = import_module(f"bin.{family}.pipeline").run(
        cfg, dataset_root=dataset_at(tmp_path / "data", classification), device="cpu",
    )
    assert cfg == original
    assert result.model is not None
    assert set(result.predictions) == {"train", "val", "test"}
    assert all(np.isfinite(values).all() for values in result.predictions.values())
    assert "test" in result.report["metrics"]


@pytest.mark.parametrize("family", ["mlp", "tabm"])
@pytest.mark.parametrize("agentic", [False, True])
def test_neural_ensemble_task_indices(family, agentic):
    pytest.importorskip("torch")
    from bin._pipeline import neural_config
    hparams = import_module(f"bin.{family}.hparams")
    method = f"agentic-{family}" if agentic else family
    path = next((ROOT / "exp/ensembled" / method / "california").glob("cfg_*.json"))
    cfg = json.loads(path.read_text())
    for task in ["regression", "binclass"]:
        params, indices = neural_config(cfg, task, hparams.INDEX_KEYS)
        branch = "regression" if task == "regression" else "classification"
        for stage, value in cfg["implementation"].get(branch, {}).items():
            assert indices[f"{stage}_idx"] == value["index"]
        assert set(indices) == set(hparams.INDEX_KEYS)
        if not agentic:
            assert all(index == 0 for index in indices.values())
        for stage, index in indices.items():
            hparams.resolve(stage, index, task)
        assert params == cfg["trial_params"]


@pytest.mark.parametrize("classification", [False, True])
@pytest.mark.parametrize("ensemble", [False, True])
def test_lightgbm_report_and_member_configs(classification, ensemble, tmp_path):
    pytest.importorskip("lightgbm")
    from bin.lightgbm.pipeline import run
    if ensemble:
        cfg = json.loads((ROOT / "exp/ensembled/agentic-lightgbm/california/cfg_0000.json").read_text())
    else:
        cfg = saved_config("agentic-lightgbm")
    cfg.update(n_estimators=12, early_stopping_rounds=3, thread_count=1)
    cfg["implementation_config"] = {"feature_idx": 3, "data_aug_idx": 0,
                                    "postproc_idx": 3 if classification else 0, "feature_mode": "replace"}
    root = dataset_at(tmp_path / "data", classification)
    original = deepcopy(cfg)
    result = run(cfg, dataset_root=root, device="cpu")
    assert cfg == original
    assert result.report["implementation_config"]["feature_mode"] == "replace"
    assert all(np.isfinite(values).all() for values in result.predictions.values())
    if classification:
        assert result.predictions["test"].shape == (24, 2)
        np.testing.assert_allclose(result.predictions["test"].sum(axis=1), 1, atol=1e-6)
    else:
        assert result.predictions["test"].mean() > 70  # inverse target scaling was applied
        y = np.load(root / "y.npy")[:64]
        assert result.report["y_standardized"]["mean"] == pytest.approx(float(y.mean()))
    cfg["implementation_config"]["feature_mode"] = "append"
    appended = run(cfg, dataset_root=root, device="cpu")
    assert appended.model.n_features_in_ == result.model.n_features_in_ + 3


@pytest.mark.parametrize("agentic", [False, True])
@pytest.mark.parametrize("refit", [False, True])
def test_tabicl_preserves_recorded_context(agentic, refit, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("huggingface_hub")
    from bin.tabicl import pipeline
    from bin.tabicl.modules import nanotabicl
    cfg = saved_config("agentic-tabicl" if agentic else "tabicl", report="report_grid.json")
    cfg["run"]["refit_context"] = refit
    cfg["run"]["parts"] = ["val", "test"]
    original = deepcopy(cfg)
    model = Mock()
    model.to.return_value = model
    model.eval.return_value = model
    monkeypatch.setitem(nanotabicl.MODEL_REGISTRY, 0, lambda **kwargs: model)
    seen = {}

    def fit_predict(self, dataset, model, device, seed, parts):
        seen["n_train"] = dataset.size("train")
        seen["indices"] = self.spec.to_dict()
        seen["parts"] = parts
        return {"test": {"rmse": 1.0}}, {"test": np.zeros(24)}

    monkeypatch.setattr(pipeline.TabICLEvaluator, "fit_predict", fit_predict)
    result = pipeline.run(cfg, dataset_root=dataset_at(tmp_path / "data"), device="cpu")
    assert cfg == original
    assert seen["n_train"] == (88 if refit else 64)
    assert seen["parts"] == ("val", "test")
    assert result.report["refit_context"] is refit
    if agentic:
        assert json.loads(json.dumps(seen["indices"])) == cfg["run"]["evaluation_spec"]


@pytest.mark.parametrize("agentic", [False, True])
@pytest.mark.parametrize("ensemble", [False, True])
def test_realmlp_report_runs_without_experiment_directory(agentic, ensemble, tmp_path):
    pytest.importorskip("delu")
    pytest.importorskip("pytabkit")
    import torch
    from bin.realmlp.pipeline import run
    torch.set_num_threads(1)
    method = "agentic-realmlp" if agentic else "realmlp"
    if ensemble:
        path = next((ROOT / "exp/ensembled" / method / "california").glob("cfg_*.json"))
        cfg = json.loads(path.read_text())
    else:
        cfg = saved_config(method)
    cfg["data"]["cache"] = False
    cfg["batch_size"] = 32
    params = cfg.get("trial_params", cfg["model"])
    params.update(hidden_width=16, n_hidden_layers=2, n_ens=1, n_epochs=1,
                  use_early_stopping=False, plr_hidden_1=4, plr_hidden_2=4)
    if "trial_params" in cfg:
        params["batch_size"] = 32
    if agentic:
        cfg["model"]["implementation_indices"] = {k: 0 for k in cfg["model"]["implementation_indices"]}
    original = deepcopy(cfg)
    result = run(cfg, dataset_root=dataset_at(tmp_path / "data"), device="cpu")
    assert cfg == original
    assert result.predictions["test"].shape == (24,)
    assert np.isfinite(result.report["metrics"]["test"]["rmse"])
    assert result.model is not None


@pytest.mark.parametrize("variant", ["claude-1", "codex-0"])
def test_ablation_report_uses_its_own_registry(variant, tmp_path):
    pytest.importorskip("delu")
    from bin.mlp.pipeline import run
    cfg = saved_config(f"mlp-ablation-{variant}")
    cfg["data"]["cache"] = False
    small_neural(cfg["model"])
    cfg["batch_size"] = 32
    cfg["model"]["implementation_indices"] = {k: 0 for k in cfg["model"]["implementation_indices"]}
    original = deepcopy(cfg)
    result = run(cfg, variant=variant, dataset_root=dataset_at(tmp_path / "data"), device="cpu")
    assert cfg == original
    assert np.isfinite(result.report["metrics"]["test"]["rmse"])


@pytest.mark.parametrize("dataset", ["microsoft", "tabarena/kddcup09_appetency"])
def test_archived_tabm_member_runs(dataset, tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from bin.tabm.pipeline import run
    path = next((ROOT / "exp/ensembled/tabm" / dataset).glob("cfg_*.json"))
    cfg = json.loads(path.read_text())
    small_neural(cfg)
    cfg["k"] = 2
    original = deepcopy(cfg)
    result = run(cfg, dataset_root=dataset_at(tmp_path / "data", dataset != "microsoft"), device="cpu")
    assert cfg == original
    assert all(np.isfinite(values).all() for values in result.predictions.values())


def test_recovered_selection_configs_are_complete():
    count = 0
    for method in ["mlp", "agentic-mlp", "tabm", "agentic-tabm", "realmlp",
                   "lightgbm", "agentic-lightgbm", "agentic-realmlp"]:
        for path in (ROOT / "exp/ensembled" / method).rglob("report.json"):
            report = json.loads(path.read_text())
            pool = json.loads(path.with_name("candidate_configs.json").read_text())
            assert len(pool["member_order"]) == report["n_configs"]
            assert set(pool["member_order"]) == set(pool["configs"])
            for member_id in report["selected_member_weights"]:
                number = int(member_id.removeprefix("evaluation-"))
                cfg = json.loads(path.with_name(f"cfg_{number:04d}.json").read_text())
                assert cfg == pool["configs"][member_id]
                assert any(key in cfg for key in ["model", "trial_params", "_implementation_indices", "lightgbm_params"])
                count += 1
    assert count == 3756
