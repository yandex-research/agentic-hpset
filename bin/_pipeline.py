"""Small shared contracts for the per-family ``pipeline.run`` entry points."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RunResult:
    """One fitted configuration: metrics, predictions, and the fitted model."""

    report: dict[str, Any]
    predictions: dict[str, np.ndarray]
    model: Any = None


def dataset_path(config: dict[str, Any], override: str | Path | None) -> Path:
    value = override if override is not None else config.get(
        "data_root", config.get("data", {}).get("path")
    )
    if value is None:
        raise ValueError("Pass dataset_root or set config.data_root/config.data.path.")
    return Path(value).expanduser()


def split_name(config: dict[str, Any]) -> str:
    split = config.get("split", config.get("data", {}).get("split_id", "default"))
    return "/".join(map(str, split)) if isinstance(split, (list, tuple)) else str(split)


def normalize_neural_indices(indices: dict[str, int]) -> dict[str, int]:
    """Adapt the baseline report schema without renumbering module choices."""
    indices = dict(indices)
    if "evaluate_idx" in indices:
        legacy = indices.pop("evaluate_idx")
        if "inference_idx" in indices and indices["inference_idx"] != legacy:
            raise ValueError("Conflicting evaluate_idx and inference_idx.")
        indices["inference_idx"] = legacy
    indices.setdefault("optimizer_idx", 0)
    indices.setdefault("loss_idx", 0)
    # Historical classification baselines did not expose a target stage.
    indices.setdefault("target_preprocess_idx", 0)
    return indices


def neural_config(config: dict[str, Any], task_type: str, index_keys) -> tuple[dict, dict]:
    """Read a single-run config or the task-split selected-ensemble schema."""
    if "trial_params" in config:
        params = deepcopy(config["trial_params"])
    elif "_implementation_indices" in config:
        # Archived prediction libraries stored the sampled parameters directly.
        params = deepcopy({k: v for k, v in config.items() if not k.startswith("_")})
    else:
        raise ValueError("Expected trial_params or an archived _implementation_indices config.")
    indices = config.get("implementation_indices", config.get("_implementation_indices"))
    if indices is None and "implementation" in config:
        impl = config["implementation"]
        task = "regression" if task_type == "regression" else "classification"
        indices = {key: value for key, value in impl.items() if key.endswith("_idx")}
        indices.update({
            f"{stage}_idx": value["index"]
            for stage, value in {**impl.get("common", {}), **impl.get(task, {})}.items()
        })
    if indices is None:
        indices = {key: 0 for key in index_keys}
    return params, normalize_neural_indices(indices)
