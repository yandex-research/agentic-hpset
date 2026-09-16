"""Assemble, train, and evaluate the modular MLP pipeline.

`train_and_eval` operates in two modes:

  - sampling mode  -- pass `trial` (an `optuna.Trial`) together with the
    per-dataset `default_hps` (from ``bin.mlp.hparams.get_default_hps``).
    Hyperparameters and implementation indices are sampled from the trial and
    persisted via `trial.user_attrs` so a tuned trial can be replayed without
    re-sampling.

  - explicit mode  -- pass `trial_params` and `implementation_indices`
    directly. Used to re-run a previously-tuned trial across multiple seeds, and
    to reconstruct a model from the indices recorded in an ``exp/`` report.

The search space, per-dataset defaults, registries, and sampler all live in
``bin.mlp.hparams``; this module is dataset-agnostic assembly logic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch

from bin._pipeline import RunResult, dataset_path, neural_config, normalize_neural_indices, split_name

from bin.mlp.core import (
    Dataset,
    TorchDataset,
    load_raw_arrays,
    load_task_info,
    to_torch,
)
from bin.mlp.hparams import DefaultHPs, resolve, sample_hps


def infer_loss_output_dim(
    loss_fn: Callable[..., Any],
    dataset_meta: Dataset,
) -> int:
    output_dim_fn = getattr(loss_fn, "model_output_dim", None)
    if callable(output_dim_fn):
        output_dim = int(output_dim_fn(dataset_meta))
    elif dataset_meta.is_multiclass:
        output_dim = dataset_meta.n_classes
    else:
        output_dim = 1
    if output_dim < 1:
        raise RuntimeError(f"Loss produced invalid model output_dim={output_dim}.")
    return output_dim


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------


def _build_dataset(
    dataset_root: Path,
    split: str,
    implementation_indices: dict[str, int],
    seed: int,
) -> Dataset:
    x_num_raw, x_cat_raw, y_raw, task_type, score_name, n_classes = load_raw_arrays(
        dataset_root, split
    )
    config = {
        "seed": seed,
        "y_train": y_raw["train"],
        "n_classes": n_classes,
        "task_type": task_type,
    }
    num_fn = resolve(
        "numerical_preprocess_idx",
        implementation_indices["numerical_preprocess_idx"],
        task_type,
    )
    cat_fn = resolve(
        "categorical_preprocess_idx",
        implementation_indices["categorical_preprocess_idx"],
        task_type,
    )
    target_fn = resolve(
        "target_preprocess_idx",
        implementation_indices["target_preprocess_idx"],
        task_type,
    )
    x_num, num_artifacts = num_fn(x_num_raw, config)
    x_cat, cat_artifacts = cat_fn(x_cat_raw, config)
    y, target_artifacts = target_fn(y_raw, config)
    return Dataset(
        x_num=x_num,
        x_cat=x_cat,
        y=y,
        y_raw=y_raw,
        task_type=task_type,
        score_name=score_name,
        n_classes=n_classes,
        preprocess_artifacts={
            "numerical": num_artifacts,
            "categorical": cat_artifacts,
            "target": target_artifacts,
        },
    )


def _build_model(
    dataset: Dataset,
    tensors: TorchDataset,
    trial_params: dict[str, Any],
    implementation_indices: dict[str, int],
    output_dim: int,
    device: torch.device,
) -> torch.nn.Module:
    task_type = dataset.task_type
    num_builder = resolve(
        "num_embedding_idx", implementation_indices["num_embedding_idx"], task_type
    )
    cat_builder = resolve(
        "cat_embedding_idx", implementation_indices["cat_embedding_idx"], task_type
    )
    model_builder = resolve(
        "model_idx", implementation_indices["model_idx"], task_type
    )
    num_embedding, num_dim, num_artifacts = num_builder(
        None if tensors.x_num is None else tensors.x_num["train"],
        dataset,
        trial_params,
        device,
    )
    cat_embedding, cat_dim, cat_artifacts = cat_builder(
        dataset.cat_cardinalities, dataset, trial_params, device
    )
    bundle = {
        "num_embedding": num_embedding,
        "cat_embedding": cat_embedding,
        "input_dim": num_dim + cat_dim,
        "artifacts": {"num_embedding": num_artifacts, "cat_embedding": cat_artifacts},
    }
    return model_builder(dataset, trial_params, bundle, device, output_dim=output_dim)


def predict(
    model: torch.nn.Module,
    dataset: Dataset,
    tensors: TorchDataset,
    trial_params: dict[str, Any],
    implementation_indices: dict[str, int],
    device: torch.device,
    parts: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, np.ndarray]:
    inference_fn = resolve(
        "inference_idx", implementation_indices["inference_idx"], dataset.task_type
    )
    model.eval()
    with torch.no_grad():
        predictions = inference_fn(
            model, tensors, dataset, trial_params, device, parts
        )
    return predictions


@dataclass
class PipelineResult:
    report: dict[str, Any]
    model: torch.nn.Module
    dataset: Dataset
    tensors: TorchDataset
    trial_params: dict[str, Any]
    implementation_indices: dict[str, int]
    predictions: dict[str, np.ndarray]


def train_and_eval(
    *,
    dataset_root: Path,
    split: str,
    device: torch.device,
    seed: int,
    trial: optuna.Trial | None = None,
    trial_params: dict[str, Any] | None = None,
    implementation_indices: dict[str, int] | None = None,
    default_hps: DefaultHPs | None = None,
    max_epochs: int = 256,
    patience: int = 16,
) -> PipelineResult:
    """Train and evaluate the modular MLP pipeline.

    Sampling mode: pass `trial` (an `optuna.Trial`) and `default_hps`. HPs and
    implementation indices are sampled from the trial and persisted as
    `trial.user_attrs` for later replay.

    Explicit mode: pass `trial_params` and `implementation_indices` directly.
    Used when re-running a tuned trial across multiple seeds.
    """
    task_type, _score_name = load_task_info(dataset_root)
    if trial is not None:
        if default_hps is None:
            raise ValueError("`default_hps` is required when sampling from a trial.")
        trial_params, implementation_indices = sample_hps(
            trial,
            default_hps=default_hps,
            task_type=task_type,
            max_epochs=max_epochs,
            patience=patience,
        )
        trial.set_user_attr("trial_params", trial_params)
        trial.set_user_attr("implementation_indices", implementation_indices)
    elif trial_params is None or implementation_indices is None:
        raise ValueError(
            "Either pass `trial` or both `trial_params` and `implementation_indices`."
        )

    implementation_indices = normalize_neural_indices(implementation_indices)
    trial_params = {**trial_params, "seed": seed}
    dataset = _build_dataset(dataset_root, split, implementation_indices, seed)
    tensors = to_torch(dataset, device)
    loss_fn = resolve("loss_idx", implementation_indices["loss_idx"], task_type)
    output_dim = infer_loss_output_dim(loss_fn, dataset)
    model = _build_model(
        dataset,
        tensors,
        trial_params,
        implementation_indices,
        output_dim,
        device,
    )
    optimizer_builder = resolve(
        "optimizer_idx", implementation_indices["optimizer_idx"], task_type
    )
    train_fn = resolve("train_idx", implementation_indices["train_idx"], task_type)
    inference_fn = resolve(
        "inference_idx", implementation_indices["inference_idx"], task_type
    )
    report, _best_state = train_fn(
        model,
        tensors,
        dataset,
        trial_params,
        inference_fn,
        loss_fn,
        optimizer_builder,
        device,
        seed,
    )
    report = {
        **report,
        "trial_params": trial_params,
        "implementation_indices": dict(implementation_indices),
    }
    predictions = predict(
        model, dataset, tensors, trial_params, implementation_indices, device
    )
    if trial is not None:
        trial.set_user_attr("report", report)
    return PipelineResult(
        report=report,
        model=model,
        dataset=dataset,
        tensors=tensors,
        trial_params=trial_params,
        implementation_indices=dict(implementation_indices),
        predictions=predictions,
    )


def run(config: dict[str, Any], *, dataset_root: str | Path | None = None,
        device: str | torch.device | None = None, variant: str | None = None) -> RunResult:
    """Fit/evaluate one saved experiment config (single model or ensemble member)."""
    from bin.mlp.hparams import INDEX_KEYS

    root = dataset_path(config, dataset_root)
    if variant is not None:
        from copy import deepcopy
        from importlib import import_module

        variants = ("claude-1", "claude-2", "claude-5", "claude-6", "claude-7",
                    "codex-0", "codex-1", "codex-2", "codex-3", "codex-4")
        if variant not in variants:
            raise ValueError(f"Unknown MLP variant {variant!r}; choose from {variants}.")
        cfg = deepcopy(config)
        cfg["data"]["path"] = str(root)
        return import_module(f"bin.mlp.{variant}.mlp").train_and_eval(cfg, device=device)
    if "trial_params" not in config:
        raise ValueError("For an ablation config, pass variant='claude-1' (or another published variant).")
    task_type, _ = load_task_info(root)
    params, indices = neural_config(config, task_type, INDEX_KEYS)
    result = train_and_eval(
        dataset_root=root, split=split_name(config),
        device=torch.device(device if device is not None else config.get("device", "cpu")),
        seed=int(config.get("seed", params.get("seed", 0))),
        trial_params=params, implementation_indices=indices,
    )
    return RunResult(result.report, result.predictions, result.model)
