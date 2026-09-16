"""Training-prior calibration with one immutable correction per inference call."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.special
import torch
from torch import Tensor, nn

from ...core import Dataset, TorchDataset


def _output_tensor(output: Any) -> Tensor:
    if isinstance(output, dict):
        for key in ('prediction', 'predictions', 'logits', 'output'):
            if key in output:
                return output[key]
        return next(iter(output.values()))
    if isinstance(output, tuple | list):
        return _output_tensor(output[0])
    if not isinstance(output, Tensor):
        raise TypeError(f'Expected tensor model output, got {type(output)}')
    return output


def _forward_parts(
    model: nn.Module,
    tensors: TorchDataset,
    trial_params: dict[str, Any],
    parts: tuple[str, ...],
) -> dict[str, np.ndarray]:
    modules = tuple(model.modules())
    modes = tuple(module.training for module in modules)
    model.eval()
    predictions: dict[str, np.ndarray] = {}
    try:
        with torch.inference_mode():
            for part in parts:
                outputs: list[Tensor] = []
                indices = torch.arange(
                    tensors.y[part].shape[0], device=tensors.y[part].device
                )
                for batch_idx in indices.split(int(trial_params['eval_batch_size'])):
                    x_num = (
                        None
                        if tensors.x_num is None
                        else tensors.x_num[part][batch_idx]
                    )
                    x_cat = (
                        None
                        if tensors.x_cat is None
                        else tensors.x_cat[part][batch_idx]
                    )
                    outputs.append(_output_tensor(model(x_num, x_cat)))
                predictions[part] = (
                    torch.cat(outputs).detach().cpu().numpy().astype(np.float32)
                )
    finally:
        for module, training in zip(modules, modes, strict=True):
            module.training = training
    return predictions


def _binary_shift(train_logits: np.ndarray, train_targets: np.ndarray) -> float:
    targets = np.asarray(train_targets).reshape(-1)
    target_prior = (float(targets.sum()) + 0.5) / (targets.size + 1.0)
    predicted_prior = float(scipy.special.expit(train_logits.reshape(-1)).mean())
    target_prior = float(np.clip(target_prior, 1e-6, 1.0 - 1e-6))
    predicted_prior = float(np.clip(predicted_prior, 1e-6, 1.0 - 1e-6))
    return float(
        scipy.special.logit(target_prior) - scipy.special.logit(predicted_prior)
    )


def _multiclass_shift(
    train_logits: np.ndarray, train_targets: np.ndarray, n_classes: int
) -> np.ndarray:
    targets = np.asarray(train_targets).reshape(-1).astype(np.int64)
    target_prior = np.bincount(targets, minlength=n_classes).astype(np.float64) + 0.5
    target_prior /= target_prior.sum()
    predicted_prior = scipy.special.softmax(train_logits, axis=1).mean(axis=0)
    predicted_prior = np.clip(predicted_prior, 1e-8, None)
    predicted_prior /= predicted_prior.sum()
    return (np.log(target_prior) - np.log(predicted_prior)).astype(np.float32)


def inference_prior_calibration_v1(
    model: nn.Module,
    tensors: TorchDataset,
    dataset: Dataset,
    trial_params: dict[str, Any],
    _device: torch.device,
    parts: tuple[str, ...],
) -> dict[str, np.ndarray]:
    needed_parts = tuple(dict.fromkeys((*parts, 'train')))
    predictions = _forward_parts(model, tensors, trial_params, needed_parts)
    if dataset.is_binclass:
        shift: float | np.ndarray = _binary_shift(
            predictions['train'], dataset.y_raw['train']
        )
    else:
        shift = _multiclass_shift(
            predictions['train'], dataset.y_raw['train'], dataset.n_classes
        )
    return {part: (predictions[part] + shift).astype(np.float32) for part in parts}
