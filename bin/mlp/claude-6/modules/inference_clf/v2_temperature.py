"""Training-only temperature calibration."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
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


def _fit_temperature(logits: np.ndarray, targets: np.ndarray, binclass: bool) -> float:
    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    targets_t = torch.as_tensor(
        targets,
        dtype=torch.float32 if binclass else torch.long,
    )
    temperatures = torch.logspace(-1.5, 1.5, 61)
    losses = []
    for temperature in temperatures:
        scaled = logits_t / temperature
        losses.append(
            F.binary_cross_entropy_with_logits(
                scaled.reshape(-1), targets_t.reshape(-1)
            ).item()
            if binclass
            else F.cross_entropy(scaled, targets_t.reshape(-1)).item()
        )
    return float(temperatures[int(np.argmin(losses))])


def inference_temperature_v2(
    model: nn.Module,
    tensors: TorchDataset,
    dataset: Dataset,
    trial_params: dict[str, Any],
    _device: torch.device,
    parts: tuple[str, ...],
) -> dict[str, np.ndarray]:
    needed_parts = tuple(dict.fromkeys((*parts, 'train')))
    predictions = _forward_parts(model, tensors, trial_params, needed_parts)
    temperature = _fit_temperature(
        predictions['train'], dataset.y_raw['train'], dataset.is_binclass
    )
    return {
        part: (predictions[part] / temperature).astype(np.float32) for part in parts
    }
