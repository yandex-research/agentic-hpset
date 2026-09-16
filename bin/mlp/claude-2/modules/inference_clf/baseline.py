"""Materialized inference implementation."""

from __future__ import annotations

from typing import Any
import numpy as np
import torch
from torch import Tensor
from ...core import Dataset, TorchDataset


def _output_tensor(output: Any) -> Tensor:
    if isinstance(output, dict):
        for key in ('prediction', 'predictions', 'logits', 'output'):
            if key in output:
                return output[key]
        return next(iter(output.values()))
    if isinstance(output, tuple | list):
        return output[0]
    return output


def _forward_parts(
    model: torch.nn.Module,
    tensors: TorchDataset,
    trial_params: dict[str, Any],
    parts: tuple[str, ...],
    *,
    passes: int = 1,
    noise_scale: float = 0.0,
    dropout: bool = False,
    median: bool = False,
) -> dict[str, np.ndarray]:
    was_training = model.training
    model.train(dropout)
    predictions: dict[str, np.ndarray] = {}
    batch_size = int(trial_params['eval_batch_size'])
    with torch.no_grad():
        for part in parts:
            pass_outputs: list[Tensor] = []
            for _ in range(passes):
                outputs: list[Tensor] = []
                indices = torch.arange(
                    tensors.y[part].shape[0], device=tensors.y[part].device
                )
                for batch_idx in indices.split(batch_size):
                    x_num = (
                        None
                        if tensors.x_num is None
                        else tensors.x_num[part][batch_idx]
                    )
                    if x_num is not None and noise_scale:
                        x_num = x_num + torch.randn_like(x_num) * noise_scale
                    x_cat = (
                        None
                        if tensors.x_cat is None
                        else tensors.x_cat[part][batch_idx]
                    )
                    outputs.append(_output_tensor(model(x_num, x_cat)))
                pass_outputs.append(torch.cat(outputs))
            stacked = torch.stack(pass_outputs)
            aggregate = stacked.median(dim=0).values if median else stacked.mean(dim=0)
            predictions[part] = aggregate.detach().cpu().numpy().astype(np.float32)
    model.train(was_training)
    return predictions


def inference_v0(
    model: torch.nn.Module,
    tensors: TorchDataset,
    _dataset: Dataset,
    trial_params: dict[str, Any],
    _device: torch.device,
    parts: tuple[str, ...],
) -> dict[str, np.ndarray]:
    return _forward_parts(model, tensors, trial_params, parts)


_implementation = inference_v0


def inference_v0(*args: Any, **kwargs: Any) -> Any:
    return _implementation(*args, **kwargs)
