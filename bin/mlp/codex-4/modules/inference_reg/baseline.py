"""Materialized inference implementation."""

from __future__ import annotations

from typing import Any
import numpy as np
import torch
from torch import Tensor
from ...core import Dataset, TorchDataset
from bin.mlp._lib.util import adjust_gpu_memory_usage


def _output_tensor(output: Any) -> Tensor:
    if isinstance(output, dict):
        for key in ('prediction', 'predictions', 'mean', 'output'):
            if key in output:
                return output[key]
        return next(iter(output.values()))
    if isinstance(output, tuple | list):
        return output[0]
    return output


@adjust_gpu_memory_usage('eval_batch_size')
def _predict_part(
    model: torch.nn.Module,
    x_num_part: Tensor | None,
    x_cat_part: Tensor | None,
    n_rows: int,
    reference_device: torch.device,
    *,
    eval_batch_size: int,
    noise_scale: float,
) -> Tensor:
    outputs: list[Tensor] = []
    indices = torch.arange(n_rows, device=reference_device)
    for batch_idx in indices.split(eval_batch_size):
        x_num = None if x_num_part is None else x_num_part[batch_idx]
        if x_num is not None and noise_scale:
            x_num = x_num + torch.randn_like(x_num) * noise_scale
        x_cat = None if x_cat_part is None else x_cat_part[batch_idx]
        output = _output_tensor(model(x_num, x_cat))
        if output.ndim == 2 and output.shape[1] >= 2:
            output = output[:, 0]
        outputs.append(output.reshape(-1))
    return torch.cat(outputs)


def _raw_predictions(
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
    modules = tuple(model.modules())
    modes = tuple(module.training for module in modules)
    model.eval()
    if dropout:
        for module in modules:
            if isinstance(module, torch.nn.modules.dropout._DropoutNd):
                module.train()
    result: dict[str, np.ndarray] = {}
    batch_size = int(trial_params['eval_batch_size'])
    try:
        with torch.no_grad():
            for part in parts:
                pass_outputs: list[Tensor] = []
                for _ in range(passes):
                    output, batch_size = _predict_part(
                        model,
                        None if tensors.x_num is None else tensors.x_num[part],
                        None if tensors.x_cat is None else tensors.x_cat[part],
                        tensors.y[part].shape[0],
                        tensors.y[part].device,
                        eval_batch_size=batch_size,
                        noise_scale=noise_scale,
                    )
                    pass_outputs.append(output)
                stacked = torch.stack(pass_outputs)
                aggregate = (
                    stacked.median(dim=0).values if median else stacked.mean(dim=0)
                )
                result[part] = aggregate.detach().cpu().numpy()
    finally:
        for module, training in zip(modules, modes, strict=True):
            module.training = training
    trial_params['eval_batch_size'] = batch_size
    return result


def _inverse(dataset: Dataset, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    inverse_transform = dataset.preprocess_artifacts['target']['inverse_transform']
    return {
        part: np.asarray(inverse_transform(prediction)).reshape(-1).astype(np.float32)
        for part, prediction in values.items()
    }


def inference_v0(
    model: torch.nn.Module,
    tensors: TorchDataset,
    dataset: Dataset,
    trial_params: dict[str, Any],
    _device: torch.device,
    parts: tuple[str, ...],
) -> dict[str, np.ndarray]:
    return _inverse(dataset, _raw_predictions(model, tensors, trial_params, parts))


_implementation = inference_v0


def inference_v0(*args: Any, **kwargs: Any) -> Any:
    return _implementation(*args, **kwargs)
