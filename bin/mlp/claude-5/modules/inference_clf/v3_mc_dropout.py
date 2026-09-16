"""Monte-Carlo dropout with frozen normalization and probability averaging."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from ...core import Dataset, TorchDataset


_DROPOUT_TYPES = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)


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


@contextmanager
def _dropout_only(model: nn.Module):
    modules = tuple(model.modules())
    modes = tuple(module.training for module in modules)
    model.eval()
    for module in modules:
        if isinstance(module, _DROPOUT_TYPES):
            module.train()
    try:
        yield
    finally:
        for module, training in zip(modules, modes, strict=True):
            module.training = training


def _as_probabilities(logits: Tensor, dataset: Dataset) -> Tensor:
    return torch.sigmoid(logits) if dataset.is_binclass else torch.softmax(logits, -1)


def _as_logits(probabilities: Tensor, dataset: Dataset) -> Tensor:
    eps = torch.finfo(probabilities.dtype).eps
    probabilities = probabilities.clamp(eps, 1.0 - eps)
    if dataset.is_binclass:
        return torch.logit(probabilities)
    return probabilities.log()


def _forward_parts(
    model: nn.Module,
    tensors: TorchDataset,
    dataset: Dataset,
    trial_params: dict[str, Any],
    parts: tuple[str, ...],
    *,
    passes: int,
) -> dict[str, np.ndarray]:
    batch_size = int(trial_params['eval_batch_size'])
    tensor_device = tensors.y['train'].device
    cuda_devices = (
        [tensor_device.index or torch.cuda.current_device()]
        if tensor_device.type == 'cuda'
        else []
    )
    predictions: dict[str, np.ndarray] = {}
    with torch.random.fork_rng(devices=cuda_devices), _dropout_only(model):
        with torch.inference_mode():
            for part in parts:
                pass_probabilities: list[Tensor] = []
                indices = torch.arange(
                    tensors.y[part].shape[0], device=tensors.y[part].device
                )
                for _ in range(passes):
                    outputs: list[Tensor] = []
                    for batch_idx in indices.split(batch_size):
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
                    pass_probabilities.append(
                        _as_probabilities(torch.cat(outputs), dataset)
                    )
                mean_probability = torch.stack(pass_probabilities).mean(0)
                predictions[part] = (
                    _as_logits(mean_probability, dataset)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
    return predictions


def inference_mc_dropout_v3(
    model: nn.Module,
    tensors: TorchDataset,
    dataset: Dataset,
    trial_params: dict[str, Any],
    _device: torch.device,
    parts: tuple[str, ...],
) -> dict[str, np.ndarray]:
    return _forward_parts(model, tensors, dataset, trial_params, parts, passes=5)
