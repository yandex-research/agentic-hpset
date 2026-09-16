"""Materialized loss implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
import torch
import torch.nn.functional as F
from torch import Tensor
from ...core import Dataset


def prediction_tensor(output: Any) -> Tensor:
    if isinstance(output, dict):
        for key in ('prediction', 'predictions', 'mean', 'output'):
            if key in output:
                return output[key]
        return next(iter(output.values()))
    if isinstance(output, tuple | list):
        return output[0]
    return output


def _mean_prediction(prediction: Tensor) -> Tensor:
    if prediction.ndim == 2 and prediction.shape[1] >= 2:
        prediction = prediction[:, 0]
    return prediction.reshape(-1)


def loss_v0(output: Any, target: Tensor, _dataset: Dataset) -> Tensor:
    return F.mse_loss(_mean_prediction(prediction_tensor(output)), target.float())


def make_loss_variant(
    strategy: str, params: dict[str, Any], symbol: str
) -> Callable[[Any, Tensor, Dataset], Tensor]:
    beta = float(params.get('huber_beta', params.get('delta', params.get('beta', 0.5))))

    def loss(output: Any, target: Tensor, dataset: Dataset) -> Tensor:
        prediction = prediction_tensor(output)
        if strategy == 'huber':
            return F.smooth_l1_loss(
                _mean_prediction(prediction), target.float(), beta=max(beta, 1e-6)
            )
        if strategy == 'heteroscedastic':
            if prediction.ndim == 2 and prediction.shape[1] >= 2:
                mean = prediction[:, 0]
                log_var = prediction[:, 1].clamp(-8.0, 8.0)
                return (
                    0.5
                    * (torch.exp(-log_var) * (target.float() - mean).square() + log_var)
                ).mean()
        return loss_v0(output, target, dataset)

    loss.__name__ = f'loss_{strategy}_{abs(hash(symbol)) & 0xFFFF:x}'
    loss.__doc__ = f'Ported from {symbol}; extracted regression loss: {strategy}.'
    return loss


_implementation = make_loss_variant(
    'huber',
    {'delta': 1.0},
    'train.train_huber.train_v3',
)


def loss_huber_v1(*args: Any, **kwargs: Any) -> Any:
    return _implementation(*args, **kwargs)
