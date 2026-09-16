"""Materialized loss implementation."""

from __future__ import annotations

from typing import Any
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


_implementation = loss_v0


def loss_v0(*args: Any, **kwargs: Any) -> Any:
    return _implementation(*args, **kwargs)
