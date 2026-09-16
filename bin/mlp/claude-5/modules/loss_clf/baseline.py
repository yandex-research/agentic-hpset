"""Materialized loss implementation."""

from __future__ import annotations

from typing import Any
import torch.nn.functional as F
from torch import Tensor
from ...core import Dataset


def prediction_tensor(output: Any) -> Tensor:
    if isinstance(output, dict):
        for key in ('prediction', 'predictions', 'logits', 'output'):
            if key in output:
                return output[key]
        return next(iter(output.values()))
    if isinstance(output, tuple | list):
        return output[0]
    return output


def loss_v0(output: Any, target: Tensor, dataset: Dataset) -> Tensor:
    prediction = prediction_tensor(output)
    if dataset.is_binclass:
        return F.binary_cross_entropy_with_logits(
            prediction.reshape(-1), target.float()
        )
    return F.cross_entropy(prediction, target.long())


_implementation = loss_v0


def loss_v0(*args: Any, **kwargs: Any) -> Any:
    return _implementation(*args, **kwargs)
