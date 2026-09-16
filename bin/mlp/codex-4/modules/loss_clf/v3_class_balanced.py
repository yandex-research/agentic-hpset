"""Materialized loss implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
import numpy as np
import torch
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


def _class_weights(dataset: Dataset, beta: float, device: torch.device) -> Tensor:
    y = np.asarray(dataset.y['train']).reshape(-1).astype(np.int64)
    n_classes = 2 if dataset.is_binclass else dataset.n_classes
    counts = np.maximum(np.bincount(y, minlength=n_classes), 1)
    effective = (1.0 - np.power(beta, counts)) / max(1.0 - beta, 1e-12)
    weights = 1.0 / effective
    weights *= n_classes / weights.sum()
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def make_loss_variant(
    strategy: str, params: dict[str, Any], symbol: str
) -> Callable[[Any, Tensor, Dataset], Tensor]:
    smoothing = float(
        params.get('smoothing', params.get('label_smoothing', params.get('eps', 0.1)))
    )
    gamma = float(params.get('gamma', params.get('focal_gamma', 2.0)))
    alpha = float(params.get('focal_alpha', params.get('alpha', 0.25)))
    beta = float(params.get('beta_cb', params.get('beta', 0.999)))

    def loss(output: Any, target: Tensor, dataset: Dataset) -> Tensor:
        prediction = prediction_tensor(output)
        if strategy == 'label_smoothing':
            if dataset.is_binclass:
                soft = target.float() * (1.0 - smoothing) + 0.5 * smoothing
                return F.binary_cross_entropy_with_logits(prediction.reshape(-1), soft)
            return F.cross_entropy(prediction, target.long(), label_smoothing=smoothing)
        if strategy == 'focal':
            if dataset.is_binclass:
                target_f = target.float()
                bce = F.binary_cross_entropy_with_logits(
                    prediction.reshape(-1), target_f, reduction='none'
                )
                prob = torch.sigmoid(prediction.reshape(-1))
                p_t = prob * target_f + (1.0 - prob) * (1.0 - target_f)
                alpha_t = alpha * target_f + (1.0 - alpha) * (1.0 - target_f)
                return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()
            ce = F.cross_entropy(prediction, target.long(), reduction='none')
            return ((1.0 - torch.exp(-ce)).pow(gamma) * ce).mean()
        if strategy in {'class_balanced', 'class_balanced_label_smoothing'}:
            weights = _class_weights(dataset, beta, prediction.device)
            if dataset.is_binclass:
                soft = target.float()
                if strategy.endswith('label_smoothing'):
                    soft = soft * (1.0 - smoothing) + 0.5 * smoothing
                sample_weight = weights[target.long()]
                raw = F.binary_cross_entropy_with_logits(
                    prediction.reshape(-1), soft, reduction='none'
                )
                return (raw * sample_weight).mean()
            return F.cross_entropy(
                prediction,
                target.long(),
                weight=weights,
                label_smoothing=(
                    smoothing if strategy.endswith('label_smoothing') else 0.0
                ),
            )
        return loss_v0(output, target, dataset)

    loss.__name__ = f'loss_{strategy}_{abs(hash(symbol)) & 0xFFFF:x}'
    loss.__doc__ = f'Ported from {symbol}; extracted classification loss: {strategy}.'
    return loss


_implementation = make_loss_variant(
    'class_balanced',
    {},
    'train.class_balanced.train_v5',
)


def loss_class_balanced_v3(*args: Any, **kwargs: Any) -> Any:
    return _implementation(*args, **kwargs)
