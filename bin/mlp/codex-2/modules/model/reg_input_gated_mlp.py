# ruff: noqa
"""Standalone implementation for ``build_input_gated_mlp``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


def _features(
    num_embedding: nn.Module | None,
    cat_embedding: nn.Module | None,
    x_num: Tensor | None,
    x_cat: Tensor | None,
) -> Tensor:
    pieces: list[Tensor] = []
    if x_num is not None and num_embedding is not None:
        pieces.append(num_embedding(x_num).flatten(1))
    if x_cat is not None and cat_embedding is not None:
        pieces.append(cat_embedding(x_cat))
    if not pieces:
        raise ValueError('Model received no input features.')
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)


def _plain_backbone(
    input_dim: int, params: dict[str, Any]
) -> tuple[nn.Sequential, int]:
    n_blocks = int(params['n_blocks'])
    d_block = int(params['d_block'])
    dropout = float(params['dropout'])
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(n_blocks):
        layers.extend([nn.Linear(current, d_block), nn.ReLU(), nn.Dropout(dropout)])
        current = d_block
    return (nn.Sequential(*layers), current)


class InputGatedMLP(nn.Module):
    def __init__(
        self,
        num_embedding: nn.Module | None,
        cat_embedding: nn.Module | None,
        input_dim: int,
        params: dict[str, Any],
    ) -> None:
        super().__init__()
        if input_dim == 0:
            raise ValueError('Model has no input features after preprocessing.')
        self.num_embedding = num_embedding
        self.cat_embedding = cat_embedding
        self.input_gate = nn.Parameter(torch.zeros(input_dim))
        self.backbone, current = _plain_backbone(input_dim, params)
        self.output = nn.Linear(current, 1)

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        x = _features(self.num_embedding, self.cat_embedding, x_num, x_cat)
        x = x * (2.0 * torch.sigmoid(self.input_gate))[None]
        return self.output(self.backbone(x)).squeeze(-1)


def _build(
    cls: type[nn.Module],
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    model = cls(
        num_embedding=embedding_bundle['num_embedding'],
        cat_embedding=embedding_bundle['cat_embedding'],
        input_dim=int(embedding_bundle['input_dim']),
        params=trial_params,
    )
    return model.to(device)


def build_input_gated_mlp(
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    return _build(InputGatedMLP, trial_params, embedding_bundle, device)
