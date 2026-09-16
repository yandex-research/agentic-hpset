# ruff: noqa
"""Standalone implementation for ``build_model_v4``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


def _assemble_input(
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
        raise RuntimeError('Model received no input features.')
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)


def _mlp_layers(
    input_dim: int, params: dict[str, Any], *, norm: bool = False
) -> tuple[nn.Sequential, int]:
    n_blocks = int(params['n_blocks'])
    d_block = int(params['d_block'])
    dropout = float(params['dropout'])
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(n_blocks):
        layers.append(nn.Linear(current, d_block))
        if norm:
            layers.append(nn.LayerNorm(d_block))
        layers.extend([nn.ReLU(), nn.Dropout(dropout)])
        current = d_block
    return (nn.Sequential(*layers), current)


class LinearSkipMLP(nn.Module):
    def __init__(
        self,
        num_embedding: nn.Module | None,
        cat_embedding: nn.Module | None,
        input_dim: int,
        output_dim: int,
        params: dict[str, Any],
    ) -> None:
        super().__init__()
        self.num_embedding = num_embedding
        self.cat_embedding = cat_embedding
        self.output_dim = output_dim
        self.backbone, current = _mlp_layers(input_dim, params)
        self.output = nn.Linear(current, output_dim)
        self.linear_skip = nn.Linear(input_dim, output_dim)

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        x = _assemble_input(self.num_embedding, self.cat_embedding, x_num, x_cat)
        output = self.output(self.backbone(x)) + self.linear_skip(x)
        return output.squeeze(-1) if self.output_dim == 1 else output


class SparseGateMLP(LinearSkipMLP):
    def __init__(
        self,
        num_embedding: nn.Module | None,
        cat_embedding: nn.Module | None,
        input_dim: int,
        output_dim: int,
        params: dict[str, Any],
    ) -> None:
        super().__init__(num_embedding, cat_embedding, input_dim, output_dim, params)
        self.feature_gate = nn.Parameter(torch.zeros(input_dim))

    def regularization_loss(self) -> Tensor:
        return 0.0001 * torch.sigmoid(self.feature_gate).mean()

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        x = _assemble_input(self.num_embedding, self.cat_embedding, x_num, x_cat)
        x = x * (2.0 * torch.sigmoid(self.feature_gate))
        output = self.output(self.backbone(x)) + self.linear_skip(x)
        return output.squeeze(-1) if self.output_dim == 1 else output


def _output_dim(*, heteroscedastic: bool = False) -> int:
    return 2 if heteroscedastic else 1


def build_model_v4(
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    return SparseGateMLP(
        embedding_bundle['num_embedding'],
        embedding_bundle['cat_embedding'],
        int(embedding_bundle['input_dim']),
        _output_dim(),
        trial_params,
    ).to(device)
