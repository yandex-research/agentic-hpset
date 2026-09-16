# ruff: noqa
"""Standalone implementation for ``build_mlp_v4``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class LowRankCrossLayer(nn.Module):
    def __init__(self, input_dim: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(input_dim, rank, bias=False)
        self.up = nn.Linear(rank, input_dim, bias=True)
        nn.init.zeros_(self.up.bias)

    def forward(self, x0: Tensor, x: Tensor) -> Tensor:
        return x + x0 * self.up(self.down(x))


class _FeatureEncoder(nn.Module):
    def __init__(
        self, num_embedding: nn.Module | None, cat_embedding: nn.Module | None
    ) -> None:
        super().__init__()
        self.num_embedding = num_embedding
        self.cat_embedding = cat_embedding

    def encode(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        pieces: list[Tensor] = []
        if x_num is not None and self.num_embedding is not None:
            pieces.append(self.num_embedding(x_num).flatten(1))
        if x_cat is not None and self.cat_embedding is not None:
            pieces.append(self.cat_embedding(x_cat))
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)


def _mlp_blocks(input_dim: int, params: dict[str, Any]) -> tuple[nn.Sequential, int]:
    n_blocks = int(params['n_blocks'])
    d_block = int(params['d_block'])
    dropout = float(params['dropout'])
    blocks: list[nn.Module] = []
    current = input_dim
    for _ in range(n_blocks):
        blocks.extend([nn.Linear(current, d_block), nn.ReLU(), nn.Dropout(dropout)])
        current = d_block
    return (nn.Sequential(*blocks), current)


class CrossMLP(_FeatureEncoder):
    def __init__(
        self,
        num_embedding: nn.Module | None,
        cat_embedding: nn.Module | None,
        input_dim: int,
        params: dict[str, Any],
    ) -> None:
        super().__init__(num_embedding, cat_embedding)
        if input_dim == 0:
            raise ValueError('Model has no input features after preprocessing.')
        rank = max(4, min(int(params['d_block']) // 4, 64, input_dim))
        n_cross = max(1, min(int(params['n_blocks']), 3))
        self.cross_layers = nn.ModuleList(
            [LowRankCrossLayer(input_dim, rank) for _ in range(n_cross)]
        )
        self.deep, current = _mlp_blocks(input_dim, params)
        self.cross_output = nn.Linear(input_dim, 1)
        self.deep_output = nn.Linear(current, 1)

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        x0 = self.encode(x_num, x_cat)
        x_cross = x0
        for layer in self.cross_layers:
            x_cross = layer(x0, x_cross)
        return (self.deep_output(self.deep(x0)) + self.cross_output(x_cross)).squeeze(
            -1
        )


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


def build_mlp_v4(
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    return _build(CrossMLP, trial_params, embedding_bundle, device)
