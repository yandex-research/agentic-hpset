# ruff: noqa
"""Standalone implementation for ``build_mlp_v1``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class _CrossLayer(nn.Module):
    def __init__(self, input_dim: int, low_rank: int) -> None:
        super().__init__()
        rank = min(low_rank, input_dim)
        self.U = nn.Parameter(torch.empty(input_dim, rank))
        self.V = nn.Parameter(torch.empty(rank, input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        bound = input_dim ** (-0.5)
        nn.init.uniform_(self.U, -bound, bound)
        nn.init.uniform_(self.V, -bound, bound)

    def forward(self, x0: Tensor, xl: Tensor) -> Tensor:
        proj = xl @ self.U @ self.V + self.bias
        return x0 * proj + xl


_LOW_RANK = 64
_N_CROSS_LAYERS = 2


class DCNv2Model(nn.Module):
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
        n_blocks = int(params['n_blocks'])
        d_block = int(params['d_block'])
        dropout = float(params['dropout'])
        self.cross_layers = nn.ModuleList(
            [_CrossLayer(input_dim, _LOW_RANK) for _ in range(_N_CROSS_LAYERS)]
        )
        blocks: list[nn.Module] = []
        current = input_dim * 2
        for _ in range(n_blocks):
            blocks.extend([nn.Linear(current, d_block), nn.ReLU(), nn.Dropout(dropout)])
            current = d_block
        self.backbone = nn.Sequential(*blocks)
        self.output = nn.Linear(current, 1)

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        pieces: list[Tensor] = []
        if x_num is not None and self.num_embedding is not None:
            pieces.append(self.num_embedding(x_num).flatten(1))
        if x_cat is not None and self.cat_embedding is not None:
            pieces.append(self.cat_embedding(x_cat))
        x0 = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
        xl = x0
        for layer in self.cross_layers:
            xl = layer(x0, xl)
        x = torch.cat([x0, xl], dim=1)
        return self.output(self.backbone(x)).squeeze(-1)


def build_mlp_v1(
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    model = DCNv2Model(
        num_embedding=embedding_bundle['num_embedding'],
        cat_embedding=embedding_bundle['cat_embedding'],
        input_dim=int(embedding_bundle['input_dim']),
        params=trial_params,
    )
    return model.to(device)
