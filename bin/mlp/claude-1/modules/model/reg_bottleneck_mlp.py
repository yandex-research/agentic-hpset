# ruff: noqa
"""Standalone implementation for ``build_bottleneck_mlp``."""

from __future__ import annotations
from typing import Any
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import torch


class BottleneckBlock(nn.Module):
    def __init__(self, d_block: int, expand: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_block)
        self.fc1 = nn.Linear(d_block, d_block * expand)
        self.fc2 = nn.Linear(d_block * expand, d_block)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return residual + x


class BottleneckMLPModel(nn.Module):
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
        self.input_proj = nn.Linear(input_dim, d_block)
        self.blocks = nn.ModuleList(
            [BottleneckBlock(d_block, expand=4) for _ in range(n_blocks)]
        )
        self.output_norm = nn.LayerNorm(d_block)
        self.output = nn.Linear(d_block, 1)

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        pieces: list[Tensor] = []
        if x_num is not None and self.num_embedding is not None:
            pieces.append(self.num_embedding(x_num).flatten(1))
        if x_cat is not None and self.cat_embedding is not None:
            pieces.append(self.cat_embedding(x_cat))
        x = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output(self.output_norm(x)).squeeze(-1)


def build_bottleneck_mlp(
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    model = BottleneckMLPModel(
        num_embedding=embedding_bundle['num_embedding'],
        cat_embedding=embedding_bundle['cat_embedding'],
        input_dim=int(embedding_bundle['input_dim']),
        params=trial_params,
    )
    return model.to(device)
