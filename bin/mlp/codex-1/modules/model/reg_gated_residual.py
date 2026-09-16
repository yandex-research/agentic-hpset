# ruff: noqa
"""Standalone implementation for ``build_mlp_v2``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class ResidualBlock(nn.Module):
    def __init__(self, d_block: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_block)
        self.linear1 = nn.Linear(d_block, d_block)
        self.linear2 = nn.Linear(d_block, d_block)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        h = torch.relu(self.linear1(h))
        h = self.dropout(self.linear2(h))
        return x + self.gate.tanh() * h


class GatedResidualMLP(nn.Module):
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
        d_block = int(params['d_block'])
        dropout = float(params['dropout'])
        self.input = nn.Linear(input_dim, d_block)
        self.blocks = nn.Sequential(
            *[ResidualBlock(d_block, dropout) for _ in range(int(params['n_blocks']))]
        )
        self.output = nn.Linear(d_block, 1)

    def _encode(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        pieces: list[Tensor] = []
        if x_num is not None and self.num_embedding is not None:
            pieces.append(self.num_embedding(x_num).flatten(1))
        if x_cat is not None and self.cat_embedding is not None:
            pieces.append(self.cat_embedding(x_cat))
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        x = torch.relu(self.input(self._encode(x_num, x_cat)))
        return self.output(self.blocks(x)).squeeze(-1)


def build_mlp_v2(
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    model = GatedResidualMLP(
        num_embedding=embedding_bundle['num_embedding'],
        cat_embedding=embedding_bundle['cat_embedding'],
        input_dim=int(embedding_bundle['input_dim']),
        params=trial_params,
    )
    return model.to(device)
