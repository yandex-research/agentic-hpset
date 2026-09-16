# ruff: noqa
"""Standalone implementation for ``build_mlp_v8``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch

_HYPER_RANK = 4


class _LowRankHyperScale(nn.Module):
    def __init__(self, input_dim: int, summary_dim: int, rank: int) -> None:
        super().__init__()
        self.U = nn.Parameter(torch.empty(summary_dim, rank))
        self.V = nn.Parameter(torch.empty(rank, input_dim))
        self.bias = nn.Parameter(torch.ones(input_dim))
        bound = (summary_dim * rank) ** (-0.5)
        nn.init.uniform_(self.U, -bound, bound)
        nn.init.zeros_(self.V)

    def forward(self, summary: Tensor) -> Tensor:
        return self.bias + summary @ self.U @ self.V


class LowRankHypernetMLP(nn.Module):
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
        summary_dim = 8
        self.summary_proj = nn.Linear(input_dim, summary_dim)
        self.hyper = _LowRankHyperScale(input_dim, summary_dim, _HYPER_RANK)
        blocks: list[nn.Module] = []
        current = input_dim
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
        x = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
        summary = torch.tanh(self.summary_proj(x))
        scales = self.hyper(summary)
        scaled = x * scales
        return self.output(self.backbone(scaled)).squeeze(-1)


def build_mlp_v8(
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    model = LowRankHypernetMLP(
        num_embedding=embedding_bundle['num_embedding'],
        cat_embedding=embedding_bundle['cat_embedding'],
        input_dim=int(embedding_bundle['input_dim']),
        params=trial_params,
    )
    return model.to(device)
