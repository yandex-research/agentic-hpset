# ruff: noqa
"""Standalone implementation for ``build_model_v3``."""

from __future__ import annotations
from typing import Any
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import torch


class GatedBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.linear(x).chunk(2, dim=1)
        return self.dropout(F.silu(value) * torch.sigmoid(gate))


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


class GatedMLP(nn.Module):
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
        current = input_dim
        blocks = []
        for _ in range(int(params['n_blocks'])):
            blocks.append(
                GatedBlock(current, int(params['d_block']), float(params['dropout']))
            )
            current = int(params['d_block'])
        self.backbone = nn.Sequential(*blocks)
        self.output = nn.Linear(current, output_dim)

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        x = _assemble_input(self.num_embedding, self.cat_embedding, x_num, x_cat)
        output = self.output(self.backbone(x))
        return output.squeeze(-1) if self.output_dim == 1 else output


def _output_dim(*, heteroscedastic: bool = False) -> int:
    return 2 if heteroscedastic else 1


def build_model_v3(
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    return GatedMLP(
        embedding_bundle['num_embedding'],
        embedding_bundle['cat_embedding'],
        int(embedding_bundle['input_dim']),
        _output_dim(),
        trial_params,
    ).to(device)
