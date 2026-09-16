# ruff: noqa
"""Standalone implementation for ``build_mlp_v4``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class ResidualGateBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip = (
            nn.Identity()
            if input_dim == output_dim
            else nn.Linear(input_dim, output_dim)
        )
        self.gate = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        transformed = self.dropout(self.activation(self.proj(x)))
        gate = torch.sigmoid(self.gate(x))
        return self.norm(self.skip(x) + gate * transformed)


class ResidualGateMLPModel(nn.Module):
    def __init__(
        self,
        num_embedding: nn.Module | None,
        cat_embedding: nn.Module | None,
        input_dim: int,
        params: dict[str, Any],
        output_dim: int,
    ) -> None:
        super().__init__()
        if input_dim == 0:
            raise ValueError('Model has no input features after preprocessing.')
        self.num_embedding = num_embedding
        self.cat_embedding = cat_embedding
        n_blocks = int(params['n_blocks'])
        d_block = int(params['d_block'])
        dropout = float(params['dropout'])
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(n_blocks):
            layers.append(ResidualGateBlock(current, d_block, dropout))
            current = d_block
        self.backbone = nn.Sequential(*layers)
        self.output = nn.Linear(current, output_dim)
        self.output_dim = output_dim

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        pieces: list[Tensor] = []
        if x_num is not None and self.num_embedding is not None:
            pieces.append(self.num_embedding(x_num).flatten(1))
        if x_cat is not None and self.cat_embedding is not None:
            pieces.append(self.cat_embedding(x_cat))
        x = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
        output = self.output(self.backbone(x))
        if self.output_dim == 1:
            return output.squeeze(-1)
        return output


def build_mlp_v4(
    dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    model = ResidualGateMLPModel(
        num_embedding=embedding_bundle['num_embedding'],
        cat_embedding=embedding_bundle['cat_embedding'],
        input_dim=int(embedding_bundle['input_dim']),
        params=trial_params,
        output_dim=1 if dataset_meta.is_binclass else int(dataset_meta.n_classes),
    )
    return model.to(device)
