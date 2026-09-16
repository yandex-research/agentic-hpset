# ruff: noqa
"""Standalone implementation for ``build_mlp_v11``."""

from __future__ import annotations
from typing import Any
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import torch


class DropConnectLinear(nn.Linear):
    def __init__(self, input_dim: int, output_dim: int, drop_prob: float) -> None:
        super().__init__(input_dim, output_dim)
        self.drop_prob = max(0.0, min(float(drop_prob), 0.5))

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight
        if self.training and self.drop_prob > 0.0:
            weight = F.dropout(weight, p=self.drop_prob, training=True)
        return F.linear(x, weight, self.bias)


class DropConnectMLPModel(nn.Module):
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
        blocks: list[nn.Module] = []
        current = input_dim
        for _ in range(n_blocks):
            blocks.extend(
                [
                    DropConnectLinear(current, d_block, dropout * 0.5),
                    nn.ReLU(),
                    nn.Dropout(dropout * 0.5),
                ]
            )
            current = d_block
        self.backbone = nn.Sequential(*blocks)
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


def build_mlp_v11(
    dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    model = DropConnectMLPModel(
        num_embedding=embedding_bundle['num_embedding'],
        cat_embedding=embedding_bundle['cat_embedding'],
        input_dim=int(embedding_bundle['input_dim']),
        params=trial_params,
        output_dim=1 if dataset_meta.is_binclass else int(dataset_meta.n_classes),
    )
    return model.to(device)
