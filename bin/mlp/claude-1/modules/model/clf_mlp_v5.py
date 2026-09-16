# ruff: noqa
"""Standalone implementation for ``build_mlp_v5``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class StochasticDepthBlock(nn.Module):
    """Residual MLP block with per-sample DropPath.

    The block computes y = x + drop_path(f(x)) where f is LayerNorm -> Linear -> ReLU -> Dropout
    -> Linear. During training a Bernoulli mask per sample with prob p is applied
    to f(x), and the output is rescaled by 1/(1-p) to keep expectations.
    Implicit ensembling within a single MLP.
    """

    def __init__(self, d_model: int, dropout: float, drop_path: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model, d_model)
        self.drop_path = drop_path

    def forward(self, x: Tensor) -> Tensor:
        h = self.fc2(self.dropout(self.act(self.fc1(self.norm(x)))))
        if self.training and self.drop_path > 0.0:
            keep = 1.0 - self.drop_path
            mask = torch.empty(
                x.shape[0], 1, device=x.device, dtype=x.dtype
            ).bernoulli_(keep)
            h = h * (mask / keep)
        return x + h


class StochasticDepthMLPModel(nn.Module):
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
        max_drop_path = 0.2
        self.input_proj = nn.Linear(input_dim, d_block)
        blocks: list[nn.Module] = []
        for i in range(n_blocks):
            dp = max_drop_path * (i + 1) / max(n_blocks, 1)
            blocks.append(StochasticDepthBlock(d_block, dropout, dp))
        self.backbone = nn.Sequential(*blocks)
        self.head_norm = nn.LayerNorm(d_block)
        self.output = nn.Linear(d_block, output_dim)
        self.output_dim = output_dim

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        pieces: list[Tensor] = []
        if x_num is not None and self.num_embedding is not None:
            pieces.append(self.num_embedding(x_num).flatten(1))
        if x_cat is not None and self.cat_embedding is not None:
            pieces.append(self.cat_embedding(x_cat))
        x = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
        x = self.input_proj(x)
        x = self.backbone(x)
        output = self.output(self.head_norm(x))
        if self.output_dim == 1:
            return output.squeeze(-1)
        return output


def build_mlp_v5(
    dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    model = StochasticDepthMLPModel(
        num_embedding=embedding_bundle['num_embedding'],
        cat_embedding=embedding_bundle['cat_embedding'],
        input_dim=int(embedding_bundle['input_dim']),
        params=trial_params,
        output_dim=1 if dataset_meta.is_binclass else int(dataset_meta.n_classes),
    )
    return model.to(device)
