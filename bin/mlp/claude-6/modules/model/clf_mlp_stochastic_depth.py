# ruff: noqa
"""Standalone implementation for ``build_mlp_v3``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class StochasticDepthBlock(nn.Module):
    def __init__(
        self, dim: int, dropout: float, drop_prob: float, residual_scale: float
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.drop_prob = drop_prob
        self.residual_scale = residual_scale

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.dropout(self.act(self.linear(self.norm(x))))
        if self.training and self.drop_prob > 0.0:
            keep = torch.rand((), device=x.device) >= self.drop_prob
            if not keep:
                return residual
            return residual + self.residual_scale * out / max(
                1.0 - self.drop_prob, 1e-06
            )
        return residual + self.residual_scale * out


class MLPStochasticDepthModel(nn.Module):
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
        max_drop = float(params.get('stochastic_depth_max_p', 0.2))
        self.input_block = nn.Sequential(
            nn.Linear(input_dim, d_block), nn.ReLU(), nn.Dropout(dropout)
        )
        residual_blocks: list[nn.Module] = []
        n_residual = max(n_blocks - 1, 0)
        for i in range(n_residual):
            drop_prob = max_drop * (i + 1) / max(n_residual, 1)
            residual_blocks.append(
                StochasticDepthBlock(
                    d_block,
                    dropout,
                    drop_prob,
                    1.0 / max(n_residual, 1) ** 0.5,
                )
            )
        self.residual_blocks = nn.Sequential(*residual_blocks)
        self.final_norm = nn.LayerNorm(d_block)
        self.output = nn.Linear(d_block, output_dim)
        self.output_dim = output_dim

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        pieces: list[Tensor] = []
        if x_num is not None and self.num_embedding is not None:
            pieces.append(self.num_embedding(x_num).flatten(1))
        if x_cat is not None and self.cat_embedding is not None:
            pieces.append(self.cat_embedding(x_cat))
        x = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
        x = self.input_block(x)
        x = self.residual_blocks(x)
        output = self.output(self.final_norm(x))
        if self.output_dim == 1:
            return output.squeeze(-1)
        return output


def build_mlp_v3(
    dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    model = MLPStochasticDepthModel(
        num_embedding=embedding_bundle['num_embedding'],
        cat_embedding=embedding_bundle['cat_embedding'],
        input_dim=int(embedding_bundle['input_dim']),
        params=trial_params,
        output_dim=1 if dataset_meta.is_binclass else int(dataset_meta.n_classes),
    )
    return model.to(device)
