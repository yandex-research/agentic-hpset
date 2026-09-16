# ruff: noqa
"""Standalone implementation for ``build_model_v7``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


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


class ResidualMLP(nn.Module):
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
        d_block = int(params['d_block'])
        dropout = float(params['dropout'])
        self.input_projection = nn.Linear(input_dim, d_block)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_block),
                    nn.Linear(d_block, d_block),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for _ in range(int(params['n_blocks']))
            ]
        )
        self.output = nn.Linear(d_block, output_dim)

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        x = self.input_projection(
            _assemble_input(self.num_embedding, self.cat_embedding, x_num, x_cat)
        )
        for block in self.blocks:
            x = x + block(x)
        output = self.output(x)
        return output.squeeze(-1) if self.output_dim == 1 else output


def _output_dim(*, heteroscedastic: bool = False) -> int:
    return 2 if heteroscedastic else 1


def build_model_v7(
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    return ResidualMLP(
        embedding_bundle['num_embedding'],
        embedding_bundle['cat_embedding'],
        int(embedding_bundle['input_dim']),
        _output_dim(),
        trial_params,
    ).to(device)
