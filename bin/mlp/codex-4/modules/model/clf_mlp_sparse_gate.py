# ruff: noqa
"""Standalone implementation for ``build_mlp_v6``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class SparseGateMLP(nn.Module):
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
        self.gate_logits = nn.Parameter(torch.full((input_dim,), 2.0))
        n_blocks = int(params['n_blocks'])
        d_block = int(params['d_block'])
        dropout = float(params['dropout'])
        blocks: list[nn.Module] = []
        current = input_dim
        for _ in range(n_blocks):
            blocks.extend([nn.Linear(current, d_block), nn.ReLU(), nn.Dropout(dropout)])
            current = d_block
        self.backbone = nn.Sequential(*blocks)
        self.output = nn.Linear(current, output_dim)
        self.output_dim = output_dim

    def _features(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        pieces: list[Tensor] = []
        if x_num is not None and self.num_embedding is not None:
            pieces.append(self.num_embedding(x_num).flatten(1))
        if x_cat is not None and self.cat_embedding is not None:
            pieces.append(self.cat_embedding(x_cat))
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        x = self._features(x_num, x_cat) * torch.sigmoid(self.gate_logits)
        output = self.output(self.backbone(x))
        if self.output_dim == 1:
            return output.squeeze(-1)
        return output


def build_mlp_v6(
    dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    return SparseGateMLP(
        num_embedding=embedding_bundle['num_embedding'],
        cat_embedding=embedding_bundle['cat_embedding'],
        input_dim=int(embedding_bundle['input_dim']),
        params=trial_params,
        output_dim=1 if dataset_meta.is_binclass else int(dataset_meta.n_classes),
    ).to(device)
