# ruff: noqa
"""Standalone implementation for ``build_mlp_v5``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class FactorizationMachineLayer(nn.Module):
    """Factorization-machine 2nd-order interaction summary.

    Projects the input through k learned vectors V_i and outputs a k-dim summary
    of pairwise interactions: 0.5 * ((sum V_i x_i)^2 - sum (V_i x_i)^2). The summary
    is concatenated with the original input to be consumed by the MLP backbone.
    """

    def __init__(self, d_in: int, k: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_in, k) * (1.0 / d_in) ** 0.5)
        self.k = k

    def forward(self, x: Tensor) -> Tensor:
        proj = x @ self.weight
        sq_sum = proj.pow(2)
        sum_sq = x.pow(2) @ self.weight.pow(2)
        return 0.5 * (sq_sum - sum_sq)


class FMMLPModel(nn.Module):
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
        k = int(params.get('fm_rank', 16))
        self.fm = FactorizationMachineLayer(input_dim, k)
        augmented_dim = input_dim + k
        blocks: list[nn.Module] = []
        current = augmented_dim
        for _ in range(n_blocks):
            blocks.extend([nn.Linear(current, d_block), nn.ReLU(), nn.Dropout(dropout)])
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
        fm = self.fm(x)
        x = torch.cat([x, fm], dim=1)
        out = self.output(self.backbone(x))
        if self.output_dim == 1:
            return out.squeeze(-1)
        return out


def build_mlp_v5(
    dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    model = FMMLPModel(
        num_embedding=embedding_bundle['num_embedding'],
        cat_embedding=embedding_bundle['cat_embedding'],
        input_dim=int(embedding_bundle['input_dim']),
        params=trial_params,
        output_dim=1 if dataset_meta.is_binclass else int(dataset_meta.n_classes),
    )
    return model.to(device)
