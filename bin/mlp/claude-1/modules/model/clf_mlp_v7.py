# ruff: noqa
"""Standalone implementation for ``build_mlp_v7``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class SEFeatureGate(nn.Module):
    """Squeeze-Excitation gate over flattened input features.

    Pools the input via mean and max along the feature dim, runs through
    a small bottleneck MLP, and produces a per-input-unit sigmoid gate.
    Multiplies elementwise with the input. Cheap, content-aware feature
    reweighting prepended to the standard MLP.
    """

    def __init__(self, input_dim: int, reduction: int = 8) -> None:
        super().__init__()
        bottleneck = max(8, input_dim // reduction)
        self.fc1 = nn.Linear(2, bottleneck)
        self.fc2 = nn.Linear(bottleneck, input_dim)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        s = torch.stack([x.mean(dim=-1), x.amax(dim=-1)], dim=-1)
        gate = torch.sigmoid(self.fc2(self.act(self.fc1(s))))
        return x * (1.0 + gate)


class SEMLPModel(nn.Module):
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
        self.se = SEFeatureGate(input_dim)
        blocks: list[nn.Module] = []
        current = input_dim
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
        x = self.se(x)
        output = self.output(self.backbone(x))
        if self.output_dim == 1:
            return output.squeeze(-1)
        return output


def build_mlp_v7(
    dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    model = SEMLPModel(
        num_embedding=embedding_bundle['num_embedding'],
        cat_embedding=embedding_bundle['cat_embedding'],
        input_dim=int(embedding_bundle['input_dim']),
        params=trial_params,
        output_dim=1 if dataset_meta.is_binclass else int(dataset_meta.n_classes),
    )
    return model.to(device)
