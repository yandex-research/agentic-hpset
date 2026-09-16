# ruff: noqa
# Source: checked-in agent-ablation source
"""Standalone implementation for ``build_mlp_v0``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class MLPModel(nn.Module):
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
        self.input_dim = input_dim
        self.output_dim = output_dim
        blocks: list[nn.Module] = []
        current = input_dim
        for _ in range(int(params['n_blocks'])):
            blocks.extend(
                [
                    nn.Linear(current, int(params['d_block'])),
                    nn.ReLU(),
                    nn.Dropout(float(params['dropout'])),
                ]
            )
            current = int(params['d_block'])
        self.backbone = nn.Sequential(*blocks)
        self.output = nn.Linear(current, output_dim)
        self.model_name = 'mlp'

    def encode_input(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        pieces: list[Tensor] = []
        if x_num is not None and self.num_embedding is not None:
            pieces.append(self.num_embedding(x_num).flatten(1))
        if x_cat is not None and self.cat_embedding is not None:
            pieces.append(self.cat_embedding(x_cat).flatten(1))
        if not pieces:
            raise RuntimeError('Model received no input features.')
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        output = self.output(self.backbone(self.encode_input(x_num, x_cat)))
        return output.squeeze(-1) if self.output_dim == 1 else output


def build_mlp_v0(
    dataset: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    output_dim = int(dataset.n_classes) if dataset.is_multiclass else 1
    return MLPModel(
        embedding_bundle['num_embedding'],
        embedding_bundle['cat_embedding'],
        int(embedding_bundle['input_dim']),
        trial_params,
        output_dim,
    ).to(device)
