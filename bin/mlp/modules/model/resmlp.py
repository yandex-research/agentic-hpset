from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class ResBlock(nn.Module):
    def __init__(self, d_block: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(d_block, d_block),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_block, d_block),
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.activation(x + self.block(x)))


class ResMLPModel(nn.Module):
    """MLP with residual skip connections.

    The first layer projects to d_block, then n_blocks ResBlocks are applied.
    Each ResBlock has a two-layer sub-network with an additive skip connection.
    This improves gradient flow and enables training deeper networks effectively.
    """

    def __init__(
        self,
        num_embedding: nn.Module | None,
        cat_embedding: nn.Module | None,
        input_dim: int,
        params: dict[str, Any],
        *,
        output_dim: int,
    ) -> None:
        super().__init__()
        if input_dim == 0:
            raise ValueError("Model has no input features after preprocessing.")
        self.num_embedding = num_embedding
        self.cat_embedding = cat_embedding
        n_blocks = int(params["n_blocks"])
        d_block = int(params["d_block"])
        dropout = float(params["dropout"])
        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_block),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            *[ResBlock(d_block, dropout) for _ in range(n_blocks)]
        )
        self.output = nn.Linear(d_block, output_dim)

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        pieces: list[Tensor] = []
        if x_num is not None and self.num_embedding is not None:
            pieces.append(self.num_embedding(x_num).flatten(1))
        if x_cat is not None and self.cat_embedding is not None:
            pieces.append(self.cat_embedding(x_cat))
        x = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
        x = self.projection(x)
        x = self.blocks(x)
        return self.output(x).squeeze(-1)


def build_mlp_v1(
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
    *,
    output_dim: int,
) -> nn.Module:
    model = ResMLPModel(
        num_embedding=embedding_bundle["num_embedding"],
        cat_embedding=embedding_bundle["cat_embedding"],
        input_dim=int(embedding_bundle["input_dim"]),
        params=trial_params,
        output_dim=output_dim,
    )
    return model.to(device)
