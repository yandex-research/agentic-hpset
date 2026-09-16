from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class BottleneckBlock(nn.Module):
    """Bottleneck block: wide → narrow → wide.

    Compresses to d_block//4, applies activation, then expands back.
    Forces the model to learn a compact intermediate representation,
    acting as implicit regularization (information bottleneck).
    """

    def __init__(self, d_block: int, dropout: float) -> None:
        super().__init__()
        d_narrow = max(d_block // 4, 8)
        self.block = nn.Sequential(
            nn.Linear(d_block, d_narrow),
            nn.ReLU(),
            nn.Linear(d_narrow, d_block),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class BottleneckMLPModel(nn.Module):
    """MLP with bottleneck blocks.

    Each block compresses the representation through a narrow layer (d_block//4)
    before expanding back. This architecture forces the model to form compact
    internal representations at each stage. Inspired by ResNet bottleneck blocks
    and the information bottleneck principle (Tishby et al.).
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
            *[BottleneckBlock(d_block, dropout) for _ in range(n_blocks)]
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


def build_mlp_v5(
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
    *,
    output_dim: int,
) -> nn.Module:
    model = BottleneckMLPModel(
        num_embedding=embedding_bundle["num_embedding"],
        cat_embedding=embedding_bundle["cat_embedding"],
        input_dim=int(embedding_bundle["input_dim"]),
        params=trial_params,
        output_dim=output_dim,
    )
    return model.to(device)
