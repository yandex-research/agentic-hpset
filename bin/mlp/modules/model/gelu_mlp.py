from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class GELUMLPModel(nn.Module):
    """MLP with GELU activation and LayerNorm.

    Replaces ReLU with GELU (smoother gradients, avoids dead neurons) and adds
    LayerNorm before each linear layer for training stability. This is the
    standard block design used in modern transformer-based tabular models.
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
        blocks: list[nn.Module] = []
        current = input_dim
        for _ in range(n_blocks):
            blocks.extend([
                nn.LayerNorm(current),
                nn.Linear(current, d_block),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current = d_block
        self.backbone = nn.Sequential(*blocks)
        self.output = nn.Sequential(
            nn.LayerNorm(current),
            nn.Linear(current, output_dim),
        )

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        pieces: list[Tensor] = []
        if x_num is not None and self.num_embedding is not None:
            pieces.append(self.num_embedding(x_num).flatten(1))
        if x_cat is not None and self.cat_embedding is not None:
            pieces.append(self.cat_embedding(x_cat))
        x = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
        return self.output(self.backbone(x)).squeeze(-1)


def build_mlp_v2(
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
    *,
    output_dim: int,
) -> nn.Module:
    model = GELUMLPModel(
        num_embedding=embedding_bundle["num_embedding"],
        cat_embedding=embedding_bundle["cat_embedding"],
        input_dim=int(embedding_bundle["input_dim"]),
        params=trial_params,
        output_dim=output_dim,
    )
    return model.to(device)
