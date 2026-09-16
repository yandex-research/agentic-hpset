from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel recalibration.

    Computes per-channel importance weights via:
    squeeze (global avg) → compress (linear to d//r) → ReLU → expand (linear to d) → sigmoid
    Then rescales the input channels by these learned weights.

    This lets the model dynamically emphasize useful hidden dimensions and
    suppress less informative ones on a per-sample basis. Adapted from
    SENet (Hu et al., CVPR 2018) for tabular hidden features.
    """

    def __init__(self, d_block: int, reduction: int = 4) -> None:
        super().__init__()
        d_squeezed = max(d_block // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(d_block, d_squeezed),
            nn.ReLU(),
            nn.Linear(d_squeezed, d_block),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, d_block)
        # "Squeeze": for 1D features, x itself acts as the global descriptor
        scale = self.fc(x)  # (batch, d_block)
        return x * scale


class SEMLPModel(nn.Module):
    """MLP with Squeeze-and-Excitation blocks after each layer.

    Each block: Linear → ReLU → Dropout → SE recalibration.
    The SE block adds per-sample, per-channel gating that recalibrates
    which hidden dimensions are important. Cheap overhead (~2*d_block//4
    parameters per block) for potentially significant expressiveness gains.
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
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(n_blocks):
            layers.append(nn.Linear(current, d_block))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(SEBlock(d_block))
            current = d_block
        self.backbone = nn.Sequential(*layers)
        self.output = nn.Linear(current, output_dim)

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        pieces: list[Tensor] = []
        if x_num is not None and self.num_embedding is not None:
            pieces.append(self.num_embedding(x_num).flatten(1))
        if x_cat is not None and self.cat_embedding is not None:
            pieces.append(self.cat_embedding(x_cat))
        x = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
        return self.output(self.backbone(x)).squeeze(-1)


def build_mlp_v6(
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
    *,
    output_dim: int,
) -> nn.Module:
    model = SEMLPModel(
        num_embedding=embedding_bundle["num_embedding"],
        cat_embedding=embedding_bundle["cat_embedding"],
        input_dim=int(embedding_bundle["input_dim"]),
        params=trial_params,
        output_dim=output_dim,
    )
    return model.to(device)
