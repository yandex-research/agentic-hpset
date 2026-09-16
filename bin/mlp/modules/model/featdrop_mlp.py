from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class FeatureDropout(nn.Module):
    """Drops entire input features (columns) during training.

    Unlike standard dropout which zeros individual neurons in hidden layers,
    this zeros entire feature channels at the input level. This forces the
    model to build redundant representations and not over-rely on any single
    feature — similar to feature bagging in random forests.

    Drop probability is set to min(dropout, 0.3) to avoid dropping too many
    features at once.
    """

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        self.p = min(p, 0.3)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        # mask: (1, n_features) — same mask for entire batch
        mask = torch.bernoulli(
            torch.full((1, x.shape[1]), 1.0 - self.p, device=x.device)
        )
        return x * mask / (1.0 - self.p)


class FeatureDropoutMLPModel(nn.Module):
    """MLP with channel-wise feature dropout at the input.

    Applies feature-level dropout before the backbone, forcing the model
    to learn robust, redundant feature representations. The backbone itself
    uses standard neuron-level dropout.
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
        dropout = float(params["dropout"])
        self.feature_dropout = FeatureDropout(p=dropout)
        n_blocks = int(params["n_blocks"])
        d_block = int(params["d_block"])
        blocks: list[nn.Module] = []
        current = input_dim
        for _ in range(n_blocks):
            blocks.extend([nn.Linear(current, d_block), nn.ReLU(), nn.Dropout(dropout)])
            current = d_block
        self.backbone = nn.Sequential(*blocks)
        self.output = nn.Linear(current, output_dim)

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        pieces: list[Tensor] = []
        if x_num is not None and self.num_embedding is not None:
            pieces.append(self.num_embedding(x_num).flatten(1))
        if x_cat is not None and self.cat_embedding is not None:
            pieces.append(self.cat_embedding(x_cat))
        x = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
        x = self.feature_dropout(x)
        return self.output(self.backbone(x)).squeeze(-1)


def build_mlp_v4(
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
    *,
    output_dim: int,
) -> nn.Module:
    model = FeatureDropoutMLPModel(
        num_embedding=embedding_bundle["num_embedding"],
        cat_embedding=embedding_bundle["cat_embedding"],
        input_dim=int(embedding_bundle["input_dim"]),
        params=trial_params,
        output_dim=output_dim,
    )
    return model.to(device)
