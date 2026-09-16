from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ._variant import build_variant
from .tabm import ElementwiseAffine, EnsembleView


class SEBlock(nn.Module):
    """Squeeze-and-Excite feed-forward block.

    Linear -> ReLU -> channel-gated dropout. The gate is a small MLP over the
    block-internal channel mean (squeeze), producing a per-channel sigmoid
    weight (excite) that re-scales the activations before dropout.
    """

    def __init__(self, d_in: int, d_block: int, dropout: float, se_reduction: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_in, d_block)
        self.dropout = nn.Dropout(dropout)
        d_reduced = max(1, d_block // se_reduction)
        self.gate = nn.Sequential(
            nn.Linear(d_block, d_reduced),
            nn.ReLU(),
            nn.Linear(d_reduced, d_block),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        hidden = F.relu(self.linear(x))
        gate = self.gate(hidden.mean(dim=-1, keepdim=True).expand_as(hidden))
        return self.dropout(hidden * gate)


class TabMSEBackbone(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        k: int,
        n_blocks: int,
        d_block: int,
        dropout: float,
        start_scaling_init: str,
        start_scaling_init_chunks: list[int],
        se_reduction: int,
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.view = EnsembleView(self.k)
        self.affine = ElementwiseAffine(
            (self.k, d_in),
            bias=False,
            scaling_init=start_scaling_init,
            scaling_init_chunks=start_scaling_init_chunks,
        )
        current_dim = d_in
        blocks: list[nn.Module] = []
        for _ in range(int(n_blocks)):
            blocks.append(SEBlock(current_dim, int(d_block), float(dropout), int(se_reduction)))
            current_dim = int(d_block)
        self.blocks = nn.ModuleList(blocks)
        self.output_dim = current_dim

    def forward(self, x: Tensor) -> Tensor:
        x = self.view(x)
        x = self.affine(x)
        for block in self.blocks:
            x = block(x)
        return x


def build_tabm_v10(
    dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    return build_variant(
        dataset_meta,
        trial_params,
        embedding_bundle,
        device,
        model_name="tabm-se",
        backbone_factory=TabMSEBackbone,
        backbone_kwargs={"se_reduction": int(trial_params.get("se_reduction", 4))},
    )


__all__ = ["SEBlock", "TabMSEBackbone", "build_tabm_v10"]
