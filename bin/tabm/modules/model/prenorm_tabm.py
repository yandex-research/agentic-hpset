from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ._variant import build_variant
from .tabm import ElementwiseAffine, EnsembleView


class PreNormBlock(nn.Module):
    """LayerNorm -> Linear -> ReLU -> Dropout (no residual).

    A simpler pre-norm ablation than the SwiGLU/RMSNorm residual variant: only
    swaps the normalization position relative to the baseline TabM block.
    """

    def __init__(self, d_in: int, d_block: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.linear = nn.Linear(d_in, d_block)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(F.relu(self.linear(self.norm(x))))


class TabMPreNormBackbone(nn.Module):
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
            blocks.append(PreNormBlock(current_dim, int(d_block), float(dropout)))
            current_dim = int(d_block)
        self.blocks = nn.ModuleList(blocks)
        self.output_dim = current_dim

    def forward(self, x: Tensor) -> Tensor:
        x = self.view(x)
        x = self.affine(x)
        for block in self.blocks:
            x = block(x)
        return x


def build_tabm_v14(
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
        model_name="tabm-prenorm",
        backbone_factory=TabMPreNormBackbone,
        backbone_kwargs={},
    )


__all__ = ["PreNormBlock", "TabMPreNormBackbone", "build_tabm_v14"]
