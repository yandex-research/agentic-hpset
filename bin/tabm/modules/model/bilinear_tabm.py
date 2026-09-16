from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ._variant import build_variant
from .tabm import ElementwiseAffine, EnsembleView


class LowRankBilinear(nn.Module):
    """Low-rank bilinear feature expansion: u(x) * v(x) with shared projections."""

    def __init__(self, d_in: int, rank: int) -> None:
        super().__init__()
        self.u = nn.Linear(d_in, rank, bias=False)
        self.v = nn.Linear(d_in, rank, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.u(x) * self.v(x)


class TabMBilinearBackbone(nn.Module):
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
        bilinear_rank: int,
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
        self.bilinear = LowRankBilinear(d_in, int(bilinear_rank))
        current_dim = d_in + int(bilinear_rank)
        blocks: list[nn.Module] = []
        for _ in range(int(n_blocks)):
            blocks.append(
                nn.Sequential(nn.Linear(current_dim, int(d_block)), nn.ReLU(), nn.Dropout(float(dropout)))
            )
            current_dim = int(d_block)
        self.blocks = nn.ModuleList(blocks)
        self.output_dim = current_dim

    def forward(self, x: Tensor) -> Tensor:
        x = self.view(x)
        x = self.affine(x)
        x = torch.cat([x, self.bilinear(x)], dim=-1)
        for block in self.blocks:
            x = block(x)
        return x


def build_tabm_v12(
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
        model_name="tabm-bilinear",
        backbone_factory=TabMBilinearBackbone,
        backbone_kwargs={"bilinear_rank": int(trial_params.get("bilinear_rank", 16))},
    )


__all__ = ["LowRankBilinear", "TabMBilinearBackbone", "build_tabm_v12"]
