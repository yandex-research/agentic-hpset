from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ._variant import build_variant
from .tabm import ElementwiseAffine, EnsembleView


class PerMemberColumnDropout(nn.Module):
    """Stochastic per-head input-feature dropout.

    Each forward pass resamples a (k, d_in) Bernoulli mask, so heads see
    different feature subsets across training steps. Distinct from a fixed
    per-head subspace mask, this is dropout with a per-head channel axis.
    """

    def __init__(self, k: int, d_in: int, p: float) -> None:
        super().__init__()
        self.k = int(k)
        self.d_in = int(d_in)
        self.p = float(p)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p <= 0.0:
            return x
        keep = torch.empty(self.k, self.d_in, device=x.device, dtype=x.dtype).bernoulli_(1.0 - self.p)
        return x * (keep / max(1.0 - self.p, 1e-6)).unsqueeze(0)


class TabMColumnDropoutBackbone(nn.Module):
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
        col_dropout: float,
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.view = EnsembleView(self.k)
        self.col_dropout = PerMemberColumnDropout(self.k, d_in, float(col_dropout))
        self.affine = ElementwiseAffine(
            (self.k, d_in),
            bias=False,
            scaling_init=start_scaling_init,
            scaling_init_chunks=start_scaling_init_chunks,
        )
        current_dim = d_in
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
        x = self.col_dropout(x)
        x = self.affine(x)
        for block in self.blocks:
            x = block(x)
        return x


def build_tabm_v11(
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
        model_name="tabm-coldrop",
        backbone_factory=TabMColumnDropoutBackbone,
        backbone_kwargs={"col_dropout": float(trial_params.get("col_dropout", 0.1))},
    )


__all__ = ["PerMemberColumnDropout", "TabMColumnDropoutBackbone", "build_tabm_v11"]
