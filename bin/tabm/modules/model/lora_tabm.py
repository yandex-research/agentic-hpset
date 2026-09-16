from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ._variant import build_variant
from .tabm import ElementwiseAffine, EnsembleView


class LoRAEnsembleLinear(nn.Module):
    """Shared Linear with per-head low-rank delta (a la LoRA).

    All heads share a single Linear(d_in, d_out). Each head also has its own
    rank-r factor A_k (r, d_in) and B_k (d_out, r) so the effective per-head
    weight is W_shared + B_k A_k. Cheap per-head specialization without
    duplicating the full weight matrix.
    """

    def __init__(self, d_in: int, d_out: int, k: int, rank: int) -> None:
        super().__init__()
        self.shared = nn.Linear(d_in, d_out)
        self.a = nn.Parameter(torch.empty(int(k), int(rank), d_in))
        self.b = nn.Parameter(torch.zeros(int(k), d_out, int(rank)))
        nn.init.kaiming_uniform_(self.a, a=5**0.5)

    def forward(self, x: Tensor) -> Tensor:
        shared = self.shared(x)
        latent = torch.einsum("bki,kri->bkr", x, self.a)
        delta = torch.einsum("bkr,kor->bko", latent, self.b)
        return shared + delta


class TabMLoRABackbone(nn.Module):
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
        lora_rank: int,
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
        layers: list[nn.Module] = []
        for _ in range(int(n_blocks)):
            layers.append(LoRAEnsembleLinear(current_dim, int(d_block), self.k, int(lora_rank)))
            current_dim = int(d_block)
        self.linears = nn.ModuleList(layers)
        self.dropout = nn.Dropout(float(dropout))
        self.output_dim = current_dim

    def forward(self, x: Tensor) -> Tensor:
        x = self.view(x)
        x = self.affine(x)
        for linear in self.linears:
            x = self.dropout(F.relu(linear(x)))
        return x


def build_tabm_v13(
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
        model_name="tabm-lora",
        backbone_factory=TabMLoRABackbone,
        backbone_kwargs={"lora_rank": int(trial_params.get("lora_rank", 8))},
    )


__all__ = ["LoRAEnsembleLinear", "TabMLoRABackbone", "build_tabm_v13"]
