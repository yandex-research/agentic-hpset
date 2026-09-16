# ruff: noqa
"""Standalone implementation for ``build_mlp_v6``."""

from __future__ import annotations
from typing import Any
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import torch

_ENTROPY_WEIGHT = 0.01


class _Expert(nn.Module):
    def __init__(
        self, input_dim: int, n_blocks: int, hidden: int, dropout: float
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(n_blocks):
            layers.extend([nn.Linear(current, hidden), nn.ReLU(), nn.Dropout(dropout)])
            current = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(current, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.body(x)).squeeze(-1)


_N_EXPERTS = 4
_TOP_K = 2


class SoftMoEMLP(nn.Module):
    def __init__(
        self,
        num_embedding: nn.Module | None,
        cat_embedding: nn.Module | None,
        input_dim: int,
        params: dict[str, Any],
    ) -> None:
        super().__init__()
        if input_dim == 0:
            raise ValueError('Model has no input features after preprocessing.')
        self.num_embedding = num_embedding
        self.cat_embedding = cat_embedding
        n_blocks = int(params['n_blocks'])
        d_block = int(params['d_block'])
        dropout = float(params['dropout'])
        hidden = max(16, d_block // 2)
        self.experts = nn.ModuleList(
            [_Expert(input_dim, n_blocks, hidden, dropout) for _ in range(_N_EXPERTS)]
        )
        self.router = nn.Linear(input_dim, _N_EXPERTS)
        self.register_buffer('entropy_penalty', torch.tensor(0.0))

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        pieces: list[Tensor] = []
        if x_num is not None and self.num_embedding is not None:
            pieces.append(self.num_embedding(x_num).flatten(1))
        if x_cat is not None and self.cat_embedding is not None:
            pieces.append(self.cat_embedding(x_cat))
        x = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
        logits = self.router(x)
        topk_vals, topk_idx = torch.topk(logits, k=_TOP_K, dim=-1)
        weights = F.softmax(topk_vals, dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        gather_idx = topk_idx
        gathered = expert_outputs.gather(-1, gather_idx)
        out = (gathered * weights).sum(dim=-1)
        if self.training:
            soft_assign = F.softmax(logits, dim=-1)
            mean_assign = soft_assign.mean(dim=0).clamp_min(1e-08)
            entropy = -(mean_assign * mean_assign.log()).sum()
            penalty = _ENTROPY_WEIGHT * -entropy
            out = out + penalty * 0.0 + (penalty * 0.0).detach()
            self.entropy_penalty = entropy.detach()
        return out


def build_mlp_v6(
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    model = SoftMoEMLP(
        num_embedding=embedding_bundle['num_embedding'],
        cat_embedding=embedding_bundle['cat_embedding'],
        input_dim=int(embedding_bundle['input_dim']),
        params=trial_params,
    )
    return model.to(device)
