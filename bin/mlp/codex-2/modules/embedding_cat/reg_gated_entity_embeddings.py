# ruff: noqa
"""Standalone implementation for ``build_gated_entity_embeddings``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class GatedEntityEmbeddings(nn.Module):
    def __init__(self, cardinalities: list[int], dims: list[int]) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList()
        self.gates = nn.ModuleList()
        for cardinality, dim in zip(cardinalities, dims):
            embedding = nn.Embedding(cardinality + 1, dim)
            gate = nn.Embedding(cardinality + 1, dim)
            nn.init.normal_(embedding.weight, std=dim ** (-0.5))
            nn.init.constant_(gate.weight, 1.0)
            self.embeddings.append(embedding)
            self.gates.append(gate)

    def forward(self, x: Tensor) -> Tensor:
        pieces: list[Tensor] = []
        for idx, (embedding, gate) in enumerate(zip(self.embeddings, self.gates)):
            values = x[:, idx].clamp_max(embedding.num_embeddings - 1)
            pieces.append(embedding(values) * torch.sigmoid(gate(values)))
        return torch.cat(pieces, dim=1)


def _embedding_dims(cardinalities: list[int], d_embedding: int) -> list[int]:
    dims: list[int] = []
    for cardinality in cardinalities:
        suggested = max(2, min(d_embedding, int(round((cardinality + 1) ** 0.5)) + 1))
        dims.append(suggested)
    return dims


def build_gated_entity_embeddings(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'variant': 'gated_entity', 'cardinalities': []})
    dims = _embedding_dims(cat_cardinalities, int(trial_params['d_embedding']))
    return (
        GatedEntityEmbeddings(cat_cardinalities, dims),
        sum(dims),
        {'variant': 'gated_entity', 'cardinalities': cat_cardinalities, 'dims': dims},
    )
