# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v2``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch


class FrequencyEncoding(nn.Module):
    """Preserve category identity and append a bounded log-frequency scalar."""

    def __init__(self, freq_tables: list[Tensor]) -> None:
        super().__init__()
        max_card = max((t.numel() for t in freq_tables))
        n_features = len(freq_tables)
        padded = torch.zeros(n_features, max_card, dtype=torch.float32)
        for i, table in enumerate(freq_tables):
            padded[i, : table.numel()] = table
        self.register_buffer('table', padded)
        self.cardinalities = [int(t.numel()) for t in freq_tables]

    def forward(self, x: Tensor) -> Tensor:
        clamped = x.clamp(0, self.table.shape[1] - 1)
        n_features = self.table.shape[0]
        out_cols = []
        for i in range(n_features):
            cardinality = self.cardinalities[i]
            index = clamped[:, i].clamp_max(cardinality - 1)
            out_cols.append(F.one_hot(index, cardinality).float())
            out_cols.append(self.table[i].index_select(0, index).unsqueeze(1))
        return torch.cat(out_cols, dim=1)


def build_cat_embedding_v2(
    cat_cardinalities: list[int],
    dataset_meta: Any,
    _trial_params: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    train_cat = dataset_meta.x_cat['train']
    freq_tables: list[Tensor] = []
    n_train = int(train_cat.shape[0])
    for i, card in enumerate(cat_cardinalities):
        col = torch.as_tensor(train_cat[:, i], dtype=torch.long)
        size = card
        counts = torch.bincount(col.clamp_min(0), minlength=size)[:size].float()
        bounded = torch.log1p(counts) / torch.log(
            torch.tensor(float(max(n_train, 1)) + 1.0)
        )
        bounded[-1] = 0.0
        freq_tables.append(bounded.clamp_(0.0, 1.0))
    embedding = FrequencyEncoding(freq_tables).to(device)
    output_dim = sum(cat_cardinalities) + len(cat_cardinalities)
    return (embedding, output_dim, {'cardinalities': cat_cardinalities})
