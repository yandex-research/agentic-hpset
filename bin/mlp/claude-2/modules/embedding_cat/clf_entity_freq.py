# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v4``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import math
import torch.nn as nn
import numpy as np
import torch


class EntityFreqEmbedding(nn.Module):
    def __init__(
        self, cardinalities: list[int], d_embedding: int, freq_tables: list[np.ndarray]
    ) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        self.d_embedding = d_embedding
        self.n_features = len(cardinalities)
        offsets: list[int] = []
        running = 0
        for c in cardinalities:
            offsets.append(running)
            running += int(c)
        self.embedding = nn.Embedding(running, d_embedding)
        nn.init.normal_(
            self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(d_embedding)
        )
        self.register_buffer('offsets_tensor', torch.tensor(offsets, dtype=torch.long))
        max_card = max(cardinalities)
        freq_buf = torch.zeros(self.n_features, max_card, dtype=torch.float32)
        for j, freqs in enumerate(freq_tables):
            arr = torch.tensor(freqs, dtype=torch.float32)
            freq_buf[j, : arr.shape[0]] = arr
        self.register_buffer('freq_table', freq_buf)
        self.freq_scale = nn.Parameter(torch.ones(self.n_features))
        self.freq_bias = nn.Parameter(torch.zeros(self.n_features))

    def forward(self, x: Tensor) -> Tensor:
        x_off = x + self.offsets_tensor
        emb = self.embedding(x_off)
        gathered = torch.gather(
            self.freq_table,
            1,
            x.t().clamp_min(0).clamp_max(self.freq_table.shape[1] - 1),
        )
        freq_scalar = gathered.t() * self.freq_scale + self.freq_bias
        out = torch.cat([emb.flatten(1), freq_scalar], dim=1)
        return out


def build_cat_embedding_v4(
    cat_cardinalities: list[int],
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    d_embedding = int(trial_params['d_embedding'])
    x_cat_train = dataset_meta.x_cat['train']
    n_train = float(x_cat_train.shape[0])
    freq_tables: list[np.ndarray] = []
    for j, c in enumerate(cat_cardinalities):
        counts = np.bincount(x_cat_train[:, j], minlength=c).astype(np.float32)[:c]
        freqs = np.log1p(counts) / max(np.log1p(n_train), 1.0)
        freq_tables.append(freqs)
    module = EntityFreqEmbedding(cat_cardinalities, d_embedding, freq_tables)
    out_dim = len(cat_cardinalities) * d_embedding + len(cat_cardinalities)
    return (
        module,
        out_dim,
        {'cardinalities': cat_cardinalities, 'd_embedding': d_embedding},
    )
