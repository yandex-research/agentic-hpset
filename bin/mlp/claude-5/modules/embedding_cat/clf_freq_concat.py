# ruff: noqa
"""Standalone implementation for ``build_cat_embedding_v4``."""

from __future__ import annotations
from typing import Any
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import numpy as np
import torch


class FreqConcatCategoricalEmbedding(nn.Module):
    """One-hot of category, concatenated with normalized log-frequency feature."""

    def __init__(self, cardinalities: list[int], freq_tables: list[np.ndarray]) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        for i, tab in enumerate(freq_tables):
            self.register_buffer(f'fq_{i}', torch.as_tensor(tab, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        pieces: list[Tensor] = []
        for i, cardinality in enumerate(self.cardinalities):
            idx = x[:, i].clamp(0, cardinality - 1)
            pieces.append(F.one_hot(idx, cardinality).float())
            freq: Tensor = getattr(self, f'fq_{i}')
            pieces.append(freq[idx].unsqueeze(-1))
        return torch.cat(pieces, dim=1)


def _frequency_table(cat_train: np.ndarray, cardinality: int) -> np.ndarray:
    """Return a bounded array of train-derived log frequencies.

    The final index is the explicitly reserved unseen bucket.
    """
    counts = np.bincount(cat_train, minlength=cardinality).astype(np.float64)
    log_freq = np.log1p(counts)
    if log_freq.size > 0:
        mu = log_freq.mean()
        sd = log_freq.std() + 1e-06
        log_freq_norm = np.clip((log_freq - mu) / sd, -4.0, 4.0)
    else:
        log_freq_norm = log_freq
    log_freq_norm[-1] = 0.0
    return log_freq_norm.astype(np.float32)


def build_cat_embedding_v4(
    cat_cardinalities: list[int],
    dataset_meta: Any,
    _trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return (None, 0, {'cardinalities': []})
    if dataset_meta.x_cat is None:
        return (None, 0, {'cardinalities': []})
    cat_train = dataset_meta.x_cat['train']
    freq_tables = [
        _frequency_table(cat_train[:, i], cardinality)
        for i, cardinality in enumerate(cat_cardinalities)
    ]
    output_dim = sum(cat_cardinalities) + len(cat_cardinalities)
    return (
        FreqConcatCategoricalEmbedding(cat_cardinalities, freq_tables),
        output_dim,
        {'cardinalities': cat_cardinalities},
    )
