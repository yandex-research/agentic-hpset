"""One-hot encoding plus per-category log-frequency feature.

Each categorical column contributes its full one-hot block (cardinality
dims) followed by one extra scalar carrying the training-set log frequency
of that category, normalized by `log(n_rows)`. Stateless — frequencies are
computed once at construction. Useful for letting the network use raw
"how common is this category" signal without learning a full embedding
table. Distinct from `freq_weighted.py` (which multiplies a learned
embedding by inverse-sqrt-frequency).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class OneHotFrequencyEncoding(nn.Module):
    def __init__(self, cardinalities: list[int], frequencies: list[Tensor]) -> None:
        super().__init__()
        self.cardinalities = cardinalities
        for index, frequency in enumerate(frequencies):
            self.register_buffer(f"frequency_{index}", frequency)

    def forward(self, x: Tensor) -> Tensor:
        pieces: list[Tensor] = []
        for i, cardinality in enumerate(self.cardinalities):
            values = x[:, i].clamp(0, cardinality)
            pieces.append(F.one_hot(values, cardinality + 1)[:, :-1].float())
            frequency = getattr(self, f"frequency_{i}")
            pieces.append(frequency[values].unsqueeze(1))
        return torch.cat(pieces, dim=1)


def _build_frequencies(x_cat_train: Tensor, cardinalities: list[int]) -> list[Tensor]:
    frequencies: list[Tensor] = []
    n_rows = max(int(x_cat_train.shape[0]), 1)
    for i, cardinality in enumerate(cardinalities):
        values = x_cat_train[:, i].clamp(0, cardinality)
        counts = torch.bincount(values, minlength=cardinality + 1).float()
        frequencies.append(torch.log1p(counts) / torch.log1p(torch.tensor(float(n_rows))))
    return frequencies


def build_cat_embedding_v8(
    cat_cardinalities: list[int],
    dataset_meta: Any,
    _trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities or dataset_meta.x_cat is None:
        return None, 0, {"cardinalities": []}
    x_cat_train = torch.as_tensor(dataset_meta.x_cat["train"], dtype=torch.long)
    frequencies = _build_frequencies(x_cat_train, cat_cardinalities)
    return (
        OneHotFrequencyEncoding(cat_cardinalities, frequencies),
        sum(cat_cardinalities) + len(cat_cardinalities),
        {"cardinalities": cat_cardinalities},
    )
