from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class BinaryCodedCatEmbedding(nn.Module):
    """Binary-coded categorical embedding.

    Represents each category index as its binary digits (e.g., 5 → [1,0,1]),
    then projects through a learned linear layer per feature. This uses
    ceil(log2(cardinality)) bits instead of cardinality dimensions (one-hot),
    drastically reducing parameter count for high-cardinality features.

    Categories with similar binary codes implicitly share representation
    structure, providing an inductive bias that pure one-hot or lookup
    embeddings lack.
    """

    def __init__(self, cardinalities: list[int], d_embedding: int) -> None:
        super().__init__()
        self.n_bits_per_feature: list[int] = []
        self.linears = nn.ModuleList()
        for card in cardinalities:
            n_bits = max(int(math.ceil(math.log2(card + 2))), 1)
            self.n_bits_per_feature.append(n_bits)
            self.linears.append(nn.Linear(n_bits, d_embedding))
        for lin in self.linears:
            nn.init.kaiming_uniform_(lin.weight, a=5**0.5)
            nn.init.zeros_(lin.bias)
        # Precompute bit masks: powers of 2
        max_bits = max(self.n_bits_per_feature)
        self.register_buffer(
            "bit_positions", 2 ** torch.arange(max_bits, dtype=torch.long)
        )

    def _to_binary(self, x_col: Tensor, n_bits: int) -> Tensor:
        """Convert integer tensor to binary representation."""
        # x_col: (batch,) → (batch, n_bits)
        bits = (x_col.unsqueeze(-1) & self.bit_positions[:n_bits]).clamp(max=1).float()
        return bits

    def forward(self, x: Tensor) -> Tensor:
        parts: list[Tensor] = []
        for i, (n_bits, linear) in enumerate(
            zip(self.n_bits_per_feature, self.linears)
        ):
            binary = self._to_binary(x[:, i], n_bits)  # (batch, n_bits)
            parts.append(linear(binary))  # (batch, d_embedding)
        return torch.cat(parts, dim=1)


def build_cat_embedding_v3(
    cat_cardinalities: list[int],
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return None, 0, {"cardinalities": []}
    d_embedding = int(trial_params["d_embedding"])
    module = BinaryCodedCatEmbedding(cat_cardinalities, d_embedding)
    output_dim = len(cat_cardinalities) * d_embedding
    return module, output_dim, {"cardinalities": cat_cardinalities}
