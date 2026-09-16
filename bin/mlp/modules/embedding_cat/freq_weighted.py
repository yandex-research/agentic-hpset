from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class FreqWeightedCatEmbedding(nn.Module):
    """Learned embeddings scaled by inverse sqrt of category frequency.

    Each category's embedding is multiplied by 1/sqrt(freq + 1), where freq is
    the training-set count for that category. This amplifies rare-but-informative
    categories and dampens frequent ones, analogous to TF-IDF weighting in NLP.
    The +1 in the denominator prevents division-by-zero for unseen categories.
    """

    def __init__(
        self,
        cardinalities: list[int],
        d_embedding: int,
        freq_weights: list[Tensor],
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card + 1, d_embedding) for card in cardinalities]
        )
        for emb in self.embeddings:
            nn.init.kaiming_uniform_(emb.weight, a=5**0.5)
        # Register frequency weights as buffers (not trainable)
        for i, w in enumerate(freq_weights):
            self.register_buffer(f"freq_weight_{i}", w)

    def forward(self, x: Tensor) -> Tensor:
        parts: list[Tensor] = []
        for i, emb in enumerate(self.embeddings):
            embedded = emb(x[:, i])  # (batch, d_embedding)
            w = getattr(self, f"freq_weight_{i}")
            scale = w[x[:, i]].unsqueeze(-1)  # (batch, 1)
            parts.append(embedded * scale)
        return torch.cat(parts, dim=1)


def _compute_freq_weights(
    x_cat_train: Tensor, cardinalities: list[int]
) -> list[Tensor]:
    """Compute inverse-sqrt-frequency weights per category per feature."""
    weights = []
    for col_idx, card in enumerate(cardinalities):
        counts = torch.zeros(card + 1, dtype=torch.float32)
        col = x_cat_train[:, col_idx]
        for val in range(card + 1):
            counts[val] = (col == val).sum().float()
        w = 1.0 / torch.sqrt(counts + 1.0)
        # Normalize so mean weight is 1.0 (preserves embedding scale)
        w = w * (len(w) / w.sum())
        weights.append(w)
    return weights


def build_cat_embedding_v2(
    cat_cardinalities: list[int],
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return None, 0, {"cardinalities": []}
    d_embedding = int(trial_params["d_embedding"])
    # Compute frequency weights from training data
    x_cat_train = torch.as_tensor(dataset_meta.x_cat["train"], dtype=torch.long)
    freq_weights = _compute_freq_weights(x_cat_train, cat_cardinalities)
    module = FreqWeightedCatEmbedding(cat_cardinalities, d_embedding, freq_weights)
    output_dim = len(cat_cardinalities) * d_embedding
    return module, output_dim, {"cardinalities": cat_cardinalities}
