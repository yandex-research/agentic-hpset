"""RealMLP categorical embedding (v0).

Learned embedding table per large-cardinality categorical feature. Each table
is shaped ``(n_ens, cat_size, embedding_size)`` with normal init; the forward
gathers per-ensemble-member rows.

Small-cardinality features are handled upstream in ``preprocess_categorical``
as one-hot blocks — this module only sees the large-cat indices.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

class LargeCatEmbedding(nn.Module):
    def __init__(self, n_models: int, cat_sizes: List[int], emb_dim: int) -> None:
        super().__init__()
        self.n_models = n_models
        self.cat_sizes = list(cat_sizes)
        self.emb_dim = int(emb_dim)
        self.embedding_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(n_models, size, emb_dim)) for size in cat_sizes]
        )
        for weight in self.embedding_weights:
            nn.init.normal_(weight, mean=0.0, std=1.0)

    @property
    def out_features(self) -> int:
        return len(self.cat_sizes) * self.emb_dim

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        """``x_cat`` is ``(batch, n_large)`` or ``(n_ens, batch, n_large)`` int64.

        Returns ``(n_ens, batch, n_large * emb_dim)``.
        """
        xs = []
        if x_cat.ndim == 2:
            xs.extend(weight[:, x_cat[:, idx], :] for idx, weight in enumerate(self.embedding_weights))
        else:
            for idx, weight in enumerate(self.embedding_weights):
                gather_idx = x_cat[:, :, idx][:, :, None].expand(-1, -1, weight.shape[-1])
                xs.append(weight.gather(1, gather_idx))
        return torch.cat(xs, dim=2)

def build_cat_embedding_v0(
    cat_sizes: List[int], cfg: Dict[str, Any]
) -> Tuple[Optional[nn.Module], int]:
    """Return ``(module, out_features)``. ``module`` is ``None`` when there are
    no large-cardinality features.
    """
    if not cat_sizes:
        return None, 0
    n_models = int(cfg.get("n_ens", 1))
    emb_dim = int(cfg.get("embedding_size", 8))
    module = LargeCatEmbedding(n_models, cat_sizes, emb_dim)
    return module, module.out_features

__all__ = ["LargeCatEmbedding", "build_cat_embedding_v0"]
