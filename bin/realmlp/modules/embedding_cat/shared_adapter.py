"""Categorical embedding (v1): shared tables with per-member sign adapters.

Hypothesis: v0 trains ``n_ens`` fully independent embedding tables, so the
gradient signal for a rare category is fragmented across members — each table
sees the category's few occurrences separately. Sharing one table per feature
pools all members' signal per category (the long-tail / high-cardinality
regime), while light per-member multiplicative adapters keep ensemble
diversity, following TabM (Gorishniy et al. 2024) / BatchEnsemble (Wen et
al. 2020): parameter sharing with per-member adapters outperforms fully
independent members.

Mechanism: per large-cat feature, one shared ``(cat_size, emb_dim)`` table
(normal init like v0) plus a per-member ``(n_ens, emb_dim)`` scale adapter
initialized with random +-1 signs (TabM's diversity-inducing init). Output
shape and ``out_features`` match v0 exactly; parameter count shrinks roughly
``n_ens``-fold on the tables.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class SharedCatEmbedding(nn.Module):
    def __init__(self, n_models: int, cat_sizes: List[int], emb_dim: int) -> None:
        super().__init__()
        self.n_models = n_models
        self.cat_sizes = list(cat_sizes)
        self.emb_dim = int(emb_dim)
        self.embedding_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(size, emb_dim)) for size in cat_sizes]
        )
        for weight in self.embedding_weights:
            nn.init.normal_(weight, mean=0.0, std=1.0)
        self.member_scales = nn.ParameterList(
            [
                nn.Parameter(
                    2.0 * torch.randint(0, 2, (n_models, emb_dim), dtype=torch.float32) - 1.0
                )
                for _ in cat_sizes
            ]
        )

    @property
    def out_features(self) -> int:
        return len(self.cat_sizes) * self.emb_dim

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        """``x_cat`` is ``(batch, n_large)`` or ``(n_ens, batch, n_large)`` int64.

        Returns ``(n_ens, batch, n_large * emb_dim)``.
        """
        xs = []
        for idx, (weight, scale) in enumerate(zip(self.embedding_weights, self.member_scales)):
            if x_cat.ndim == 2:
                emb = weight[x_cat[:, idx]][None, :, :] * scale[:, None, :]
            else:
                emb = weight[x_cat[:, :, idx]] * scale[:, None, :]
            xs.append(emb)
        return torch.cat(xs, dim=2)


def build_cat_embedding_shared_adapter(
    cat_sizes: List[int], cfg: Dict[str, Any]
) -> Tuple[Optional[nn.Module], int]:
    """Return ``(module, out_features)`` mirroring ``build_cat_embedding_v0``."""
    if not cat_sizes:
        return None, 0
    n_models = int(cfg.get("n_ens", 1))
    emb_dim = int(cfg.get("embedding_size", 8))
    module = SharedCatEmbedding(n_models, cat_sizes, emb_dim)
    return module, module.out_features


__all__ = ["SharedCatEmbedding", "build_cat_embedding_shared_adapter"]
