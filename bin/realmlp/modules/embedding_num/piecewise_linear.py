"""Numerical embedding (v1): piecewise-linear embeddings with a densenet skip.

Hypothesis: v0's periodic PLR features impose an oscillatory inductive bias
everywhere; on features with threshold / piecewise-constant relationships
(common in tree-friendly tabular data) the piecewise-linear encoding of
Gorishniy et al. 2022 ("On Embeddings for Numerical Features") matches the
target function class better and beats PLR on a substantial minority of
datasets. The meta-tuned default froze one embedding family for all datasets;
the per-dataset tuner can select PLE exactly where it wins.

Adaptation: the upstream numerical preprocessing bounds inputs (smooth clip
to +-3, or a standard-normal quantile map), so bins are fixed uniform on
``[-ple_bound, ple_bound]`` (default 3.0) instead of data-derived quantile
bins — the embedding builder receives no data. First/last components stay
linear outside the range so out-of-bound values extrapolate.

Layout mirrors v0: per member and per feature, the PLE vector (``ple_n_bins``,
default 16) maps through a learned linear layer to ``plr_hidden_2 - 1``
channels, and the raw feature block is concatenated at the end (densenet
skip), so ``out_features == n_num * plr_hidden_2`` exactly like v0. The same
``num_emb_type`` gate as v0 is honored so configs that disable numeric
embeddings keep working.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class PiecewiseLinearEmbeddings(nn.Module):
    def __init__(
        self,
        n_models: int,
        n_cont: int,
        n_bins: int,
        hidden_2: int,
        bound: float,
    ):
        super().__init__()
        self.n_models = n_models
        self.n_cont = n_cont
        self.n_bins = n_bins
        self.hidden_2 = hidden_2
        hidden_2_without_dense = max(0, hidden_2 - 1)
        edges = torch.linspace(-bound, bound, n_bins + 1)
        self.register_buffer("edges_left", edges[:-1])
        self.register_buffer("bin_width", edges[1:] - edges[:-1])
        self.weight = nn.Parameter(
            (-1.0 + 2.0 * torch.rand(n_models, n_cont, n_bins, hidden_2_without_dense))
            / math.sqrt(n_bins)
        )
        self.bias = nn.Parameter(
            (-1.0 + 2.0 * torch.rand(n_models, n_cont, hidden_2_without_dense))
            / math.sqrt(n_bins)
        )

    @property
    def out_features(self) -> int:
        return self.n_cont * self.hidden_2

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (..., n_cont) -> PLE codes (..., n_cont, n_bins)."""
        t = ((x[..., None] - self.edges_left) / self.bin_width).clamp(0.0, 1.0)
        # First/last bins stay linear outside the bounded range (extrapolation).
        t_first = ((x - self.edges_left[0]) / self.bin_width[0]).clamp(max=1.0)
        t_last = ((x - self.edges_left[-1]) / self.bin_width[-1]).clamp(min=0.0)
        return torch.cat([t_first[..., None], t[..., 1:-1], t_last[..., None]], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_cont == 0:
            batch = x.shape[-2]
            return x.new_zeros((self.n_models, batch, 0))
        if x.ndim == 2:
            x = x[None, :, :].expand(self.n_models, -1, -1)
        codes = self._encode(x)  # (e, b, f, T)
        z = torch.einsum("ebft,eftd->ebfd", codes, self.weight) + self.bias[:, None, :, :]
        z = z.reshape(self.n_models, x.shape[-2], -1)
        return torch.cat([z, x], dim=-1)


def build_num_embedding_ple(n_num: int, cfg: Dict[str, Any]) -> Tuple[Optional[nn.Module], int]:
    """Return ``(module, out_features)`` mirroring ``build_num_embedding_v0``."""
    n_models = int(cfg.get("n_ens", 1))
    if n_num == 0:
        return None, 0
    if cfg.get("num_emb_type", "pbld") not in ("pbld", "pblrd", "pl", "plr"):
        return None, n_num
    module = PiecewiseLinearEmbeddings(
        n_models=n_models,
        n_cont=n_num,
        n_bins=max(2, int(cfg.get("ple_n_bins", 16))),
        hidden_2=int(cfg.get("plr_hidden_2", 4)),
        bound=float(cfg.get("ple_bound", 3.0)),
    )
    return module, module.out_features


__all__ = ["PiecewiseLinearEmbeddings", "build_num_embedding_ple"]
