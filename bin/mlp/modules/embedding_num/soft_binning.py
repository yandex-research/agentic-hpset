"""Soft-binning numeric embedding (fully differentiable).

Replaces hard piecewise-linear bins with a soft assignment: each feature
learns per-bin centers and (log) inverse-widths; the value's bin probability
is a softmax of negative squared distance to each center (RBF-like). The
probabilities are mixed via a learnable per-feature linear layer into a
d_embedding vector. Centers are initialized from quantile midpoints, so the
embedding starts close to PLR but is end-to-end trainable.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .piecewise_linear import LinearEmbeddings, compute_bins


class SoftBinningEmbedding(nn.Module):
    def __init__(self, bins: list[Tensor], d_embedding: int) -> None:
        super().__init__()
        n_features = len(bins)
        n_bins_each = [len(edges) - 1 for edges in bins]
        max_n_bins = max(n_bins_each)
        centers = torch.zeros(n_features, max_n_bins)
        log_inv_widths = torch.zeros(n_features, max_n_bins)
        for i, edges in enumerate(bins):
            mids = 0.5 * (edges[1:] + edges[:-1])
            widths = edges.diff().clamp_min(1e-3)
            log_iw = -torch.log(widths)
            pad_len = max_n_bins - mids.numel()
            if pad_len > 0:
                mids = torch.cat([mids, mids[-1:].expand(pad_len)])
                log_iw = torch.cat([log_iw, log_iw[-1:].expand(pad_len)])
            centers[i] = mids
            log_inv_widths[i] = log_iw
        self.centers = nn.Parameter(centers)
        self.log_inv_widths = nn.Parameter(log_inv_widths)
        self.mix = nn.Parameter(torch.zeros(n_features, max_n_bins, d_embedding))
        nn.init.normal_(self.mix, std=max_n_bins ** -0.5)
        self.linear0 = LinearEmbeddings(n_features, d_embedding)

    def forward(self, x: Tensor) -> Tensor:
        x_e = x[..., None]
        diffs = x_e - self.centers[None, ...]
        scale = torch.exp(self.log_inv_widths)[None, ...]
        logits = -(diffs ** 2) * scale
        probs = F.softmax(logits, dim=-1)
        emb = torch.einsum("bfk,fkd->bfd", probs, self.mix)
        return emb + self.linear0(x)


def build_num_embedding_v10(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return None, 0, {"bins": []}
    bins = compute_bins(x_num_train, int(trial_params["n_bins"]))
    embedding = SoftBinningEmbedding(bins, int(trial_params["d_embedding"]))
    output_dim = dataset_meta.n_num_features * int(trial_params["d_embedding"])
    return embedding, output_dim, {"bins": bins}
