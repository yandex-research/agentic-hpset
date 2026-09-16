"""PLR with trainable bin edges.

Initial edges come from quantiles, but are stored as parameters using a
softplus-cumulative reparametrization that guarantees strict monotonicity
during training. This lets the model adapt the discretization to the loss
instead of being locked to the empirical quantiles.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .piecewise_linear import LinearEmbeddings, compute_bins


class LearnablePiecewiseLinearEncoding(nn.Module):
    """PLR encoding where the bin edges are learnable, parametrized as

        e_0 = first_edge
        e_k = e_{k-1} + softplus(raw_gap_k)

    so the edges are strictly monotone. `raw_gap_*` is initialized so that
    softplus(raw_gap_*) reproduces the original empirical-quantile gaps.
    """

    def __init__(self, bins: list[Tensor]) -> None:
        super().__init__()
        n_features = len(bins)
        n_bins = [len(edges) - 1 for edges in bins]
        max_n_bins = max(n_bins)
        first_edges = torch.zeros(n_features)
        raw_gap = torch.zeros(n_features, max_n_bins)
        bin_count = torch.zeros(n_features, dtype=torch.long)
        for i, edges in enumerate(bins):
            first_edges[i] = edges[0]
            actual_gaps = edges.diff()
            actual_gaps = torch.clamp_min(actual_gaps, 1e-6)
            raw = torch.log(torch.expm1(actual_gaps))
            padded = torch.full(
                (max_n_bins,), float(torch.log(torch.expm1(torch.tensor(1e-3))))
            )
            padded[: actual_gaps.numel()] = raw
            raw_gap[i] = padded
            bin_count[i] = len(edges) - 1
        self.first_edges = nn.Parameter(first_edges)
        self.raw_gap = nn.Parameter(raw_gap)
        self.register_buffer("single_bin_mask", bin_count == 1)
        self.register_buffer("bin_count", bin_count)
        self.register_buffer(
            "mask",
            None
            if (bin_count == max_n_bins).all()
            else torch.row_stack(
                [
                    torch.cat(
                        [
                            torch.ones(max(int(n) - 1, 0), dtype=torch.bool),
                            torch.zeros(max_n_bins - int(n), dtype=torch.bool),
                            torch.ones(1, dtype=torch.bool),
                        ]
                    )
                    for n in bin_count
                ]
            ),
        )
        self.max_n_bins = max_n_bins

    def _edges_and_params(self) -> tuple[Tensor, Tensor, Tensor]:
        gaps = nn.functional.softplus(self.raw_gap) + 1e-6
        edges = torch.cat(
            [self.first_edges[:, None], self.first_edges[:, None] + gaps.cumsum(dim=1)],
            dim=1,
        )
        widths = edges[:, 1:] - edges[:, :-1]
        weights = 1.0 / widths
        biases = -edges[:, :-1] / widths
        gather_idx = (self.bin_count - 1).clamp_min(0)[:, None]
        last_w = weights.gather(1, gather_idx)
        last_b = biases.gather(1, gather_idx)
        weight_out = weights[:, : self.max_n_bins].clone()
        bias_out = biases[:, : self.max_n_bins].clone()
        weight_out[:, -1:] = last_w
        bias_out[:, -1:] = last_b
        return weight_out, bias_out, edges

    def forward(self, x: Tensor) -> Tensor:
        weight, bias, _ = self._edges_and_params()
        encoded = torch.addcmul(bias, weight, x[..., None])
        if encoded.shape[-1] == 1:
            return encoded
        last = torch.where(
            self.single_bin_mask[..., None],
            encoded[..., -1:],
            encoded[..., -1:].clamp_min(0.0),
        )
        encoded = torch.cat(
            [encoded[..., :1].clamp_max(1.0), encoded[..., 1:-1].clamp(0.0, 1.0), last],
            dim=-1,
        )
        if self.mask is not None:
            encoded = encoded * self.mask
        return encoded


class LearnablePLREmbeddings(nn.Module):
    def __init__(self, bins: list[Tensor], d_embedding: int) -> None:
        super().__init__()
        self.linear0 = LinearEmbeddings(len(bins), d_embedding)
        self.encoding = LearnablePiecewiseLinearEncoding(bins)
        self.linear = nn.Parameter(
            torch.zeros(len(bins), self.encoding.max_n_bins, d_embedding)
        )

    def forward(self, x: Tensor) -> Tensor:
        x_linear = self.linear0(x)
        x_ple = self.encoding(x).transpose(0, 1)
        x_ple = (x_ple @ self.linear).transpose(0, 1)
        return x_linear + x_ple


def build_num_embedding_v7(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return None, 0, {"bins": []}
    bins = compute_bins(x_num_train, int(trial_params["n_bins"]))
    embedding = LearnablePLREmbeddings(bins, int(trial_params["d_embedding"]))
    output_dim = dataset_meta.n_num_features * int(trial_params["d_embedding"])
    return embedding, output_dim, {"bins": bins}
