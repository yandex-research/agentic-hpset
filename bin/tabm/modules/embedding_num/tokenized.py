from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


def compute_bin_edges(x: Tensor, n_bins: int) -> list[Tensor]:
    quantiles = torch.linspace(0.0, 1.0, n_bins + 1, device=x.device, dtype=x.dtype)
    edges = torch.quantile(x, quantiles, dim=0).T  # (n_features, n_bins+1)
    return [e.unique() for e in edges]


class TokenizedBinEmbedding(nn.Module):
    """Discretize each feature into bins and look up per-bin embeddings.

    For feature ``i``, value ``x`` is assigned to its bucket via ``torch.bucketize``
    and the embedding is the linearly-interpolated mix of two adjacent bin tokens.
    Linear interpolation provides a continuous, differentiable signal w.r.t. the
    raw value while the table form gives each bin its own arbitrary representation.
    """

    def __init__(self, bins: list[Tensor], d_embedding: int) -> None:
        super().__init__()
        self.n_features = len(bins)
        self.d_embedding = int(d_embedding)
        self.n_bins_per_feature: list[int] = [int(len(b) - 1) for b in bins]
        max_bins = max(self.n_bins_per_feature)
        self.max_bins = int(max_bins)
        # Pad shorter bin lists by repeating last edge so searchsorted is well defined
        # on a shared (n_features, max_bins+1) tensor. Padding with the last real edge
        # keeps sequences non-decreasing and makes out-of-range values bucket into the
        # top bin, matching the per-feature search on the trimmed edge list.
        edge_tensor = torch.zeros(self.n_features, max_bins + 1)
        for i, edges in enumerate(bins):
            n = edges.shape[0]
            edge_tensor[i, :n] = edges
            edge_tensor[i, n:] = edges[-1]
        self.register_buffer("edges", edge_tensor)
        self.register_buffer(
            "n_bins", torch.tensor(self.n_bins_per_feature, dtype=torch.long)
        )
        # Token table: (n_features, max_bins, d_embedding).
        self.token_table = nn.Parameter(
            torch.empty(self.n_features, self.max_bins, self.d_embedding)
        )
        bound = self.d_embedding**-0.5
        nn.init.uniform_(self.token_table, -bound, bound)

    def get_output_shape(self) -> torch.Size:
        return torch.Size((self.n_features, self.d_embedding))

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., n_features). Batched per-feature bucket lookup, fully vectorized:
        # one searchsorted + a couple of gathers over all features at once.
        original_shape = x.shape
        flat = x.reshape(-1, self.n_features)
        values = flat.transpose(0, 1).contiguous()  # (n_features, N)
        n_bins = self.n_bins  # (n_features,)
        top_edge = n_bins.unsqueeze(-1)  # (n_features, 1)
        top_token = (n_bins - 1).unsqueeze(-1)
        raw_idx = torch.searchsorted(self.edges, values, right=False) - 1
        idx = raw_idx.clamp_min(0).minimum(top_token)  # (n_features, N)
        right_edge_idx = (idx + 1).minimum(top_edge)
        left = self.edges.gather(1, idx)
        right = self.edges.gather(1, right_edge_idx)
        width = (right - left).clamp_min(1e-6)
        t = ((values - left) / width).clamp(0.0, 1.0).unsqueeze(-1)  # (n_features, N, 1)
        right_token_idx = (idx + 1).minimum(top_token)
        d = self.d_embedding
        left_gather = idx.unsqueeze(-1).expand(-1, -1, d)
        right_gather = right_token_idx.unsqueeze(-1).expand(-1, -1, d)
        left_token = self.token_table.gather(1, left_gather)
        right_token = self.token_table.gather(1, right_gather)
        mixed = left_token * (1.0 - t) + right_token * t  # (n_features, N, d)
        out = mixed.transpose(0, 1).contiguous()  # (N, n_features, d)
        return out.reshape(*original_shape, self.d_embedding)


def build_num_embedding_v3(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    """Bin-token embedding with linear interpolation (a learnable PLR).

    Hypothesis: PLR caps each bin's representation at a single learned linear
    function. Replacing each bin with a free embedding vector (and interpolating
    between adjacent bin tokens) gives the model strictly more expressivity while
    keeping a continuous, differentiable mapping from x. This often closes the
    gap to attention-based encoders without their compute cost.
    """
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return None, 0, {"bins": []}
    bins = compute_bin_edges(x_num_train, int(trial_params["n_bins"]))
    embedding = TokenizedBinEmbedding(bins, int(trial_params["d_embedding"]))
    output_dim = dataset_meta.n_num_features * int(trial_params["d_embedding"])
    return embedding, output_dim, {
        "bins": bins,
        "d_embedding": int(trial_params["d_embedding"]),
    }
