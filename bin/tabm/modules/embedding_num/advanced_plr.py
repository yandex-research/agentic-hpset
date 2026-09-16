from __future__ import annotations

import math
from typing import Any

import numpy as np
import sklearn.tree
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .piecewise_linear import (
    LinearEmbeddings,
    PLREmbeddings,
    PiecewiseLinearEncoding,
    compute_bins,
)


def _tree_thresholds(
    column: np.ndarray,
    target: np.ndarray,
    n_bins: int,
    task_type: str,
    seed: int,
) -> np.ndarray:
    tree_cls = (
        sklearn.tree.DecisionTreeClassifier
        if task_type in {"binclass", "multiclass"}
        else sklearn.tree.DecisionTreeRegressor
    )
    tree = tree_cls(
        max_leaf_nodes=max(2, int(n_bins)),
        random_state=seed,
        min_samples_leaf=max(1, column.shape[0] // (max(2, int(n_bins)) * 8)),
    )
    tree.fit(column.reshape(-1, 1), target)
    thresholds = tree.tree_.threshold
    feature = tree.tree_.feature
    selected = np.array(
        sorted({float(threshold) for threshold, feat in zip(thresholds, feature) if feat != -2}),
        dtype=np.float64,
    )
    col_min = float(np.nanmin(column))
    col_max = float(np.nanmax(column))
    if selected.size == 0:
        return np.array([col_min, col_max + 1e-6], dtype=np.float64)
    edges = np.unique(np.concatenate([[col_min], selected, [col_max + 1e-6]]))
    if edges.size < 2:
        edges = np.array([col_min, col_min + 1.0], dtype=np.float64)
    return edges


def build_num_embedding_v4(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return None, 0, {"bins": []}
    x_train = x_num_train.detach().cpu().numpy()
    target = np.asarray(dataset_meta.y["train"]).reshape(-1)
    if dataset_meta.is_regression:
        target = target.astype(np.float32)
    else:
        target = target.astype(np.int64)
    bins = [
        torch.tensor(
            _tree_thresholds(
                x_train[:, feature_idx],
                target,
                int(trial_params["n_bins"]),
                dataset_meta.task_type,
                int(trial_params.get("seed", 0)) + feature_idx,
            ),
            dtype=x_num_train.dtype,
            device=x_num_train.device,
        )
        for feature_idx in range(x_train.shape[1])
    ]
    embedding = PLREmbeddings(bins, int(trial_params["d_embedding"]))
    output_dim = dataset_meta.n_num_features * int(trial_params["d_embedding"])
    return embedding, output_dim, {
        "bins": bins,
        "d_embedding": int(trial_params["d_embedding"]),
    }


class LearnablePiecewiseLinearEncoding(nn.Module):
    def __init__(self, bins: list[Tensor]) -> None:
        super().__init__()
        n_features = len(bins)
        n_bins_per_feature = [len(edges) - 1 for edges in bins]
        max_n_bins = max(n_bins_per_feature)
        self.max_n_bins = max_n_bins
        first = torch.zeros(n_features, dtype=bins[0].dtype, device=bins[0].device)
        last = torch.zeros_like(first)
        gap_logits = torch.zeros(n_features, max_n_bins, dtype=bins[0].dtype, device=bins[0].device)
        for feature_idx, edges in enumerate(bins):
            first[feature_idx] = edges[0]
            last[feature_idx] = edges[-1]
            gaps = (edges[1:] - edges[:-1]).clamp_min(1e-4)
            gap_logits[feature_idx, : gaps.shape[0]] = torch.nan_to_num(
                torch.log(torch.expm1(gaps)), nan=0.0, posinf=10.0, neginf=-10.0
            )
            if gaps.shape[0] < max_n_bins:
                gap_logits[feature_idx, gaps.shape[0] :] = -10.0
        self.register_buffer("first", first)
        self.register_buffer("last", last)
        self.register_buffer(
            "single_bin_mask", torch.tensor(n_bins_per_feature, dtype=torch.long) == 1
        )
        self.gap_logits = nn.Parameter(gap_logits)
        if len(set(n_bins_per_feature)) == 1:
            self.register_buffer("mask", None)
        else:
            mask = torch.zeros(n_features, max_n_bins, dtype=torch.bool)
            for feature_idx, n_bins in enumerate(n_bins_per_feature):
                if n_bins - 1 > 0:
                    mask[feature_idx, : n_bins - 1] = True
                mask[feature_idx, -1] = True
            self.register_buffer("mask", mask)

    def get_edges(self) -> Tensor:
        gaps = F.softplus(self.gap_logits) + 1e-6
        cumulative = torch.cumsum(gaps, dim=1)
        scale = (self.last - self.first).unsqueeze(-1) / cumulative[:, -1:].clamp_min(1e-6)
        return torch.cat(
            [self.first.unsqueeze(-1), self.first.unsqueeze(-1) + cumulative * scale],
            dim=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        edges = self.get_edges()
        widths = (edges[:, 1:] - edges[:, :-1]).clamp_min(1e-6)
        encoded = torch.addcmul(-edges[:, :-1] / widths, 1.0 / widths, x[..., None])
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


class PLREmbeddingsLearnable(nn.Module):
    def __init__(self, bins: list[Tensor], d_embedding: int) -> None:
        super().__init__()
        self.n_features = len(bins)
        self.d_embedding = int(d_embedding)
        self.linear0 = LinearEmbeddings(len(bins), d_embedding)
        self.encoding = LearnablePiecewiseLinearEncoding(bins)
        self.linear = nn.Parameter(
            torch.zeros(len(bins), self.encoding.max_n_bins, d_embedding)
        )

    def get_output_shape(self) -> torch.Size:
        return torch.Size((self.n_features, self.d_embedding))

    def forward(self, x: Tensor) -> Tensor:
        x_ple = self.encoding(x).transpose(0, 1)
        x_ple = (x_ple @ self.linear).transpose(0, 1)
        return self.linear0(x) + x_ple


def build_num_embedding_v5(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return None, 0, {"bins": []}
    bins = compute_bins(x_num_train, int(trial_params["n_bins"]))
    embedding = PLREmbeddingsLearnable(bins, int(trial_params["d_embedding"]))
    output_dim = dataset_meta.n_num_features * int(trial_params["d_embedding"])
    return embedding, output_dim, {
        "bins": bins,
        "d_embedding": int(trial_params["d_embedding"]),
    }


class FourierFeatures(nn.Module):
    def __init__(self, n_features: int, d_embedding: int, sigma: float) -> None:
        super().__init__()
        if d_embedding < 2:
            raise ValueError("d_embedding must be >= 2 for Fourier features.")
        n_freq = d_embedding // 2
        self.register_buffer("frequencies", torch.randn(n_features, n_freq) * sigma)
        self.n_features = int(n_features)
        self.d_embedding = int(d_embedding)

    def forward(self, x: Tensor) -> Tensor:
        scaled = x[..., None] * self.frequencies * 2.0 * math.pi
        output = torch.cat([torch.cos(scaled), torch.sin(scaled)], dim=-1)
        if output.shape[-1] < self.d_embedding:
            pad = torch.zeros(
                *output.shape[:-1],
                self.d_embedding - output.shape[-1],
                device=output.device,
                dtype=output.dtype,
            )
            output = torch.cat([output, pad], dim=-1)
        return output


class PLRPlusFourierEmbeddings(nn.Module):
    def __init__(self, bins: list[Tensor], d_embedding: int, fourier_sigma: float) -> None:
        super().__init__()
        self.n_features = len(bins)
        d_fourier = max(2, int(d_embedding) // 2)
        d_plr = max(1, int(d_embedding) - d_fourier)
        self.linear0 = LinearEmbeddings(len(bins), d_plr)
        self.encoding = PiecewiseLinearEncoding(bins)
        self.linear = nn.Parameter(torch.zeros(len(bins), self.encoding.weight.shape[1], d_plr))
        self.fourier = FourierFeatures(len(bins), d_fourier, sigma=fourier_sigma)
        self.proj = nn.Linear(d_fourier, d_fourier)
        self.d_embedding = d_plr + d_fourier

    def get_output_shape(self) -> torch.Size:
        return torch.Size((self.n_features, self.d_embedding))

    def forward(self, x: Tensor) -> Tensor:
        x_ple = self.encoding(x).transpose(0, 1)
        x_ple = (x_ple @ self.linear).transpose(0, 1)
        return torch.cat([self.linear0(x) + x_ple, self.proj(self.fourier(x))], dim=-1)


def build_num_embedding_v6(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return None, 0, {"bins": []}
    embedding = PLRPlusFourierEmbeddings(
        compute_bins(x_num_train, int(trial_params["n_bins"])),
        int(trial_params["d_embedding"]),
        float(trial_params.get("fourier_sigma", 1.0)),
    )
    output_dim = dataset_meta.n_num_features * int(embedding.get_output_shape()[-1])
    return embedding, output_dim, {"d_embedding": int(embedding.get_output_shape()[-1])}


class SoftBinningEncoding(nn.Module):
    def __init__(self, bins: list[Tensor]) -> None:
        super().__init__()
        n_features = len(bins)
        n_bins_per_feature = [max(2, len(edges) - 1) for edges in bins]
        max_n_bins = max(n_bins_per_feature)
        centers = torch.zeros(n_features, max_n_bins, dtype=bins[0].dtype, device=bins[0].device)
        valid_mask = torch.zeros(n_features, max_n_bins, dtype=torch.bool)
        widths: list[float] = []
        for feature_idx, edges in enumerate(bins):
            if len(edges) == 1:
                edges = torch.tensor([float(edges[0]), float(edges[0]) + 1.0], dtype=edges.dtype, device=edges.device)
            mids = 0.5 * (edges[:-1] + edges[1:])
            centers[feature_idx, : mids.shape[0]] = mids
            valid_mask[feature_idx, : mids.shape[0]] = True
            widths.extend(((edges[1:] - edges[:-1]) ** 2).tolist())
        mean_width_sq = max(sum(widths) / max(1, len(widths)), 1e-8)
        self.register_buffer("centers", centers)
        self.register_buffer("valid_mask", valid_mask)
        self.log_temp = nn.Parameter(torch.full((n_features,), float(-math.log(mean_width_sq))))
        self.max_n_bins = max_n_bins

    def forward(self, x: Tensor) -> Tensor:
        logits = -torch.exp(self.log_temp).unsqueeze(-1) * (x[..., None] - self.centers).pow(2)
        return F.softmax(logits.masked_fill(~self.valid_mask, float("-inf")), dim=-1)


class PLRSoftEmbeddings(nn.Module):
    def __init__(self, bins: list[Tensor], d_embedding: int) -> None:
        super().__init__()
        self.n_features = len(bins)
        self.d_embedding = int(d_embedding)
        self.linear0 = LinearEmbeddings(len(bins), d_embedding)
        self.encoding = SoftBinningEncoding(bins)
        self.linear = nn.Parameter(
            torch.zeros(len(bins), self.encoding.max_n_bins, d_embedding)
        )

    def get_output_shape(self) -> torch.Size:
        return torch.Size((self.n_features, self.d_embedding))

    def forward(self, x: Tensor) -> Tensor:
        x_soft = self.encoding(x).transpose(0, 1)
        x_soft = (x_soft @ self.linear).transpose(0, 1)
        return self.linear0(x) + x_soft


def build_num_embedding_v7(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return None, 0, {"bins": []}
    bins = compute_bins(x_num_train, int(trial_params["n_bins"]))
    embedding = PLRSoftEmbeddings(bins, int(trial_params["d_embedding"]))
    output_dim = dataset_meta.n_num_features * int(trial_params["d_embedding"])
    return embedding, output_dim, {"bins": bins, "d_embedding": int(trial_params["d_embedding"])}


class PLREmbeddingsWithLayerNorm(nn.Module):
    def __init__(self, bins: list[Tensor], d_embedding: int) -> None:
        super().__init__()
        self.n_features = len(bins)
        self.d_embedding = int(d_embedding)
        self.linear0 = LinearEmbeddings(len(bins), d_embedding)
        self.encoding = PiecewiseLinearEncoding(bins)
        self.linear = nn.Parameter(torch.zeros(len(bins), self.encoding.weight.shape[1], d_embedding))
        self.layer_norm = nn.LayerNorm(d_embedding)

    def get_output_shape(self) -> torch.Size:
        return torch.Size((self.n_features, self.d_embedding))

    def forward(self, x: Tensor) -> Tensor:
        x_ple = self.encoding(x).transpose(0, 1)
        x_ple = (x_ple @ self.linear).transpose(0, 1)
        return self.layer_norm(self.linear0(x) + x_ple)


def build_num_embedding_v8(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return None, 0, {"bins": []}
    bins = compute_bins(x_num_train, int(trial_params["n_bins"]))
    embedding = PLREmbeddingsWithLayerNorm(bins, int(trial_params["d_embedding"]))
    output_dim = dataset_meta.n_num_features * int(trial_params["d_embedding"])
    return embedding, output_dim, {"bins": bins, "d_embedding": int(trial_params["d_embedding"])}


__all__ = [
    "build_num_embedding_v4",
    "build_num_embedding_v5",
    "build_num_embedding_v6",
    "build_num_embedding_v7",
    "build_num_embedding_v8",
]
