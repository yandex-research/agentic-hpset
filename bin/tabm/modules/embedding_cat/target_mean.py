from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def _compute_target_mean_table(
    cat_train: np.ndarray,
    target_train: np.ndarray,
    cardinalities: list[int],
    n_classes: int,
    is_regression: bool,
    is_binclass: bool,
    smoothing: float,
) -> np.ndarray:
    n_features = cat_train.shape[1]
    max_cardinality = max(int(cardinality) + 1 for cardinality in cardinalities)
    if is_regression or is_binclass:
        n_targets = 1
        target_matrix = target_train.reshape(-1, 1).astype(np.float32)
        global_mean = np.array([float(target_train.mean())], dtype=np.float32)
    else:
        n_targets = int(n_classes)
        target_matrix = np.eye(n_targets, dtype=np.float32)[
            target_train.astype(np.int64)
        ]
        global_mean = target_matrix.mean(axis=0)
    table = np.broadcast_to(
        global_mean.reshape(1, 1, n_targets),
        (n_features, max_cardinality, n_targets),
    ).copy()
    for feature_idx in range(n_features):
        column = cat_train[:, feature_idx].astype(np.int64)
        for value in range(max_cardinality):
            mask = column == value
            count = int(mask.sum())
            if count == 0:
                continue
            weight = count / (count + smoothing)
            table[feature_idx, value] = (
                weight * target_matrix[mask].mean(axis=0) + (1.0 - weight) * global_mean
            )
    return table


class TargetMeanCategoricalEmbedding(nn.Module):
    def __init__(self, table: np.ndarray) -> None:
        super().__init__()
        n_features, max_cardinality, n_targets = table.shape
        self.n_features = int(n_features)
        self.max_cardinality = int(max_cardinality)
        self.n_targets = int(n_targets)
        self.tables = nn.ParameterList(
            [
                nn.Parameter(torch.tensor(table[feature_idx], dtype=torch.float32))
                for feature_idx in range(n_features)
            ]
        )
        self.d_features = [self.n_targets] * self.n_features

    def get_output_shape(self) -> torch.Size:
        return torch.Size((self.n_features, self.n_targets))

    def forward(self, x: Tensor) -> Tensor:
        return torch.stack(
            [
                table[x[:, feature_idx].clamp(0, self.max_cardinality - 1)]
                for feature_idx, table in enumerate(self.tables)
            ],
            dim=1,
        )


def build_cat_embedding_v3(
    cat_cardinalities: list[int],
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if not cat_cardinalities:
        return None, 0, {"cardinalities": []}
    if dataset_meta.x_cat is None:
        return None, 0, {"cardinalities": cat_cardinalities}
    cat_train = np.asarray(dataset_meta.x_cat["train"]).astype(np.int64)
    y_train = np.asarray(dataset_meta.y["train"]).reshape(-1)
    smoothing = float(trial_params.get("target_smoothing", 10.0))
    table = _compute_target_mean_table(
        cat_train,
        y_train,
        cat_cardinalities,
        int(dataset_meta.n_classes),
        bool(dataset_meta.is_regression),
        bool(dataset_meta.is_binclass),
        smoothing,
    )
    embedding = TargetMeanCategoricalEmbedding(table)
    return (
        embedding,
        embedding.n_features * embedding.n_targets,
        {
            "cardinalities": cat_cardinalities,
            "n_targets": embedding.n_targets,
            "target_smoothing": smoothing,
        },
    )


__all__ = ["TargetMeanCategoricalEmbedding", "build_cat_embedding_v3"]
