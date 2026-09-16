"""Tree-based PLR bins.

Replaces quantile-based bin edges with target-aware bin edges from a shallow
DecisionTree per numeric feature. The tree finds split points that are most
informative for the target, giving PLR a better discretization than quantiles.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from torch import Tensor

from .piecewise_linear import PLREmbeddings


def _compute_tree_bins(
    x: Tensor,
    y: np.ndarray,
    n_bins: int,
    task_type: str,
    seed: int,
) -> list[Tensor]:
    x_np = x.detach().cpu().numpy()
    edges_per_feature: list[Tensor] = []
    for col in range(x_np.shape[1]):
        column = x_np[:, col]
        unique_vals = np.unique(column)
        if unique_vals.shape[0] <= 2:
            edge_arr = np.array([column.min(), column.max()], dtype=np.float64)
            edges_per_feature.append(
                torch.as_tensor(edge_arr, device=x.device, dtype=x.dtype).unique()
            )
            continue
        if task_type in ("binclass", "multiclass"):
            tree = DecisionTreeClassifier(
                max_leaf_nodes=max(2, n_bins),
                min_samples_leaf=max(1, column.shape[0] // (n_bins * 8)),
                random_state=seed,
            )
        else:
            tree = DecisionTreeRegressor(
                max_leaf_nodes=max(2, n_bins),
                min_samples_leaf=max(1, column.shape[0] // (n_bins * 8)),
                random_state=seed,
            )
        try:
            tree.fit(column.reshape(-1, 1), y)
            thresholds = tree.tree_.threshold[tree.tree_.feature >= 0]
        except Exception:
            thresholds = np.array([], dtype=np.float64)
        edges = np.concatenate([[column.min()], np.sort(thresholds), [column.max()]])
        edges_per_feature.append(
            torch.as_tensor(edges, device=x.device, dtype=x.dtype).unique()
        )
    return edges_per_feature


def build_num_embedding_v6(
    x_num_train: Tensor | None,
    dataset_meta: Any,
    trial_params: dict[str, Any],
    _device: torch.device,
) -> tuple[nn.Module | None, int, dict[str, object]]:
    if x_num_train is None or dataset_meta.n_num_features == 0:
        return None, 0, {"bins": []}
    y_train = dataset_meta.y_raw["train"]
    bins = _compute_tree_bins(
        x_num_train,
        y_train,
        int(trial_params["n_bins"]),
        dataset_meta.task_type,
        seed=0,
    )
    embedding = PLREmbeddings(bins, int(trial_params["d_embedding"]))
    output_dim = dataset_meta.n_num_features * int(trial_params["d_embedding"])
    return embedding, output_dim, {"bins": bins}
