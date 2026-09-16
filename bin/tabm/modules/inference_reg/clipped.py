from __future__ import annotations

from typing import Any

import numpy as np
import torch

from bin.tabm.core import Dataset, TorchDataset

from .inference import _predict_with_reducer
from .median import _median_heads


def inference_v7(
    model: torch.nn.Module,
    dataset_tensors: TorchDataset,
    dataset_meta: Dataset,
    trial_params: dict[str, Any],
    _device: torch.device,
    parts: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Median head aggregation followed by train-target quantile clipping."""
    clip_quantile = float(trial_params.get("clip_quantile", 0.005))
    clip_quantile = min(max(clip_quantile, 0.0), 0.49)
    lower, upper = np.quantile(
        dataset_meta.y_raw["train"],
        [clip_quantile, 1.0 - clip_quantile],
    )
    predictions = _predict_with_reducer(
        model, dataset_tensors, dataset_meta, trial_params, parts, _median_heads
    )
    return {
        part: np.clip(preds, lower, upper).astype(np.float32)
        for part, preds in predictions.items()
    }


__all__ = ["inference_v7"]
