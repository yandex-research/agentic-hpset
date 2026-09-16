from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy import stats

from bin.tabm.core import Dataset, TorchDataset

from .inference import _predict_with_reducer, _squeeze_output


def inference_v2(
    model: torch.nn.Module,
    dataset_tensors: TorchDataset,
    dataset_meta: Dataset,
    trial_params: dict[str, Any],
    _device: torch.device,
    parts: tuple[str, ...],
) -> dict[str, np.ndarray]:
    trim_fraction = float(trial_params.get("trim_fraction", 0.2))

    def reducer(preds: np.ndarray) -> np.ndarray:
        preds = _squeeze_output(preds)
        if preds.ndim == 2:
            return stats.trim_mean(preds, proportiontocut=trim_fraction, axis=1)
        return preds

    return _predict_with_reducer(
        model, dataset_tensors, dataset_meta, trial_params, parts, reducer
    )


__all__ = ["inference_v2"]
