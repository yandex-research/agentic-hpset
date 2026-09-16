from __future__ import annotations

from typing import Any

import numpy as np
import torch

from bin.tabm.core import Dataset, TorchDataset

from .inference import _fit_temperature, _predict_aggregated_logits


def inference_v1(
    model: torch.nn.Module,
    dataset_tensors: TorchDataset,
    dataset_meta: Dataset,
    trial_params: dict[str, Any],
    _device: torch.device,
    parts: tuple[str, ...],
) -> dict[str, np.ndarray]:
    parts_to_run = tuple(dict.fromkeys((*parts, "train")))
    raw_predictions = _predict_aggregated_logits(
        model, dataset_tensors, dataset_meta, trial_params, parts_to_run
    )
    temperature = _fit_temperature(
        raw_predictions["train"],
        np.asarray(dataset_meta.y_raw["train"]).reshape(-1),
        dataset_meta,
    )
    trial_params["temperature"] = temperature
    return {
        part: (raw_predictions[part] / temperature).astype(np.float32)
        for part in parts
    }


__all__ = ["inference_v1"]
