from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from bin.mlp.core import Dataset, TorchDataset

from .utils import mean_regression_output


def _enable_dropout(model: torch.nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def inference_v1(
    model: torch.nn.Module,
    dataset_tensors: TorchDataset,
    dataset_meta: Dataset,
    trial_params: dict[str, Any],
    _device: torch.device,
    parts: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """MC Dropout: average predictions over multiple stochastic forward passes."""
    n_passes = 5
    batch_size = int(trial_params["eval_batch_size"])
    target_inverse_transform = dataset_meta.preprocess_artifacts["target"]["inverse_transform"]
    predictions: dict[str, np.ndarray] = {}
    for part in parts:
        all_pass_preds: list[np.ndarray] = []
        indices = torch.arange(
            dataset_tensors.y[part].shape[0], device=dataset_tensors.y[part].device
        )
        for _pass in range(n_passes):
            outputs: list[Tensor] = []
            model.eval()
            _enable_dropout(model)
            with torch.no_grad():
                for batch_idx in indices.split(batch_size):
                    outputs.append(
                        model(
                            None if dataset_tensors.x_num is None else dataset_tensors.x_num[part][batch_idx],
                            None if dataset_tensors.x_cat is None else dataset_tensors.x_cat[part][batch_idx],
                        )
                    )
            preds = torch.cat(outputs).detach().cpu().numpy()
            preds = mean_regression_output(preds)
            all_pass_preds.append(target_inverse_transform(preds).astype(np.float32))
        predictions[part] = np.mean(all_pass_preds, axis=0).astype(np.float32)
    model.eval()
    return predictions


__all__ = ["inference_v1"]
