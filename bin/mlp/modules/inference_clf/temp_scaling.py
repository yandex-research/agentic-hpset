"""Post-hoc temperature scaling.

After collecting raw logits for the requested parts (always including train),
fits a scalar temperature `T` on the training set by minimizing
NLL(logits / T) with LBFGS. `T` is clamped to `[0.7, 10.0]` to prevent
extreme scaling. All parts' logits are then divided by the same `T`.
Validation labels are never used to fit the calibrator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from bin.mlp.core import Dataset, TorchDataset


def _fit_temperature(
    logits_train: Tensor,
    y_train: Tensor,
    is_binclass: bool,
    t_min: float = 0.7,
    t_max: float = 10.0,
) -> float:
    raw = torch.zeros(1, requires_grad=True, device=logits_train.device)
    optimizer = torch.optim.LBFGS([raw], lr=0.1, max_iter=50)

    def closure() -> Tensor:
        optimizer.zero_grad()
        T = t_min + (t_max - t_min) * torch.sigmoid(raw)
        if is_binclass:
            loss = F.binary_cross_entropy_with_logits(logits_train / T, y_train.float())
        else:
            loss = F.cross_entropy(logits_train / T, y_train.long())
        loss.backward()
        return loss

    try:
        with torch.enable_grad():
            optimizer.step(closure)
    except RuntimeError:
        return 1.0
    return float(t_min + (t_max - t_min) * float(torch.sigmoid(raw).detach().cpu()))


def inference_v2(
    model: torch.nn.Module,
    dataset_tensors: TorchDataset,
    dataset_meta: Dataset,
    trial_params: dict[str, Any],
    _device: torch.device,
    parts: tuple[str, ...],
) -> dict[str, np.ndarray]:
    raw_logits: dict[str, Tensor] = {}
    batch_size = int(trial_params["eval_batch_size"])
    parts_to_compute = tuple(dict.fromkeys((*parts, "train")))
    for part in parts_to_compute:
        outputs: list[Tensor] = []
        indices = torch.arange(
            dataset_tensors.y[part].shape[0], device=dataset_tensors.y[part].device
        )
        for batch_idx in indices.split(batch_size):
            outputs.append(
                model(
                    None if dataset_tensors.x_num is None else dataset_tensors.x_num[part][batch_idx],
                    None if dataset_tensors.x_cat is None else dataset_tensors.x_cat[part][batch_idx],
                )
            )
        raw_logits[part] = torch.cat(outputs).detach()
    T = _fit_temperature(
        raw_logits["train"], dataset_tensors.y["train"], dataset_meta.is_binclass
    )
    predictions: dict[str, np.ndarray] = {}
    for part in parts:
        predictions[part] = (raw_logits[part] / T).cpu().numpy().astype(np.float32)
    return predictions


__all__ = ["inference_v2"]
