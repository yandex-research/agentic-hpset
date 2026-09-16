from __future__ import annotations

import torch

from bin.mlp.core import Dataset


def loss_reg_v2(
    preds: torch.Tensor,
    targets: torch.Tensor,
    dataset_meta: Dataset,
) -> torch.Tensor:
    if not dataset_meta.is_regression:
        raise RuntimeError(
            f"Gaussian-NLL regression loss does not support "
            f"task_type={dataset_meta.task_type!r}."
        )
    if preds.ndim < 2 or preds.shape[-1] != 2:
        raise RuntimeError(
            "Gaussian-NLL regression loss requires model outputs with shape "
            f"(..., 2), got {tuple(preds.shape)}."
        )
    mu = preds[..., 0]
    log_var = preds[..., 1].clamp(-7.0, 7.0)
    inv_var = torch.exp(-log_var)
    return (0.5 * (log_var + (targets - mu).pow(2) * inv_var)).mean()


def model_output_dim(dataset_meta: Dataset) -> int:
    if not dataset_meta.is_regression:
        raise RuntimeError(
            f"Gaussian-NLL regression loss does not support "
            f"task_type={dataset_meta.task_type!r}."
        )
    return 2


loss_reg_v2.model_output_dim = model_output_dim  # type: ignore[attr-defined]


__all__ = ["loss_reg_v2", "model_output_dim"]
