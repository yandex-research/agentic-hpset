from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bin.tabm.core import Dataset, TorchDataset

from .train import InferenceFn, LossFn, OptimizerBuilder, train_v0


def _off_diagonal_mean_square(gram: Tensor) -> Tensor:
    if gram.shape[0] <= 1:
        return gram.new_zeros(())
    eye = torch.eye(gram.shape[0], dtype=torch.bool, device=gram.device)
    return gram.masked_select(~eye).pow(2).mean()


def _decorrelation_penalty(model: nn.Module) -> Tensor:
    penalties: list[Tensor] = []
    for name, parameter in model.named_parameters():
        if not (name.endswith("backbone.affine.weight") and parameter.ndim == 2):
            continue
        vectors = parameter.float()
        if vectors.shape[0] <= 1:
            continue
        vectors = F.normalize(vectors, dim=1, eps=1e-6)
        penalties.append(_off_diagonal_mean_square(vectors @ vectors.T))
    if not penalties:
        return next(model.parameters()).new_zeros(())
    return torch.stack(penalties).mean()


def train_v7(
    model: nn.Module,
    dataset_tensors: TorchDataset,
    dataset_meta: Dataset,
    trial_params: dict[str, Any],
    inference_fn: InferenceFn,
    loss_fn: LossFn,
    optimizer_builder: OptimizerBuilder,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Train with the selected loss plus an orthogonality penalty on the per-head input affine."""
    decorrelation_lambda = float(trial_params.get("weight_decorrelation_lambda", 0.01))

    def decorrelated_loss(
        preds: Tensor, targets: Tensor, loss_dataset_meta: Dataset
    ) -> Tensor:
        return loss_fn(preds, targets, loss_dataset_meta) + (
            decorrelation_lambda * _decorrelation_penalty(model)
        )

    report, best_state = train_v0(
        model,
        dataset_tensors,
        dataset_meta,
        trial_params,
        inference_fn,
        decorrelated_loss,
        optimizer_builder,
        device,
        seed,
    )
    report["weight_decorrelation_lambda"] = decorrelation_lambda
    return report, best_state


__all__ = ["train_v7"]
