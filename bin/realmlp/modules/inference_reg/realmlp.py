"""RealMLP regression inference (v0).

- ``score_regressor`` returns the RMSE on the (denormalized, optionally clamped)
  ensemble-mean prediction. Used inside the train loop for best-checkpoint
  selection.
- ``predict_members_reg_v0`` consumes each ``FittedMember``'s ``y_mean`` /
  ``y_std`` / ``y_min`` / ``y_max`` to denormalize + clamp, averages across
  members, and reshapes back to the original target shape.
- ``predict_ensemble_reg`` returns the per-member predictions stacked along
  the ensemble dim (used by ``predict_ensemble``).
- ``raw_member_predict`` is re-exported from ``inference_clf`` so the two
  inference stages share the same forward-pass primitive.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..inference_clf.realmlp import _forward_with_oom_retry, raw_member_predict

def score_regressor(
    model: torch.nn.Module,
    x_cont: torch.Tensor,
    x_cat: torch.Tensor,
    y: torch.Tensor,
    cfg: Dict[str, Any],
    y_mean: Optional[np.ndarray],
    y_std: Optional[np.ndarray],
    y_min: Optional[np.ndarray],
    y_max: Optional[np.ndarray],
) -> float:
    model.eval()
    with torch.no_grad():
        raw = _forward_with_oom_retry(model, x_cont, x_cat, cfg)
        pred = raw.mean(dim=0)
        if y_mean is not None and y_std is not None:
            mean = torch.as_tensor(y_mean, dtype=pred.dtype, device=pred.device)
            std = torch.as_tensor(y_std, dtype=pred.dtype, device=pred.device)
            pred = pred * std[None, :] + mean[None, :]
        if cfg.get("clamp_output", False) and y_min is not None and y_max is not None:
            low = torch.as_tensor(y_min, dtype=pred.dtype, device=pred.device)
            high = torch.as_tensor(y_max, dtype=pred.dtype, device=pred.device)
            pred = torch.minimum(torch.maximum(pred, low[None, :]), high[None, :])
        return float(torch.sqrt(F.mse_loss(pred, y)).item())

def predict_members_reg_v0(
    members: List[Any],
    X: pd.DataFrame,
    params: Dict[str, Any],
    *,
    is_y_1d: bool,
    is_y_float64: bool,
) -> np.ndarray:
    """Denormalize + optionally clamp, then average across members.

    Returns ``(N,)`` when ``is_y_1d`` and the model has a 1-dim output, else
    ``(N, out_dim)``.
    """
    preds: List[np.ndarray] = []
    for member in members:
        pred = raw_member_predict(member.preprocessor, member.model, X, params).numpy()
        if member.y_mean is not None and member.y_std is not None:
            pred = pred * member.y_std[None, None, :] + member.y_mean[None, None, :]
        if params.get("clamp_output", False) and member.y_min is not None and member.y_max is not None:
            pred = np.clip(pred, member.y_min[None, None, :], member.y_max[None, None, :])
        preds.append(pred.mean(axis=0))
    out = np.mean(preds, axis=0)
    if is_y_1d and out.shape[1] == 1:
        out = out[:, 0]
    if is_y_float64:
        out = out.astype(np.float64)
    return out

def predict_ensemble_reg(
    members: List[Any],
    X: pd.DataFrame,
    params: Dict[str, Any],
    *,
    is_y_1d: bool,
) -> np.ndarray:
    """Per-member denormalized predictions stacked along the ensemble dim.

    Returns ``(sum_member_n_ens, N)`` when ``is_y_1d`` else ``(sum_member_n_ens, N, out_dim)``.
    """
    preds: List[np.ndarray] = []
    for member in members:
        pred = raw_member_predict(member.preprocessor, member.model, X, params).numpy()
        if member.y_mean is not None and member.y_std is not None:
            pred = pred * member.y_std[None, None, :] + member.y_mean[None, None, :]
        preds.append(pred[:, :, 0] if is_y_1d and pred.shape[-1] == 1 else pred)
    return np.concatenate(preds, axis=0)

__all__ = [
    "score_regressor",
    "predict_members_reg_v0",
    "predict_ensemble_reg",
    "raw_member_predict",
    ]
