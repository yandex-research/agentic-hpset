from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegressionTargetTransform:
    y_context: np.ndarray
    y_mean: float
    y_std: float
    log_active: bool

    def inverse(self, values: np.ndarray) -> np.ndarray:
        raw = np.asarray(values, dtype=np.float32) * self.y_std + self.y_mean
        if self.log_active:
            raw = np.expm1(raw)
        return raw.astype(np.float32)


@dataclass(frozen=True)
class RegressionPostproc:
    """A regression target postprocessor characterized by one flag.

    `log` applies log1p to the target (only when all targets are non-negative).
    Normalization statistics and the in-context targets always come from the
    train part only.
    """

    log: bool

    def prepare(self, dataset, part: str) -> RegressionTargetTransform:
        y_raw = np.asarray(dataset.y_raw["train"], dtype=np.float32).reshape(-1)
        log_active = self.log and bool(np.all(y_raw >= 0.0))
        y_model = np.log1p(y_raw) if log_active else y_raw
        y_mean = float(y_model.mean())
        y_std = float(y_model.std() + 1e-6)
        y_context = ((y_model - y_mean) / y_std).astype(np.float32)
        return RegressionTargetTransform(
            y_context=y_context,
            y_mean=y_mean,
            y_std=y_std,
            log_active=log_active,
        )


__all__ = ["RegressionTargetTransform", "RegressionPostproc"]
