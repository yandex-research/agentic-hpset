from __future__ import annotations

import numpy as np

from ._base import softmax


class ProbaPostproc:
    """Plain softmax at a fixed temperature of 0.9; no fitting."""

    def fit(self, train_logits, y_train, n_classes, prior) -> "ProbaPostproc":
        self.temperature = 0.9
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return softmax(logits, temperature=self.temperature)


__all__ = ["ProbaPostproc"]
