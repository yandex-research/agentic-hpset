from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import LabelEncoder


@dataclass
class ClassificationTargetEncoder:
    encoder: LabelEncoder
    y_train: np.ndarray
    n_classes: int
    prior: np.ndarray

    @classmethod
    def fit(cls, dataset) -> "ClassificationTargetEncoder":
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(dataset.y["train"]).astype(np.int64)
        n_classes = len(encoder.classes_)
        prior = np.bincount(y_train, minlength=n_classes).astype(np.float64)
        prior = prior / max(float(prior.sum()), 1.0)
        return cls(
            encoder=encoder,
            y_train=y_train,
            n_classes=n_classes,
            prior=prior.astype(np.float32),
        )

    def transform(self, values) -> np.ndarray:
        return self.encoder.transform(values).astype(np.int64)


def softmax(logits: np.ndarray, *, temperature: float = 0.9) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64) / float(temperature)
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(np.float32)


def undo_class_permutation(
    logits: np.ndarray, class_permutation: tuple[int, ...] | None
) -> np.ndarray:
    if class_permutation is None:
        return logits
    return logits[..., list(class_permutation)]


__all__ = [
    "ClassificationTargetEncoder",
    "softmax",
    "undo_class_permutation",
]
