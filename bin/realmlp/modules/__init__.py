"""RealMLP modular building blocks with a per-stage registry layout.

Each subfolder owns one pipeline stage (preprocess_numerical, ..., inference_reg)
and exposes a ``*_MAP`` dict mapping integer indices to builder functions. This
file aggregates them into ``IMPLEMENTATION_REGISTRIES`` and provides ``resolve()``
so the wrapper can dispatch on indices uniformly.

Hyperparameters are not sampled in Python: every stage reads what it needs from
the shared ``cfg`` dict (with in-code defaults), and the search space — including
the per-stage ``implementation_indices`` — is declared entirely in the tuning
config's ``[space]`` and sampled by ``bin/tune.py``.

Adding a new variant is purely additive: drop a new ``foo.py`` under e.g.
``optimizer/``, define ``build_optimizer_foo(model, cfg)``, and append the map
entry in ``optimizer/__init__.py``.
"""

from __future__ import annotations

from typing import Any

from .embedding_cat import CAT_EMBEDDING_MAP
from .embedding_num import NUM_EMBEDDING_MAP
from .inference_clf import INFERENCE_CLF_MAP
from .inference_reg import INFERENCE_REG_MAP
from .loss_clf import LOSS_CLF_MAP
from .loss_reg import LOSS_REG_MAP
from .model import MODEL_MAP
from .optimizer import OPTIMIZER_MAP
from .preprocess import TabPreprocessor
from .preprocess_categorical import CAT_PREPROCESS_MAP
from .preprocess_numerical import NUM_PREPROCESS_MAP
from .preprocess_target_clf import TARGET_PREPROCESS_CLF_MAP
from .preprocess_target_reg import TARGET_PREPROCESS_REG_MAP
from .train import TRAIN_MAP


STAGE_KEYS: tuple[str, ...] = (
    "numerical_preprocess",
    "categorical_preprocess",
    "target_preprocess",      # dispatched to clf/reg by task_type
    "num_embedding",
    "cat_embedding",
    "model",
    "train",
    "optimizer",
    "inference",              # dispatched to clf/reg by task_type
    "loss",                   # dispatched to clf/reg by task_type
)


IMPLEMENTATION_REGISTRIES: dict[str, dict[int, Any]] = {
    "numerical_preprocess":   NUM_PREPROCESS_MAP,
    "categorical_preprocess": CAT_PREPROCESS_MAP,
    "target_preprocess_clf":  TARGET_PREPROCESS_CLF_MAP,
    "target_preprocess_reg":  TARGET_PREPROCESS_REG_MAP,
    "num_embedding":          NUM_EMBEDDING_MAP,
    "cat_embedding":          CAT_EMBEDDING_MAP,
    "model":                  MODEL_MAP,
    "train":                  TRAIN_MAP,
    "optimizer":              OPTIMIZER_MAP,
    "inference_clf":          INFERENCE_CLF_MAP,
    "inference_reg":          INFERENCE_REG_MAP,
    "loss_clf":               LOSS_CLF_MAP,
    "loss_reg":               LOSS_REG_MAP,
}


_TASK_DISPATCHED: set[str] = {"inference", "loss", "target_preprocess"}


def registry_name(stage_key: str, task_type: str) -> str:
    """Map a stage key to a concrete registry key, splitting clf/reg by ``task_type``.

    ``task_type`` is anything containing ``"regression"`` (reg suffix) or anything
    else (clf suffix). Stage keys not in ``_TASK_DISPATCHED`` are returned as-is.
    """
    if stage_key in _TASK_DISPATCHED:
        suffix = "reg" if task_type == "regression" else "clf"
        return f"{stage_key}_{suffix}"
    return stage_key


def resolve(stage_key: str, idx: int, task_type: str) -> Any:
    """Return the builder registered at index ``idx`` for the given stage.

    Raises ``ValueError`` with the available indices when ``idx`` is unknown.
    """
    name = registry_name(stage_key, task_type)
    registry = IMPLEMENTATION_REGISTRIES[name]
    if idx not in registry:
        choices = ", ".join(str(k) for k in sorted(registry))
        raise ValueError(f"Unknown {name} index {idx}. Available: [{choices}]")
    return registry[idx]


__all__ = [
    "STAGE_KEYS",
    "IMPLEMENTATION_REGISTRIES",
    "registry_name",
    "resolve",
    "TabPreprocessor",
]
