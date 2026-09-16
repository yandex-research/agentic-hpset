from __future__ import annotations

from .model import (
    DEFAULT_CHECKPOINT_VERSION,
    DEFAULT_CLASSIFIER_CHECKPOINT_VERSION,
    DEFAULT_REGRESSOR_CHECKPOINT_VERSION,
    NanoTabICLv2,
    build_classifier_v0,
    build_model_v0,
    build_regressor_v0,
    load_pretrained_tabiclv2_classifier,
    load_pretrained_tabiclv2_regressor,
)

MODEL_REGISTRY = {0: build_model_v0}

__all__ = [
    "DEFAULT_CHECKPOINT_VERSION",
    "DEFAULT_CLASSIFIER_CHECKPOINT_VERSION",
    "DEFAULT_REGRESSOR_CHECKPOINT_VERSION",
    "MODEL_REGISTRY",
    "NanoTabICLv2",
    "build_classifier_v0",
    "build_model_v0",
    "build_regressor_v0",
    "load_pretrained_tabiclv2_classifier",
    "load_pretrained_tabiclv2_regressor",
]
