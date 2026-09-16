from .batch_ensemble import (
    BatchEnsembleLinear,
    BatchEnsembleMLPNet,
    build_model_batch_ensemble,
)
from .realmlp import (
    BatchedLinear,
    ParametricMish,
    RealMLPNet,
    SmoothDropout,
    TrainableScale,
    build_model_v0,
)
from .residual import ResidualBlock, ResidualMLPNet, build_model_residual


MODEL_MAP = {
    0: build_model_v0,
    1: build_model_batch_ensemble,
    2: build_model_residual,
}

__all__ = [
    "BatchedLinear",
    "BatchEnsembleLinear",
    "BatchEnsembleMLPNet",
    "ParametricMish",
    "RealMLPNet",
    "SmoothDropout",
    "TrainableScale",
    "ResidualBlock",
    "ResidualMLPNet",
    "build_model_v0",
    "build_model_batch_ensemble",
    "build_model_residual",
    "MODEL_MAP",
]
