from .grad_clip import build_optimizer_grad_clip
from .realmlp import (
    OptimizerBundle,
    apply_decoupled_weight_decay_v0,
    build_optimizer_v0,
    update_realmlp_optimizer_v0,
)


OPTIMIZER_MAP = {
    0: build_optimizer_v0,
    1: build_optimizer_grad_clip,
}

__all__ = [
    "OptimizerBundle",
    "build_optimizer_v0",
    "build_optimizer_grad_clip",
    "apply_decoupled_weight_decay_v0",
    "update_realmlp_optimizer_v0",
    "OPTIMIZER_MAP",
]
