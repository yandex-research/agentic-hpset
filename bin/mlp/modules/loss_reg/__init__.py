from .gaussian_nll import loss_reg_v2
from .huber import loss_reg_v1
from .loss_reg import loss_reg_v0

LOSS_REG_MAP = {0: loss_reg_v0, 1: loss_reg_v1, 2: loss_reg_v2}

__all__ = ["loss_reg_v0", "loss_reg_v1", "loss_reg_v2", "LOSS_REG_MAP"]
