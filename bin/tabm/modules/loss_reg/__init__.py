from .huber import loss_reg_v1
from .loss_reg import loss_reg_v0

LOSS_REG_MAP = {0: loss_reg_v0, 1: loss_reg_v1}

__all__ = ["loss_reg_v0", "loss_reg_v1", "LOSS_REG_MAP"]
