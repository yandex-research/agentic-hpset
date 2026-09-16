"""Schedule helpers shared by the train loop and the optimizer step."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def schedule_value(name: Optional[str], t: float) -> float:
    """Schedule multiplier at fractional training progress ``t in [0, 1]``.

    Supported names: ``constant``/``flat``, ``flat_cos``, ``cos``, ``coslog<N>``.
    Used for lr/wd/dropout/label-smoothing scheduling.
    """
    t = float(np.clip(t, 0.0, 1.0))
    if name is None or name in ("constant", "flat"):
        return 1.0
    if name == "flat_cos":
        if t <= 0.5:
            return 1.0
        u = 2.0 * (t - 0.5)
        return 0.5 * (1.0 + math.cos(math.pi * u))
    if name == "cos":
        return 0.5 * (1.0 + math.cos(math.pi * t))
    if name.startswith("coslog"):
        n_cycles = int(name[len("coslog"):])
        return 0.5 * (1.0 - math.cos(2.0 * math.pi * math.log2(1.0 + (2**n_cycles - 1.0) * t)))
    raise ValueError(f"Unsupported schedule {name!r}")


def mish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.tanh(F.softplus(x))


__all__ = ["schedule_value", "mish"]
