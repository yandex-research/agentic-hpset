"""RealMLP optimizer (v1): v0 Adam + global gradient-norm clipping.

Hypothesis: heavy-tailed features or targets occasionally produce exploding
gradients that a fixed meta-tuned learning rate cannot guard against on
outlier datasets; a loose global-norm clip (Pascanu et al. 2013 — universal
in NLP training stacks) bounds those steps without touching the tuned
mechanism. The entire v0 bundle (Adam, per-parameter lr/wd factor groups,
schedules, decoupled decay) is reused unchanged — clipping is added to the
pre-step hook, so it composes with the tuned structure instead of overriding
it. A too-tight clip slows learning; the default (``grad_clip_norm=10.0``)
rarely binds and the tuner deselects where it hurts.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from .realmlp import OptimizerBundle, build_optimizer_v0


def build_optimizer_grad_clip(model: torch.nn.Module, cfg: Dict[str, Any]) -> OptimizerBundle:
    """Mirror ``build_optimizer_v0``'s contract; clip before decay + step."""
    bundle = build_optimizer_v0(model, cfg)
    max_norm = float(cfg.get("grad_clip_norm", 10.0))
    params = [p for group in bundle.optimizer.param_groups for p in group["params"]]

    def clip_then_decay(optimizer: torch.optim.Optimizer) -> None:
        torch.nn.utils.clip_grad_norm_(params, max_norm)
        bundle.weight_decay_fn(optimizer)

    return OptimizerBundle(
        optimizer=bundle.optimizer,
        update_fn=bundle.update_fn,
        weight_decay_fn=clip_then_decay,
    )


__all__ = ["build_optimizer_grad_clip"]
