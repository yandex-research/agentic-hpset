"""RealMLP optimizer (v0): Adam with per-parameter lr/wd factors.

The model and embeddings tag each parameter group with ``lr_factor`` and
``wd_factor`` based on which layer it belongs to (PLR embedding, front scale,
weight vs. bias, first hidden layer, parametric activation, learned cat
embedding). The train loop calls ``update_fn`` every step to set the live
``lr`` from the ``coslog4`` schedule, the live ``betas`` / ``eps``, and the
per-group ``decoupled_wd_step``. Decoupled lr-coupled weight decay is applied
manually via ``weight_decay_fn`` **before** ``optimizer.step()``:

    p *= 1 - (wd * wd_sched(t) * f_wd) * (lr * lr_sched(t) * f_lr) * f_wd * f_lr

So new optimizer variants can override either the optimizer object, the update
schedule, or the decay treatment without touching the train loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

import torch

from .._schedules import schedule_value
from ..embedding_num import PLREmbeddings
from ..model import BatchedLinear, ParametricMish, TrainableScale

@dataclass
class OptimizerBundle:
    optimizer: torch.optim.Optimizer
    update_fn: Callable[[torch.optim.Optimizer, Dict[str, Any], float], None]
    weight_decay_fn: Callable[[torch.optim.Optimizer], None]

def _build_param_groups(model: torch.nn.Module, cfg: Dict[str, Any]) -> list[Dict[str, Any]]:
    groups: list[Dict[str, Any]] = []
    first_linear_seen = False
    for module_name, module in model.named_modules():
        for local_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            lr_factor = 1.0
            wd_factor = 1.0
            if isinstance(module, PLREmbeddings):
                lr_factor *= float(cfg.get("plr_lr_factor", 0.1))
                wd_factor *= float(cfg.get("plr_wd_factor", 1.0))
            elif isinstance(module, TrainableScale):
                lr_factor *= float(cfg.get("scale_lr_factor", 6.0))
                wd_factor *= float(cfg.get("scale_wd_factor", 1.0))
            elif isinstance(module, BatchedLinear):
                if local_name == "weight":
                    if not first_linear_seen:
                        lr_factor *= float(cfg.get("first_layer_lr_factor", 1.0))
                        first_linear_seen = True
                    lr_factor *= float(cfg.get("weight_lr_factor", 1.0))
                    wd_factor *= float(cfg.get("weight_wd_factor", 1.0))
                elif local_name == "bias":
                    lr_factor *= float(cfg.get("bias_lr_factor", 0.1))
                    wd_factor *= float(cfg.get("bias_wd_factor", 0.0))
            elif isinstance(module, ParametricMish):
                lr_factor *= float(cfg.get("act_lr_factor", 0.1))
                wd_factor *= float(cfg.get("act_wd_factor", 1.0))
            elif "embedding_weights" in module_name:
                lr_factor *= float(cfg.get("emb_lr_factor", 1.0))
            groups.append({"params": [param], "lr_factor": lr_factor, "wd_factor": wd_factor})
    if not groups:
        raise ValueError("No trainable parameters found")
    return groups

def update_realmlp_optimizer_v0(
    optimizer: torch.optim.Optimizer, cfg: Dict[str, Any], t: float
) -> None:
    base_lr = float(cfg.get("lr", 1e-3)) * schedule_value(cfg.get("lr_sched", "coslog4"), t)
    base_wd = float(cfg.get("wd", 0.0)) * schedule_value(cfg.get("wd_sched", "flat_cos"), t)
    mom = float(cfg.get("mom", 0.9)) * schedule_value(cfg.get("mom_sched", "constant"), t)
    sq_mom = float(cfg.get("sq_mom", 0.95)) * schedule_value(cfg.get("sq_mom_sched", "constant"), t)
    opt_eps = float(cfg.get("opt_eps", 1e-8)) * schedule_value(cfg.get("opt_eps_sched", "constant"), t)
    for group in optimizer.param_groups:
        lr_factor = group.get("lr_factor", 1.0)
        wd_factor = group.get("wd_factor", 1.0)
        group["lr"] = base_lr * lr_factor
        group["betas"] = (mom, sq_mom)
        group["eps"] = opt_eps
        group["decoupled_wd_step"] = base_wd * base_lr * (wd_factor ** 2) * (lr_factor ** 2)

def apply_decoupled_weight_decay_v0(optimizer: torch.optim.Optimizer) -> None:
    with torch.no_grad():
        for group in optimizer.param_groups:
            wd_step = group.get("decoupled_wd_step", 0.0)
            if wd_step == 0.0:
                continue
            for param in group["params"]:
                param.mul_(1.0 - wd_step)

def build_optimizer_v0(model: torch.nn.Module, cfg: Dict[str, Any]) -> OptimizerBundle:
    opt_name = cfg.get("opt", "adam")
    if opt_name != "adam":
        raise ValueError(f"RealMLP v0 optimizer only supports opt='adam', got {opt_name!r}")
    groups = _build_param_groups(model, cfg)
    optimizer = torch.optim.Adam(
        groups,
        lr=float(cfg.get("lr", 1e-3)),
        betas=(float(cfg.get("mom", 0.9)), float(cfg.get("sq_mom", 0.95))),
        eps=float(cfg.get("opt_eps", 1e-8)),
        weight_decay=0.0,
    )
    return OptimizerBundle(
        optimizer=optimizer,
        update_fn=update_realmlp_optimizer_v0,
        weight_decay_fn=apply_decoupled_weight_decay_v0,
    )

__all__ = [
    "OptimizerBundle",
    "build_optimizer_v0",
    "update_realmlp_optimizer_v0",
    "apply_decoupled_weight_decay_v0",
    ]
