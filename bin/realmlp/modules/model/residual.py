"""RealMLP backbone (v2): pre-activation residual blocks with NTK init.

Hypothesis: the tabular ResNet of Gorishniy et al. 2021 is a top-tier
replicated baseline, and skip connections are what make depth a usable
dimension. v0's plain MLP was meta-tuned at 3 hidden layers; when the
per-dataset tuner samples deeper configs, plain-MLP signal propagation
degrades while residual paths keep the identity component intact. At shallow
depth the skip adds little and can slightly perturb the tuned dynamics — the
tuner deselects it there.

Mechanism: the first hidden layer stays a plain projection
(BatchedLinear -> ParametricMish -> SmoothDropout, exactly v0), and each
further hidden layer becomes a pre-activation residual block
``x + Linear(Drop(Act(x)))``. No norm layers are added (RealMLP's
data-dependent init handles scale): each block's linear is NTK-initialized on
the actual block input, and the head's NTK init absorbs the residual variance
growth. With ``n_hidden_layers == 1`` the model is identical to v0.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .realmlp import (
    BatchedLinear,
    ParametricMish,
    RealMLPNet,
    SmoothDropout,
)


class ResidualBlock(nn.Module):
    def __init__(self, n_models: int, width: int, p_drop: float, use_parametric_act: bool):
        super().__init__()
        self.act = ParametricMish(n_models, width) if use_parametric_act else nn.Mish()
        self.drop = SmoothDropout(p_drop) if p_drop > 0.0 else None
        self.linear = BatchedLinear(n_models, width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(x)
        if self.drop is not None:
            h = self.drop(h)
        return x + self.linear(h)

    def initialize_ntk(self, x: torch.Tensor) -> torch.Tensor:
        # eval mode during the init walk -> dropout is inactive.
        return x + self.linear.initialize_ntk_std(self.act(x))


class ResidualMLPNet(RealMLPNet):
    """RealMLPNet whose hidden layers 2..L are pre-activation residual blocks."""

    def __init__(self, *args: Any, cfg: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(*args, cfg=cfg, **kwargs)
        hidden_sizes = cfg.get("hidden_sizes", "rectangular")
        if hidden_sizes == "rectangular":
            hidden_sizes = [int(cfg.get("hidden_width", 256))] * int(
                cfg.get("n_hidden_layers", 3)
            )
        else:
            hidden_sizes = list(hidden_sizes)
        p_drop = float(cfg.get("p_drop", 0.0))
        use_parametric_act = bool(cfg.get("use_parametric_act", True))

        linears = [m for m in self.net if isinstance(m, BatchedLinear)]
        in_features = linears[0].weight.shape[1]
        out_dim = linears[-1].weight.shape[2]
        width = hidden_sizes[0]
        if any(w != width for w in hidden_sizes):
            raise ValueError("ResidualMLPNet requires rectangular hidden sizes")

        layers: List[nn.Module] = [BatchedLinear(self.n_models, in_features, width)]
        layers.append(
            ParametricMish(self.n_models, width) if use_parametric_act else nn.Mish()
        )
        if p_drop > 0.0:
            layers.append(SmoothDropout(p_drop))
        for _ in range(len(hidden_sizes) - 1):
            layers.append(ResidualBlock(self.n_models, width, p_drop, use_parametric_act))
        layers.append(BatchedLinear(self.n_models, width, out_dim))
        self.net = nn.Sequential(*layers)

    def initialize_from_data(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> None:
        self.eval()
        with torch.no_grad():
            x = self._features(x_cont, x_cat)
            for module in self.net:
                if isinstance(module, ResidualBlock):
                    x = module.initialize_ntk(x)
                elif isinstance(module, BatchedLinear):
                    x = module.initialize_ntk_std(x)
                elif isinstance(module, SmoothDropout):
                    continue
                else:
                    x = module(x)


def build_model_residual(
    num_embedding: Optional[nn.Module],
    cat_embedding: Optional[nn.Module],
    *,
    num_embedding_out: int,
    n_num: int,
    n_one_hot: int,
    cat_embedding_out: int,
    cfg: Dict[str, Any],
    out_dim: int,
) -> nn.Module:
    """Mirror ``build_model_v0``'s signature and returned-object contract."""
    return ResidualMLPNet(
        num_embedding=num_embedding,
        cat_embedding=cat_embedding,
        num_embedding_out=num_embedding_out,
        n_num=n_num,
        n_one_hot=n_one_hot,
        cat_embedding_out=cat_embedding_out,
        cfg=cfg,
        out_dim=out_dim,
    )


__all__ = ["ResidualBlock", "ResidualMLPNet", "build_model_residual"]
