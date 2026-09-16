"""RealMLP backbone (v0).

Block layout (``hidden_sizes='rectangular'``, ``use_parametric_act=True``,
``add_front_scale=True``):

    front_scale -> (BatchedLinear -> ParametricMish -> SmoothDropout) * n_hidden_layers -> BatchedLinear

The model assumes the input feature tensor already includes the small-cat
one-hot block (handled by ``preprocess_categorical``). The embeddings for the
continuous block (PLR-style) and the large-cat block (learned tables) are
passed in pre-built so this module never imports from the other stages.

Data-dependent init: ``initialize_from_data(x_cont, x_cat)`` walks the network
once with a small training slice to set ``BatchedLinear`` weights (NTK std init
so the post-linear stddev is 1) and biases (``he+5``: negative simplex combo of
five random pre-activations).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def _mish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.tanh(F.softplus(x))

def _heplus_bias(x: torch.Tensor, n_simplex: int = 5) -> torch.Tensor:
    n_models, n_samples, n_features = x.shape
    if n_samples == 0:
        return x.new_zeros((n_models, n_features))
    idxs = torch.randint(0, n_samples, size=(n_models, n_features, n_simplex), device=x.device)
    simplex_weights = torch.distributions.Exponential(1.0).sample((n_models, n_features, n_simplex)).to(x.device)
    simplex_weights = simplex_weights / simplex_weights.sum(dim=2, keepdim=True)
    selected = x.transpose(1, 2).gather(2, idxs)
    return -(selected * simplex_weights).sum(dim=2)

class ParametricMish(nn.Module):
    def __init__(self, n_models: int, n_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_models, n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (_mish(x) - x) * self.weight[:, None, :]

class TrainableScale(nn.Module):
    def __init__(self, n_models: int, n_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_models, n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight[:, None, :]

class BatchedLinear(nn.Module):
    def __init__(self, n_models: int, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_models, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(n_models, out_features))
        self.factor = 1.0 / math.sqrt(in_features) if in_features > 0 else 1.0
        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x[None, :, :].expand(self.weight.shape[0], -1, -1)
        return self.factor * torch.einsum("ebi,eio->ebo", x, self.weight) + self.bias[:, None, :]

    def initialize_ntk_std(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == 0:
            return self.forward(x)
        with torch.no_grad():
            raw_weight = torch.randn_like(self.weight)
            pre = self.factor * torch.einsum("ebi,eio->ebo", x, raw_weight)
            std = pre.std(dim=1, correction=0, keepdim=True).clamp_min(1e-30)
            self.weight.copy_(raw_weight / std)
            pre = self.factor * torch.einsum("ebi,eio->ebo", x, self.weight)
            self.bias.copy_(_heplus_bias(pre))
            return pre + self.bias[:, None, :]

class SmoothDropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, p=self.p, training=self.training)

class RealMLPNet(nn.Module):
    def __init__(
        self,
        num_embedding: Optional[nn.Module],
        cat_embedding: Optional[nn.Module],
        num_embedding_out: int,
        n_num: int,
        n_one_hot: int,
        cat_embedding_out: int,
        cfg: Dict[str, Any],
        *,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.n_models = int(cfg.get("n_ens", 1))
        self.n_num = n_num
        self.n_one_hot = n_one_hot
        self.num_embedding = num_embedding
        self.cat_embedding = cat_embedding

        n_num_features = num_embedding_out if num_embedding is not None else n_num
        in_features = n_num_features + n_one_hot + cat_embedding_out

        self.front_scale = (
            TrainableScale(self.n_models, in_features) if cfg.get("add_front_scale", True) else None
        )

        hidden_sizes = cfg.get("hidden_sizes", "rectangular")
        if hidden_sizes == "rectangular":
            hidden_sizes = [int(cfg.get("hidden_width", 256))] * int(cfg.get("n_hidden_layers", 3))
        else:
            hidden_sizes = list(hidden_sizes)

        layers: List[nn.Module] = []
        p_drop = float(cfg.get("p_drop", 0.0))
        use_parametric_act = bool(cfg.get("use_parametric_act", True))
        prev = in_features
        for width in hidden_sizes:
            layers.append(BatchedLinear(self.n_models, prev, width))
            layers.append(ParametricMish(self.n_models, width) if use_parametric_act else nn.Mish())
            if p_drop > 0.0:
                layers.append(SmoothDropout(p_drop))
            prev = width
        layers.append(BatchedLinear(self.n_models, prev, out_dim))
        self.net = nn.Sequential(*layers)

    def set_dropout(self, p: float) -> None:
        for module in self.modules():
            if isinstance(module, SmoothDropout):
                module.p = float(p)

    def _features(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        xs: List[torch.Tensor] = []
        x_is_batched = x_cont.ndim == 3
        if self.n_num:
            x_num = x_cont[..., : self.n_num]
            if self.num_embedding is not None:
                xs.append(self.num_embedding(x_num))
            else:
                xs.append(
                    x_num if x_is_batched else x_num[None, :, :].expand(self.n_models, -1, -1)
                )
        if self.n_one_hot:
            x_one_hot = x_cont[..., self.n_num : self.n_num + self.n_one_hot]
            xs.append(
                x_one_hot if x_is_batched else x_one_hot[None, :, :].expand(self.n_models, -1, -1)
            )
        if self.cat_embedding is not None:
            xs.append(self.cat_embedding(x_cat))
        x = (
            torch.cat(xs, dim=2)
            if xs
            else torch.zeros((self.n_models, x_cont.shape[-2], 0), device=x_cont.device)
        )
        if self.front_scale is not None:
            x = self.front_scale(x)
        return x

    def initialize_from_data(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> None:
        self.eval()
        with torch.no_grad():
            x = self._features(x_cont, x_cat)
            for module in self.net:
                if isinstance(module, BatchedLinear):
                    x = module.initialize_ntk_std(x)
                elif isinstance(module, SmoothDropout):
                    continue
                else:
                    x = module(x)

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        x = self._features(x_cont, x_cat)
        return self.net(x)

def build_model_v0(
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
    """Compose the v0 RealMLPNet around the pre-built embeddings.

    Embeddings are built and passed in by the wrapper so the model module never
    imports from ``embedding_num``/``embedding_cat`` directly — that keeps each
    stage swappable in isolation.
    """
    return RealMLPNet(
        num_embedding=num_embedding,
        cat_embedding=cat_embedding,
        num_embedding_out=num_embedding_out,
        n_num=n_num,
        n_one_hot=n_one_hot,
        cat_embedding_out=cat_embedding_out,
        cfg=cfg,
        out_dim=out_dim,
    )

__all__ = [
    "ParametricMish",
    "TrainableScale",
    "BatchedLinear",
    "SmoothDropout",
    "RealMLPNet",
    "build_model_v0",
    ]
