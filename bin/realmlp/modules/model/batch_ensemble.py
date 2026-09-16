"""RealMLP backbone (v1): BatchEnsemble hidden layers (TabM-style sharing).

Hypothesis: v0 trains ``n_ens`` fully independent members — the pre-TabM
design. TabM (Gorishniy et al. 2024) shows that sharing the MLP weights
across members and keeping only light per-member rank-1 multiplicative
adapters (BatchEnsemble; Wen et al. 2020) *outperforms* independent deep
ensembles on the standard tabular benchmarks while shrinking parameters
roughly ``n_ens``-fold: sharing regularizes on small-n where independent wide
members overfit. The per-dataset tuner selects it where that trade wins; on
datasets where independent-member diversity is the main gain it is deselected.

Mechanism: every hidden ``BatchedLinear`` becomes a ``BatchEnsembleLinear`` —
one shared ``(in, out)`` weight plus per-member input/output scale vectors
``r``/``s`` initialized with random +-1 signs (TabM's diversity-inducing
init) and per-member biases. The output head stays a fully per-member
``BatchedLinear`` (TabM keeps non-shared heads), as do the front scale and
parametric activations. The NTK data-dependent init is preserved: the shared
weight is std-normalized on pooled (member, batch) pre-activations, and the
he+5 bias init stays per-member.

``BatchEnsembleLinear`` subclasses ``BatchedLinear`` so the v0 optimizer's
per-parameter factor groups (weight / bias / first-layer) apply unchanged;
the r/s adapters fall into the default factor group.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .realmlp import BatchedLinear, RealMLPNet, _heplus_bias


def _sign_init(*shape: int) -> torch.Tensor:
    return 2.0 * torch.randint(0, 2, shape, dtype=torch.float32) - 1.0


class BatchEnsembleLinear(BatchedLinear):
    """Shared-weight linear with per-member rank-1 adapters and biases."""

    def __init__(self, n_models: int, in_features: int, out_features: int):
        nn.Module.__init__(self)  # skip BatchedLinear's per-member allocation
        self.n_models = n_models
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(n_models, out_features))
        self.r = nn.Parameter(_sign_init(n_models, in_features))
        self.s = nn.Parameter(_sign_init(n_models, out_features))
        self.factor = 1.0 / math.sqrt(in_features) if in_features > 0 else 1.0
        nn.init.normal_(self.weight)

    def _pre_activation(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        xr = x * self.r[:, None, :]
        return self.factor * torch.einsum("ebi,io->ebo", xr, weight) * self.s[:, None, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x[None, :, :].expand(self.n_models, -1, -1)
        return self._pre_activation(x, self.weight) + self.bias[:, None, :]

    def initialize_ntk_std(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == 0:
            return self.forward(x)
        if x.ndim == 2:
            x = x[None, :, :].expand(self.n_models, -1, -1)
        with torch.no_grad():
            raw_weight = torch.randn_like(self.weight)
            pre = self._pre_activation(x, raw_weight)
            # Shared weight -> pool the std over members and batch per output unit.
            std = pre.reshape(-1, pre.shape[-1]).std(dim=0, correction=0).clamp_min(1e-30)
            self.weight.copy_(raw_weight / std)
            pre = self._pre_activation(x, self.weight)
            self.bias.copy_(_heplus_bias(pre))
            return pre + self.bias[:, None, :]


class BatchEnsembleMLPNet(RealMLPNet):
    """RealMLPNet with hidden layers swapped to ``BatchEnsembleLinear``.

    The last linear (output head) stays fully per-member.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        linear_positions = [
            i for i, m in enumerate(self.net) if isinstance(m, BatchedLinear)
        ]
        for i in linear_positions[:-1]:  # keep the head per-member
            old = self.net[i]
            n_models, in_features, out_features = old.weight.shape
            self.net[i] = BatchEnsembleLinear(n_models, in_features, out_features)


def build_model_batch_ensemble(
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
    return BatchEnsembleMLPNet(
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
    "BatchEnsembleLinear",
    "BatchEnsembleMLPNet",
    "build_model_batch_ensemble",
]
