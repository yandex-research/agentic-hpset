"""RealMLP numerical embedding (v0): PBLD-style PLR with densenet skip.

Per numerical feature ``f`` (treated as a batched dim):

    z = cos(2*pi * x_f * W1_f + b1_f)              # W1_f: (1, plr_hidden_1)
    z = z @ W2_f + b2_f                             # W2_f: (plr_hidden_1, plr_hidden_2 - 1)
    out_f = concat([z, x_f])                        # densenet -> plr_hidden_2 channels

Output is flattened to ``(n_ens, batch, n_num * plr_hidden_2)``.

The embedding is built only when ``cfg["num_emb_type"]`` is one of
``pbld | pblrd | pl | plr``; otherwise the wrapper passes raw numerical features
straight to the MLP (the v0 default is ``pbld``).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

class PLREmbeddings(nn.Module):
    def __init__(self, n_models: int, n_cont: int, plr_hidden_1: int, plr_hidden_2: int, plr_sigma: float):
        super().__init__()
        self.n_models = n_models
        self.n_cont = n_cont
        self.plr_hidden_1 = plr_hidden_1
        self.plr_hidden_2 = plr_hidden_2
        hidden_2_without_dense = max(0, plr_hidden_2 - 1)
        self.weight_1 = nn.Parameter(plr_sigma * torch.randn(n_models, n_cont, plr_hidden_1))
        self.bias_1 = nn.Parameter(math.pi * (-1.0 + 2.0 * torch.rand(n_models, n_cont, plr_hidden_1)))
        self.weight_2 = nn.Parameter(
            (-1.0 + 2.0 * torch.rand(n_models, n_cont, plr_hidden_1, hidden_2_without_dense))
            / math.sqrt(plr_hidden_1)
        )
        self.bias_2 = nn.Parameter(
            (-1.0 + 2.0 * torch.rand(n_models, n_cont, hidden_2_without_dense)) / math.sqrt(plr_hidden_1)
        )

    @property
    def out_features(self) -> int:
        return self.n_cont * self.plr_hidden_2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_cont == 0:
            batch = x.shape[-2]
            return x.new_zeros((self.n_models, batch, 0))
        orig = x
        if x.ndim == 2:
            x_work = x[None, :, :, None]
            orig_work = orig[None, :, :, None].expand(self.n_models, -1, -1, -1)
        else:
            x_work = x[:, :, :, None]
            orig_work = orig[:, :, :, None]
        z = 2.0 * math.pi * x_work * self.weight_1[:, None, :, :] + self.bias_1[:, None, :, :]
        z = torch.cos(z)
        z = torch.einsum("ebfh,efho->ebfo", z, self.weight_2) + self.bias_2[:, None, :, :]
        z = z.reshape(self.n_models, x.shape[-2], -1)
        return torch.cat([z, orig_work.squeeze(-1)], dim=-1)

def build_num_embedding_v0(n_num: int, cfg: Dict[str, Any]) -> Tuple[Optional[nn.Module], int]:
    """Return ``(module, out_features)``. ``module`` is ``None`` when ``n_num == 0``
    or when ``num_emb_type`` falls outside the PLR family (in which case raw
    features pass through and ``out_features == n_num``).
    """
    n_models = int(cfg.get("n_ens", 1))
    if n_num == 0:
        return None, 0
    if cfg.get("num_emb_type", "pbld") not in ("pbld", "pblrd", "pl", "plr"):
        return None, n_num
    module = PLREmbeddings(
        n_models=n_models,
        n_cont=n_num,
        plr_hidden_1=int(cfg.get("plr_hidden_1", 16)),
        plr_hidden_2=int(cfg.get("plr_hidden_2", 4)),
        plr_sigma=float(cfg.get("plr_sigma", 0.1)),
    )
    return module, module.out_features

__all__ = ["PLREmbeddings", "build_num_embedding_v0"]
