# ruff: noqa
"""Standalone implementation for ``build_mlp_v8``."""

from __future__ import annotations
from typing import Any
import torch.nn.functional as F
from torch import Tensor
import math
import torch.nn as nn
import torch

BETA = 2.0 / 3.0
GAMMA = -0.1
ZETA = 1.1


class L0Linear(nn.Module):
    """Linear layer with L0-regularized weight gates via the Hard-Concrete distribution.

    During training, each weight has a stochastic 0/1 gate sampled from a stretched
    binary concrete distribution; in eval mode we use its mean. The expected L0
    penalty is exposed via `l0_penalty()` for use in the training loss.
    """

    def __init__(self, d_in: int, d_out: int, init_log_alpha: float = 1.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        self.bias = nn.Parameter(torch.zeros(d_out))
        self.log_alpha = nn.Parameter(torch.full((d_out, d_in), init_log_alpha))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        bound = 1.0 / d_in**0.5
        nn.init.uniform_(self.bias, -bound, bound)

    def _sample_mask(self) -> Tensor:
        if self.training:
            u = torch.rand_like(self.log_alpha).clamp(1e-06, 1 - 1e-06)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) / BETA)
        else:
            s = torch.sigmoid(self.log_alpha)
        s_bar = s * (ZETA - GAMMA) + GAMMA
        return s_bar.clamp(0.0, 1.0)

    def l0_penalty(self) -> Tensor:
        prob_open = torch.sigmoid(self.log_alpha - BETA * math.log(-GAMMA / ZETA))
        return prob_open.sum()

    def forward(self, x: Tensor) -> Tensor:
        mask = self._sample_mask()
        return F.linear(x, self.weight * mask, self.bias)


class L0MLP(nn.Module):
    def __init__(
        self,
        num_embedding: nn.Module | None,
        cat_embedding: nn.Module | None,
        input_dim: int,
        params: dict[str, Any],
        output_dim: int,
    ) -> None:
        super().__init__()
        if input_dim == 0:
            raise ValueError('Model has no input features after preprocessing.')
        self.num_embedding = num_embedding
        self.cat_embedding = cat_embedding
        n_blocks = int(params['n_blocks'])
        d_block = int(params['d_block'])
        dropout = float(params['dropout'])
        self.l0_layers: list[L0Linear] = []
        blocks: list[nn.Module] = []
        current = input_dim
        for _ in range(n_blocks):
            layer = L0Linear(current, d_block)
            self.l0_layers.append(layer)
            blocks.extend([layer, nn.ReLU(), nn.Dropout(dropout)])
            current = d_block
        self.backbone = nn.Sequential(*blocks)
        self.output = nn.Linear(current, output_dim)
        self.output_dim = output_dim
        self.l0_lambda = float(params.get('l0_lambda', 1e-05))

    def l0_loss(self) -> Tensor:
        total = self.output.weight.new_zeros(())
        for layer in self.l0_layers:
            total = total + layer.l0_penalty()
        return self.l0_lambda * total

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        pieces: list[Tensor] = []
        if x_num is not None and self.num_embedding is not None:
            pieces.append(self.num_embedding(x_num).flatten(1))
        if x_cat is not None and self.cat_embedding is not None:
            pieces.append(self.cat_embedding(x_cat))
        x = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
        out = self.output(self.backbone(x))
        if self.output_dim == 1:
            return out.squeeze(-1)
        return out


def build_mlp_v8(
    dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    model = L0MLP(
        num_embedding=embedding_bundle['num_embedding'],
        cat_embedding=embedding_bundle['cat_embedding'],
        input_dim=int(embedding_bundle['input_dim']),
        params=trial_params,
        output_dim=1 if dataset_meta.is_binclass else int(dataset_meta.n_classes),
    )
    return model.to(device)
