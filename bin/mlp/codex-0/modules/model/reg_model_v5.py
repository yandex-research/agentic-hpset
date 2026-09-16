# ruff: noqa
"""Standalone implementation for ``build_model_v5``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class MixerBlock(nn.Module):
    def __init__(self, n_tokens: int, d_token: int, dropout: float) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(d_token)
        self.token_mixer = nn.Linear(n_tokens, n_tokens)
        self.channel_norm = nn.LayerNorm(d_token)
        self.channel_mixer = nn.Sequential(
            nn.Linear(d_token, d_token * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token * 2, d_token),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.token_norm(x).transpose(1, 2)
        x = x + self.token_mixer(y).transpose(1, 2)
        x = x + self.channel_mixer(self.channel_norm(x))
        return x


def _assemble_input(
    num_embedding: nn.Module | None,
    cat_embedding: nn.Module | None,
    x_num: Tensor | None,
    x_cat: Tensor | None,
) -> Tensor:
    pieces: list[Tensor] = []
    if x_num is not None and num_embedding is not None:
        pieces.append(num_embedding(x_num).flatten(1))
    if x_cat is not None and cat_embedding is not None:
        pieces.append(cat_embedding(x_cat))
    if not pieces:
        raise RuntimeError('Model received no input features.')
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)


class FeatureMixerMLP(nn.Module):
    def __init__(
        self,
        num_embedding: nn.Module | None,
        cat_embedding: nn.Module | None,
        input_dim: int,
        output_dim: int,
        params: dict[str, Any],
    ) -> None:
        super().__init__()
        self.num_embedding = num_embedding
        self.cat_embedding = cat_embedding
        self.output_dim = output_dim
        self.n_tokens = 16
        self.d_token = max(16, min(64, int(params['d_block'])))
        self.tokenizer = nn.Linear(input_dim, self.n_tokens * self.d_token)
        self.blocks = nn.Sequential(
            *[
                MixerBlock(self.n_tokens, self.d_token, float(params['dropout']))
                for _ in range(max(1, min(int(params['n_blocks']), 4)))
            ]
        )
        self.output = nn.Linear(self.n_tokens * self.d_token, output_dim)

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        x = _assemble_input(self.num_embedding, self.cat_embedding, x_num, x_cat)
        tokens = self.tokenizer(x).reshape(x.shape[0], self.n_tokens, self.d_token)
        output = self.output(self.blocks(tokens).flatten(1))
        return output.squeeze(-1) if self.output_dim == 1 else output


def _output_dim(*, heteroscedastic: bool = False) -> int:
    return 2 if heteroscedastic else 1


def build_model_v5(
    _dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    return FeatureMixerMLP(
        embedding_bundle['num_embedding'],
        embedding_bundle['cat_embedding'],
        int(embedding_bundle['input_dim']),
        _output_dim(),
        trial_params,
    ).to(device)
