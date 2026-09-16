# ruff: noqa
"""Standalone implementation for ``build_mlp_v1``."""

from __future__ import annotations
from typing import Any
from torch import Tensor
import torch.nn as nn
import torch


class GroupedLinear(nn.Module):
    """Grouped linear with explicit mixing.

    Splits the d_in dim into G chunks, applies an independent
    linear per group to produce d_out / G features each, then a
    1x1 mixing linear to allow cross-group communication.
    Lower FLOPs than a full linear at fixed (d_in, d_out).
    """

    def __init__(self, d_in: int, d_out: int, groups: int = 4) -> None:
        super().__init__()
        groups = max(1, min(groups, d_in, d_out))
        while d_in % groups != 0 or d_out % groups != 0:
            groups -= 1
            if groups < 1:
                groups = 1
                break
        self.groups = groups
        in_per = d_in // groups
        out_per = d_out // groups
        self.weight = nn.Parameter(torch.empty(groups, out_per, in_per))
        self.bias = nn.Parameter(torch.zeros(d_out))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.mix = nn.Linear(d_out, d_out, bias=False)
        nn.init.kaiming_uniform_(self.mix.weight, a=5**0.5)
        self.d_in = d_in
        self.d_out = d_out

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x = x.view(B, self.groups, self.d_in // self.groups)
        out = torch.einsum('bgi,goi->bgo', x, self.weight)
        out = out.reshape(B, -1) + self.bias
        return self.mix(out)


class GroupedMLPModel(nn.Module):
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
        groups = 4
        blocks: list[nn.Module] = []
        current = input_dim
        blocks.extend([nn.Linear(current, d_block), nn.ReLU(), nn.Dropout(dropout)])
        current = d_block
        for _ in range(max(0, n_blocks - 1)):
            blocks.extend(
                [
                    GroupedLinear(current, d_block, groups=groups),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current = d_block
        self.backbone = nn.Sequential(*blocks)
        self.output = nn.Linear(current, output_dim)
        self.output_dim = output_dim

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        pieces: list[Tensor] = []
        if x_num is not None and self.num_embedding is not None:
            pieces.append(self.num_embedding(x_num).flatten(1))
        if x_cat is not None and self.cat_embedding is not None:
            pieces.append(self.cat_embedding(x_cat))
        x = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
        output = self.output(self.backbone(x))
        if self.output_dim == 1:
            return output.squeeze(-1)
        return output


def build_mlp_v1(
    dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    model = GroupedMLPModel(
        num_embedding=embedding_bundle['num_embedding'],
        cat_embedding=embedding_bundle['cat_embedding'],
        input_dim=int(embedding_bundle['input_dim']),
        params=trial_params,
        output_dim=1 if dataset_meta.is_binclass else int(dataset_meta.n_classes),
    )
    return model.to(device)
