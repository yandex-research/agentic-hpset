from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ._utils import resolve_cat_embedding, task_output_dim
from .tabm import LinearEnsemble, OneHotEncoding, TabMMiniBackbone


class LinearResidualTabMModel(nn.Module):
    """TabM with a per-head linear residual that bypasses the shared backbone."""

    def __init__(
        self,
        *,
        num_embedding: nn.Module | None,
        cat_embedding: nn.Module | None,
        n_num_features: int,
        cat_cardinalities: list[int],
        params: dict[str, Any],
        output_dim: int,
    ) -> None:
        super().__init__()
        self.num_embedding = num_embedding
        self.n_num_features = int(n_num_features)
        self.n_cat_features = len(cat_cardinalities)
        self.k = int(params["k"])
        self.model_name = "tabm-linres"
        self.output_dim = int(output_dim)

        d_features: list[int] = []
        if self.n_num_features:
            if self.num_embedding is None:
                d_features.extend(1 for _ in range(self.n_num_features))
            else:
                if not hasattr(self.num_embedding, "get_output_shape"):
                    raise ValueError(
                        "Numerical embedding must define get_output_shape() for TabM."
                    )
                n_emb_features, d_embedding = self.num_embedding.get_output_shape()
                if int(n_emb_features) != self.n_num_features:
                    raise ValueError(
                        "Numerical embedding feature count does not match dataset: "
                        f"{n_emb_features} != {self.n_num_features}."
                    )
                d_features.extend(int(d_embedding) for _ in range(self.n_num_features))

        self.cat_embedding, cat_d_features = resolve_cat_embedding(
            cat_embedding, cat_cardinalities, OneHotEncoding
        )
        d_features.extend(cat_d_features)
        if not d_features:
            raise ValueError("Model has no input features after preprocessing.")

        d_in = sum(d_features)
        start_scaling_init = params.get(
            "start_scaling_init",
            "normal" if self.num_embedding is not None else "random-signs",
        )
        self.backbone = TabMMiniBackbone(
            d_in=d_in,
            k=self.k,
            n_blocks=int(params["n_blocks"]),
            d_block=int(params["d_block"]),
            dropout=float(params["dropout"]),
            start_scaling_init=start_scaling_init,
            start_scaling_init_chunks=d_features,
        )
        self.output = LinearEnsemble(self.backbone.output_dim, self.output_dim, k=self.k)
        self.residual = LinearEnsemble(d_in, self.output_dim, k=self.k)

    def _reshape_input_to_2d(self, x: Tensor | None) -> Tensor | None:
        if x is None or x.ndim == 2:
            return x
        if x.ndim == 3:
            return x.flatten(0, 1)
        raise ValueError(f"Expected a 2D or 3D tensor, got ndim={x.ndim}.")

    def _get_batch_info(
        self, x_num: Tensor | None, x_cat: Tensor | None
    ) -> tuple[int, int]:
        if x_num is not None:
            return x_num.shape[0], x_num.ndim
        if x_cat is not None:
            return x_cat.shape[0], x_cat.ndim
        raise ValueError("Both x_num and x_cat are None.")

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        batch_size, ndim = self._get_batch_info(x_num, x_cat)
        x_num = self._reshape_input_to_2d(x_num)
        x_cat = self._reshape_input_to_2d(x_cat)

        pieces: list[Tensor] = []
        if x_num is not None:
            num_features = x_num if self.num_embedding is None else self.num_embedding(x_num)
            pieces.append(num_features.flatten(1))
        if x_cat is not None and self.cat_embedding is not None:
            cat_out = self.cat_embedding(x_cat)
            if cat_out.ndim == 3:
                cat_out = cat_out.flatten(1)
            pieces.append(cat_out)

        x = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
        if ndim == 3:
            x = x.unflatten(0, (batch_size, self.k))

        x_view = self.backbone.view(x)
        residual = self.residual(x_view)
        x_processed = self.backbone.affine(x_view)
        for block in self.backbone.blocks:
            x_processed = block(x_processed)
        x = self.output(x_processed) + residual
        if self.output_dim == 1:
            return x.squeeze(-1)
        return x


def build_tabm_v7(
    dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    model = LinearResidualTabMModel(
        num_embedding=embedding_bundle["num_embedding"],
        cat_embedding=embedding_bundle.get("cat_embedding"),
        n_num_features=int(dataset_meta.n_num_features),
        cat_cardinalities=list(dataset_meta.cat_cardinalities),
        params=trial_params,
        output_dim=task_output_dim(dataset_meta),
    )
    return model.to(device)


__all__ = ["LinearResidualTabMModel", "build_tabm_v7"]
