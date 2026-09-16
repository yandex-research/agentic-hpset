from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ._utils import resolve_cat_embedding, task_output_dim
from .tabm import LinearEnsemble, OneHotEncoding


class TabMVariantModel(nn.Module):
    """Shared wrapper for TabM variants that only customize the backbone.

    Variants supply a ``backbone_factory`` that returns an object exposing the
    same contract as ``TabMMiniBackbone``: it takes a 3D tensor ``(batch, k, d_in)``
    and produces ``(batch, k, output_dim)``. The wrapper handles embedding
    resolution, input reshaping, and the final per-head linear head.
    """

    def __init__(
        self,
        *,
        num_embedding: nn.Module | None,
        cat_embedding: nn.Module | None,
        n_num_features: int,
        cat_cardinalities: list[int],
        params: dict[str, Any],
        output_dim: int,
        model_name: str,
        backbone_factory: Callable[..., nn.Module],
        backbone_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.num_embedding = num_embedding
        self.n_num_features = int(n_num_features)
        self.n_cat_features = len(cat_cardinalities)
        self.k = int(params["k"])
        self.model_name = model_name
        self.output_dim = int(output_dim)
        feature_dims: list[int] = []
        if self.n_num_features:
            if self.num_embedding is None:
                feature_dims.extend(1 for _ in range(self.n_num_features))
            else:
                n_emb_features, d_embedding = self.num_embedding.get_output_shape()
                if int(n_emb_features) != self.n_num_features:
                    raise ValueError("Numerical embedding feature count mismatch.")
                feature_dims.extend(int(d_embedding) for _ in range(self.n_num_features))
        self.cat_embedding, cat_feature_dims = resolve_cat_embedding(
            cat_embedding, cat_cardinalities, OneHotEncoding
        )
        feature_dims.extend(cat_feature_dims)
        if not feature_dims:
            raise ValueError("Model has no input features after preprocessing.")
        start_scaling_init = params.get(
            "start_scaling_init",
            "normal" if self.num_embedding is not None else "random-signs",
        )
        self.backbone = backbone_factory(
            d_in=sum(feature_dims),
            k=self.k,
            n_blocks=int(params["n_blocks"]),
            d_block=int(params["d_block"]),
            dropout=float(params["dropout"]),
            start_scaling_init=start_scaling_init,
            start_scaling_init_chunks=feature_dims,
            **backbone_kwargs,
        )
        self.output = LinearEnsemble(self.backbone.output_dim, self.output_dim, k=self.k)

    def _reshape_input_to_2d(self, x: Tensor | None) -> Tensor | None:
        if x is None or x.ndim == 2:
            return x
        if x.ndim == 3:
            return x.flatten(0, 1)
        raise ValueError(f"Expected a 2D or 3D tensor, got ndim={x.ndim}.")

    def _get_batch_info(self, x_num: Tensor | None, x_cat: Tensor | None) -> tuple[int, int]:
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
        x = self.output(self.backbone(x))
        if self.output_dim == 1:
            return x.squeeze(-1)
        return x


def build_variant(
    dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
    *,
    model_name: str,
    backbone_factory: Callable[..., nn.Module],
    backbone_kwargs: dict[str, Any],
) -> nn.Module:
    model = TabMVariantModel(
        num_embedding=embedding_bundle["num_embedding"],
        cat_embedding=embedding_bundle.get("cat_embedding"),
        n_num_features=int(dataset_meta.n_num_features),
        cat_cardinalities=list(dataset_meta.cat_cardinalities),
        params=trial_params,
        output_dim=task_output_dim(dataset_meta),
        model_name=model_name,
        backbone_factory=backbone_factory,
        backbone_kwargs=backbone_kwargs,
    )
    return model.to(device)
