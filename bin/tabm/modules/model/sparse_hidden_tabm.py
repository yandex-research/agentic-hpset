from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ._utils import resolve_cat_embedding, task_output_dim
from .tabm import ElementwiseAffine, EnsembleView, LinearEnsemble, OneHotEncoding


class TopKActivation(nn.Module):
    def __init__(self, keep_fraction: float) -> None:
        super().__init__()
        self.keep_fraction = float(keep_fraction)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(x)
        if self.keep_fraction >= 1.0 or x.shape[-1] <= 1:
            return x
        k = max(1, int(round(x.shape[-1] * self.keep_fraction)))
        if k >= x.shape[-1]:
            return x
        threshold = x.topk(k, dim=-1).values[..., -1:]
        return x * (x >= threshold)


class SparseTabMMiniBackbone(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        k: int,
        n_blocks: int,
        d_block: int,
        dropout: float,
        start_scaling_init: str,
        start_scaling_init_chunks: list[int] | None,
        sparse_keep_fraction: float,
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.view = EnsembleView(self.k)
        self.affine = ElementwiseAffine(
            (self.k, d_in),
            bias=False,
            scaling_init=start_scaling_init,
            scaling_init_chunks=start_scaling_init_chunks,
        )
        blocks: list[nn.Module] = []
        current_dim = d_in
        for _ in range(int(n_blocks)):
            blocks.append(
                nn.Sequential(
                    nn.Linear(current_dim, int(d_block)),
                    TopKActivation(sparse_keep_fraction),
                    nn.Dropout(float(dropout)),
                )
            )
            current_dim = int(d_block)
        self.blocks = nn.ModuleList(blocks)
        self.output_dim = current_dim

    def forward(self, x: Tensor) -> Tensor:
        x = self.view(x)
        x = self.affine(x)
        for block in self.blocks:
            x = block(x)
        return x


class SparseHiddenTabMModel(nn.Module):
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
        self.model_name = "tabm-sparse-hidden"
        self.output_dim = int(output_dim)

        feature_dims: list[int] = []
        if self.n_num_features:
            if self.num_embedding is None:
                feature_dims.extend(1 for _ in range(self.n_num_features))
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
                feature_dims.extend(int(d_embedding) for _ in range(self.n_num_features))

        self.cat_embedding, cat_d_features = resolve_cat_embedding(
            cat_embedding, cat_cardinalities, OneHotEncoding
        )
        feature_dims.extend(cat_d_features)
        if not feature_dims:
            raise ValueError("Model has no input features after preprocessing.")

        start_scaling_init = params.get(
            "start_scaling_init",
            "normal" if self.num_embedding is not None else "random-signs",
        )
        self.backbone = SparseTabMMiniBackbone(
            d_in=sum(feature_dims),
            k=self.k,
            n_blocks=int(params["n_blocks"]),
            d_block=int(params["d_block"]),
            dropout=float(params["dropout"]),
            start_scaling_init=start_scaling_init,
            start_scaling_init_chunks=feature_dims,
            sparse_keep_fraction=float(params.get("sparse_keep_fraction", 0.5)),
        )
        self.output = LinearEnsemble(self.backbone.output_dim, self.output_dim, k=self.k)

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
        x = self.backbone(x)
        x = self.output(x)
        if self.output_dim == 1:
            return x.squeeze(-1)
        return x


def build_tabm_v9(
    dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    model = SparseHiddenTabMModel(
        num_embedding=embedding_bundle["num_embedding"],
        cat_embedding=embedding_bundle.get("cat_embedding"),
        n_num_features=int(dataset_meta.n_num_features),
        cat_cardinalities=list(dataset_meta.cat_cardinalities),
        params=trial_params,
        output_dim=task_output_dim(dataset_meta),
    )
    return model.to(device)


__all__ = [
    "SparseHiddenTabMModel",
    "SparseTabMMiniBackbone",
    "TopKActivation",
    "build_tabm_v9",
]
