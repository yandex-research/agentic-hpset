from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ._utils import resolve_cat_embedding, task_output_dim


def _init_random_signs_(tensor: Tensor) -> Tensor:
    return tensor.bernoulli_(0.5).mul_(2).add_(-1)


def _init_scaling_(tensor: Tensor, distribution: str, chunks: list[int] | None = None) -> Tensor:
    if distribution == "ones":
        init_fn = nn.init.ones_
    elif distribution == "normal":
        init_fn = nn.init.normal_
    elif distribution == "random-signs":
        init_fn = _init_random_signs_
    else:
        raise ValueError(f"Unknown scaling initialization: {distribution}")
    if chunks is None:
        return init_fn(tensor)
    if sum(chunks) != tensor.shape[-1]:
        raise ValueError("Scaling chunks must sum to the last tensor dimension.")
    with torch.inference_mode():
        offset = 0
        for chunk_size in chunks:
            tensor[..., offset : offset + chunk_size] = init_fn(
                torch.empty(*tensor.shape[:-1], 1, device=tensor.device, dtype=tensor.dtype)
            )
            offset += chunk_size
    return tensor


class ElementwiseAffine(nn.Module):
    def __init__(self, shape: tuple[int, ...], *, scaling_init: str, scaling_init_chunks: list[int] | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(shape))
        self.scaling_init = scaling_init
        self.scaling_init_chunks = scaling_init_chunks
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_scaling_(self.weight, self.scaling_init, self.scaling_init_chunks)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.weight


class EnsembleView(nn.Module):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = int(k)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            return x.unsqueeze(1).expand(-1, self.k, -1)
        if x.ndim == 3 and x.shape[1] == self.k:
            return x
        raise ValueError(f"Bad shape {tuple(x.shape)}")


class LinearEnsemble(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, k: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(k, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(k, out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = self.weight.shape[1] ** -0.5
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1] != self.weight.shape[0]:
            raise ValueError(f"Bad shape {tuple(x.shape)} for k={self.weight.shape[0]}")
        x = x.transpose(0, 1) @ self.weight
        x = x.transpose(0, 1)
        if self.bias is not None:
            x = x + self.bias
        return x


class OneHotEncoding(nn.Module):
    def __init__(self, cardinalities: list[int]) -> None:
        super().__init__()
        self.cardinalities = cardinalities

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat(
            [F.one_hot(x[:, i], cardinality) for i, cardinality in enumerate(self.cardinalities)],
            dim=1,
        ).float()


class PerHeadFeatureMask(nn.Module):
    """Fixed per-head Bernoulli feature mask, applied multiplicatively.

    Each ensemble head k has its own static mask of shape (d_in,) drawn at init
    from Bernoulli(p_keep). The mask is registered as a non-trainable buffer so
    every batch sees the same head-feature subsets — this is bagging-style
    diversity rather than per-step dropout. Forward divides by the mean keep
    rate so the expected scale is preserved.
    """

    def __init__(self, k: int, d_in: int, p_keep: float = 0.85, seed: int = 0) -> None:
        super().__init__()
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        mask = (torch.rand((k, d_in), generator=gen) < p_keep).float()
        # Ensure no head has all-zero mask.
        for i in range(k):
            if mask[i].sum() == 0:
                mask[i, torch.randint(0, d_in, (1,), generator=gen)] = 1.0
        scale = 1.0 / max(p_keep, 1e-3)
        self.register_buffer("mask", mask * scale)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.mask


class TabMFeatureDropoutBackbone(nn.Module):
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
        feature_keep: float,
        seed: int,
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.view = EnsembleView(self.k)
        self.feature_mask = PerHeadFeatureMask(self.k, d_in, feature_keep, seed)
        self.affine = ElementwiseAffine(
            (self.k, d_in),
            scaling_init=start_scaling_init,
            scaling_init_chunks=start_scaling_init_chunks,
        )
        blocks: list[nn.Module] = []
        current_dim = d_in
        for _ in range(int(n_blocks)):
            blocks.append(
                nn.Sequential(
                    nn.Linear(current_dim, int(d_block)),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                )
            )
            current_dim = int(d_block)
        self.blocks = nn.ModuleList(blocks)
        self.output_dim = current_dim

    def forward(self, x: Tensor) -> Tensor:
        x = self.view(x)
        x = self.feature_mask(x)
        x = self.affine(x)
        for block in self.blocks:
            x = block(x)
        return x


class TabMFeatureDropoutModel(nn.Module):
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
        self.model_name = "tabm-fdrop"
        self.output_dim = int(output_dim)

        d_features: list[int] = []
        if self.n_num_features:
            if self.num_embedding is None:
                d_features.extend(1 for _ in range(self.n_num_features))
            else:
                if not hasattr(self.num_embedding, "get_output_shape"):
                    raise ValueError("Numerical embedding must define get_output_shape().")
                n_emb_features, d_embedding = self.num_embedding.get_output_shape()
                if int(n_emb_features) != self.n_num_features:
                    raise ValueError("Numerical embedding feature count mismatch.")
                d_features.extend(int(d_embedding) for _ in range(self.n_num_features))

        self.cat_embedding, cat_d_features = resolve_cat_embedding(
            cat_embedding, cat_cardinalities, OneHotEncoding
        )
        d_features.extend(cat_d_features)
        if not d_features:
            raise ValueError("Model has no input features after preprocessing.")

        start_scaling_init = params.get(
            "start_scaling_init",
            "normal" if self.num_embedding is not None else "random-signs",
        )
        self.backbone = TabMFeatureDropoutBackbone(
            d_in=sum(d_features),
            k=self.k,
            n_blocks=int(params["n_blocks"]),
            d_block=int(params["d_block"]),
            dropout=float(params["dropout"]),
            start_scaling_init=start_scaling_init,
            start_scaling_init_chunks=d_features,
            feature_keep=float(params.get("feature_keep", 0.85)),
            seed=int(params.get("feature_mask_seed", 0)),
        )
        self.output = LinearEnsemble(self.backbone.output_dim, self.output_dim, k=self.k)

    def _reshape_input_to_2d(self, x: Tensor | None) -> Tensor | None:
        if x is None or x.ndim == 2:
            return x
        if x.ndim == 3:
            return x.flatten(0, 1)
        raise ValueError(f"Expected 2D/3D tensor, got ndim={x.ndim}.")

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
        x = self.backbone(x)
        x = self.output(x)
        if self.output_dim == 1:
            return x.squeeze(-1)
        return x


def build_tabm_v3(
    dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """TabM with a fixed per-head Bernoulli feature mask (random-subspace bagging).

    Hypothesis: TabM's only source of head diversity is data permutation and the
    starting affine init. Random forests show that random feature subsets give
    much stronger ensemble diversity (de-correlated heads) than just data
    bagging. Applying a fixed Bernoulli mask per head before the affine layer
    forces each head to specialize on a feature subspace, while averaging at the
    output recombines complementary predictions.
    """
    model = TabMFeatureDropoutModel(
        num_embedding=embedding_bundle["num_embedding"],
        cat_embedding=embedding_bundle.get("cat_embedding"),
        n_num_features=int(dataset_meta.n_num_features),
        cat_cardinalities=list(dataset_meta.cat_cardinalities),
        params=trial_params,
        output_dim=task_output_dim(dataset_meta),
    )
    return model.to(device)
