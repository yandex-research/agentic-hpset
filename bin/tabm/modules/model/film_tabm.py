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


class FiLMBlock(nn.Module):
    """Feed-forward block whose hidden activations are conditioned on the raw input via FiLM.

    Given block input ``x`` and an external context ``c`` (the raw model input
    summary), compute scale and shift vectors from ``c`` and apply them
    element-wise to the post-ReLU activation. This injects input information at
    every depth, helping the network maintain a clean residual signal.
    """

    def __init__(self, d_in: int, d_out: int, d_context: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.film = nn.Linear(d_context, 2 * d_out)
        # Initialize FiLM to identity (gamma=1, beta=0) by zeroing weights and
        # setting bias to [1, ..., 1, 0, ..., 0].
        nn.init.zeros_(self.film.weight)
        with torch.no_grad():
            self.film.bias[:d_out].fill_(1.0)
            self.film.bias[d_out:].zero_()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        h = F.relu(self.linear(x))
        gamma, beta = self.film(context).chunk(2, dim=-1)
        h = h * gamma + beta
        return self.dropout(h)


class TabMFiLMBackbone(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        k: int,
        n_blocks: int,
        d_block: int,
        dropout: float,
        d_context: int,
        start_scaling_init: str,
        start_scaling_init_chunks: list[int] | None,
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.view = EnsembleView(self.k)
        self.affine = ElementwiseAffine(
            (self.k, d_in),
            scaling_init=start_scaling_init,
            scaling_init_chunks=start_scaling_init_chunks,
        )
        # Compress raw input into a small context vector once per block-stack.
        self.context_proj = nn.Linear(d_in, d_context)
        blocks: list[FiLMBlock] = []
        current_dim = d_in
        for _ in range(int(n_blocks)):
            blocks.append(FiLMBlock(current_dim, int(d_block), d_context, float(dropout)))
            current_dim = int(d_block)
        self.blocks = nn.ModuleList(blocks)
        self.output_dim = current_dim

    def forward(self, x: Tensor) -> Tensor:
        x = self.view(x)
        x = self.affine(x)
        context = F.relu(self.context_proj(x))
        for block in self.blocks:
            x = block(x, context)
        return x


class TabMFiLMModel(nn.Module):
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
        self.model_name = "tabm-film"
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
        d_context = max(16, int(params["d_block"]) // 4)
        self.backbone = TabMFiLMBackbone(
            d_in=sum(d_features),
            k=self.k,
            n_blocks=int(params["n_blocks"]),
            d_block=int(params["d_block"]),
            dropout=float(params["dropout"]),
            d_context=d_context,
            start_scaling_init=start_scaling_init,
            start_scaling_init_chunks=d_features,
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


def build_tabm_v5(
    dataset_meta: Any,
    trial_params: dict[str, Any],
    embedding_bundle: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """TabM whose blocks are conditioned on the raw input via FiLM.

    Hypothesis: deep feed-forward stacks progressively wash out the original input signal. FiLM
    (Perez et al. 2018) re-injects per-channel scale and shift conditioned on a
    small projection of the original input at every block, giving the model a
    persistent skip-like reference to raw features. This reliably improves
    performance on tasks with rich feature interactions.
    """
    model = TabMFiLMModel(
        num_embedding=embedding_bundle["num_embedding"],
        cat_embedding=embedding_bundle.get("cat_embedding"),
        n_num_features=int(dataset_meta.n_num_features),
        cat_cardinalities=list(dataset_meta.cat_cardinalities),
        params=trial_params,
        output_dim=task_output_dim(dataset_meta),
    )
    return model.to(device)
