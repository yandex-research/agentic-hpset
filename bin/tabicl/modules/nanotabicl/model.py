import math
import typing
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError
from torch.nn.attention import SDPBackend, sdpa_kernel

HF_REPO_ID = "jingang/TabICL"
DEFAULT_REGRESSOR_CHECKPOINT_VERSION = "tabicl-regressor-v2-20260212.ckpt"
DEFAULT_CLASSIFIER_CHECKPOINT_VERSION = "tabicl-classifier-v2-20260212.ckpt"
DEFAULT_CHECKPOINT_VERSION = DEFAULT_REGRESSOR_CHECKPOINT_VERSION

# CUDA grid y/z dim limit. SDPA fused kernels (Flash / mem-efficient) crash
# with "invalid configuration argument" when batch exceeds this.
_SDPA_MAX_BATCH = 65535


class NanoTabICLv2(nn.Module):
    def __init__(
        self,
        max_classes: int,
        out_dim: int,
        embed_dim: int = 128,
        col_num_blocks: int = 3,
        row_num_blocks: int = 3,
        icl_num_blocks: int = 12,
        col_nhead: int = 8,
        row_nhead: int = 8,
        icl_nhead: int = 8,
        feature_group_size: int = 3,
        n_cls_cols: int = 4,
        n_cls_rows: int = 128,
        standardize_x: bool = True,
        feature_group_mode: str = "nano",
        bias_free_ln: bool = False,
    ):
        # classification: max_classes = out_dim (= 10 typically); regression: max_classes = 0, out_dim = n_quantiles
        super().__init__()
        self.max_classes = max_classes
        self.out_dim = out_dim
        self.feature_group_size = feature_group_size
        self.standardize_x = standardize_x
        self.feature_group_mode = feature_group_mode
        icl_dim = embed_dim * n_cls_cols

        self.x_embed = nn.Linear(feature_group_size, embed_dim)
        self.y_embed_in = (
            ClassEmbedding(max_classes, embed_dim)
            if max_classes > 0
            else nn.Linear(1, embed_dim)
        )
        self.y_embed_icl = (
            ClassEmbedding(max_classes, icl_dim)
            if max_classes > 0
            else nn.Linear(1, icl_dim)
        )

        self.col_blocks = nn.ModuleList(
            [
                InducedTransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=col_nhead,
                    n_inducing=n_cls_rows,
                    ssmax=True,
                    bias_free_ln=bias_free_ln,
                )
                for _ in range(col_num_blocks)
            ]
        )
        self.row_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=row_nhead,
                    use_rope=True,
                    bias_free_ln=bias_free_ln,
                )
                for _ in range(row_num_blocks)
            ]
        )
        self.icl_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=icl_dim,
                    num_heads=icl_nhead,
                    ssmax=True,
                    bias_free_ln=bias_free_ln,
                )
                for _ in range(icl_num_blocks)
            ]
        )

        self.row_cls_tokens = nn.Parameter(
            0.02 * torch.randn(1, 1, n_cls_cols, embed_dim)
        )
        self.row_ln = nn.LayerNorm(embed_dim, bias=not bias_free_ln)
        self.out_ln = nn.LayerNorm(icl_dim, bias=not bias_free_ln)
        self.out_mlp = get_mlp(icl_dim, icl_dim * 2, out_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n_batch, n_rows, n_cols = x.shape
        n_batch, n_train = y.shape

        # ----- Embedding: repeated feature grouping -> x embedding -> add y embedding to train
        if self.standardize_x:
            x = x / (
                x[:, :n_train].std(dim=1, unbiased=False, keepdim=True) + 1e-8
            )  # standardize x based on train
        idxs = torch.arange(n_cols, dtype=torch.long, device=x.device)
        shift = 1 if self.feature_group_mode == "nano" else 0
        x = torch.stack(
            [
                x[:, :, (idxs + 2**i - shift) % n_cols]
                for i in range(self.feature_group_size)
            ],
            dim=-1,
        )
        emb = self.x_embed(x)  # emb.shape = (n_batch, n_rows, n_cols, embed_dim)
        emb[:, :n_train] += self.y_embed_in(y[:, :, None, None])

        # ----- TF_col: induced self-attention within each column
        for block in self.col_blocks:
            emb = block.col_attn(
                emb, kv_max_idx=n_train
            )  # all rows only attend to training rows

        # ----- TF_row: concat CLS tokens as extra columns -> row attention -> norm + merge cls tokens
        emb = torch.cat(
            [self.row_cls_tokens.expand(n_batch, n_rows, -1, -1), emb], dim=2
        )
        for block in self.row_blocks[:-1]:
            emb = block.row_attn(emb)
        emb = self.row_blocks[-1].row_attn(
            emb, q_max_idx=self.row_cls_tokens.size(-2)
        )  # need only the cls token values
        emb = self.row_ln(emb).flatten(
            -2, -1
        )  # norm + merge cls tokens into one bigger token

        # ----- TF_icl: add y embedding -> self-attention
        # now emb.shape = (n_batch, n_rows, icl_dim)
        emb[:, :n_train] += self.y_embed_icl(y[:, :, None])  # add y embeddings again
        for block in self.icl_blocks[:-1]:
            emb = block(
                emb, kv_max_idx=n_train
            )  # all rows only attend to training rows
        emb = self.icl_blocks[-1](
            emb[:, n_train:], emb[:, :n_train]
        )  # need only test predictions

        return self.out_mlp(self.out_ln(emb))  # output MLP


class ClassEmbedding(nn.Linear):
    def __init__(self, max_classes: int, embed_dim: int):
        super().__init__(max_classes, embed_dim)
        self.max_classes = max_classes

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(y.squeeze(-1).long(), self.max_classes).to(y.dtype)
        return F.linear(one_hot.float(), self.weight, self.bias)


def get_mlp(n_in: int, n_hidden: int, n_out: int):
    return nn.Sequential(
        nn.Linear(n_in, n_hidden), nn.GELU(), nn.Linear(n_hidden, n_out)
    )


class TableAttnBase(
    nn.Module
):  # base class with functions to apply attention on 2D tables instead of 1D sequences
    def row_attn(
        self, q, kv=None, **kwargs
    ):  # apply attention within each row separately
        n_batch, n_rows, n_cols, embed_dim = q.shape
        q, kv = (
            None if t is None else t.flatten(0, 1) for t in [q, kv]
        )  # merge rows dim into batch dim
        # apply attention -> unmerge rows dim from batch dim; dimension -2 might differ because of q_max_idx in kwargs
        return self(q, kv, **kwargs).reshape(n_batch, n_rows, -1, embed_dim)

    def col_attn(
        self, q, kv=None, **kwargs
    ):  # apply attention within each column separately
        return self.row_attn(
            q.transpose(1, 2), None if kv is None else kv.transpose(1, 2), **kwargs
        ).transpose(1, 2)


class InducedTransformerBlock(TableAttnBase):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        n_inducing: int,
        ssmax: bool = False,
        bias_free_ln: bool = False,
    ):
        super().__init__()
        self.tfm1 = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ssmax=ssmax,
            bias_free_ln=bias_free_ln,
        )
        self.tfm2 = TransformerBlock(
            embed_dim=embed_dim, num_heads=num_heads, bias_free_ln=bias_free_ln
        )
        self.inducing_vectors = nn.Parameter(
            0.02 * torch.randn(1, n_inducing, embed_dim)
        )

    def forward(
        self,
        q,
        kv=None,
        q_max_idx: typing.Optional[int] = None,
        kv_max_idx: typing.Optional[int] = None,
    ):
        kv = self.tfm1(
            self.inducing_vectors.expand(q.shape[0], -1, -1),
            q if kv is None else kv,
            kv_max_idx=kv_max_idx,
        )
        return self.tfm2(q, kv, q_max_idx=q_max_idx)


class TransformerBlock(nn.MultiheadAttention, TableAttnBase):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        use_rope: bool = False,
        ssmax: bool = False,
        bias_free_ln: bool = False,
    ):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads)
        self.rope = (
            Rope(head_dim=embed_dim // num_heads, theta=100_000.0) if use_rope else None
        )
        self.ssmax_layer = (
            QASSMax(num_heads=num_heads, head_dim=embed_dim // num_heads)
            if ssmax
            else None
        )
        self.mlp = get_mlp(embed_dim, embed_dim * 2, embed_dim)
        self.ln_attn = nn.LayerNorm(embed_dim, bias=not bias_free_ln)
        self.ln_mlp = nn.LayerNorm(embed_dim, bias=not bias_free_ln)

    def forward(
        self,
        q,
        kv=None,
        q_max_idx: typing.Optional[int] = None,
        kv_max_idx: typing.Optional[int] = None,
    ):
        # q.shape: (batch_size, q_len, embed_dim), kv.shape: (batch_size, kv_len, embed_dim)
        x, q = q, self.ln_attn(q)
        kv = q if kv is None else self.ln_attn(kv)
        if kv_max_idx is not None:
            kv = kv[..., :kv_max_idx, :]
        if q_max_idx is not None:
            x, q = x[..., :q_max_idx, :], q[..., :q_max_idx, :]

        x = x + self.attn(q, kv)
        del q, kv  # save memory during inference
        return x + self.mlp(
            self.ln_mlp(x)
        )  # we use pre-norm here and for the attention as well

    def attn(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # Joint projection of (q, k, v), then transpose heads to (batch_size, num_heads, len, head_dim)
        q, k, v = nn.functional._in_projection_packed(
            q, k, k, self.in_proj_weight, self.in_proj_bias
        )
        q, k, v = (
            t.unflatten(-1, (self.num_heads, self.head_dim)).transpose(-3, -2)
            for t in [q, k, v]
        )

        q = (
            q if self.ssmax_layer is None else self.ssmax_layer(q=q, n=k.size(-2))
        )  # SSMax (optional)
        q, k = (
            t if self.rope is None else self.rope(t) for t in [q, k]
        )  # RoPE (optional)

        # PyTorch's fused SDPA kernels need 4D (B, H, T, D) AND fail with
        # "invalid configuration argument" once B exceeds 65535 (CUDA grid y/z
        # limit). Row/col attention can hit B=batch*rows up to >100k, so chunk
        # along the batch dim and run Flash multiple times rather than fall
        # back to math (which materializes the O(B*H*T*T) score matrix).
        # bf16 cast on CUDA enables FlashAttention; we cast back before out_proj.
        orig_dtype = q.dtype
        if q.is_cuda and orig_dtype in (torch.float32, torch.float64):
            q, k, v = (t.to(torch.bfloat16) for t in (q, k, v))
        with sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        ):
            B = q.shape[0]
            if B <= _SDPA_MAX_BATCH or not q.is_cuda:
                attn_output = nn.functional.scaled_dot_product_attention(q, k, v)
            else:
                chunks = [
                    nn.functional.scaled_dot_product_attention(
                        q[start : start + _SDPA_MAX_BATCH],
                        k[start : start + _SDPA_MAX_BATCH],
                        v[start : start + _SDPA_MAX_BATCH],
                    )
                    for start in range(0, B, _SDPA_MAX_BATCH)
                ]
                attn_output = torch.cat(chunks, dim=0)
                del chunks
        attn_output = attn_output.to(orig_dtype)
        del q, k, v  # save memory during inference
        return self.out_proj(
            attn_output.transpose(-3, -2).flatten(-2, -1)
        )  # (batch_size, q_len, embed_dim)


class Rope(nn.Module):  # rotary positional encoding
    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        self.half = head_dim // 2
        self.register_buffer(
            "inv_freq",
            theta ** torch.linspace(0.0, -1.0, self.half + 1)[:-1],
            persistent=False,
        )
        self.register_buffer("sin", torch.empty(0), persistent=False)
        self.register_buffer("cos", torch.empty(0), persistent=False)

    @torch.autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = x.shape

        if (
            self.cos.numel() == 0
            or self.cos.device != x.device
            or self.cos.size(0) < seq_len
        ):  # need to extend cache
            pos = torch.arange(
                seq_len, device=x.device, dtype=self.inv_freq.dtype
            )  # (seq_len,)
            angles = pos[:, None] * self.inv_freq[None, :]  # (seq_len, half_head_dim)
            self.sin, self.cos = angles.sin(), angles.cos()  # (seq_len, half_head_dim)

        sin, cos = self.sin[:seq_len], self.cos[:seq_len]
        x1, x2 = (
            x[..., : self.half],
            x[..., self.half :],
        )  # (batch_size, num_heads, seq_len, half_head_dim)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)


class QASSMax(
    nn.Module
):  # query-aware scalable softmax for better context length scaling
    def __init__(self, num_heads: int, head_dim: int, n_hidden: int = 64):
        super().__init__()
        self.base_mlp = get_mlp(1, n_hidden, num_heads * head_dim)
        self.query_mlp = get_mlp(head_dim, n_hidden, head_dim)
        nn.init.zeros_(self.query_mlp[-1].weight)  # ensures initial modulation is zero
        nn.init.zeros_(self.query_mlp[-1].bias)

    def forward(self, q: torch.Tensor, n: int) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = q.shape
        logn = q.new_tensor(math.log(max(1, n))).view(1, 1)
        return (
            self.base_mlp(logn).view(1, num_heads, 1, head_dim)
            * (1 + torch.tanh(self.query_mlp(q)))
            * q
        )


def _check_config(c):
    expected = dict(
        col_affine=False,
        col_feature_group="same",
        col_target_aware=True,
        norm_first=True,
        ff_factor=2,
        dropout=0.0,
        activation="gelu",
        row_rope_interleaved=False,
        recompute=False,
        col_ssmax="qassmax-mlp-elementwise",
        icl_ssmax="qassmax-mlp-elementwise",
    )
    bad = {k: (c.get(k), v) for k, v in expected.items() if c.get(k) != v}
    if bad:
        raise ValueError(f"Unsupported TabICLv2 checkpoint config: {bad}")
    if int(c.get("max_classes", 0)) < 0:
        raise ValueError("Unsupported TabICLv2 checkpoint config: max_classes < 0")
    if int(c.get("max_classes", 0)) == 0 and int(c.get("num_quantiles", 0)) <= 0:
        raise ValueError(
            "Unsupported TabICLv2 regression checkpoint: num_quantiles <= 0"
        )


def _map_block(src, dst, state, out):
    names = {
        "linear1.weight": "mlp.0.weight",
        "linear1.bias": "mlp.0.bias",
        "linear2.weight": "mlp.2.weight",
        "linear2.bias": "mlp.2.bias",
        "norm1.weight": "ln_attn.weight",
        "norm2.weight": "ln_mlp.weight",
        "attn.in_proj_weight": "in_proj_weight",
        "attn.in_proj_bias": "in_proj_bias",
        "attn.out_proj.weight": "out_proj.weight",
        "attn.out_proj.bias": "out_proj.bias",
    }
    for a, b in names.items():
        out[f"{dst}.{b}"] = state[f"{src}.{a}"]
    for a, b in {
        "norm1.bias": "ln_attn.bias",
        "norm2.bias": "ln_mlp.bias",
    }.items():
        key = f"{src}.{a}"
        if key in state:
            out[f"{dst}.{b}"] = state[key]
    ss = f"{src}.attn.ssmax_layer."
    for k, v in state.items():
        if k.startswith(ss):
            out[f"{dst}.ssmax_layer.{k[len(ss) :]}"] = v


def convert_tabiclv2_state_dict(state, config):
    _check_config(config)
    out = {
        "x_embed.weight": state["col_embedder.in_linear.weight"],
        "x_embed.bias": state["col_embedder.in_linear.bias"],
        "y_embed_in.weight": state["col_embedder.y_encoder.weight"],
        "y_embed_in.bias": state["col_embedder.y_encoder.bias"],
        "y_embed_icl.weight": state["icl_predictor.y_encoder.weight"],
        "y_embed_icl.bias": state["icl_predictor.y_encoder.bias"],
        "row_cls_tokens": state["row_interactor.cls_tokens"].view(
            1, 1, config["row_num_cls"], config["embed_dim"]
        ),
        "row_ln.weight": state["row_interactor.out_ln.weight"],
        "out_ln.weight": state["icl_predictor.ln.weight"],
        "out_mlp.0.weight": state["icl_predictor.decoder.0.weight"],
        "out_mlp.0.bias": state["icl_predictor.decoder.0.bias"],
        "out_mlp.2.weight": state["icl_predictor.decoder.2.weight"],
        "out_mlp.2.bias": state["icl_predictor.decoder.2.bias"],
    }
    if "row_interactor.out_ln.bias" in state:
        out["row_ln.bias"] = state["row_interactor.out_ln.bias"]
    if "icl_predictor.ln.bias" in state:
        out["out_ln.bias"] = state["icl_predictor.ln.bias"]
    for i in range(config["col_num_blocks"]):
        out[f"col_blocks.{i}.inducing_vectors"] = state[
            f"col_embedder.tf_col.blocks.{i}.ind_vectors"
        ].unsqueeze(0)
        _map_block(
            f"col_embedder.tf_col.blocks.{i}.multihead_attn1",
            f"col_blocks.{i}.tfm1",
            state,
            out,
        )
        _map_block(
            f"col_embedder.tf_col.blocks.{i}.multihead_attn2",
            f"col_blocks.{i}.tfm2",
            state,
            out,
        )
    for i in range(config["row_num_blocks"]):
        _map_block(f"row_interactor.tf_row.blocks.{i}", f"row_blocks.{i}", state, out)
    for i in range(config["icl_num_blocks"]):
        _map_block(f"icl_predictor.tf_icl.blocks.{i}", f"icl_blocks.{i}", state, out)
    return out


def _model_from_config(c, model_cls=NanoTabICLv2):
    return model_cls(
        max_classes=c["max_classes"],
        out_dim=c["max_classes"] if c["max_classes"] > 0 else c["num_quantiles"],
        embed_dim=c["embed_dim"],
        col_num_blocks=c["col_num_blocks"],
        row_num_blocks=c["row_num_blocks"],
        icl_num_blocks=c["icl_num_blocks"],
        col_nhead=c["col_nhead"],
        row_nhead=c["row_nhead"],
        icl_nhead=c["icl_nhead"],
        feature_group_size=c["col_feature_group_size"],
        n_cls_cols=c["row_num_cls"],
        n_cls_rows=c["col_num_inds"],
        standardize_x=False,
        feature_group_mode="tabicl",
        bias_free_ln=c["bias_free_ln"],
    )


def _normalize_task_type(task_type: str | None) -> str:
    if task_type in {None, "regression"}:
        return "regression"
    if task_type in {"classification", "binclass", "multiclass"}:
        return "classification"
    raise ValueError(f"Unsupported task type: {task_type!r}")


def _default_checkpoint_version(task_type: str | None) -> str:
    return (
        DEFAULT_REGRESSOR_CHECKPOINT_VERSION
        if _normalize_task_type(task_type) == "regression"
        else DEFAULT_CLASSIFIER_CHECKPOINT_VERSION
    )


def build_model_v0(
    model_path: Optional[str | Path] = None,
    checkpoint_version: str | None = None,
    task_type: str | None = "regression",
    allow_auto_download: bool = True,
    map_location: str | torch.device = "cpu",
    model_cls=NanoTabICLv2,
) -> NanoTabICLv2:
    task_type = _normalize_task_type(task_type)
    checkpoint_version = checkpoint_version or _default_checkpoint_version(task_type)
    if model_path is None:
        try:
            path = hf_hub_download(
                HF_REPO_ID, checkpoint_version, local_files_only=True
            )
        except LocalEntryNotFoundError:
            if not allow_auto_download:
                raise ValueError(
                    f"Checkpoint {checkpoint_version!r} is not cached locally."
                )
            path = hf_hub_download(HF_REPO_ID, checkpoint_version)
    else:
        path = Path(model_path)
        if not path.exists():
            if not allow_auto_download:
                raise ValueError(f"Checkpoint not found: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            got = Path(
                hf_hub_download(HF_REPO_ID, checkpoint_version, local_dir=path.parent)
            )
            if got != path:
                got.rename(path)
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    config, state = ckpt["config"], ckpt["state_dict"]
    checkpoint_task_type = (
        "classification" if int(config.get("max_classes", 0)) > 0 else "regression"
    )
    if checkpoint_task_type != task_type:
        raise ValueError(
            f"Requested a {task_type} model but checkpoint {checkpoint_version!r} "
            f"is for {checkpoint_task_type}."
        )
    model = _model_from_config(config, model_cls)
    model.load_state_dict(convert_tabiclv2_state_dict(state, config), strict=True)
    model.eval()
    model.config = config
    model.checkpoint_version = checkpoint_version
    return model


def build_regressor_v0(**kwargs) -> NanoTabICLv2:
    return build_model_v0(task_type="regression", **kwargs)


def build_classifier_v0(**kwargs) -> NanoTabICLv2:
    return build_model_v0(task_type="classification", **kwargs)


load_pretrained_tabiclv2_regressor = build_regressor_v0
load_pretrained_tabiclv2_classifier = build_classifier_v0
