"""Assemble the modular RealMLP pipeline and expose `main()` for tune/evaluate.

Each pipeline stage (preprocess_numerical, ..., inference_reg) is registered
under an integer index in `bin/realmlp/modules/`. The
`[space.model.implementation_indices]` sub-table in the config picks one index
per stage; all other keys under `[space.model]` flow through as the shared `cfg`
dict every stage reads (see `bin/realmlp/search_space.toml`). Hyperparameters are
not sampled here — an Optuna TPE loop (not shipped) draws from the search space.
"""

import math
import time
from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy

from typing import Any, NotRequired, TypedDict

import delu
import numpy as np
import pandas as pd
import torch
from loguru import logger

from bin._pipeline import RunResult, dataset_path

import bin.realmlp._lib.data
import bin.realmlp._lib.experiment
import bin.realmlp._lib.util
from bin.realmlp.modules import TabPreprocessor, resolve
from bin.realmlp.modules.inference_clf.realmlp import (
    score_classifier,
)
from bin.realmlp.modules.inference_reg.realmlp import (
    score_regressor,
)
from bin.realmlp.base import (
    _DEFAULT_PREDICT_BATCH_SIZE,
    _to_pytabkit_frames,
    _to_pytabkit_labels,
)
from bin.realmlp._lib.types import KWArgs

# NTK init only needs enough samples to estimate per-layer activation std.
# Using the full train tensor blows GPU memory on wide num-embeddings (e.g.
# PLR at plr_hidden_1=64 with ~100 numeric features).
_INIT_BATCH_SIZE = 4096
# Floor for the OOM failsafe retry (batch sizes never shrink below this).
_MIN_BATCH_SIZE = 1


def _is_oom(err: BaseException) -> bool:
    """True for a CUDA out-of-memory error (by type or message)."""
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True
    return 'out of memory' in str(err).lower()


def _clear_cuda_cache(device: torch.device) -> None:
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class FittedMember:
    preprocessor: TabPreprocessor
    model: torch.nn.Module
    y_mean: np.ndarray | None = None
    y_std: np.ndarray | None = None
    y_min: np.ndarray | None = None
    y_max: np.ndarray | None = None


class Config(TypedDict):
    seed: int
    data: KWArgs
    # `model` mirrors `bin/realmlp/base/realmlp.py`'s schema plus one required
    # sub-dict, `implementation_indices` (per-stage int index sampled from
    # `[space.model.implementation_indices]`). `main` also post-processes the
    # base's derived keys:
    #   `one_minus_sq_mom`         -> `sq_mom = 1 - .`
    #   `max_one_hot_cat_size_raw` -> `floor(.)`
    model: KWArgs
    batch_size: int
    predict_batch_size: NotRequired[int]
    patience: NotRequired[int]


def _fit_realmlp(
    *,
    cfg: dict[str, Any],
    x_frames: dict[str, pd.DataFrame],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_val_eval: np.ndarray,
    cat_indicator: list[bool],
    task: bin.realmlp._lib.data.Task,
    device: torch.device,
    out_dim: int,
    init_batch_size: int = _INIT_BATCH_SIZE,
) -> FittedMember:
    task_type = 'regression' if task.is_regression else 'multiclass'
    impl = cfg['implementation_indices']

    # 1) Preprocess (num + cat)
    num_stage = resolve(
        'numerical_preprocess', impl['numerical_preprocess'], task_type
    )(cfg)
    cat_stage = resolve(
        'categorical_preprocess', impl['categorical_preprocess'], task_type
    )(cfg)
    prep = TabPreprocessor(num_stage, cat_stage)
    prep.fit(x_frames['train'], cat_indicator=cat_indicator)
    x_cont_train, x_cat_train = prep.transform(x_frames['train'])
    x_cont_val, x_cat_val = prep.transform(x_frames['val'])

    # 2) Regression target preprocess (classification label encoding is done
    # once at main() level, above).
    y_mean = y_std = y_min = y_max = None
    if task.is_regression:
        target_fn = resolve('target_preprocess', impl['target_preprocess'], task_type)
        y_train, y_val, artifacts = target_fn(y_train, y_val, cfg)
        y_mean = artifacts['y_mean']
        y_std = artifacts['y_std']
        y_min = artifacts['y_min']
        y_max = artifacts['y_max']

    # 3) Tensorize
    tx = torch.as_tensor(x_cont_train, dtype=torch.float32, device=device)
    tc = torch.as_tensor(x_cat_train, dtype=torch.long, device=device)
    vx = torch.as_tensor(x_cont_val, dtype=torch.float32, device=device)
    vc = torch.as_tensor(x_cat_val, dtype=torch.long, device=device)
    if task.is_regression:
        ty = torch.as_tensor(y_train, dtype=torch.float32, device=device)
        vy = torch.as_tensor(y_val, dtype=torch.float32, device=device)
        score_target = torch.as_tensor(y_val_eval, dtype=torch.float32, device=device)
    else:
        ty = torch.as_tensor(y_train, dtype=torch.long, device=device)
        vy = torch.as_tensor(y_val, dtype=torch.long, device=device)
        score_target = vy

    # 4) Embeddings
    num_emb, num_emb_out = resolve('num_embedding', impl['num_embedding'], task_type)(
        prep.n_num_, cfg
    )
    cat_emb, cat_emb_out = resolve('cat_embedding', impl['cat_embedding'], task_type)(
        prep.large_cat_sizes_, cfg
    )

    # 5) Model
    model = resolve('model', impl['model'], task_type)(
        num_emb,
        cat_emb,
        num_embedding_out=num_emb_out,
        n_num=prep.n_num_,
        n_one_hot=prep.n_one_hot_,
        cat_embedding_out=cat_emb_out,
        cfg=cfg,
        out_dim=out_dim,
    ).to(device)
    n_init = min(init_batch_size, tx.shape[0])
    init_idx = torch.randperm(tx.shape[0], device=device)[:n_init]
    model.initialize_from_data(tx[init_idx], tc[init_idx])

    # 6) Optimizer + loss + val-scorer
    opt_bundle = resolve('optimizer', impl['optimizer'], task_type)(model, cfg)
    loss_fn = resolve('loss', impl['loss'], task_type)
    if task.is_regression:
        def score_fn(m, xc, xt, yv):
            return score_regressor(m, xc, xt, yv, cfg, y_mean, y_std, y_min, y_max)
    else:
        def score_fn(m, xc, xt, yv):
            return score_classifier(m, xc, xt, yv, cfg)

    # 7) Train
    train_fn = resolve('train', impl['train'], task_type)
    result = train_fn(
        model, opt_bundle, loss_fn,
        tx, tc, ty,
        vx, vc, score_target,
        score_fn, cfg,
        has_val=True,
        time_limit_s=None,
    )

    # 8) Load best state, move to CPU
    model.load_state_dict(result.best_state)
    model.eval()
    model.to('cpu')
    return FittedMember(
        preprocessor=prep, model=model,
        y_mean=y_mean, y_std=y_std, y_min=y_min, y_max=y_max,
    )


def _predict(
    *,
    task: bin.realmlp._lib.data.Task,
    member: FittedMember,
    x_part: pd.DataFrame,
    cfg: dict[str, Any],
) -> np.ndarray:
    impl = cfg['implementation_indices']
    if task.is_regression:
        pred_fn = resolve('inference', impl['inference'], 'regression')
        return np.asarray(
            pred_fn([member], x_part, cfg, is_y_1d=True, is_y_float64=False)
        ).reshape(-1)
    proba_fn = resolve('inference', impl['inference'], 'multiclass')
    proba = np.asarray(proba_fn([member], x_part, cfg))
    # Float32 aggregation over ensemble columns can exceed 1.0 by an ulp when
    # members saturate, and sklearn's log_loss validation rejects any value
    # > 1. Probabilities are <= 1 in exact arithmetic, so clipping only
    # removes rounding noise.
    proba = np.clip(proba, 0.0, 1.0)
    # `lib/metrics.py` expects binclass probs as the 1D positive-class
    # probability, not sklearn-style (n, 2).
    if task.is_binclass:
        return proba[:, 1]
    return proba


def train_and_eval(config: Config, *, device: str | torch.device | None = None) -> RunResult:
    report = bin.realmlp._lib.experiment.create_report(main, add_gpu_info=True)

    seed = int(config['seed'])
    delu.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = bin.realmlp._lib.util.get_device() if device is None else torch.device(device)
    logger.info(f'Device: {device}')
    if device.type == 'cuda':
        # Wide tabular inputs make the flattened num-embedding -> first-layer
        # GEMM the dominant, matmul-bound cost, multiplied by the n_ens
        # ensemble. `bin.realmlp._lib.util.configure_torch()` disables TF32 for bitwise
        # reproducibility; here we trade that for the large TF32 tensor-core
        # speedup on Ampere+ GPUs (also lowers memory pressure). The numeric
        # change is far below seed-to-seed variance and touches no tuned
        # hyperparameter.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # >>> Data
    dataset = bin.realmlp._lib.data.build_dataset(**config['data'])
    task = dataset.task
    x_frames, cat_indicator = _to_pytabkit_frames(dataset)
    y = _to_pytabkit_labels(dataset)
    logger.info(
        f'train={x_frames["train"].shape} val={x_frames["val"].shape}'
        f' test={x_frames["test"].shape} cat_dims={sum(cat_indicator)}'
    )

    # >>> Resolve cfg (derived keys + harness-supplied fields)
    cfg: dict[str, Any] = dict(config['model'])
    if 'one_minus_sq_mom' in cfg:
        cfg['sq_mom'] = 1.0 - cfg.pop('one_minus_sq_mom')
    if 'max_one_hot_cat_size_raw' in cfg:
        cfg['max_one_hot_cat_size'] = math.floor(cfg.pop('max_one_hot_cat_size_raw'))
    # `is_large` is a sampling-side gate for the `_if_` DSL; no stage reads it.
    cfg.pop('is_large', None)
    predict_batch_size = int(
        config.get('predict_batch_size', _DEFAULT_PREDICT_BATCH_SIZE)
    )
    cfg['batch_size'] = int(config['batch_size'])
    cfg['predict_batch_size'] = predict_batch_size
    if 'patience' in config:
        cfg['use_early_stopping'] = True
        cfg['early_stopping_additive_patience'] = int(config['patience'])

    # >>> Labels + out_dim
    if task.is_regression:
        # `target_preprocess_reg_v0` expects 2D (N, out_dim); reshape 1D → (N, 1).
        y_train = y['train'].astype(np.float32).reshape(-1, 1)
        y_val = y['val'].astype(np.float32).reshape(-1, 1)
        y_val_eval = y_val.copy()
        out_dim = 1
    else:
        # Encode labels once. `bin.realmlp._lib.data` with `cat_policy='ordinal'` already
        # produced deterministic integer labels, so fitting the encoder on
        # train+val together is safe and matches bin.realmlp._lib.data's ordering.
        target_fn = resolve(
            'target_preprocess',
            cfg['implementation_indices']['target_preprocess'],
            'multiclass',
        )
        y_all = np.concatenate([y['train'], y['val']], axis=0)
        y_int, classes, _ = target_fn(y_all)
        n_train = len(y['train'])
        y_train = y_int[:n_train]
        y_val = y_int[n_train:]
        y_val_eval = y_val
        out_dim = len(classes)

    # >>> Fit (with OOM failsafe: on CUDA OOM, halve the eval/init forward
    # batch sizes and retry the whole fit. If the batches hit the floor it
    # re-raises, so `bin/tune.py` prunes the trial / `bin/evaluate.py` skips
    # the seed instead of the study aborting.)
    start = time.perf_counter()
    init_batch_size = _INIT_BATCH_SIZE
    while True:
        try:
            member = _fit_realmlp(
                cfg=cfg,
                x_frames=x_frames,
                y_train=y_train,
                y_val=y_val,
                y_val_eval=y_val_eval,
                cat_indicator=cat_indicator,
                task=task,
                device=device,
                out_dim=out_dim,
                init_batch_size=init_batch_size,
            )
            break
        except Exception as err:
            predict_bs = int(cfg['predict_batch_size'])
            if not _is_oom(err) or predict_bs <= _MIN_BATCH_SIZE:
                raise
            next_predict = max(_MIN_BATCH_SIZE, predict_bs // 2)
            next_init = max(_MIN_BATCH_SIZE, init_batch_size // 2)
            logger.warning(
                f'fit OOM with predict_batch_size={predict_bs}'
                f' init_batch_size={init_batch_size};'
                f' retrying with predict_batch_size={next_predict}'
                f' init_batch_size={next_init}'
            )
            cfg['predict_batch_size'] = next_predict
            init_batch_size = next_init
            _clear_cuda_cache(device)
    fit_seconds = time.perf_counter() - start
    logger.info(f'fit time: {fit_seconds:.1f}s')

    # >>> Evaluate
    predictions = {
        part: _predict(task=task, member=member, x_part=x_frames[part], cfg=cfg)
        for part in ('train', 'val', 'test')
    }
    # Failsafe: surface a diverged trial as a classified non-finite error (so
    # the harness prunes/skips it) rather than a downstream sklearn traceback.
    for part, values in predictions.items():
        if not np.isfinite(values).all():
            raise RuntimeError(
                f'Encountered non-finite RealMLP predictions for part={part!r}.'
            )
    prediction_type = 'labels' if task.is_regression else 'probs'
    report['prediction_type'] = prediction_type
    report['metrics'] = task.calculate_metrics(predictions, prediction_type)
    report['time'] = fit_seconds
    report['n_parameters'] = int(sum(p.numel() for p in member.model.parameters()))
    report['predict_batch_size'] = predict_batch_size

    return RunResult(report, predictions, member)


def main(config: Config, exp: str | Path) -> bin.realmlp._lib.experiment.Report:
    """Retain the research driver's file-based interface."""
    result = train_and_eval(config)
    bin.realmlp._lib.experiment.dump_predictions(exp, result.predictions)
    bin.realmlp._lib.experiment.finish(exp, result.report)
    return result.report


def run(config: dict[str, Any], *, dataset_root: str | Path | None = None,
        device: str | torch.device | None = None) -> RunResult:
    """Fit a saved baseline or agentic config; return metrics and predictions in memory."""
    cfg = deepcopy(config)
    root = dataset_path(cfg, dataset_root)
    if "trial_params" in cfg:
        # The baseline ensemble export stores resolved pytabkit parameters here.
        params = cfg["trial_params"]
        cfg = {
            "seed": cfg.get("seed", 0), "data": cfg.get("data", {}),
            "model": params, "batch_size": params.pop("batch_size", 256),
            "predict_batch_size": cfg.get("predict_batch_size", 8192),
        }
    cfg["data"] = {**cfg.get("data", {}), "path": str(root)}
    if "implementation_indices" in cfg["model"]:
        return train_and_eval(cfg, device=device)
    from bin.realmlp.base import train_and_eval as train_baseline
    return train_baseline(cfg, device=device)


__all__ = ['Config', 'main', 'train_and_eval', 'run']


if __name__ == '__main__':
    bin.realmlp._lib.util.init()
    bin.realmlp._lib.experiment.run_cli(main)
