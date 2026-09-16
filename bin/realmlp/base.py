"""Train and evaluate a RealMLP (pytabkit) model."""

import math
import time
import warnings
from pathlib import Path

from typing import Any, NotRequired, TypedDict

import delu
import numpy as np
import pandas as pd
import torch
from loguru import logger

from bin._pipeline import RunResult

import bin.realmlp._lib.data
import bin.realmlp._lib.experiment
import bin.realmlp._lib.util
from bin.realmlp._lib.data import Dataset, Score
from bin.realmlp._lib.types import KWArgs

from pytabkit import RealMLP_TD_Classifier, RealMLP_TD_Regressor

_DEFAULT_PREDICT_BATCH_SIZE = 8192
_MIN_PREDICT_BATCH_SIZE = 1


class Config(TypedDict):
    seed: int
    data: KWArgs
    # Fully-resolved RealMLP params (the tuning DSL produces this via `_if_`
    # for the `is_large` branch). `main` post-processes two derived keys:
    #   `one_minus_sq_mom`      -> `sq_mom = 1 - one_minus_sq_mom`
    #   `max_one_hot_cat_size_raw` -> `max_one_hot_cat_size = floor(.)`
    model: KWArgs
    batch_size: int
    predict_batch_size: NotRequired[int]  # default 8192, halved on OOM
    patience: NotRequired[int]
    realmlp_verbosity: NotRequired[int]


# ======================================================================================
# Dataset → pytabkit conversion
# ======================================================================================
def _impute_x_num(x_num: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Mean-impute NaNs in x_num using train statistics."""
    train = np.asarray(x_num['train'], dtype=np.float32)
    finite = ~np.isnan(train)
    counts = finite.sum(axis=0)
    sums = np.where(finite, train, 0.0).sum(axis=0, dtype=np.float64)
    fill = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0).astype(
        np.float32
    )
    fill = np.where(np.isnan(fill), np.float32(0.0), fill).astype(np.float32)

    out = {}
    for part, values in x_num.items():
        values = np.asarray(values, dtype=np.float32)
        if np.isnan(values).any():
            values = np.where(np.isnan(values), fill, values)
        out[part] = values.astype(np.float32, copy=False)
    return out


def _combine_features(
    x_num_part: np.ndarray | None, x_cat_part: np.ndarray | None
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if x_num_part is not None and x_num_part.shape[1] > 0:
        frames.append(
            pd.DataFrame(
                np.asarray(x_num_part, dtype=np.float32),
                columns=[f'num_{i}' for i in range(x_num_part.shape[1])],
            )
        )
    if x_cat_part is not None and x_cat_part.shape[1] > 0:
        cat_df = pd.DataFrame(
            x_cat_part, columns=[f'cat_{i}' for i in range(x_cat_part.shape[1])]
        )
        for column in cat_df.columns:
            cat_df[column] = cat_df[column].astype('category')
        frames.append(cat_df)
    if not frames:
        raise RuntimeError('Dataset has neither numerical nor categorical features.')
    return frames[0] if len(frames) == 1 else pd.concat(frames, axis=1)


def _to_pytabkit_frames(
    dataset: Dataset,
) -> tuple[dict[str, pd.DataFrame], list[bool]]:
    x_num = dataset.data.get('x_num')
    x_cat = dataset.data.get('x_cat')
    if x_num is not None:
        x_num = _impute_x_num(x_num)
    n_num = 0 if x_num is None else int(x_num['train'].shape[1])
    n_cat = 0 if x_cat is None else int(x_cat['train'].shape[1])
    cat_indicator = [False] * n_num + [True] * n_cat
    frames = {
        part: _combine_features(
            None if x_num is None else x_num[part],
            None if x_cat is None else x_cat[part],
        )
        for part in ('train', 'val', 'test')
    }
    return frames, cat_indicator


def _to_pytabkit_labels(dataset: Dataset) -> dict[str, np.ndarray]:
    y = dataset.data['y']
    dtype = np.float32 if dataset.task.is_regression else np.int64
    return {part: values.astype(dtype).reshape(-1) for part, values in y.items()}


# ======================================================================================
# pytabkit estimator wrapper
# ======================================================================================
_SCORE_TO_PYTABKIT = {
    Score.RMSE: 'rmse',
    Score.MAE: 'mae',
    Score.R2: 'r2',
    Score.CROSS_ENTROPY: 'cross_entropy',
    Score.ACCURACY: 'class_error',
    Score.ROC_AUC: '1-auc_ovr',
}


def _pytabkit_val_metric_name(task: bin.realmlp._lib.data.Task) -> str | None:
    return _SCORE_TO_PYTABKIT.get(task.score)


def _build_estimator(
    task: bin.realmlp._lib.data.Task,
    params: dict[str, Any],
    *,
    seed: int,
    device: str,
    batch_size: int,
    predict_batch_size: int,
    patience: int | None,
    val_metric_name: str | None,
    verbosity: int,
):
    cls = RealMLP_TD_Regressor if task.is_regression else RealMLP_TD_Classifier
    kwargs: dict[str, Any] = {
        'device': device,
        'random_state': seed,
        'n_cv': 1,
        'verbosity': int(verbosity),
        **params,
    }
    if val_metric_name is not None:
        kwargs['val_metric_name'] = val_metric_name
    if patience is not None:
        kwargs['use_early_stopping'] = True
        kwargs['early_stopping_additive_patience'] = int(patience)
    kwargs['batch_size'] = int(batch_size)
    kwargs['predict_batch_size'] = int(predict_batch_size)
    return cls(**kwargs)


def _is_oom(err: BaseException) -> bool:
    try:
        import torch

        if isinstance(err, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    message = str(err).lower()
    return 'out of memory' in message


def _clear_cache(device: str) -> None:
    try:
        import torch
    except Exception:
        return
    if device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif (
        device == 'mps' and hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache')
    ):
        torch.mps.empty_cache()


def _fit_with_oom_retry(
    *,
    task: bin.realmlp._lib.data.Task,
    params: dict[str, Any],
    seed: int,
    batch_size: int,
    predict_batch_size: int,
    patience: int | None,
    device: str,
    val_metric_name: str | None,
    verbosity: int,
    x: dict[str, pd.DataFrame],
    y: dict[str, np.ndarray],
    cat_indicator: list[bool],
):
    start = time.perf_counter()
    current_predict_batch_size = int(predict_batch_size)
    while True:
        estimator = _build_estimator(
            task,
            params,
            seed=seed,
            device=device,
            batch_size=batch_size,
            predict_batch_size=current_predict_batch_size,
            patience=patience,
            val_metric_name=val_metric_name,
            verbosity=verbosity,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                estimator.fit(
                    x['train'],
                    y['train'],
                    X_val=x['val'],
                    y_val=y['val'],
                    cat_indicator=cat_indicator,
                )
            return estimator, time.perf_counter() - start
        except Exception as err:
            if (
                not _is_oom(err)
                or current_predict_batch_size <= _MIN_PREDICT_BATCH_SIZE
            ):
                raise
            next_size = max(_MIN_PREDICT_BATCH_SIZE, current_predict_batch_size // 2)
            logger.warning(
                f'fit OOM with predict_batch_size={current_predict_batch_size};'
                f' retrying with predict_batch_size={next_size}'
            )
            current_predict_batch_size = next_size
            _clear_cache(device)


def _predict_with_oom_retry(
    task: bin.realmlp._lib.data.Task, estimator, x_part: pd.DataFrame
) -> np.ndarray:
    while True:
        try:
            if task.is_regression:
                return np.asarray(estimator.predict(x_part)).reshape(-1)
            proba = np.asarray(estimator.predict_proba(x_part))
            # `lib/metrics.py` expects binclass `probs` as the 1D
            # positive-class probability, not the sklearn-style (n, 2) array.
            if task.is_binclass:
                return proba[:, 1]
            return proba
        except Exception as err:
            current = int(
                getattr(estimator, 'predict_batch_size', _DEFAULT_PREDICT_BATCH_SIZE)
            )
            if not _is_oom(err) or current <= _MIN_PREDICT_BATCH_SIZE:
                raise
            next_size = max(_MIN_PREDICT_BATCH_SIZE, current // 2)
            logger.warning(
                f'predict OOM with predict_batch_size={current};'
                f' retrying with predict_batch_size={next_size}'
            )
            estimator.predict_batch_size = next_size
            _clear_cache(str(getattr(estimator, 'device', '')))


# ======================================================================================
# Main
# ======================================================================================
def train_and_eval(config: Config, *, device: str | torch.device | None = None) -> RunResult:
    report = bin.realmlp._lib.experiment.create_report(main, add_gpu_info=True)

    delu.random.seed(config['seed'])
    device = str(bin.realmlp._lib.util.get_device() if device is None else device)
    logger.info(f'Device: {device}')

    # >>> Data
    dataset = bin.realmlp._lib.data.build_dataset(**config['data'])
    task = dataset.task
    x_frames, cat_indicator = _to_pytabkit_frames(dataset)
    y = _to_pytabkit_labels(dataset)
    logger.info(
        f'train={x_frames["train"].shape} val={x_frames["val"].shape}'
        f' test={x_frames["test"].shape} cat_dims={sum(cat_indicator)}'
    )

    # >>> Resolve derived params (the only Python-side post-processing).
    realmlp_params = dict(config['model'])
    if 'one_minus_sq_mom' in realmlp_params:
        realmlp_params['sq_mom'] = 1.0 - realmlp_params.pop('one_minus_sq_mom')
    if 'max_one_hot_cat_size_raw' in realmlp_params:
        realmlp_params['max_one_hot_cat_size'] = math.floor(realmlp_params.pop('max_one_hot_cat_size_raw'))
    # `is_large` is a sampling control flag (used by the `_if_` rule in the
    # tuning DSL to gate the large-branch params); pytabkit's RealMLP_TD_*
    # constructor does not accept it.
    realmlp_params.pop('is_large', None)

    # >>> Fit
    val_metric_name = _pytabkit_val_metric_name(task)
    predict_batch_size = int(
        config.get('predict_batch_size', _DEFAULT_PREDICT_BATCH_SIZE)
    )
    estimator, fit_seconds = _fit_with_oom_retry(
        task=task,
        params=realmlp_params,
        seed=config['seed'],
        batch_size=int(config['batch_size']),
        predict_batch_size=predict_batch_size,
        patience=config.get('patience'),
        device=device,
        val_metric_name=val_metric_name,
        verbosity=int(config.get('realmlp_verbosity', 0)),
        x=x_frames,
        y=y,
        cat_indicator=cat_indicator,
    )
    logger.info(f'fit time: {fit_seconds:.1f}s')

    # >>> Evaluate
    predictions = {
        part: _predict_with_oom_retry(task, estimator, x_frames[part])
        for part in ('train', 'val', 'test')
    }
    prediction_type = 'labels' if task.is_regression else 'probs'
    report['prediction_type'] = prediction_type
    report['metrics'] = task.calculate_metrics(predictions, prediction_type)
    report['time'] = fit_seconds
    report['predict_batch_size'] = int(
        getattr(estimator, 'predict_batch_size', predict_batch_size)
    )

    return RunResult(report, predictions, estimator)


def main(config: Config, exp: str | Path) -> bin.realmlp._lib.experiment.Report:
    """Retain the research driver's file-based interface."""
    result = train_and_eval(config)
    bin.realmlp._lib.experiment.dump_predictions(exp, result.predictions)
    bin.realmlp._lib.experiment.finish(exp, result.report)
    return result.report


if __name__ == '__main__':
    bin.realmlp._lib.util.init()
    bin.realmlp._lib.experiment.run_cli(main)
