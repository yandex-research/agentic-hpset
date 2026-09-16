"""Assemble one self-contained modular MLP package for tune/evaluate."""

import time
from pathlib import Path

from typing import Any, NotRequired, TypedDict

import delu
import numpy as np
import scipy.special
import torch
from loguru import logger

from bin._pipeline import RunResult

from bin.mlp._lib import data as lib_data
from bin.mlp._lib import experiment as lib_experiment
from bin.mlp._lib import util as lib_util
from bin.mlp._lib.types import KWArgs

from .core import Dataset, seed_everything, to_torch, validate_and_prepare_labels
from .modules import (
    IMPLEMENTATION_REGISTRIES,
    STAGE_KEYS,
    registry_name,
    resolve,
)


_DEFAULT_PREDICT_BATCH_SIZE = 32768


class Config(TypedDict):
    seed: int
    data: KWArgs
    model: KWArgs
    batch_size: int
    predict_batch_size: NotRequired[int]
    patience: NotRequired[int]


def _prepare_labels(
    y_raw: dict[str, np.ndarray], task_type: str
) -> tuple[dict[str, np.ndarray], int]:
    if task_type == 'regression':
        return (
            {
                part: np.asarray(values).reshape(-1).astype(np.float32)
                for part, values in y_raw.items()
            },
            0,
        )
    prepared = validate_and_prepare_labels(y_raw, task_type)
    n_classes = len(np.unique(np.concatenate(list(prepared.values()))))
    return prepared, int(n_classes)


def _build_dataset_from_yr(
    dataset_yr: lib_data.Dataset, trial_params: dict[str, Any], seed: int
) -> Dataset:
    task_type = dataset_yr.task.type_.value
    score_name = dataset_yr.task.score.value
    y_prepared, n_classes = _prepare_labels(dataset_yr.data['y'], task_type)
    stage_config = {
        'seed': seed,
        'y_train': y_prepared['train'],
        'n_classes': n_classes,
        'task_type': task_type,
    }
    impl = trial_params['implementation_indices']
    num_fn = resolve('numerical_preprocess', impl['numerical_preprocess'], task_type)
    cat_fn = resolve(
        'categorical_preprocess', impl['categorical_preprocess'], task_type
    )
    target_fn = resolve('target_preprocess', impl['target_preprocess'], task_type)
    x_num, num_artifacts = num_fn(dataset_yr.data.get('x_num'), stage_config)
    x_cat, cat_artifacts = cat_fn(dataset_yr.data.get('x_cat'), stage_config)
    y, target_artifacts = target_fn(y_prepared, stage_config)
    if 'inverse_transform' not in target_artifacts:
        if task_type == 'regression':
            raise RuntimeError(
                'Regression target preprocessing must expose inverse_transform.'
            )
        target_artifacts['inverse_transform'] = lambda values: np.asarray(values)
    return Dataset(
        x_num=x_num,
        x_cat=x_cat,
        y=y,
        y_raw=y_prepared,
        task_type=task_type,
        score_name=score_name,
        n_classes=n_classes,
        preprocess_artifacts={
            'numerical': num_artifacts,
            'categorical': cat_artifacts,
            'target': target_artifacts,
        },
    )


def _build_model(
    dataset: Dataset,
    tensors,
    trial_params: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    task_type = dataset.task_type
    impl = trial_params['implementation_indices']
    num_builder = resolve('num_embedding', impl['num_embedding'], task_type)
    cat_builder = resolve('cat_embedding', impl['cat_embedding'], task_type)
    model_builder = resolve('model', impl['model'], task_type)
    num_embedding, num_dim, num_artifacts = num_builder(
        None if tensors.x_num is None else tensors.x_num['train'],
        dataset,
        trial_params,
        device,
    )
    cat_embedding, cat_dim, cat_artifacts = cat_builder(
        dataset.cat_cardinalities, dataset, trial_params, device
    )
    bundle = {
        'num_embedding': num_embedding,
        'cat_embedding': cat_embedding,
        'input_dim': int(num_dim) + int(cat_dim),
        'artifacts': {
            'num_embedding': num_artifacts,
            'cat_embedding': cat_artifacts,
        },
    }
    return model_builder(dataset, trial_params, bundle, device).to(device)


def _to_yr_predictions(
    dataset: Dataset, raw_predictions: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], str]:
    if dataset.is_regression:
        return (
            {
                part: np.asarray(values).reshape(-1).astype(np.float32)
                for part, values in raw_predictions.items()
            },
            'labels',
        )
    if dataset.is_binclass:
        return (
            {
                part: scipy.special.expit(np.asarray(values).reshape(-1)).astype(
                    np.float32
                )
                for part, values in raw_predictions.items()
            },
            'probs',
        )
    return (
        {
            part: scipy.special.softmax(np.asarray(values), axis=1).astype(np.float32)
            for part, values in raw_predictions.items()
        },
        'probs',
    )


def train_and_eval(config: Config, *, device: str | torch.device | None = None) -> RunResult:
    report = lib_experiment.create_report(main, add_gpu_info=True)
    seed = int(config['seed'])
    delu.random.seed(seed)
    seed_everything(seed)
    device = lib_util.get_device() if device is None else torch.device(device)
    logger.info(f'Device: {device}')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dataset_yr = lib_data.build_dataset(**config['data'])
    task_type = dataset_yr.task.type_.value
    trial_params: dict[str, Any] = dict(config['model'])
    defaults = {
        'lr': 1e-3,
        'weight_decay': 0.0,
        'n_blocks': 2,
        'd_block': 128,
        'dropout': 0.0,
        'n_bins': 16,
        'd_embedding': 16,
        'max_epochs': 256,
        'patience': 16,
        'gradient_clip': 1.0,
    }
    for key, value in defaults.items():
        trial_params.setdefault(key, value)
    trial_params['batch_size'] = int(config['batch_size'])
    trial_params['eval_batch_size'] = int(
        config.get('predict_batch_size', _DEFAULT_PREDICT_BATCH_SIZE)
    )
    trial_params['seed'] = seed
    if 'patience' in config:
        trial_params['patience'] = int(config['patience'])

    dataset = _build_dataset_from_yr(dataset_yr, trial_params, seed)
    tensors = to_torch(dataset, device)
    model = _build_model(dataset, tensors, trial_params, device)
    impl = trial_params['implementation_indices']
    loss_fn = resolve('loss', impl['loss'], task_type)
    optimizer_builder = resolve('optimizer', impl['optimizer'], task_type)
    train_fn = resolve('train', impl['train'], task_type)
    inference_fn = resolve('inference', impl['inference'], task_type)

    start = time.perf_counter()
    train_report, _ = train_fn(
        model,
        tensors,
        dataset,
        trial_params,
        inference_fn,
        loss_fn,
        optimizer_builder,
        device,
        seed,
    )
    fit_seconds = time.perf_counter() - start
    model.eval()
    with torch.no_grad():
        raw_predictions = inference_fn(
            model,
            tensors,
            dataset,
            trial_params,
            device,
            ('train', 'val', 'test'),
        )
    predictions, prediction_type = _to_yr_predictions(dataset, raw_predictions)
    if not all(np.isfinite(values).all() for values in predictions.values()):
        raise RuntimeError('Encountered non-finite final predictions.')
    report['prediction_type'] = prediction_type
    report['metrics'] = dataset_yr.task.calculate_metrics(predictions, prediction_type)
    report['time'] = fit_seconds
    report['n_parameters'] = int(sum(p.numel() for p in model.parameters()))
    report['predict_batch_size'] = int(trial_params['eval_batch_size'])
    report['implementation_indices'] = dict(impl)
    report['train'] = train_report
    return RunResult(report, predictions, model)


def main(config: Config, exp: str | Path) -> lib_experiment.Report:
    """Retain the research driver's file-based interface."""
    result = train_and_eval(config)
    lib_experiment.dump_predictions(exp, result.predictions)
    lib_experiment.finish(exp, result.report)
    return result.report


__all__ = [
    'IMPLEMENTATION_REGISTRIES',
    'STAGE_KEYS',
    'Config',
    'main',
    'registry_name',
    'resolve',
]


if __name__ == '__main__':
    lib_util.init()
    lib_experiment.run_cli(main)
