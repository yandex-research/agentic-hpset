"""Materialized inference implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from ...core import Dataset, TorchDataset


def _output_tensor(output: Any) -> Tensor:
    if isinstance(output, dict):
        for key in ('prediction', 'predictions', 'logits', 'output'):
            if key in output:
                return output[key]
        return next(iter(output.values()))
    if isinstance(output, tuple | list):
        return output[0]
    return output


def _forward_parts(
    model: torch.nn.Module,
    tensors: TorchDataset,
    trial_params: dict[str, Any],
    parts: tuple[str, ...],
    *,
    passes: int = 1,
    noise_scale: float = 0.0,
    dropout: bool = False,
    median: bool = False,
) -> dict[str, np.ndarray]:
    was_training = model.training
    model.train(dropout)
    predictions: dict[str, np.ndarray] = {}
    batch_size = int(trial_params['eval_batch_size'])
    with torch.no_grad():
        for part in parts:
            pass_outputs: list[Tensor] = []
            for _ in range(passes):
                outputs: list[Tensor] = []
                indices = torch.arange(
                    tensors.y[part].shape[0], device=tensors.y[part].device
                )
                for batch_idx in indices.split(batch_size):
                    x_num = (
                        None
                        if tensors.x_num is None
                        else tensors.x_num[part][batch_idx]
                    )
                    if x_num is not None and noise_scale:
                        x_num = x_num + torch.randn_like(x_num) * noise_scale
                    x_cat = (
                        None
                        if tensors.x_cat is None
                        else tensors.x_cat[part][batch_idx]
                    )
                    outputs.append(_output_tensor(model(x_num, x_cat)))
                pass_outputs.append(torch.cat(outputs))
            stacked = torch.stack(pass_outputs)
            aggregate = stacked.median(dim=0).values if median else stacked.mean(dim=0)
            predictions[part] = aggregate.detach().cpu().numpy().astype(np.float32)
    model.train(was_training)
    return predictions


def _fit_temperature(logits: np.ndarray, targets: np.ndarray, binclass: bool) -> float:
    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    targets_t = torch.as_tensor(
        targets,
        dtype=torch.float32 if binclass else torch.long,
    )
    temperatures = torch.logspace(-1.5, 1.5, 61)
    losses = []
    for temperature in temperatures:
        scaled = logits_t / temperature
        losses.append(
            F.binary_cross_entropy_with_logits(scaled.reshape(-1), targets_t).item()
            if binclass
            else F.cross_entropy(scaled, targets_t).item()
        )
    return float(temperatures[int(np.argmin(losses))])


def make_inference_variant(
    strategy: str, params: dict[str, Any], symbol: str
) -> Callable[..., dict[str, np.ndarray]]:
    passes = int(params.get('passes', params.get('n_passes', 8)))
    noise_scale = float(params.get('noise_scale', params.get('noise_std', 0.01)))

    def inference(
        model: torch.nn.Module,
        tensors: TorchDataset,
        dataset: Dataset,
        trial_params: dict[str, Any],
        _device: torch.device,
        parts: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        if strategy == 'mc_dropout':
            return _forward_parts(
                model, tensors, trial_params, parts, passes=passes, dropout=True
            )
        if strategy in {'tta', 'input_noise_tta', 'median_tta'}:
            return _forward_parts(
                model,
                tensors,
                trial_params,
                parts,
                passes=passes,
                noise_scale=noise_scale,
                median=strategy == 'median_tta',
            )
        predictions = _forward_parts(model, tensors, trial_params, parts)
        if strategy == 'temperature':
            calibration = _forward_parts(model, tensors, trial_params, ('val',))['val']
            temperature = _fit_temperature(
                calibration, dataset.y_raw['val'], dataset.is_binclass
            )
            return {part: values / temperature for part, values in predictions.items()}
        if strategy == 'prior_calibration' and dataset.is_binclass:
            prior = float(np.asarray(dataset.y_raw['train']).mean())
            for part, values in predictions.items():
                probs = 1.0 / (1.0 + np.exp(-values.reshape(-1)))
                predicted_prior = float(np.clip(probs.mean(), 1e-4, 1 - 1e-4))
                shift = np.log(prior / (1 - prior + 1e-12) + 1e-12) - np.log(
                    predicted_prior / (1 - predicted_prior)
                )
                predictions[part] = (values.reshape(-1) + shift).astype(np.float32)
        return predictions

    inference.__name__ = f'inference_{strategy}_{abs(hash(symbol)) & 0xFFFF:x}'
    inference.__doc__ = f'Ported from {symbol}; extracted inference: {strategy}.'
    return inference


_implementation = make_inference_variant(
    'temperature',
    {},
    'evaluate.evaluate_temperature.evaluate_v2',
)


def inference_temperature_v2(*args: Any, **kwargs: Any) -> Any:
    return _implementation(*args, **kwargs)
