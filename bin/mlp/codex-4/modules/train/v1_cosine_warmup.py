"""Materialized training implementation."""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from ...core import (
    Dataset,
    TorchDataset,
    capture_gpu_stats,
    compute_metrics_for_parts,
    maybe_sync,
    reset_gpu_stats,
    seed_everything,
)

InferenceFn = Callable[..., dict[str, np.ndarray]]

LossFn = Callable[[Any, Tensor, Dataset], Tensor]

OptimizerBuilder = Callable[[nn.Module, dict[str, Any]], torch.optim.Optimizer]


def _prediction_tensor(output: Any) -> Tensor:
    if isinstance(output, dict):
        for key in ('prediction', 'predictions', 'logits', 'mean', 'output'):
            if key in output:
                return output[key]
        return next(iter(output.values()))
    if isinstance(output, tuple | list):
        return output[0]
    return output


def _model_regularization(model: nn.Module, device: torch.device) -> Tensor:
    regularizer = getattr(model, 'regularization_loss', None)
    if callable(regularizer):
        value = regularizer()
        if isinstance(value, Tensor):
            return value
    return torch.zeros((), device=device)


def _copy_state(model: nn.Module) -> dict[str, Tensor]:
    return copy.deepcopy(model.state_dict())


def _update_average(
    average: dict[str, Tensor], model: nn.Module, decay: float | None, count: int
) -> None:
    for key, value in model.state_dict().items():
        if not torch.is_floating_point(value):
            average[key].copy_(value)
        elif decay is None:
            average[key].add_(value.detach() - average[key], alpha=1.0 / count)
        else:
            average[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)


def _evaluate(
    model: nn.Module,
    tensors: TorchDataset,
    dataset: Dataset,
    params: dict[str, Any],
    inference_fn: InferenceFn,
    device: torch.device,
    parts: tuple[str, ...],
    state: dict[str, Tensor] | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    current = _copy_state(model) if state is not None else None
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        predictions = inference_fn(model, tensors, dataset, params, device, parts)
    if current is not None:
        model.load_state_dict(current)
    if not all(np.isfinite(value).all() for value in predictions.values()):
        raise RuntimeError('Encountered non-finite MLP predictions.')
    return compute_metrics_for_parts(dataset, predictions, parts), predictions


def _scheduler(
    optimizer: torch.optim.Optimizer,
    strategy: str,
    params: dict[str, Any],
    total_steps: int,
):
    if 'onecycle' in strategy:
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max(group['lr'] for group in optimizer.param_groups),
            total_steps=total_steps,
            pct_start=float(params.get('warmup_fraction', 0.1)),
        )
    if 'snapshot' in strategy:
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, total_steps // int(params.get('n_cycles', 3))),
        )
    if 'cosine' not in strategy and 'warmup' not in strategy:
        return None
    warmup = max(1, int(total_steps * float(params.get('warmup_fraction', 0.05))))

    def schedule(step: int) -> float:
        if step < warmup:
            return max(1e-3, (step + 1) / warmup)
        if 'cosine' not in strategy:
            return 1.0
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def _consistency_loss(first: Tensor, second: Tensor, dataset: Dataset) -> Tensor:
    if dataset.is_regression:
        return F.mse_loss(first, second)
    if dataset.is_binclass:
        return F.mse_loss(torch.sigmoid(first), torch.sigmoid(second))
    return 0.5 * (
        F.kl_div(
            F.log_softmax(first, -1),
            F.softmax(second.detach(), -1),
            reduction='batchmean',
        )
        + F.kl_div(
            F.log_softmax(second, -1),
            F.softmax(first.detach(), -1),
            reduction='batchmean',
        )
    )


def run_training_loop(
    model: nn.Module,
    tensors: TorchDataset,
    dataset: Dataset,
    trial_params: dict[str, Any],
    inference_fn: InferenceFn,
    loss_fn: LossFn,
    optimizer_builder: OptimizerBuilder,
    device: torch.device,
    seed: int,
    *,
    strategy: str = 'standard',
    strategy_params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    strategy_params = strategy_params or {}
    seed_everything(seed)
    optimizer = optimizer_builder(model, trial_params)
    batch_size = int(trial_params['batch_size'])
    gradient_clip = float(trial_params['gradient_clip'])
    train_size = dataset.size('train')
    max_epochs = int(trial_params['max_epochs'])
    steps_per_epoch = math.ceil(train_size / batch_size)
    scheduler_params = dict(strategy_params)
    if 'warmup_epochs' in scheduler_params:
        scheduler_params['warmup_fraction'] = (
            float(scheduler_params['warmup_epochs']) / max_epochs
        )
    scheduler = _scheduler(
        optimizer, strategy, scheduler_params, max(1, max_epochs * steps_per_epoch)
    )
    use_ema = 'ema' in strategy
    use_swa = 'swa' in strategy
    average = _copy_state(model) if use_ema or use_swa else None
    average_count = 0
    ema_decay = float(
        strategy_params.get('decay', strategy_params.get('ema_decay', 0.995))
    )
    swa_start_epoch = int(
        strategy_params.get(
            'swa_start_epoch',
            max(
                1,
                math.ceil(
                    max_epochs * float(strategy_params.get('swa_start_fraction', 0.5))
                ),
            ),
        )
    )
    best_score = float('inf')
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    last_it_s = 0
    reset_gpu_stats(device)
    maybe_sync(device)
    start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        epoch_losses: list[float] = []
        for batch_idx in torch.randperm(train_size, device=device).split(batch_size):
            x_num = None if tensors.x_num is None else tensors.x_num['train'][batch_idx]
            x_cat = None if tensors.x_cat is None else tensors.x_cat['train'][batch_idx]
            target = tensors.y['train'][batch_idx]
            alternate_target: Tensor | None = None
            mix_lambda = 1.0
            if 'mixup' in strategy and x_num is not None and len(batch_idx) > 1:
                mix_alpha = float(
                    strategy_params.get(
                        'mixup_alpha', strategy_params.get('alpha', 0.2)
                    )
                )
                mix_lambda = float(np.random.beta(mix_alpha, mix_alpha))
                permutation = torch.randperm(len(batch_idx), device=device)
                x_num = mix_lambda * x_num + (1.0 - mix_lambda) * x_num[permutation]
                alternate_target = target[permutation]
            if strategy == 'feature_noise' and x_num is not None:
                scale = float(
                    strategy_params.get(
                        'noise_scale',
                        strategy_params.get(
                            'noise_std', strategy_params.get('adv_epsilon', 0.01)
                        ),
                    )
                )
                x_num = x_num + torch.randn_like(x_num) * scale
            if 'masked' in strategy and x_num is not None:
                mask_probability = float(strategy_params.get('mask_probability', 0.15))
                x_num = x_num.masked_fill(
                    torch.rand_like(x_num) < mask_probability, 0.0
                )
            noisy_target = target
            if 'target_noise' in strategy and dataset.is_regression:
                noisy_target = target + torch.randn_like(target) * float(
                    strategy_params.get('target_noise_std', 0.01)
                )

            def closure() -> Tensor:
                if strategy == 'adversarial' and x_num is not None:
                    clean_x_num = x_num.detach().clone().requires_grad_(True)
                    output = model(clean_x_num, x_cat)
                    clean_loss = loss_fn(output, noisy_target, dataset)
                    input_gradient = torch.autograd.grad(
                        clean_loss, clean_x_num, retain_graph=True
                    )[0]
                    adversarial_x_num = (
                        clean_x_num
                        + float(strategy_params.get('adv_epsilon', 0.01))
                        * input_gradient.sign()
                    ).detach()
                    adversarial_output = model(adversarial_x_num, x_cat)
                    adversarial_loss = loss_fn(
                        adversarial_output, noisy_target, dataset
                    )
                    adversarial_weight = float(strategy_params.get('adv_weight', 0.5))
                    loss = (
                        1.0 - adversarial_weight
                    ) * clean_loss + adversarial_weight * adversarial_loss
                else:
                    output = model(x_num, x_cat)
                    loss = loss_fn(output, noisy_target, dataset)
                if strategy == 'gradient_boost' and dataset.is_regression:
                    prediction = _prediction_tensor(output)
                    if prediction.ndim == 2:
                        prediction = prediction[:, 0]
                    residual = prediction.reshape(-1) - noisy_target.float().reshape(-1)
                    n_rounds = max(1, int(strategy_params.get('n_boost_rounds', 3)))
                    round_index = min(
                        n_rounds - 1, (epoch - 1) * n_rounds // max_epochs
                    )
                    emphasis = 1.0 + (
                        round_index
                        * float(strategy_params.get('learning_rate_scale', 0.5))
                        * residual.detach().abs()
                        / residual.detach().abs().mean().clamp_min(1e-6)
                    )
                    loss = (emphasis * residual.square()).mean()
                if alternate_target is not None:
                    loss = mix_lambda * loss + (1.0 - mix_lambda) * loss_fn(
                        output, alternate_target, dataset
                    )
                if 'rdrop' in strategy or 'distill' in strategy or 'masked' in strategy:
                    second = model(x_num, x_cat)
                    loss = 0.5 * (loss + loss_fn(second, noisy_target, dataset))
                    loss = loss + float(
                        strategy_params.get(
                            'consistency_weight',
                            strategy_params.get(
                                'aux_weight', strategy_params.get('distill_alpha', 0.1)
                            ),
                        )
                    ) * _consistency_loss(
                        _prediction_tensor(output), _prediction_tensor(second), dataset
                    )
                return loss + _model_regularization(model, device)

            optimizer.zero_grad(set_to_none=True)
            loss = closure()
            if not torch.isfinite(loss):
                raise RuntimeError('Encountered non-finite MLP loss.')
            if 'sam' in strategy:
                loss.backward()
                parameters = [p for p in model.parameters() if p.grad is not None]
                norm = torch.norm(torch.stack([p.grad.norm() for p in parameters]))
                scale = float(strategy_params.get('rho', 0.05)) / (norm + 1e-12)
                perturbations: list[tuple[nn.Parameter, Tensor]] = []
                with torch.no_grad():
                    for parameter in parameters:
                        perturbation = parameter.grad * scale.to(parameter)
                        parameter.add_(perturbation)
                        perturbations.append((parameter, perturbation))
                optimizer.zero_grad(set_to_none=True)
                second_loss = closure()
                second_loss.backward()
                with torch.no_grad():
                    for parameter, perturbation in perturbations:
                        parameter.sub_(perturbation)
                loss = second_loss
            else:
                loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if average is not None and (use_ema or epoch >= swa_start_epoch):
                average_count += 1
                _update_average(
                    average,
                    model,
                    ema_decay if use_ema else None,
                    average_count,
                )
            epoch_losses.append(float(loss.detach().cpu()))

        eval_state = average if average is not None and average_count else None
        metrics, _ = _evaluate(
            model,
            tensors,
            dataset,
            trial_params,
            inference_fn,
            device,
            ('val',),
            eval_state,
        )
        val_score = float(metrics['val']['score'])
        improved = val_score < best_score
        if improved:
            best_score = val_score
            best_epoch = epoch
            best_state = copy.deepcopy(
                eval_state if eval_state is not None else model.state_dict()
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        elapsed = time.perf_counter() - start
        epoch_time = max(time.perf_counter() - epoch_start, 1e-6)
        last_it_s = max(1, math.trunc(steps_per_epoch / epoch_time))
        print(
            f'{"*" if improved else " "} [mlp epoch] {epoch:<3}'
            f' [val {dataset.score_name}] {metrics["val"][dataset.score_name]:.4f}'
            f' [loss] {float(np.mean(epoch_losses)):.4f}'
            f' [time] {elapsed:.1f}s [it/s] {last_it_s:>3}'
        )
        if epochs_without_improvement >= int(trial_params['patience']):
            break

    if best_state is None:
        raise RuntimeError('Training did not produce a valid checkpoint.')
    model.load_state_dict(best_state)
    model.eval()
    metrics, _ = _evaluate(
        model,
        tensors,
        dataset,
        trial_params,
        inference_fn,
        device,
        ('train', 'val', 'test'),
    )
    maybe_sync(device)
    report = {
        'model': getattr(model, 'model_name', 'mlp'),
        'strategy': strategy,
        'score_name': dataset.score_name,
        'score': float(metrics['val']['score']),
        'device': str(device),
        'n_parameters': int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        'best_epoch': best_epoch,
        'time_seconds': time.perf_counter() - start,
        'it_s': last_it_s,
        'gpu': capture_gpu_stats(device),
        'metrics': metrics,
    }
    return report, best_state


def train_cosine_warmup_v1(*args: Any, **kwargs: Any) -> Any:
    return run_training_loop(
        *args,
        **kwargs,
        strategy='cosine_warmup',
        strategy_params={},
    )
