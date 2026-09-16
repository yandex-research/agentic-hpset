"""Search space, per-dataset defaults, and sampler for the modular TabM.

This single module defines *how hyperparameters are sampled*, so a run is fully
specified by:

  * the implementation registries (which candidate module each ``*_idx`` selects),
  * the continuous/integer HP ranges in :func:`sample_hps`,
  * the per-dataset fixed defaults in :func:`get_default_hps`,
  * the optimizer (Optuna TPE, :func:`make_sampler`) and trial budget
    (:func:`n_trials_for`).

The result JSONs under ``exp/tuned/tabm`` and ``exp/tuned/agentic-tabm`` record the
sampled ``implementation_indices`` and ``trial_params``; feed those back through
``bin.tabm.pipeline`` to reconstruct any evaluated model. Index 0 of every stage is
the base reference implementation, so the non-agentic ``tabm`` runs are this same
pipeline with every ``*_idx`` pinned to 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import optuna

from bin.tabm.modules.embedding_cat import CAT_EMBEDDING_MAP
from bin.tabm.modules.embedding_num import NUM_EMBEDDING_MAP
from bin.tabm.modules.inference_clf import INFERENCE_MAP as INFERENCE_CLF_MAP
from bin.tabm.modules.inference_reg import INFERENCE_MAP as INFERENCE_REG_MAP
from bin.tabm.modules.loss_clf import LOSS_CLF_MAP
from bin.tabm.modules.loss_reg import LOSS_REG_MAP
from bin.tabm.modules.model import MODEL_MAP
from bin.tabm.modules.optimizer import OPTIMIZER_MAP
from bin.tabm.modules.preprocess_categorical import CAT_PREPROCESS_MAP
from bin.tabm.modules.preprocess_numerical import NUM_PREPROCESS_MAP
from bin.tabm.modules.preprocess_target_clf import TARGET_PREPROCESS_CLF_MAP
from bin.tabm.modules.preprocess_target_reg import TARGET_PREPROCESS_REG_MAP
from bin.tabm.modules.train import TRAIN_MAP


# ---------------------------------------------------------------------------
# Implementation registries (the targets of the sampled ``*_idx`` indices)
# ---------------------------------------------------------------------------

IMPLEMENTATION_REGISTRIES: dict[str, dict[int, object]] = {
    "numerical_preprocess": NUM_PREPROCESS_MAP,
    "categorical_preprocess": CAT_PREPROCESS_MAP,
    "target_preprocess_reg": TARGET_PREPROCESS_REG_MAP,
    "target_preprocess_clf": TARGET_PREPROCESS_CLF_MAP,
    "num_embedding": NUM_EMBEDDING_MAP,
    "cat_embedding": CAT_EMBEDDING_MAP,
    "model": MODEL_MAP,
    "train": TRAIN_MAP,
    "inference_reg": INFERENCE_REG_MAP,
    "inference_clf": INFERENCE_CLF_MAP,
    "loss_reg": LOSS_REG_MAP,
    "loss_clf": LOSS_CLF_MAP,
    "optimizer": OPTIMIZER_MAP,
}

INDEX_KEYS: tuple[str, ...] = (
    "numerical_preprocess_idx",
    "categorical_preprocess_idx",
    "target_preprocess_idx",
    "num_embedding_idx",
    "cat_embedding_idx",
    "model_idx",
    "train_idx",
    "optimizer_idx",
    "inference_idx",
    "loss_idx",
)

_TASK_DISPATCHED = {"inference", "loss", "target_preprocess"}


def _registry_name(arg_name: str, task_type: str) -> str:
    name = arg_name.removesuffix("_idx")
    if name in _TASK_DISPATCHED:
        suffix = "reg" if task_type == "regression" else "clf"
        return f"{name}_{suffix}"
    return name


def registry_for(arg_name: str, task_type: str) -> dict[int, object]:
    return IMPLEMENTATION_REGISTRIES[_registry_name(arg_name, task_type)]


def resolve(arg_name: str, idx: int, task_type: str) -> Any:
    """Return the builder registered for ``arg_name`` at ``idx`` for ``task_type``."""
    name = _registry_name(arg_name, task_type)
    registry = IMPLEMENTATION_REGISTRIES[name]
    if idx not in registry:
        choices = ", ".join(str(key) for key in sorted(registry))
        raise ValueError(f"Unknown {name} index {idx}. Available: [{choices}]")
    return registry[idx]


# ---------------------------------------------------------------------------
# Per-dataset fixed defaults (batch size + PLE bin range)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntRange:
    low: int
    high: int
    step: int | None = None


@dataclass(frozen=True)
class DefaultHPs:
    batch_size: int
    n_bins: IntRange


STANDARD_N_BINS = IntRange(16, 128, step=4)
TABRED_N_BINS = IntRange(8, 32, step=1)


# Batch size is fixed per dataset (chosen by dataset scale); everything else is
# tuned. Add a dataset's batch size here to run it.
DATASET_TO_BATCH_SIZE: dict[str, int] = {
    "churn": 256,
    "california": 256,
    "house": 256,
    "adult": 256,
    "diamond": 512,
    "otto": 512,
    "higgs-small": 512,
    "black-friday": 512,
    "covtype2": 1024,
    "microsoft": 1024,
    # TabReD
    "tabred/sberbank-housing": 256,
    "tabred/ecom-offers": 1024,
    "tabred/maps-routing": 1024,
    "tabred/homesite-insurance": 1024,
    "tabred/cooking-time": 1024,
    "tabred/homecredit-default": 1024,
    "tabred/delivery-eta": 1024,
    "tabred/weather": 1024,
    # TabArena
    "tabarena/airfoil_self_noise": 16,
    "tabarena/Amazon_employee_access": 256,
    "tabarena/anneal": 16,
    "tabarena/Another-Dataset-on-used-Fiat-500": 16,
    "tabarena/APSFailure": 512,
    "tabarena/bank-marketing": 256,
    "tabarena/Bank_Customer_Churn": 64,
    "tabarena/Bioresponse": 32,
    "tabarena/blood-transfusion-service-center": 16,
    "tabarena/churn": 32,
    "tabarena/coil2000_insurance_policies": 64,
    "tabarena/concrete_compressive_strength": 16,
    "tabarena/credit-g": 16,
    "tabarena/credit_card_clients_default": 256,
    "tabarena/customer_satisfaction_in_airline": 512,
    "tabarena/diabetes": 16,
    "tabarena/Diabetes130US": 512,
    "tabarena/diamonds": 256,
    "tabarena/E-CommereShippingData": 128,
    "tabarena/Fitness_Club": 16,
    "tabarena/Food_Delivery_Time": 256,
    "tabarena/GiveMeSomeCredit": 512,
    "tabarena/hazelnut-spread-contaminant-detection": 16,
    "tabarena/healthcare_insurance_expenses": 16,
    "tabarena/heloc": 64,
    "tabarena/hiva_agnostic": 32,
    "tabarena/houses": 128,
    "tabarena/HR_Analytics_Job_Change_of_Data_Scientists": 128,
    "tabarena/in_vehicle_coupon_recommendation": 128,
    "tabarena/Is-this-a-good-customer": 16,
    "tabarena/kddcup09_appetency": 256,
    "tabarena/Marketing_Campaign": 16,
    "tabarena/maternal_health_risk": 16,
    "tabarena/miami_housing": 128,
    "tabarena/NATICUSdroid": 64,
    "tabarena/online_shoppers_intention": 128,
    "tabarena/physiochemical_protein": 256,
    "tabarena/polish_companies_bankruptcy": 64,
    "tabarena/qsar-biodeg": 16,
    "tabarena/QSAR-TID-11": 64,
    "tabarena/QSAR_fish_toxicity": 16,
    "tabarena/SDSS17": 512,
    "tabarena/seismic-bumps": 16,
    "tabarena/splice": 32,
    "tabarena/students_dropout_and_academic_success": 32,
    "tabarena/superconductivity": 128,
    "tabarena/taiwanese_bankruptcy_prediction": 64,
    "tabarena/website_phishing": 16,
    "tabarena/wine_quality": 64,
    "tabarena/MIC": 16,
    "tabarena/jm1": 128,
}


def _n_bins_range(dataset_name: str) -> IntRange:
    return TABRED_N_BINS if dataset_name.startswith("tabred/") else STANDARD_N_BINS


def get_default_hps(dataset_name: str) -> DefaultHPs:
    if dataset_name not in DATASET_TO_BATCH_SIZE:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Add its batch_size to "
            "DATASET_TO_BATCH_SIZE in bin/tabm/hparams.py."
        )
    return DefaultHPs(
        batch_size=DATASET_TO_BATCH_SIZE[dataset_name],
        n_bins=_n_bins_range(dataset_name),
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_hps(
    trial: optuna.Trial,
    *,
    default_hps: DefaultHPs,
    task_type: str,
    max_epochs: int = 256,
    patience: int = 16,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Sample one trial's ``(trial_params, implementation_indices)``.

    ``k`` is the TabM ensemble size (fixed at 32). The continuous/integer ranges
    plus the per-stage categorical index choices are the full TabM search space.
    """
    n_bins_kw: dict[str, int] = {}
    if default_hps.n_bins.step is not None:
        n_bins_kw["step"] = default_hps.n_bins.step
    n_bins = trial.suggest_int(
        "n_bins", default_hps.n_bins.low, default_hps.n_bins.high, **n_bins_kw
    )
    use_dropout = trial.suggest_categorical("use_dropout", [True, False])
    dropout = trial.suggest_float("dropout", 0.0, 0.5) if use_dropout else 0.0
    use_wd = trial.suggest_categorical("use_weight_decay", [True, False])
    weight_decay = (
        trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True) if use_wd else 0.0
    )
    trial_params: dict[str, Any] = {
        "batch_size": default_hps.batch_size,
        "gradient_clip": 1.0,
        "max_epochs": max_epochs,
        "patience": patience,
        "eval_batch_size": 32768,
        "n_bins": n_bins,
        "lr": trial.suggest_float("lr", 3e-5, 1e-3, log=True),
        "weight_decay": weight_decay,
        "n_blocks": trial.suggest_int("n_blocks", 1, 6),
        "d_block": trial.suggest_int("d_block", 64, 1024, step=16),
        "dropout": dropout,
        "d_embedding": trial.suggest_int("d_embedding", 8, 128, step=4),
        "k": 32,
    }
    implementation_indices: dict[str, int] = {}
    for key in INDEX_KEYS:
        registry = IMPLEMENTATION_REGISTRIES[_registry_name(key, task_type)]
        implementation_indices[key] = trial.suggest_categorical(key, sorted(registry))
    return trial_params, implementation_indices


# ---------------------------------------------------------------------------
# Optimizer + budget
# ---------------------------------------------------------------------------

# Optuna study: minimize the validation metric with a TPE sampler. Each trial is
# evaluated with a per-trial seed of ``base_seed + trial.number``.
N_TRIALS_DEFAULT = 200
N_TRIALS_LARGE = 100  # tabred/* and microsoft (large datasets)


def is_large_dataset(dataset_name: str) -> bool:
    return dataset_name.startswith("tabred/") or dataset_name == "microsoft"


def n_trials_for(dataset_name: str) -> int:
    return N_TRIALS_LARGE if is_large_dataset(dataset_name) else N_TRIALS_DEFAULT


def make_sampler(seed: int, *, n_startup_trials: int = 1, multivariate: bool = False):
    """The TPE sampler used for the search (``direction="minimize"``)."""
    return optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=n_startup_trials,
        multivariate=multivariate,
        group=multivariate,
    )
