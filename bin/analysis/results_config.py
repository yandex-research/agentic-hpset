from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = ROOT / "exp"
TUNED_ROOT = EXP_ROOT / "tuned"
ENSEMBLED_ROOT = EXP_ROOT / "ensembled"
ARTIFACTS_ROOT = ROOT / "artifacts"
DATA_ROOT = ROOT / "data"

MAX_BUDGET = 200
TABRED_BUDGET = 100
SPLIT = "test"
BOOTSTRAP_SEED = 20260615
ELO_ANCHOR_METHOD = "mlp"
ELO_ANCHOR_VALUE = 1000.0

PLAIN_DATASETS = (
    "black-friday", "california", "churn", "diamond", "house", "microsoft",
)
TABARENA_DATASETS = (
    "APSFailure", "Amazon_employee_access", "Another-Dataset-on-used-Fiat-500",
    "Bioresponse", "Diabetes130US", "E-CommereShippingData", "Food_Delivery_Time",
    "GiveMeSomeCredit", "HR_Analytics_Job_Change_of_Data_Scientists", "NATICUSdroid",
    "QSAR-TID-11", "QSAR_fish_toxicity", "airfoil_self_noise", "bank-marketing",
    "concrete_compressive_strength", "credit_card_clients_default",
    "customer_satisfaction_in_airline", "diabetes", "healthcare_insurance_expenses",
    "heloc", "in_vehicle_coupon_recommendation", "jm1", "kddcup09_appetency",
    "miami_housing", "online_shoppers_intention", "physiochemical_protein",
    "polish_companies_bankruptcy", "qsar-biodeg", "splice",
    "taiwanese_bankruptcy_prediction", "wine_quality",
)
TABRED_DATASETS = (
    "cooking-time", "delivery-eta", "ecom-offers", "homecredit-default",
    "homesite-insurance", "maps-routing", "sberbank-housing", "weather",
)
DATASETS = (
    PLAIN_DATASETS
    + tuple(f"tabarena/{name}" for name in TABARENA_DATASETS)
    + tuple(f"tabred/{name}" for name in TABRED_DATASETS)
)


@dataclass(frozen=True)
class MethodSpec:
    group: str
    method: str


# Provenance of the committed single-model results in exp/tuned/<method>. The
# published repo ships the corresponding model code under bin/<group>/ (see each
# family's README); the tuning/eval drivers that produced these reports are not
# included. Every agentic result uses the family's `unified-v1` search space —
# the larger exploratory search spaces did not contribute any committed result.
METHOD_SOURCE = {
    "realmlp": "RealMLP-TD baseline (bin/realmlp/base.py)",
    "agentic-realmlp": "agentic RealMLP (bin/realmlp/)",
    "mlp": "MLP baseline (bin/mlp/, indices = 0)",
    "agentic-mlp": "agentic MLP unified-v1 (bin/mlp/)",
    "lightgbm": "LightGBM baseline (bin/lightgbm/, indices = 0)",
    "agentic-lightgbm": "agentic LightGBM (bin/lightgbm/)",
    "tabm": "TabM baseline (bin/tabm/, indices = 0)",
    "agentic-tabm": "agentic TabM unified-v1 (bin/tabm/)",
    "tabicl": "NanoTabICL default recipe (bin/tabicl/)",
    "agentic-tabicl": "agentic NanoTabICL (bin/tabicl/)",
}

SINGLE_METHOD_SPECS = (
    MethodSpec(group="realmlp", method="realmlp"),
    MethodSpec(group="realmlp", method="agentic-realmlp"),
    MethodSpec(group="mlp", method="mlp"),
    MethodSpec(group="mlp", method="agentic-mlp"),
    MethodSpec(group="lightgbm", method="lightgbm"),
    MethodSpec(group="lightgbm", method="agentic-lightgbm"),
    MethodSpec(group="tabm", method="tabm"),
    MethodSpec(group="tabm", method="agentic-tabm"),
    MethodSpec(group="tabicl", method="tabicl"),
    MethodSpec(group="tabicl", method="agentic-tabicl"),
)

SINGLE_METHODS = tuple(spec.method for spec in SINGLE_METHOD_SPECS)
# Agent-ablation methods: the MLP agentic protocol re-run end-to-end by two coding
# agents (claude, codex), five independent runs each. Module packages live at
# bin/mlp/<variant>/; results at exp/tuned/mlp-ablation-<variant>/. Kept out of
# SINGLE_METHOD_SPECS so the main figures and tables are unaffected; consumed only
# by bin.analysis.gen_agent_ablation.
AGENT_ABLATION_VARIANTS = (
    "claude-1", "claude-2", "claude-5", "claude-6", "claude-7",
    "codex-0", "codex-1", "codex-2", "codex-3", "codex-4",
)
AGENT_ABLATION_METHODS = tuple(f"mlp-ablation-{v}" for v in AGENT_ABLATION_VARIANTS)
# Extra comparison methods are excluded from rank-based grouped tables, which
# require full per-dataset method coverage.
EXTRA_SINGLE_METHODS = ("agentic-realmlp",)
CORE_SINGLE_METHODS = tuple(m for m in SINGLE_METHODS if m not in EXTRA_SINGLE_METHODS)
# Methods drawn in the budget-curve figures: the core comparison plus the agentic
# RealMLP version with milestone evaluations. Both panels use this set; each budget
# keeps the datasets where every method has a report (large panel: 6 of 9 at budget
# 50, all 9 at 100 where agentic RealMLP's full-budget evaluations fill in).
BUDGET_CURVE_METHODS = CORE_SINGLE_METHODS + ("agentic-realmlp",)
TABICL_METHODS = ("tabicl", "agentic-tabicl")

ENSEMBLE_METHODS = (
    "BaseMLP-PLE",
    "BaseTabM",
    "ModularMLP-v1",
    "ModularTabM-v1",
    "RealMLP-tabarena",
    "BaseLightGBM",
    "ModularLightGBM",
    "GesRealMLP",
)
ENSEMBLE_TO_MAIN_METHOD = {
    "BaseMLP-PLE": "mlp",
    "ModularMLP-v1": "agentic-mlp",
    "BaseTabM": "tabm",
    "ModularTabM-v1": "agentic-tabm",
    "RealMLP-tabarena": "realmlp",
    "BaseLightGBM": "lightgbm",
    "ModularLightGBM": "agentic-lightgbm",
    "GesRealMLP": "agentic-realmlp",
}

METHOD_DISPLAY = {
    "realmlp": "RealMLP",
    "agentic-realmlp": "RealMLP (agentic)",
    "mlp": "MLP",
    "agentic-mlp": "MLP (agentic)",
    "lightgbm": "LightGBM",
    "agentic-lightgbm": "LightGBM (agentic)",
    "tabm": "TabM",
    "agentic-tabm": "TabM (agentic)",
    "tabicl": "TabICL",
    "agentic-tabicl": "TabICL (agentic)",
}
FAMILY_DISPLAY = {
    "realmlp": "RealMLP",
    "mlp": "MLP",
    "lightgbm": "LightGBM",
    "tabm": "TabM",
    "tabicl": "TabICL",
}
FAMILY_PAIRS = (
    ("realmlp", "agentic-realmlp", "realmlp"),
    ("mlp", "agentic-mlp", "mlp"),
    ("lightgbm", "agentic-lightgbm", "lightgbm"),
    ("tabm", "agentic-tabm", "tabm"),
    ("tabicl", "agentic-tabicl", "tabicl"),
)


def is_tabred_dataset(dataset: str) -> bool:
    return dataset.startswith("tabred/") or dataset == "microsoft"


def budget_for_dataset(dataset: str, budget: int = MAX_BUDGET) -> int:
    return TABRED_BUDGET if is_tabred_dataset(dataset) else budget


def budgets_for_dataset(dataset: str) -> tuple[int, ...]:
    if is_tabred_dataset(dataset):
        return (50, TABRED_BUDGET)
    return tuple(range(50, MAX_BUDGET + 1, 50))


def report_name_for_budget(budget: int | str) -> str:
    return "report_grid.json" if budget == "grid" else f"report_{int(budget)}.json"
