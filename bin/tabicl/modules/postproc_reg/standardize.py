from __future__ import annotations

from ._base import RegressionPostproc

regression_postproc_standardize = RegressionPostproc(log=False)

__all__ = ["regression_postproc_standardize"]
