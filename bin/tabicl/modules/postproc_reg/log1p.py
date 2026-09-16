from __future__ import annotations

from ._base import RegressionPostproc

regression_postproc_log1p = RegressionPostproc(log=True)

__all__ = ["regression_postproc_log1p"]
