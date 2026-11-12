"""Quota share treaty."""

from __future__ import annotations

import numpy as np

from .base import ReinsuranceTreaty


class QuotaShareTreaty(ReinsuranceTreaty):
    """Cedes a fixed proportion ``quota`` of every loss (0 < quota <= 1)."""

    def __init__(self, quota: float) -> None:
        if not 0 < quota <= 1:
            raise ValueError(f"quota must be in (0, 1], got {quota}")
        self.quota = float(quota)

    def apply(self, gross_losses: np.ndarray, year_index: np.ndarray | None = None) -> np.ndarray:
        return np.asarray(gross_losses, dtype=float) * self.quota

    def __repr__(self) -> str:
        return f"QuotaShareTreaty({self.quota:.0%})"
