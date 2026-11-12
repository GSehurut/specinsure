"""Reserving results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.types import FloatArray, IntArray
from ..risk.measures import tvar, var
from .development import DevelopmentPattern


@dataclass(frozen=True)
class ReserveResult:
    """Ultimate claims and reserves by origin from a deterministic method.

    ``reserve`` = ultimate - latest. On an incurred triangle that is IBNR
    (including IBNER); on a paid triangle it is the total outstanding reserve.
    """

    method: str
    origins: IntArray
    latest: FloatArray
    ultimate: FloatArray
    pattern: DevelopmentPattern
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def reserve(self) -> FloatArray:
        return self.ultimate - self.latest

    @property
    def total_reserve(self) -> float:
        return float(self.reserve.sum())

    @property
    def total_ultimate(self) -> float:
        return float(self.ultimate.sum())

    def loss_ratios(self, premium: FloatArray) -> FloatArray:
        """Ultimate loss ratios for the given premium by origin."""
        return self.ultimate / np.asarray(premium, dtype=float)

    def summary(self) -> list[dict[str, float]]:
        """One row per origin plus a total row, for :func:`specinsure.reporting.format_table`."""
        rows: list[dict[str, Any]] = [
            {"origin": int(o), "latest": lat, "ultimate": ult, "reserve": ult - lat}
            for o, lat, ult in zip(self.origins, self.latest, self.ultimate, strict=False)
        ]
        rows.append(
            {
                "origin": "total",
                "latest": float(self.latest.sum()),
                "ultimate": self.total_ultimate,
                "reserve": self.total_reserve,
            }
        )
        return rows


@dataclass(frozen=True)
class MackResult(ReserveResult):
    """Chain ladder result with Mack (1993) standard errors of the reserve.

    ``standard_error`` holds the root mean squared error of prediction per
    origin (process plus parameter error); ``total_standard_error`` allows for
    the correlation between origins induced by the shared factor estimates.
    """

    standard_error: FloatArray = field(default_factory=lambda: np.empty(0))
    process_error: FloatArray = field(default_factory=lambda: np.empty(0))
    parameter_error: FloatArray = field(default_factory=lambda: np.empty(0))
    total_standard_error: float = 0.0
    sigma2: FloatArray = field(default_factory=lambda: np.empty(0))

    @property
    def cv(self) -> FloatArray:
        """Coefficient of variation of each origin's reserve (NaN where the reserve is zero)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.reserve != 0, self.standard_error / self.reserve, np.nan)

    @property
    def total_cv(self) -> float:
        return self.total_standard_error / self.total_reserve

    def summary(self) -> list[dict[str, float]]:
        rows = super().summary()
        for row, se in zip(rows, [*self.standard_error, self.total_standard_error], strict=False):
            row["standard_error"] = float(se)
            row["cv"] = float(se / row["reserve"]) if row["reserve"] else float("nan")
        return rows


@dataclass(frozen=True)
class BootstrapResult:
    """Simulated reserves by origin from the ODP bootstrap, shape ``(n_sims, n_origins)``.

    Includes both parameter and process error, giving a full predictive
    distribution for reserve risk.
    """

    origins: IntArray
    reserves: FloatArray
    scale: float
    deterministic: ReserveResult

    @property
    def total(self) -> FloatArray:
        """Simulated total reserve across origins."""
        return self.reserves.sum(axis=1)

    @property
    def mean(self) -> FloatArray:
        return self.reserves.mean(axis=0)

    @property
    def standard_error(self) -> FloatArray:
        return self.reserves.std(axis=0, ddof=1)

    def percentile(self, level: float) -> float:
        """Total reserve at ``level`` (e.g. 0.75 for the 75th percentile)."""
        return var(self.total, level)

    def total_tvar(self, level: float) -> float:
        return tvar(self.total, level)
