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
class MerzWuthrichResult(ReserveResult):
    """One-year claims development result (CDR) standard errors, Merz & Wüthrich (2008).

    ``standard_error`` is the root mean squared error of prediction of the
    next calendar year's CDR by origin; ``total_standard_error`` is for the
    whole portfolio. ``mack`` holds the ultimate (run-off) view on the same
    factors, so ``one_year_ratio`` shows how much of the ultimate reserve
    risk emerges in the first year.
    """

    standard_error: FloatArray = field(default_factory=lambda: np.empty(0))
    process_error: FloatArray = field(default_factory=lambda: np.empty(0))
    parameter_error: FloatArray = field(default_factory=lambda: np.empty(0))
    total_standard_error: float = 0.0
    total_process_error: float = 0.0
    total_parameter_error: float = 0.0
    mack: MackResult | None = None

    @property
    def one_year_ratio(self) -> float:
        """Total one-year standard error divided by Mack's ultimate standard error."""
        if self.mack is None:
            return float("nan")
        return self.total_standard_error / self.mack.total_standard_error

    def summary(self) -> list[dict[str, float]]:
        rows = super().summary()
        one_year = [*self.standard_error, self.total_standard_error]
        ultimate = (
            [*self.mack.standard_error, self.mack.total_standard_error]
            if self.mack is not None
            else [float("nan")] * len(rows)
        )
        for row, cdr, mack in zip(rows, one_year, ultimate, strict=False):
            row["cdr_standard_error"] = float(cdr)
            row["mack_standard_error"] = float(mack)
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
