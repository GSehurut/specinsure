"""Frequency and severity simulation producing a year loss table."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .treaties import ReinsuranceTreaty


@dataclass(frozen=True)
class YearLossTable:
    """Individual simulated losses and the year each one belongs to."""

    losses: np.ndarray
    year_index: np.ndarray
    n_years: int

    def annual_totals(self, values: np.ndarray | None = None) -> np.ndarray:
        """Sum per simulated year (of ``values`` if given, else gross losses)."""
        v = self.losses if values is None else values
        return np.bincount(self.year_index, weights=v, minlength=self.n_years)

    def annual_max(self, values: np.ndarray | None = None) -> np.ndarray:
        """Largest single loss per simulated year (0 for years with no losses)."""
        v = self.losses if values is None else values
        out = np.zeros(self.n_years)
        np.maximum.at(out, self.year_index, v)
        return out

    def ceded(self, treaty: ReinsuranceTreaty) -> np.ndarray:
        """Ceded amount per loss, applying annual terms year by year."""
        return treaty.apply(self.losses, self.year_index)


def simulate(frequency, severity, n_years: int, seed: int | None = None) -> YearLossTable:
    """Simulate ``n_years`` of losses.

    Parameters
    ----------
    frequency:
        A frozen ``scipy.stats`` discrete distribution for the number of losses
        per year, e.g. ``scipy.stats.poisson(12)`` or ``scipy.stats.nbinom(n, p)``.
    severity:
        A frozen ``scipy.stats`` continuous distribution for individual loss size,
        e.g. ``scipy.stats.lognorm(s=1.2, scale=1e6)``.
    n_years:
        Number of simulated years.
    seed:
        Seed for reproducible results.
    """
    if n_years <= 0:
        raise ValueError(f"n_years must be positive, got {n_years}")
    rng = np.random.default_rng(seed)
    counts = np.asarray(frequency.rvs(size=n_years, random_state=rng), dtype=np.intp)
    if counts.min() < 0:
        raise ValueError("frequency distribution produced negative counts")
    losses = np.asarray(severity.rvs(size=int(counts.sum()), random_state=rng), dtype=float)
    year_index = np.repeat(np.arange(n_years, dtype=np.intp), counts)
    return YearLossTable(losses=losses, year_index=year_index, n_years=n_years)
