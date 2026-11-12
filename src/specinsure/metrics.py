"""Risk metrics for simulated annual losses."""

from __future__ import annotations

import numpy as np


def aal(annual_losses: np.ndarray) -> float:
    """Average annual loss."""
    return float(np.mean(annual_losses))


def var(annual_losses: np.ndarray, level: float) -> float:
    """Value at Risk: the ``level`` quantile of annual losses (e.g. 0.995)."""
    _check_level(level)
    return float(np.quantile(annual_losses, level))


def tvar(annual_losses: np.ndarray, level: float) -> float:
    """Tail Value at Risk: mean of annual losses at or above the ``level`` quantile."""
    threshold = var(annual_losses, level)
    values = np.asarray(annual_losses)
    return float(values[values >= threshold].mean())


def return_period_loss(annual_losses: np.ndarray, return_period: float) -> float:
    """Loss exceeded on average once every ``return_period`` years."""
    if return_period <= 1:
        raise ValueError("return_period must be greater than 1")
    return var(annual_losses, 1 - 1 / return_period)


def exceedance_curve(annual_losses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (loss levels, exceedance probabilities), losses sorted descending.

    Use annual totals for an AEP curve and annual maxima for an OEP curve.
    """
    values = np.sort(np.asarray(annual_losses, dtype=float))[::-1]
    probs = np.arange(1, values.size + 1) / values.size
    return values, probs


def _check_level(level: float) -> None:
    if not 0 < level < 1:
        raise ValueError(f"level must be between 0 and 1, got {level}")
