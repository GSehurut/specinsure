"""Risk measures on simulated outcomes.

Convention: inputs are **losses, positive = bad** (annual gross, ceded or net
losses). To measure an underwriting result where profit is positive, pass
``-result``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from ..core.base import ValidationError
from ..core.types import ArrayLike, FloatArray
from ..core.validation import as_float_array, check_level


def _values(losses: ArrayLike) -> FloatArray:
    x = as_float_array("losses", losses)
    if x.size == 0:
        raise ValidationError("losses must not be empty")
    return x


def aal(losses: ArrayLike) -> float:
    """Average annual loss (the mean of annual losses)."""
    return float(_values(losses).mean())


def standard_deviation(losses: ArrayLike) -> float:
    """Sample standard deviation (``ddof=1``)."""
    x = _values(losses)
    return float(x.std(ddof=1)) if x.size > 1 else 0.0


def standard_error(losses: ArrayLike) -> float:
    """Monte Carlo standard error of the mean: s / sqrt(n)."""
    x = _values(losses)
    return standard_deviation(x) / math.sqrt(x.size)


def aal_confidence_interval(losses: ArrayLike, confidence: float = 0.95) -> tuple[float, float]:
    """Normal-approximation confidence interval for the AAL, reflecting simulation error only."""
    check_level(confidence)
    z = float(stats.norm.ppf(0.5 + confidence / 2))
    mean, se = aal(losses), standard_error(losses)
    return mean - z * se, mean + z * se


def var(losses: ArrayLike, level: float) -> float:
    """Value at Risk: the ``level`` quantile of losses (e.g. 0.995 for 1-in-200).

    Uses numpy's default linear interpolation between order statistics.
    """
    check_level(level)
    return float(np.quantile(_values(losses), level))


def tvar(losses: ArrayLike, level: float) -> float:
    """Tail Value at Risk (expected shortfall): mean of the worst ``(1 - level)`` share of outcomes.

    Computed as the mean of the ``k = ceil(n (1 - level))`` largest values, so
    ties at the threshold are handled consistently and the measure is coherent
    (Acerbi and Tasche, 2002). Matches the Rockafellar-Uryasev LP used in
    :mod:`specinsure.optimisation` when ``n (1 - level)`` is a whole number.
    """
    check_level(level)
    x = _values(losses)
    k = tail_count(x.size, level)
    return float(np.sort(x)[-k:].mean())


def tail_count(n: int, level: float) -> int:
    """Number of scenarios in the ``(1 - level)`` tail of ``n`` simulations (at least 1)."""
    return max(1, math.ceil(n * (1 - level) - 1e-9))


def capital_requirement(
    losses: ArrayLike, level: float = 0.995, *, relative_to_mean: bool = False
) -> float:
    """Capital as VaR of losses at ``level``; the Solvency II SCR basis at the default 99.5%.

    Pass the one-year loss of own funds (``-net_result``). With
    ``relative_to_mean=True`` the expected loss is deducted, giving the
    "unexpected loss" capital used in some internal models.
    """
    capital = var(losses, level)
    if relative_to_mean:
        capital -= aal(losses)
    return capital
