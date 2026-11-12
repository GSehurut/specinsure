"""As-if adjustments to experience: trend, on-levelling, development and credibility."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping

import numpy as np

from ..core.base import ValidationError
from ..core.types import ArrayLike, FloatArray
from ..core.validation import as_float_array, check_non_negative


def trend_factor(from_year: ArrayLike, to_year: int, annual_rate: float) -> FloatArray:
    """Compound trend factor (1 + rate) ** (to_year - from_year)."""
    if annual_rate <= -1:
        raise ValidationError(f"annual_rate must be greater than -100%, got {annual_rate}")
    return (1 + annual_rate) ** (to_year - np.asarray(from_year, dtype=float))


def trend_losses(
    losses: ArrayLike, loss_years: ArrayLike, target_year: int, annual_rate: float
) -> FloatArray:
    """Bring historical losses to ``target_year`` cost levels with compound claims inflation.

    Apply this to individual losses *before* applying an XoL layer: inflation
    has a leveraged effect on excess layers because losses below the
    attachment are pushed into the layer.
    """
    x = as_float_array("losses", losses, non_negative=True)
    return x * trend_factor(loss_years, target_year, annual_rate)


def on_level_factors(
    years: ArrayLike, target_year: int, rate_changes: Mapping[int, float]
) -> FloatArray:
    """Factors bringing each year's premium to ``target_year`` rate level.

    ``rate_changes[t]`` is the rate change applied at the start of year ``t``
    (0.05 = +5%). The factor for year ``y`` is the product of ``1 + r_t`` for
    ``y < t <= target_year``. Missing years mean no change.
    """
    out = []
    for y in np.asarray(years).tolist():
        factor = 1.0
        for t in range(int(y) + 1, target_year + 1):
            factor *= 1 + rate_changes.get(t, 0.0)
        out.append(factor)
    return np.asarray(out, dtype=float)


def develop_losses(
    losses: ArrayLike, loss_years: ArrayLike, development_factors: Mapping[int, float]
) -> FloatArray:
    """Apply incurred-to-ultimate factors (IBNER) by year; years not in the mapping use 1.0.

    Applying a uniform factor to each individual loss is the usual simplification
    for burning cost; it does not create the pure IBNR claims that are still to
    be reported, which should be allowed for separately.
    """
    x = as_float_array("losses", losses, non_negative=True)
    factors = np.array([development_factors.get(int(y), 1.0) for y in np.asarray(loss_years)])
    if (factors < 1).any():
        # Negative development happens, but is unusual enough to be worth surfacing.
        warnings.warn("development factor below 1 (negative development)", stacklevel=2)
    return x * factors


def credibility_weight(n_claims: float, full_credibility_claims: float = 1082.0) -> float:
    """Limited-fluctuation credibility: Z = min(1, sqrt(n / n_full)).

    The default standard of 1,082 claims gives a 90% probability that the
    observed frequency is within 5% of the true value (Longley-Cook, 1962).
    Specialty XoL layers rarely have that many claims; pass a lower standard
    if your methodology justifies one.
    """
    check_non_negative("n_claims", n_claims)
    if full_credibility_claims <= 0:
        raise ValidationError("full_credibility_claims must be positive")
    return min(1.0, math.sqrt(n_claims / full_credibility_claims))
