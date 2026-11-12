"""Return periods and the losses associated with them.

A 1-in-N year loss is the loss exceeded with annual probability 1/N, so the
1-in-200 loss is VaR at 99.5%.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..core.base import ValidationError
from ..core.types import ArrayLike
from ..core.validation import as_float_array
from .measures import var

STANDARD_RETURN_PERIODS = (2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000)


def level_for_return_period(return_period: float) -> float:
    """Non-exceedance probability for a return period: 200 -> 0.995."""
    if return_period <= 1:
        raise ValidationError(f"return_period must be greater than 1, got {return_period}")
    return 1 - 1 / return_period


def return_period_for_level(level: float) -> float:
    """Return period for a non-exceedance probability: 0.995 -> 200."""
    if not 0 < level < 1:
        raise ValidationError(f"level must be between 0 and 1, got {level}")
    return 1 / (1 - level)


def return_period_loss(annual_losses: ArrayLike, return_period: float) -> float:
    """Loss exceeded on average once every ``return_period`` years.

    Pass annual totals for an AEP figure, or annual maxima for an OEP figure.
    """
    return var(annual_losses, level_for_return_period(return_period))


def empirical_return_period(annual_losses: ArrayLike, loss: float) -> float:
    """Empirical return period of ``loss``: 1 / P(annual loss >= loss); ``inf`` if never reached."""
    x = as_float_array("annual_losses", annual_losses)
    p = np.mean(x >= loss)
    return float("inf") if p == 0 else float(1 / p)


def return_period_table(
    annual_losses: ArrayLike, return_periods: Iterable[float] = STANDARD_RETURN_PERIODS
) -> dict[float, float]:
    """Return period losses for each return period, e.g. ``{200: 1.2e7, ...}``."""
    return {rp: return_period_loss(annual_losses, rp) for rp in return_periods}
