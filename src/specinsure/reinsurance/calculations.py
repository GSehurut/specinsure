"""Market metrics and premium calculations for reinsurance layers."""

from __future__ import annotations

import numpy as np

from ..core.types import ArrayLike, FloatArray
from ..core.validation import as_float_array, check_non_negative, check_positive
from .treaties import ExcessOfLoss


def rate_on_line(premium: float, limit: float) -> float:
    """ROL = premium / limit."""
    return check_non_negative("premium", premium) / check_positive("limit", limit)


def loss_on_line(expected_loss: float, limit: float) -> float:
    """LOL = expected loss / limit."""
    return check_non_negative("expected_loss", expected_loss) / check_positive("limit", limit)


def payback_years(premium: float, limit: float) -> float:
    """Years of premium needed to pay one full limit loss: 1 / ROL."""
    return 1 / rate_on_line(check_positive("premium", premium), limit)


def reinstatement_premium_factors(treaty: ExcessOfLoss, annual_ceded: ArrayLike) -> FloatArray:
    """Reinstatement premium per year as a multiple of the deposit premium.

    Pro rata as to amount, no time adjustment: reinstatement k is used by the
    recoveries between ``(k - 1) * limit`` and ``k * limit`` in the year, and
    costs ``rate_k * used / limit`` of the deposit premium. Only the first
    ``reinstatements`` limits are reinstated; the final limit is not.
    """
    recovered = as_float_array("annual_ceded", annual_ceded, non_negative=True)
    n = treaty.reinstatements
    if n is None or n == 0:
        return np.zeros_like(recovered)
    rates = np.broadcast_to(treaty.reinstatement_rates, (n,))
    k = np.arange(n)
    used = np.clip(recovered[:, None] - k * treaty.limit, 0.0, treaty.limit) / treaty.limit
    return used @ rates


def reinstatement_premiums(
    treaty: ExcessOfLoss, annual_ceded: ArrayLike, deposit_premium: float
) -> FloatArray:
    """Reinstatement premium payable each year for a given deposit (upfront) premium."""
    check_non_negative("deposit_premium", deposit_premium)
    return deposit_premium * reinstatement_premium_factors(treaty, annual_ceded)
