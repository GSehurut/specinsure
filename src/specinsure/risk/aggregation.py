"""Combining risks across lines: diversification and capital allocation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..core.base import ValidationError
from ..core.types import ArrayLike, FloatArray
from ..core.validation import check_level
from .measures import tail_count


def _matrix(losses: ArrayLike) -> FloatArray:
    m = np.asarray(losses, dtype=float)
    if m.ndim != 2 or m.shape[0] == 0 or m.shape[1] == 0:
        raise ValidationError(f"losses must be a non-empty (n_sims, n_lines) matrix, got {m.shape}")
    if not np.all(np.isfinite(m)):
        raise ValidationError("losses contain NaN or infinite values")
    return m


def diversification_benefit(losses: ArrayLike, measure: Callable[[FloatArray], float]) -> float:
    """Sum of stand-alone risk minus the risk of the total.

    ``losses`` has shape (n_sims, n_lines).

    ``measure`` is e.g. ``lambda x: tvar(x, 0.99)``. Can be negative for VaR,
    which is not subadditive.
    """
    m = _matrix(losses)
    standalone = sum(measure(m[:, j]) for j in range(m.shape[1]))
    return float(standalone - measure(m.sum(axis=1)))


def euler_tvar_allocation(losses: ArrayLike, level: float) -> FloatArray:
    """Allocate portfolio TVaR to lines by the Euler (co-TVaR) principle.

    Allocation_j = E[L_j | portfolio loss in its worst (1 - level) scenarios].
    The allocations add up exactly to the portfolio TVaR, so return on
    allocated capital is consistent across lines (Tasche, 1999).
    """
    check_level(level)
    m = _matrix(losses)
    total = m.sum(axis=1)
    k = tail_count(total.size, level)
    tail = np.argsort(total, kind="stable")[-k:]
    return m[tail].mean(axis=0)
