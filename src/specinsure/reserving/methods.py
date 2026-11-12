"""Deterministic reserving methods: chain ladder, Bornhuetter-Ferguson, Cape Cod."""

from __future__ import annotations

import numpy as np

from ..core.base import ValidationError
from ..core.types import ArrayLike, FloatArray
from ..core.validation import as_float_array
from .development import DevelopmentPattern, development_factors
from .results import ReserveResult
from .triangle import Triangle


def _pattern(triangle: Triangle, pattern: DevelopmentPattern | None) -> DevelopmentPattern:
    pattern = pattern or development_factors(triangle)
    if pattern.factors.size != triangle.shape[1] - 1:
        raise ValidationError(
            f"pattern has {pattern.factors.size} factors, "
            f"the triangle needs {triangle.shape[1] - 1}"
        )
    return pattern


def _premium(triangle: Triangle, premium: ArrayLike) -> FloatArray:
    p = as_float_array("premium", premium, non_negative=True)
    if p.shape != (triangle.shape[0],):
        raise ValidationError("premium must have one value per origin")
    return p


def cdf_for_latest(triangle: Triangle, pattern: DevelopmentPattern) -> FloatArray:
    """Cumulative development factor to ultimate from each origin's latest age."""
    return pattern.cdf_to_ultimate[triangle.latest_index]


def chain_ladder(triangle: Triangle, pattern: DevelopmentPattern | None = None) -> ReserveResult:
    """Chain ladder: ultimate = latest x CDF to ultimate.

    Uses volume-weighted factors with no tail unless ``pattern`` is given.
    """
    pattern = _pattern(triangle, pattern)
    cdf = cdf_for_latest(triangle, pattern)
    return ReserveResult(
        "chain_ladder",
        triangle.origins,
        triangle.latest,
        triangle.latest * cdf,
        pattern,
        {"cdf": cdf},
    )


def expected_loss_ratio(
    triangle: Triangle, premium: ArrayLike, loss_ratio: float | ArrayLike
) -> ReserveResult:
    """Ultimate = premium x a priori loss ratio, ignoring the claims to date.

    Often used for the most recent, immature origin years.
    """
    p = _premium(triangle, premium)
    elr = np.broadcast_to(np.asarray(loss_ratio, dtype=float), p.shape)
    pattern = development_factors(triangle)
    return ReserveResult(
        "expected_loss_ratio", triangle.origins, triangle.latest, p * elr, pattern, {"elr": elr}
    )


def bornhuetter_ferguson(
    triangle: Triangle,
    premium: ArrayLike,
    loss_ratio: float | ArrayLike,
    pattern: DevelopmentPattern | None = None,
) -> ReserveResult:
    """Bornhuetter-Ferguson (1972): reserve = premium x ELR x (1 - 1 / CDF).

    The a priori ``loss_ratio`` (one value, or one per origin) usually comes
    from the pricing basis. Credibility moves from the a priori to the chain
    ladder as an origin develops.
    """
    pattern = _pattern(triangle, pattern)
    p = _premium(triangle, premium)
    elr = np.broadcast_to(np.asarray(loss_ratio, dtype=float), p.shape)
    if (elr < 0).any():
        raise ValidationError("loss_ratio must be non-negative")
    cdf = cdf_for_latest(triangle, pattern)
    unreported = 1 - 1 / cdf
    ultimate = triangle.latest + p * elr * unreported
    return ReserveResult(
        "bornhuetter_ferguson",
        triangle.origins,
        triangle.latest,
        ultimate,
        pattern,
        {"cdf": cdf, "elr": elr},
    )


def cape_cod(
    triangle: Triangle, premium: ArrayLike, pattern: DevelopmentPattern | None = None
) -> ReserveResult:
    """Cape Cod (Stanard-Bühlmann): BF with the loss ratio estimated from the data.

    ELR = sum(latest) / sum(premium / CDF), the claims to date over the
    "used-up" premium, applied to every origin.
    """
    pattern = _pattern(triangle, pattern)
    p = _premium(triangle, premium)
    cdf = cdf_for_latest(triangle, pattern)
    used_up = (p / cdf).sum()
    if used_up <= 0:
        raise ValidationError("used-up premium is zero")
    elr = float(triangle.latest.sum() / used_up)
    result = bornhuetter_ferguson(triangle, p, elr, pattern)
    return ReserveResult(
        "cape_cod",
        result.origins,
        result.latest,
        result.ultimate,
        pattern,
        {**result.details, "elr": elr},
    )
