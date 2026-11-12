"""Development factor selection and tail factors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..core.base import ValidationError
from ..core.types import FloatArray
from .triangle import Triangle


@dataclass(frozen=True)
class DevelopmentPattern:
    """Age-to-age factors and a tail factor, with derived cumulative patterns.

    ``factors[k]`` develops age ``k`` to age ``k + 1``; ``tail`` develops the
    last age to ultimate.
    """

    factors: FloatArray
    tail: float = 1.0

    def __post_init__(self) -> None:
        f = np.asarray(self.factors, dtype=float)
        if f.ndim != 1 or not np.all(np.isfinite(f)) or (f <= 0).any():
            raise ValidationError("factors must be a 1-d array of positive finite numbers")
        if not np.isfinite(self.tail) or self.tail <= 0:
            raise ValidationError(f"tail must be positive, got {self.tail}")
        object.__setattr__(self, "factors", f)
        object.__setattr__(self, "tail", float(self.tail))

    @property
    def cdf_to_ultimate(self) -> FloatArray:
        """Cumulative development factor from each age to ultimate (length ``n_ages``)."""
        to_last = np.cumprod(np.r_[self.factors, 1.0][::-1])[::-1]
        return to_last * self.tail

    @property
    def percent_developed(self) -> FloatArray:
        """Expected share of ultimate observed at each age: 1 / CDF."""
        return 1 / self.cdf_to_ultimate


def development_factors(
    triangle: Triangle,
    *,
    average: Literal["volume", "simple"] = "volume",
    n_periods: int | None = None,
    tail: float = 1.0,
) -> DevelopmentPattern:
    """Select age-to-age factors from a triangle.

    ``average="volume"`` is the chain ladder estimator sum C[i, k+1] / sum C[i, k]
    over origins observed at both ages (Mack, 1993); ``"simple"`` averages the
    individual link ratios. ``n_periods`` restricts each average to the most
    recent ``n_periods`` origins with a link ratio, a common choice when the
    pattern has changed.
    """
    if average not in ("volume", "simple"):
        raise ValidationError("average must be 'volume' or 'simple'")
    if n_periods is not None and n_periods < 1:
        raise ValidationError("n_periods must be at least 1")
    c = triangle.values
    n_ages = triangle.shape[1]
    factors = np.empty(n_ages - 1)
    for k in range(n_ages - 1):
        rows = np.flatnonzero(~np.isnan(c[:, k + 1]))
        if rows.size == 0:
            raise ValidationError(f"no origin is observed at age index {k + 1}")
        if n_periods is not None:
            rows = rows[-n_periods:]
        num, den = c[rows, k + 1], c[rows, k]
        if average == "volume":
            if den.sum() <= 0:
                raise ValidationError(f"cannot develop from age index {k}: claims sum to zero")
            factors[k] = num.sum() / den.sum()
        else:
            if (den <= 0).any():
                raise ValidationError(f"zero claims at age index {k}: use average='volume'")
            factors[k] = np.mean(num / den)
    return DevelopmentPattern(factors, tail)


def exponential_tail(
    pattern: DevelopmentPattern, n_fit: int | None = None, max_ages: int = 100
) -> float:
    """Tail factor from an exponential decay fitted to (factor - 1).

    Fits log(f_k - 1) = a + b k by least squares over the last ``n_fit``
    factors greater than 1 and multiplies the extrapolated factors for the
    next ``max_ages`` ages. A common, simple curve; compare with benchmark
    tails before relying on it (Clark, 2003, for a curve-fitting alternative).
    """
    f = pattern.factors
    k = np.flatnonzero(f > 1)
    if n_fit is not None:
        k = k[-n_fit:]
    if k.size < 2:
        raise ValidationError("need at least two factors above 1 to fit a tail")
    slope, intercept = np.polyfit(k, np.log(f[k] - 1), 1)
    if slope >= 0:
        raise ValidationError(
            "factors are not decaying towards 1; an exponential tail does not fit"
        )
    future = np.arange(f.size, f.size + max_ages)
    return float(np.prod(1 + np.exp(intercept + slope * future)))
