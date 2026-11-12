"""Adapter for frozen ``scipy.stats`` distributions, plus actuarial parametrisations."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from ..core.base import ValidationError
from ..core.types import ArrayLike, FloatArray
from ..core.validation import check_positive
from .base import Distribution


class ScipyDistribution(Distribution):
    """Wrap a frozen ``scipy.stats`` distribution, e.g. ``stats.lognorm(s=1.2, scale=1e6)``."""

    def __init__(self, frozen) -> None:
        if not all(hasattr(frozen, attr) for attr in ("rvs", "cdf", "ppf", "mean")):
            raise ValidationError(
                f"expected a frozen scipy.stats distribution, got {type(frozen).__name__}"
            )
        self.frozen = frozen
        self.is_discrete = isinstance(getattr(frozen, "dist", None), stats.rv_discrete)

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return np.asarray(self.frozen.rvs(size=size, random_state=rng))

    def mean(self) -> float:
        return float(self.frozen.mean())

    def cdf(self, x: ArrayLike) -> FloatArray:
        return np.asarray(self.frozen.cdf(x), dtype=float)

    def sf(self, x: ArrayLike) -> FloatArray:
        return np.asarray(self.frozen.sf(x), dtype=float)

    def ppf(self, q: ArrayLike) -> FloatArray:
        return np.asarray(self.frozen.ppf(q), dtype=float)

    def limited_expected_value(self, limit: float) -> float:
        if self.is_discrete and math.isfinite(limit) and limit > 0:
            # S is a right-continuous step function on the integers, so
            # E[min(N, u)] = sum_{k < floor(u)} S(k) + (u - floor(u)) * S(floor(u))
            whole = math.floor(limit)
            total = float(self.frozen.sf(np.arange(whole)).sum())
            return total + (limit - whole) * float(self.frozen.sf(whole))
        return super().limited_expected_value(limit)

    def __repr__(self) -> str:
        name = getattr(getattr(self.frozen, "dist", None), "name", "scipy")
        kwds = ", ".join(f"{k}={v:g}" for k, v in getattr(self.frozen, "kwds", {}).items())
        return f"ScipyDistribution({name}({kwds}))"


def as_distribution(obj) -> Distribution:
    """Return ``obj`` as a :class:`Distribution`, wrapping frozen scipy distributions."""
    if isinstance(obj, Distribution):
        return obj
    return ScipyDistribution(obj)


# --- Actuarial parametrisations -------------------------------------------------------


def poisson(mean: float) -> ScipyDistribution:
    """Poisson claim count with the given mean."""
    return ScipyDistribution(stats.poisson(check_positive("mean", mean)))


def negative_binomial(mean: float, variance: float) -> ScipyDistribution:
    """Negative binomial claim count from mean and variance (variance > mean: over-dispersed)."""
    mean, variance = check_positive("mean", mean), check_positive("variance", variance)
    if variance <= mean:
        raise ValidationError("negative binomial needs variance > mean; use poisson otherwise")
    p = mean / variance
    return ScipyDistribution(stats.nbinom(n=mean * p / (1 - p), p=p))


def lognormal(mean: float, cv: float) -> ScipyDistribution:
    """Lognormal severity from its mean and coefficient of variation."""
    mean, cv = check_positive("mean", mean), check_positive("cv", cv)
    sigma2 = math.log1p(cv**2)
    return ScipyDistribution(stats.lognorm(s=math.sqrt(sigma2), scale=mean * math.exp(-sigma2 / 2)))


def gamma(mean: float, cv: float) -> ScipyDistribution:
    """Gamma severity from its mean and coefficient of variation."""
    mean, cv = check_positive("mean", mean), check_positive("cv", cv)
    shape = 1 / cv**2
    return ScipyDistribution(stats.gamma(a=shape, scale=mean / shape))


def pareto(alpha: float, threshold: float) -> ScipyDistribution:
    """Single-parameter Pareto: P(X > x) = (threshold / x) ** alpha for x >= threshold.

    The standard model for large losses above a modelling threshold. The mean is
    infinite for ``alpha <= 1``.
    """
    alpha, threshold = check_positive("alpha", alpha), check_positive("threshold", threshold)
    return ScipyDistribution(stats.pareto(b=alpha, scale=threshold))


def generalised_pareto(shape: float, scale: float, threshold: float = 0.0) -> ScipyDistribution:
    """Generalised Pareto (GPD) for excesses over ``threshold`` (peaks over threshold)."""
    return ScipyDistribution(
        stats.genpareto(c=shape, scale=check_positive("scale", scale), loc=threshold)
    )
