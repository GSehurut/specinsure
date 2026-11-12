"""Frequency and severity distributions, fitting and comparison."""

from .base import Distribution
from .empirical import Empirical
from .fitting import FitResult, compare_fits, fit
from .mixture import Mixture
from .scipy import (
    ScipyDistribution,
    as_distribution,
    gamma,
    generalised_pareto,
    lognormal,
    negative_binomial,
    pareto,
    poisson,
)

__all__ = [
    "Distribution",
    "Empirical",
    "FitResult",
    "Mixture",
    "ScipyDistribution",
    "as_distribution",
    "compare_fits",
    "fit",
    "gamma",
    "generalised_pareto",
    "lognormal",
    "negative_binomial",
    "pareto",
    "poisson",
]
