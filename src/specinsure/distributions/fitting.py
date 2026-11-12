"""Maximum likelihood fitting of severity distributions and model comparison."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..core.base import ConvergenceError, ValidationError
from ..core.types import ArrayLike
from ..core.validation import as_float_array
from .scipy import ScipyDistribution

#: Families tried by :func:`compare_fits` when none are given.
DEFAULT_FAMILIES = ("lognorm", "gamma", "weibull_min", "expon")


@dataclass(frozen=True)
class FitResult:
    """Outcome of fitting one family to data.

    ``ks_pvalue`` is optimistic because the parameters were estimated from the
    same data; use it to rank fits, not as a formal test.
    """

    family: str
    distribution: ScipyDistribution
    params: tuple[float, ...]
    n_params: int
    n: int
    log_likelihood: float
    aic: float
    bic: float
    ks_statistic: float
    ks_pvalue: float


def fit(data: ArrayLike, family: str, *, fix_loc_at_zero: bool = True) -> FitResult:
    """Fit a ``scipy.stats`` continuous ``family`` (by name) to positive ``data`` by MLE.

    The location parameter is fixed at zero by default, the usual choice for
    loss sizes. For losses above a threshold, fit the excesses (``data -
    threshold``) or use ``family="pareto"`` on the losses themselves.
    """
    x = as_float_array("data", data, non_negative=True)
    if x.size < 2:
        raise ValidationError("need at least two observations to fit a distribution")
    dist = getattr(stats, family, None)
    if not isinstance(dist, stats.rv_continuous):
        raise ValidationError(f"{family!r} is not a scipy.stats continuous distribution")

    kwargs = {"floc": 0.0} if fix_loc_at_zero else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            params = tuple(float(p) for p in dist.fit(x, **kwargs))
        except Exception as exc:  # scipy raises a variety of errors on failed fits
            raise ConvergenceError(f"fitting {family!r} failed: {exc}") from exc

    frozen = dist(*params)
    log_likelihood = float(np.sum(frozen.logpdf(x)))
    if not math.isfinite(log_likelihood):
        raise ConvergenceError(f"fitted {family!r} gives zero likelihood to some observations")
    k = len(params) - len(kwargs)
    ks = stats.kstest(x, frozen.cdf)
    return FitResult(
        family=family,
        distribution=ScipyDistribution(frozen),
        params=params,
        n_params=k,
        n=int(x.size),
        log_likelihood=log_likelihood,
        aic=2 * k - 2 * log_likelihood,
        bic=k * math.log(x.size) - 2 * log_likelihood,
        ks_statistic=float(ks.statistic),
        ks_pvalue=float(ks.pvalue),
    )


def compare_fits(data: ArrayLike, families: Sequence[str] = DEFAULT_FAMILIES) -> list[FitResult]:
    """Fit each family and return results sorted by AIC (best first).

    Families that fail to fit are skipped with a warning rather than aborting
    the comparison.
    """
    results = []
    for family in families:
        try:
            results.append(fit(data, family))
        except ConvergenceError as exc:
            warnings.warn(str(exc), stacklevel=2)
    if not results:
        raise ConvergenceError("no family could be fitted to the data")
    return sorted(results, key=lambda r: r.aic)
