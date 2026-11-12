"""Mack (1993) chain ladder standard errors.

Mack, T. (1993), "Distribution-free calculation of the standard error of
chain ladder reserve estimates", ASTIN Bulletin 23(2).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..core.base import ValidationError
from ..core.types import FloatArray
from .development import development_factors
from .results import MackResult
from .triangle import Triangle


def _sigma2(
    triangle: Triangle, factors: FloatArray, extrapolation: Literal["mack", "loglinear"]
) -> FloatArray:
    c = triangle.values
    n = factors.size
    sigma2 = np.full(n, np.nan)
    for k in range(n):
        rows = np.flatnonzero(~np.isnan(c[:, k + 1]))
        if rows.size > 1:
            ratios = c[rows, k + 1] / c[rows, k]
            sigma2[k] = np.sum(c[rows, k] * (ratios - factors[k]) ** 2) / (rows.size - 1)
    missing = np.flatnonzero(np.isnan(sigma2))
    if missing.size == 0:
        return sigma2
    known = np.flatnonzero(~np.isnan(sigma2))
    if extrapolation == "mack":
        # Mack's rule for the last factor(s): min(s_{k-1}^4 / s_{k-2}^2, s_{k-2}^2, s_{k-1}^2)
        if known.size < 2:
            raise ValidationError("need at least two estimable variances to extrapolate")
        for k in missing:
            a, b = sigma2[k - 2], sigma2[k - 1]
            sigma2[k] = min(b**2 / a if a > 0 else 0.0, a, b)
    elif extrapolation == "loglinear":
        positive = known[sigma2[known] > 0]
        if positive.size < 2:
            raise ValidationError(
                "need at least two positive variances for log-linear extrapolation"
            )
        slope, intercept = np.polyfit(positive, np.log(sigma2[positive]), 1)
        sigma2[missing] = np.exp(intercept + slope * missing)
    else:
        raise ValidationError("extrapolation must be 'mack' or 'loglinear'")
    return sigma2


def mack_chain_ladder(
    triangle: Triangle, *, sigma_extrapolation: Literal["mack", "loglinear"] = "mack"
) -> MackResult:
    """Chain ladder reserves with Mack standard errors (volume-weighted factors, no tail).

    ``sigma_extrapolation`` sets how the variance parameter is estimated where
    only one link ratio exists: Mack's original rule or a log-linear
    regression (the default in the R ChainLadder package). Tail uncertainty
    is not included; add a tail-factor allowance separately.
    """
    if triangle.shape[0] < 3 or triangle.shape[1] < 3:
        raise ValidationError("Mack needs at least 3 origins and 3 ages")
    pattern = development_factors(triangle)
    f = pattern.factors
    sigma2 = _sigma2(triangle, f, sigma_extrapolation)
    c = triangle.values
    n_origins, n_ages = triangle.shape
    latest_idx = triangle.latest_index

    # S_k: the sum of C[., k] over origins used to estimate f_k
    s = np.array([c[~np.isnan(c[:, k + 1]), k].sum() for k in range(n_ages - 1)])

    # Complete the square with chain ladder projections
    full = c.copy()
    for k in range(n_ages - 1):
        future = np.isnan(full[:, k + 1])
        full[future, k + 1] = full[future, k] * f[k]
    ultimate = full[:, -1]

    process = np.zeros(n_origins)
    parameter = np.zeros(n_origins)
    for i in range(n_origins):
        for k in range(latest_idx[i], n_ages - 1):
            w = sigma2[k] / f[k] ** 2
            process[i] += w / full[i, k]
            parameter[i] += w / s[k]
    process_var = ultimate**2 * process
    parameter_var = ultimate**2 * parameter
    mse = process_var + parameter_var

    # Covariance between origins through the shared factor estimates
    total_mse = mse.sum()
    # Rows are ordered most to least developed, so for a pair i < j the shared
    # future factors start at row i's latest age (Mack 1993, corollary to theorem 3).
    for i in range(n_origins):
        later = ultimate[i + 1 :].sum()
        ks = np.arange(latest_idx[i], n_ages - 1)
        total_mse += ultimate[i] * later * np.sum(2 * sigma2[ks] / f[ks] ** 2 / s[ks])

    return MackResult(
        method="mack_chain_ladder",
        origins=triangle.origins,
        latest=triangle.latest,
        ultimate=ultimate,
        pattern=pattern,
        details={"projected": full},
        standard_error=np.sqrt(mse),
        process_error=np.sqrt(process_var),
        parameter_error=np.sqrt(parameter_var),
        total_standard_error=float(np.sqrt(total_mse)),
        sigma2=sigma2,
    )
