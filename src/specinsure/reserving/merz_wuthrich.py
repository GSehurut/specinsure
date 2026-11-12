"""Merz-Wüthrich one-year reserve risk: the claims development result (CDR).

Merz, M. and Wüthrich, M. V. (2008), "Modelling the claims development result
for solvency purposes", CAS E-Forum, Fall 2008.

The CDR for the next calendar year is the change in the chain ladder ultimate
once one more diagonal is observed and the factors are re-estimated:
``CDR_i = U_i(I) - U_i(I + 1)``. Its expected value is zero; its root mean
squared error of prediction is the one-year reserve risk used for Solvency II
(which measures the change in own funds over one year), as opposed to Mack's
ultimate view of the full run-off.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .mack import _factor_volumes, mack_chain_ladder
from .results import MerzWuthrichResult
from .triangle import Triangle


def merz_wuthrich(
    triangle: Triangle, *, sigma_extrapolation: Literal["mack", "loglinear"] = "mack"
) -> MerzWuthrichResult:
    """One-year claims development result standard errors (Merz & Wüthrich 2008).

    Uses the same chain ladder factors and variance parameters as
    :func:`mack_chain_ladder` (volume-weighted, no tail), so the result also
    carries the Mack ultimate view for comparison. The process part uses the
    paper's multiplicative form; the parameter part uses Mack's estimator of
    factor uncertainty. For the most developed open origin the one-year and
    ultimate errors coincide, because only one development period remains.

    Rows with the same latest age (non-square triangles) are handled by
    pooling them in the next-diagonal update of each factor.
    """
    mack = mack_chain_ladder(triangle, sigma_extrapolation=sigma_extrapolation)
    c = triangle.values
    n_ages = triangle.shape[1]
    f = mack.pattern.factors
    sigma2 = mack.sigma2
    ultimate = mack.ultimate
    age = triangle.latest_index
    latest = triangle.latest

    s_now = _factor_volumes(c)  # S_j^I: volume behind f_j today
    # Volume on the current diagonal at each age, which joins f_j next year
    diagonal = np.zeros(n_ages - 1)
    open_rows = age < n_ages - 1
    np.add.at(diagonal, age[open_rows], latest[open_rows])
    s_next = s_now + diagonal  # S_j^{I+1}

    w = sigma2 / f**2
    with np.errstate(divide="ignore", invalid="ignore"):
        # Contribution of factor j's re-estimation to the CDR of later origins
        process_term = np.where(diagonal > 0, w * diagonal / s_next**2, 0.0)
        parameter_term = np.where(diagonal > 0, (diagonal / s_next) ** 2 * w / s_now, 0.0)
        own_weight = np.where(s_next > 0, diagonal / s_next, 0.0)
    # Tail sums/products over factors strictly after age j
    after_param = np.concatenate([np.cumsum(parameter_term[::-1])[::-1][1:], [0.0]])
    after_prod = np.concatenate([np.cumprod((1 + process_term)[::-1])[::-1][1:], [1.0]])

    n_origins = triangle.shape[0]
    process_rel = np.zeros(n_origins)
    parameter_rel = np.zeros(n_origins)
    for i in np.flatnonzero(open_rows):
        a = age[i]
        process_rel[i] = (1 + w[a] / c[i, a]) * after_prod[a] - 1
        parameter_rel[i] = w[a] / s_now[a] + after_param[a]
    process_var = ultimate**2 * process_rel
    parameter_var = ultimate**2 * parameter_rel

    # Cross terms: origin i with each less (or equally) developed origin k;
    # Triangle guarantees rows run from most to least developed.
    total_process = process_var.sum()
    total_parameter = parameter_var.sum()
    open_idx = np.flatnonzero(open_rows)
    for pos, i in enumerate(open_idx):
        a = age[i]
        for k in open_idx[pos + 1 :]:
            later = age[k] < a
            proc = (1 + (w[a] / s_next[a] if later else 0.0)) * after_prod[a] - 1
            coef = own_weight[a] if later else 1.0
            param = coef * w[a] / s_now[a] + after_param[a]
            total_process += 2 * ultimate[i] * ultimate[k] * proc
            total_parameter += 2 * ultimate[i] * ultimate[k] * param

    return MerzWuthrichResult(
        method="merz_wuthrich",
        origins=triangle.origins,
        latest=triangle.latest,
        ultimate=ultimate,
        pattern=mack.pattern,
        details=mack.details,
        standard_error=np.sqrt(process_var + parameter_var),
        process_error=np.sqrt(process_var),
        parameter_error=np.sqrt(parameter_var),
        total_standard_error=float(np.sqrt(total_process + total_parameter)),
        total_process_error=float(np.sqrt(total_process)),
        total_parameter_error=float(np.sqrt(total_parameter)),
        mack=mack,
    )
