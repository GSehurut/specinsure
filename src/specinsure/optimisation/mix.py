"""Line-of-business mix optimisation with a mean-TVaR linear programme.

Implements Rockafellar and Uryasev (2000), "Optimization of conditional
value-at-risk", Journal of Risk 2(3). For simulated results R (n_sims x
n_lines, profit positive) and weights w, portfolio loss in scenario s is
L_s = -R_s . w and

    TVaR_a(w) = min over z of  z + 1 / ((1 - a) n) * sum_s max(L_s - z, 0),

which is linear in (w, z, u) with u_s >= L_s - z, u_s >= 0. Scaling a line's
results linearly with its weight assumes rate adequacy, expense ratio and
loss profile do not change with volume; treat weights far from 1 with care.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, sparse

from ..core.base import ConvergenceError, ValidationError
from ..core.types import ArrayLike, FloatArray
from ..core.validation import check_level
from ..risk.measures import tvar, var
from .results import MixResult


def _setup(results: ArrayLike, bounds) -> tuple[FloatArray, FloatArray, FloatArray]:
    r = np.asarray(results, dtype=float)
    if r.ndim != 2 or r.shape[0] < 2 or r.shape[1] < 1:
        raise ValidationError(f"results must be an (n_sims, n_lines) matrix, got shape {r.shape}")
    if not np.all(np.isfinite(r)):
        raise ValidationError("results contain NaN or infinite values")
    d = r.shape[1]
    lower, upper = (np.broadcast_to(np.asarray(b, dtype=float), (d,)).copy() for b in bounds)
    if (lower > upper).any() or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValidationError("bounds must be finite with lower <= upper for every line")
    return r, lower, upper


def optimise_mix(
    results: ArrayLike,
    level: float = 0.99,
    *,
    target_return: float | None = None,
    tvar_budget: float | None = None,
    bounds: tuple[ArrayLike, ArrayLike] = (0.0, 1.0),
) -> MixResult:
    """Choose line weights on the mean-TVaR efficient frontier.

    Give exactly one of:

    ``target_return``
        Minimise TVaR subject to expected result >= target.
    ``tvar_budget``
        Maximise expected result subject to TVaR <= budget.

    ``bounds`` are (lower, upper) weights, scalars or one value per line
    (0 = exit the line, 1 = plan volume). Raises :class:`ConvergenceError`
    if the problem is infeasible.
    """
    check_level(level)
    if (target_return is None) == (tvar_budget is None):
        raise ValidationError("give exactly one of target_return or tvar_budget")
    r, lower, upper = _setup(results, bounds)
    n, d = r.shape
    tail_weight = 1.0 / ((1 - level) * n)
    mean = r.mean(axis=0)

    # Variables: [w (d), z (1), u (n)]; rows: -R w - z - u <= 0
    a_tail = sparse.hstack(
        [
            sparse.csr_matrix(-r),
            sparse.csr_matrix(-np.ones((n, 1))),
            -sparse.identity(n, format="csr"),
        ],
        format="csr",
    )
    tvar_row = np.concatenate([np.zeros(d), [1.0], np.full(n, tail_weight)])
    var_bounds = (
        [(lo, hi) for lo, hi in zip(lower, upper, strict=False)] + [(None, None)] + [(0, None)] * n
    )

    if target_return is not None:
        cost = tvar_row
        extra_row = np.concatenate([-mean, [0.0], np.zeros(n)])
        extra_rhs = -float(target_return)
    else:
        cost = np.concatenate([-mean, [0.0], np.zeros(n)])
        extra_row = tvar_row
        extra_rhs = float(tvar_budget)

    a_ub = sparse.vstack([a_tail, sparse.csr_matrix(extra_row)], format="csr")
    b_ub = np.concatenate([np.zeros(n), [extra_rhs]])
    solution = optimize.linprog(cost, A_ub=a_ub, b_ub=b_ub, bounds=var_bounds, method="highs")
    if not solution.success:
        raise ConvergenceError(f"mix optimisation failed: {solution.message}")

    w = solution.x[:d]
    losses = -(r @ w)
    # HiGHS meets bounds to around 1e-7; use a looser tolerance to report binding bounds
    tol = 1e-6 * max(1.0, float(np.abs(upper).max()))
    return MixResult(
        weights=w,
        expected_result=float(mean @ w),
        tvar=tvar(losses, level),
        var=var(losses, level),
        level=level,
        at_lower=np.isclose(w, lower, atol=tol),
        at_upper=np.isclose(w, upper, atol=tol),
        message=str(solution.message),
    )


def efficient_frontier(
    results: ArrayLike,
    level: float = 0.99,
    *,
    n_points: int = 20,
    bounds: tuple[ArrayLike, ArrayLike] = (0.0, 1.0),
) -> list[MixResult]:
    """Mean-TVaR efficient frontier from the minimum-risk mix to the maximum-return mix.

    Points are spaced evenly in expected result, so TVaR is non-decreasing
    along the returned list.
    """
    if n_points < 2:
        raise ValidationError("n_points must be at least 2")
    r, lower, upper = _setup(results, bounds)
    mean = r.mean(axis=0)
    max_return = float(np.where(mean > 0, mean * upper, mean * lower).sum())
    # Minimum-TVaR mix: the lowest target is never binding below the return of
    # the worst feasible mix.
    floor = float(np.where(mean > 0, mean * lower, mean * upper).sum())
    min_risk = optimise_mix(r, level, target_return=floor, bounds=(lower, upper))
    targets = np.linspace(min_risk.expected_result, max_return, n_points)
    return [min_risk] + [
        optimise_mix(r, level, target_return=t, bounds=(lower, upper)) for t in targets[1:]
    ]
