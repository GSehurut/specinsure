"""Over-dispersed Poisson bootstrap of the chain ladder.

England, P. and Verrall, R. (2002), "Stochastic claims reserving in general
insurance", British Actuarial Journal 8(3), section 7 and appendix 3.
"""

from __future__ import annotations

import numpy as np

from ..core.base import ValidationError
from ..core.random import make_rng
from ..core.types import FloatArray, SeedLike
from .development import development_factors
from .methods import chain_ladder
from .results import BootstrapResult
from .triangle import Triangle


def _fitted_incremental(triangle: Triangle, factors: FloatArray) -> FloatArray:
    """Fitted cumulative values back-cast from the latest diagonal, as incrementals."""
    c = triangle.values
    fitted = np.full_like(c, np.nan)
    for i, last in enumerate(triangle.latest_index):
        fitted[i, last] = c[i, last]
        for k in range(last - 1, -1, -1):
            fitted[i, k] = fitted[i, k + 1] / factors[k]
    return np.diff(fitted, axis=1, prepend=0.0)


def odp_bootstrap(
    triangle: Triangle, n_sims: int = 10_000, seed: SeedLike = None
) -> BootstrapResult:
    """Simulate reserves with the ODP bootstrap (parameter and process error).

    1. Fit the chain ladder and back-cast fitted incrementals m.
    2. Form Pearson residuals r = (X - m) / sqrt(m), scaled by sqrt(N / (N - p))
       for degrees of freedom, and the scale parameter phi = sum r^2 / (N - p).
    3. For each simulation, resample residuals with replacement, rebuild a
       pseudo triangle X* = m + r* sqrt(m), refit the chain ladder factors,
       project future incrementals and add process error from a gamma
       distribution with mean m* and variance phi m*.

    Requires positive fitted incrementals. Results are per origin, without a
    tail factor.
    """
    if n_sims < 1:
        raise ValidationError("n_sims must be at least 1")
    rng = make_rng(seed)
    observed = triangle.observed
    factors = development_factors(triangle).factors
    actual = triangle.incremental()
    fitted = _fitted_incremental(triangle, factors)
    if np.any(fitted[observed] <= 0):
        raise ValidationError("ODP bootstrap needs positive fitted incrementals")

    n_obs = int(observed.sum())
    n_params = triangle.shape[0] + triangle.shape[1] - 1
    dof = n_obs - n_params
    if dof <= 0:
        raise ValidationError("triangle too small: no degrees of freedom left for the bootstrap")
    residuals = ((actual - fitted) / np.sqrt(fitted))[observed]
    scale = float(np.sum(residuals**2) / dof)
    residuals = residuals * np.sqrt(n_obs / dof)

    # Vectorised over simulations: pseudo cumulative triangles, shape (n_sims, n_origins, n_ages)
    sampled = rng.choice(residuals, size=(n_sims, n_obs))
    pseudo = np.zeros((n_sims, *triangle.shape))
    pseudo[:, observed] = fitted[observed] + sampled * np.sqrt(fitted[observed])
    cumulative = np.cumsum(pseudo, axis=2)

    n_ages = triangle.shape[1]
    pseudo_factors = np.empty((n_sims, n_ages - 1))
    for k in range(n_ages - 1):
        rows = observed[:, k + 1]
        denominator = cumulative[:, rows, k].sum(axis=1)
        if np.any(denominator <= 0):
            raise ValidationError(f"a resampled triangle has non-positive claims at age index {k}")
        pseudo_factors[:, k] = cumulative[:, rows, k + 1].sum(axis=1) / denominator

    latest_index = triangle.latest_index
    level = cumulative[:, np.arange(triangle.shape[0]), latest_index]  # (n_sims, n_origins)
    reserves = np.zeros((n_sims, triangle.shape[0]))
    for k in range(n_ages - 1):
        developing = latest_index <= k  # origins whose age k is at or beyond the latest diagonal
        if not developing.any():
            continue
        nxt = level[:, developing] * pseudo_factors[:, [k]]
        reserves[:, developing] += _gamma_process(nxt - level[:, developing], scale, rng)
        level[:, developing] = nxt

    return BootstrapResult(triangle.origins, reserves, scale, chain_ladder(triangle))


def _gamma_process(mean: FloatArray, scale: float, rng: np.random.Generator) -> FloatArray:
    """Gamma draws with mean ``mean`` and variance ``scale * |mean|``, keeping the sign."""
    out = np.zeros_like(mean)
    nonzero = mean != 0
    m = np.abs(mean[nonzero])
    out[nonzero] = np.sign(mean[nonzero]) * rng.gamma(shape=m / scale, scale=scale)
    return out
