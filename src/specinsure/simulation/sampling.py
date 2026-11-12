"""Low-level sampling: claim counts, severities and copula dependency."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ..core.base import ValidationError
from ..core.types import ArrayLike, FloatArray, IntArray
from ..distributions.base import Distribution


def sample_counts(frequency: Distribution, n_years: int, rng: np.random.Generator) -> IntArray:
    """Draw annual claim counts and check they are non-negative integers."""
    raw = np.asarray(frequency.sample(n_years, rng))
    counts = raw.astype(np.intp)
    if raw.shape != (n_years,) or not np.array_equal(counts, raw):
        raise ValidationError("frequency distribution must return integer counts")
    if counts.size and counts.min() < 0:
        raise ValidationError("frequency distribution produced negative counts")
    return counts


def sample_severities(severity: Distribution, n: int, rng: np.random.Generator) -> FloatArray:
    """Draw ``n`` individual losses and check they are finite and non-negative."""
    losses = np.asarray(severity.sample(n, rng), dtype=float)
    if losses.shape != (n,):
        raise ValidationError(f"severity returned shape {losses.shape}, expected ({n},)")
    if n and (not np.all(np.isfinite(losses)) or losses.min() < 0):
        raise ValidationError("severity distribution produced negative or non-finite losses")
    return losses


def validate_correlation(correlation: ArrayLike) -> FloatArray:
    """Check a correlation matrix is square, symmetric, unit-diagonal and positive definite."""
    corr = np.asarray(correlation, dtype=float)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValidationError(f"correlation must be a square matrix, got shape {corr.shape}")
    if not np.allclose(corr, corr.T) or not np.allclose(np.diag(corr), 1.0):
        raise ValidationError("correlation must be symmetric with ones on the diagonal")
    if np.abs(corr).max() > 1:
        raise ValidationError("correlations must lie in [-1, 1]")
    try:
        np.linalg.cholesky(corr)
    except np.linalg.LinAlgError as exc:
        raise ValidationError("correlation matrix is not positive definite") from exc
    return corr


def gaussian_copula(correlation: ArrayLike, n: int, rng: np.random.Generator) -> FloatArray:
    """``n`` draws of uniforms with a Gaussian copula, shape ``(n, d)``."""
    corr = validate_correlation(correlation)
    z = rng.standard_normal((n, corr.shape[0])) @ np.linalg.cholesky(corr).T
    return stats.norm.cdf(z)


def student_t_copula(
    correlation: ArrayLike, dof: float, n: int, rng: np.random.Generator
) -> FloatArray:
    """``n`` draws of uniforms with a Student t copula (tail dependence), shape ``(n, d)``."""
    if dof <= 0:
        raise ValidationError(f"dof must be positive, got {dof}")
    corr = validate_correlation(correlation)
    z = rng.standard_normal((n, corr.shape[0])) @ np.linalg.cholesky(corr).T
    w = rng.chisquare(dof, size=(n, 1)) / dof
    return stats.t.cdf(z / np.sqrt(w), df=dof)


def reorder_years(annual_totals: FloatArray, uniforms: FloatArray) -> IntArray:
    """Year permutation that gives ``annual_totals`` the same ranks as ``uniforms``.

    Returns ``new_year`` such that the year ``y`` of one line becomes year
    ``new_year[y]``. This is the reordering step of Iman and Conover (1982): it
    keeps each line's simulated distribution exactly and imposes the copula's
    rank dependence on annual totals.
    """
    n = annual_totals.size
    if uniforms.shape != (n,):
        raise ValidationError("uniforms must have one value per simulated year")
    by_loss = np.argsort(annual_totals, kind="stable")
    by_uniform = np.argsort(uniforms, kind="stable")
    new_year = np.empty(n, dtype=np.intp)
    new_year[by_loss] = by_uniform
    return new_year
