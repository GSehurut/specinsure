"""Monte Carlo engine: turn loss models into year loss tables."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from ..core.base import ValidationError
from ..core.random import make_rng, spawn_rngs
from ..core.types import ArrayLike, SeedLike
from .models import FrequencySeverityModel
from .results import SimulationResult, YearLossTable
from .sampling import (
    gaussian_copula,
    reorder_years,
    sample_counts,
    sample_severities,
    student_t_copula,
)


def simulate(model: FrequencySeverityModel, n_years: int, seed: SeedLike = None) -> YearLossTable:
    """Simulate ``n_years`` of losses from a single frequency-severity model."""
    if n_years <= 0:
        raise ValidationError(f"n_years must be positive, got {n_years}")
    rng = make_rng(seed)
    counts = sample_counts(model.frequency, n_years, rng)
    losses = sample_severities(model.severity, int(counts.sum()), rng)
    year_index = np.repeat(np.arange(n_years, dtype=np.intp), counts)
    return YearLossTable(losses=losses, year_index=year_index, n_years=n_years, name=model.name)


def simulate_portfolio(
    models: Sequence[FrequencySeverityModel],
    n_years: int,
    seed: SeedLike = None,
    *,
    correlation: ArrayLike | None = None,
    copula: Literal["gaussian", "t"] = "gaussian",
    dof: float = 4.0,
) -> SimulationResult:
    """Simulate several lines over the same ``n_years``.

    Each model draws from its own random stream spawned from ``seed``, so the
    losses for a line do not change when other lines are added or removed.

    Dependency
    ----------
    Without ``correlation`` the lines are independent. With a correlation
    matrix (one row per model), each line's years are reordered so annual
    totals follow a Gaussian or Student t copula (Iman-Conover). Marginal
    distributions are unchanged; only which years coincide changes. The t
    copula adds tail dependence: bad years cluster more than under a Gaussian
    with the same correlation.
    """
    if not models:
        raise ValidationError("models must not be empty")
    names = [m.name for m in models]
    if len(set(names)) != len(names):
        raise ValidationError(f"model names must be unique, got {names}")

    *line_rngs, copula_rng = spawn_rngs(seed, len(models) + 1)
    tables = [simulate(m, n_years, rng) for m, rng in zip(models, line_rngs, strict=False)]

    if correlation is not None:
        corr = np.asarray(correlation, dtype=float)
        if corr.shape != (len(models), len(models)):
            raise ValidationError(
                f"correlation must be {len(models)}x{len(models)}, got shape {corr.shape}"
            )
        if copula == "gaussian":
            u = gaussian_copula(corr, n_years, copula_rng)
        elif copula == "t":
            u = student_t_copula(corr, dof, n_years, copula_rng)
        else:
            raise ValidationError(f"copula must be 'gaussian' or 't', got {copula!r}")
        tables = [_reordered(t, u[:, j]) for j, t in enumerate(tables)]

    return SimulationResult({t.name: t for t in tables})


def _reordered(table: YearLossTable, uniforms: np.ndarray) -> YearLossTable:
    new_year = reorder_years(table.annual_totals(), uniforms)
    year_index = new_year[table.year_index]
    order = np.argsort(year_index, kind="stable")  # keep occurrence order within each year
    return YearLossTable(
        losses=table.losses[order],
        year_index=year_index[order],
        n_years=table.n_years,
        name=table.name,
    )
