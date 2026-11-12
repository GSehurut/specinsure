"""Claims development triangles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.base import ValidationError
from ..core.types import ArrayLike, FloatArray, IntArray


@dataclass(frozen=True)
class Triangle:
    """Cumulative claims by origin period (rows) and development age (columns).

    ``values`` has shape ``(n_origins, n_ages)`` with ``NaN`` for cells not yet
    observed. Each row's observed cells must be contiguous from the first age,
    and later origins may not be observed at more ages than earlier ones. That
    covers the usual triangle and rectangular shapes (more origins than ages).

    ``origins`` label the rows (underwriting or accident years) and ``ages``
    the columns (for example 12, 24, 36 months). Whether the triangle holds
    paid or incurred claims is up to the caller: an incurred triangle gives
    IBNR (including IBNER), a paid triangle gives the total outstanding reserve.
    """

    values: FloatArray
    origins: IntArray
    ages: IntArray

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=float)
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
            raise ValidationError(
                f"values must be a 2-d array with at least 2 ages, got {values.shape}"
            )
        origins, ages = np.asarray(self.origins), np.asarray(self.ages)
        if origins.shape != (values.shape[0],) or ages.shape != (values.shape[1],):
            raise ValidationError("origins and ages must label every row and column")
        if np.unique(origins).size != origins.size or np.any(np.diff(ages) <= 0):
            raise ValidationError("origins must be unique and ages strictly increasing")
        observed = ~np.isnan(values)
        counts = observed.sum(axis=1)
        contiguous = observed == (np.arange(values.shape[1]) < counts[:, None])
        if not contiguous.all():
            raise ValidationError(
                "each row's observed values must be contiguous from the first age"
            )
        if (counts == 0).any():
            raise ValidationError("every origin needs at least one observed value")
        if np.any(np.diff(counts) > 0):
            raise ValidationError(
                "a later origin cannot be observed at more ages than an earlier one"
            )
        if not np.all(np.isfinite(values[observed])):
            raise ValidationError("observed values must be finite")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "origins", origins.astype(np.intp))
        object.__setattr__(self, "ages", ages.astype(np.intp))

    # --- constructors ---------------------------------------------------------------

    @classmethod
    def from_cumulative(
        cls, values: ArrayLike, origins: ArrayLike | None = None, ages: ArrayLike | None = None
    ) -> Triangle:
        """Build from a cumulative array; origins default to 0, 1, ... and ages to 1, 2, ..."""
        v = np.asarray(values, dtype=float)
        origins = np.arange(v.shape[0]) if origins is None else origins
        ages = np.arange(1, v.shape[1] + 1) if ages is None else ages
        return cls(v, np.asarray(origins), np.asarray(ages))

    @classmethod
    def from_incremental(
        cls, values: ArrayLike, origins: ArrayLike | None = None, ages: ArrayLike | None = None
    ) -> Triangle:
        """Build from incremental amounts (NaN for unobserved cells)."""
        v = np.asarray(values, dtype=float)
        return cls.from_cumulative(np.cumsum(v, axis=1), origins, ages)

    @classmethod
    def from_claims(
        cls, origin: ArrayLike, age: ArrayLike, amount: ArrayLike, *, cumulative: bool = False
    ) -> Triangle:
        """Build from long-format records (one row per origin, age and amount).

        Incremental records (the default) are summed per cell, so transaction
        level data can be passed directly. Missing cells inside the observed
        region count as zero; ages beyond each origin's latest record are unobserved.
        """
        origin, age, amount = np.asarray(origin), np.asarray(age), np.asarray(amount, dtype=float)
        if not (origin.shape == age.shape == amount.shape) or origin.ndim != 1:
            raise ValidationError("origin, age and amount must be 1-d arrays of equal length")
        origins, row = np.unique(origin, return_inverse=True)
        ages, col = np.unique(age, return_inverse=True)
        grid = np.zeros((origins.size, ages.size))
        if cumulative:
            grid[:] = np.nan
            grid[row, col] = amount
        else:
            np.add.at(grid, (row, col), amount)
        latest = np.zeros(origins.size, dtype=np.intp)
        np.maximum.at(latest, row, col)
        grid[np.arange(ages.size) > latest[:, None]] = np.nan
        if cumulative:
            return cls(grid, origins, ages)
        return cls.from_incremental(grid, origins, ages)

    # --- views ----------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape  # type: ignore[return-value]

    @property
    def observed(self) -> np.ndarray:
        return ~np.isnan(self.values)

    @property
    def latest_index(self) -> IntArray:
        """Column index of each origin's latest observed age."""
        return self.observed.sum(axis=1) - 1

    @property
    def latest(self) -> FloatArray:
        """Latest diagonal: the most recent cumulative value for each origin."""
        return self.values[np.arange(self.shape[0]), self.latest_index]

    def incremental(self) -> FloatArray:
        """Incremental amounts, NaN where unobserved."""
        inc = np.diff(self.values, axis=1, prepend=0.0)
        return np.where(self.observed, inc, np.nan)

    def link_ratios(self) -> FloatArray:
        """Individual age-to-age factors C[i, k+1] / C[i, k], shape ``(n_origins, n_ages - 1)``."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.values[:, 1:] / self.values[:, :-1]

    def __repr__(self) -> str:
        return (
            f"Triangle({self.shape[0]} origins {self.origins[0]}-{self.origins[-1]}, "
            f"ages {self.ages[0]}-{self.ages[-1]})"
        )
