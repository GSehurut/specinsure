"""Simulation outputs: year loss tables for one line or a whole portfolio."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np

from ..core.base import ValidationError
from ..core.types import FloatArray, IntArray
from ..core.validation import as_float_array, as_year_index


@dataclass(frozen=True)
class YearLossTable:
    """Individual simulated ground-up losses and the simulated year of each.

    Losses within a year are stored in occurrence order, which matters for
    treaties with aggregate terms (AAD, aggregate limit, reinstatements).
    """

    losses: FloatArray
    year_index: IntArray
    n_years: int
    name: str = "losses"

    def __post_init__(self) -> None:
        losses = as_float_array("losses", self.losses, non_negative=True)
        years = as_year_index(self.year_index, losses.size)
        if self.n_years <= 0:
            raise ValidationError(f"n_years must be positive, got {self.n_years}")
        if years.size and years.max() >= self.n_years:
            raise ValidationError("year_index contains a year >= n_years")
        object.__setattr__(self, "losses", losses)
        object.__setattr__(self, "year_index", years)

    @property
    def n_losses(self) -> int:
        return int(self.losses.size)

    def annual_totals(self, values: FloatArray | None = None) -> FloatArray:
        """Aggregate per simulated year (of ``values`` if given, else gross losses): AEP basis."""
        return np.bincount(self.year_index, weights=self._values(values), minlength=self.n_years)

    def annual_max(self, values: FloatArray | None = None) -> FloatArray:
        """Largest single loss per simulated year, 0 when there is none: OEP basis."""
        out = np.zeros(self.n_years)
        np.maximum.at(out, self.year_index, self._values(values))
        return out

    def annual_counts(self) -> IntArray:
        return np.bincount(self.year_index, minlength=self.n_years)

    def ceded(self, treaty) -> FloatArray:
        """Ceded amount per loss under ``treaty``, with annual terms applied year by year."""
        return treaty.apply(self.losses, self.year_index)

    def _values(self, values: FloatArray | None) -> FloatArray:
        if values is None:
            return self.losses
        values = np.asarray(values, dtype=float)
        if values.shape != self.losses.shape:
            raise ValidationError("values must have one entry per loss")
        return values


@dataclass(frozen=True, eq=False)
class SimulationResult(Mapping[str, YearLossTable]):
    """Year loss tables for several lines simulated over the same years.

    Year ``y`` is the same simulated year in every table, so dependency
    between lines is preserved when results are combined.
    """

    tables: dict[str, YearLossTable]
    n_years: int = field(init=False)

    def __post_init__(self) -> None:
        if not self.tables:
            raise ValidationError("a simulation result needs at least one table")
        n_years = {t.n_years for t in self.tables.values()}
        if len(n_years) != 1:
            raise ValidationError("all tables must have the same number of simulated years")
        object.__setattr__(self, "n_years", n_years.pop())

    def __getitem__(self, name: str) -> YearLossTable:
        return self.tables[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.tables)

    def __len__(self) -> int:
        return len(self.tables)

    @property
    def names(self) -> list[str]:
        return list(self.tables)

    def annual_matrix(self) -> FloatArray:
        """Annual gross totals, shape ``(n_years, n_lines)``, columns in :attr:`names` order."""
        return np.column_stack([t.annual_totals() for t in self.tables.values()])

    def annual_total(self) -> FloatArray:
        """Portfolio annual gross total across all lines."""
        return self.annual_matrix().sum(axis=1)
