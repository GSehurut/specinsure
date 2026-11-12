"""Base class for reinsurance treaties."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core.types import ArrayLike, FloatArray, IntArray
from ..core.validation import as_float_array, as_year_index


class Treaty(ABC):
    """A reinsurance contract that cedes part of each loss.

    Treaties work on arrays of individual losses. When the losses span many
    simulated years, pass ``year_index`` so annual terms (AAD, aggregate
    limits, reinstatements) are applied year by year and not across the whole
    simulation. Losses within a year are taken to be in occurrence order.
    """

    def __init__(self, name: str | None = None) -> None:
        self._name = name

    @property
    def name(self) -> str:
        """Label used in programme and report output; defaults to ``repr(self)``."""
        return self._name or repr(self)

    def apply(self, losses: ArrayLike, year_index: ArrayLike | None = None) -> FloatArray:
        """Return the ceded amount for each loss."""
        x = as_float_array("losses", losses, non_negative=True)
        years = as_year_index(year_index, x.size)
        return self._ceded(x, years)

    @abstractmethod
    def _ceded(self, losses: FloatArray, years: IntArray) -> FloatArray:
        """Ceded amounts for validated losses and year index."""

    def net(self, losses: ArrayLike, year_index: ArrayLike | None = None) -> FloatArray:
        """Retained amount for each loss (subject loss minus ceded)."""
        x = as_float_array("losses", losses, non_negative=True)
        return x - self.apply(x, year_index)


def cumulative_within_year(values: FloatArray, years: IntArray) -> FloatArray:
    """Running total of ``values`` within each year, in array (occurrence) order."""
    if values.size == 0:
        return values.copy()
    order = np.argsort(years, kind="stable")
    v, y = values[order], years[order]
    running = np.cumsum(v)
    is_first = np.r_[True, y[1:] != y[:-1]]
    group = np.cumsum(is_first) - 1
    before_year = (running - v)[is_first]
    out = np.empty_like(values)
    out[order] = running - before_year[group]
    return out


def allocate_pro_rata(
    losses: FloatArray, years: IntArray, annual_subject: FloatArray, annual_ceded: FloatArray
) -> FloatArray:
    """Spread each year's ceded amount over that year's losses in proportion to their size."""
    safe = np.where(annual_subject > 0, annual_subject, 1.0)
    ratio = np.where(annual_subject > 0, annual_ceded / safe, 0.0)
    return losses * ratio[years]
