"""Pricing inputs: historical experience and premium loadings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..core.base import ValidationError
from ..core.types import ArrayLike, FloatArray, IntArray
from ..core.validation import as_float_array, check_fraction, check_level, check_non_negative


@dataclass(frozen=True)
class Loadings:
    """Loadings that turn an expected loss into a technical premium.

    ``brokerage``, ``internal_expenses`` and ``profit_margin`` are shares of
    gross premium. The capital charge is ``cost_of_capital`` times the capital
    the layer consumes, measured as TVaR or VaR of annual layer losses at
    ``capital_level`` minus the expected loss.
    """

    brokerage: float = 0.0
    internal_expenses: float = 0.0
    profit_margin: float = 0.0
    cost_of_capital: float = 0.0
    capital_level: float = 0.99
    capital_measure: Literal["tvar", "var"] = "tvar"

    def __post_init__(self) -> None:
        for name in ("brokerage", "internal_expenses", "profit_margin"):
            check_fraction(name, getattr(self, name))
        check_non_negative("cost_of_capital", self.cost_of_capital)
        check_level(self.capital_level)
        if self.capital_measure not in ("tvar", "var"):
            raise ValidationError("capital_measure must be 'tvar' or 'var'")
        if self.premium_loads >= 1:
            raise ValidationError(
                "brokerage + internal_expenses + profit_margin must be below 100%"
            )

    @property
    def premium_loads(self) -> float:
        """Total proportional loads on premium."""
        return self.brokerage + self.internal_expenses + self.profit_margin


@dataclass(frozen=True)
class ExperienceData:
    """Historical individual losses and subject premium by underwriting year.

    ``losses`` are ground-up incurred amounts as at the data date, with the
    underwriting (or accident) year of each in ``loss_years``. ``years`` and
    ``subject_premium`` give the exposure for every year in the experience
    period, including years with no losses.
    """

    years: IntArray
    subject_premium: FloatArray
    loss_years: IntArray
    losses: FloatArray

    def __post_init__(self) -> None:
        years = np.asarray(self.years)
        if years.ndim != 1 or years.size == 0 or not np.issubdtype(years.dtype, np.integer):
            raise ValidationError("years must be a non-empty 1-d array of integers")
        if np.unique(years).size != years.size:
            raise ValidationError("years must not repeat")
        premium = as_float_array("subject_premium", self.subject_premium, non_negative=True)
        if premium.shape != years.shape:
            raise ValidationError("subject_premium must have one value per year")
        losses = as_float_array("losses", self.losses, non_negative=True)
        loss_years = np.asarray(self.loss_years)
        if loss_years.shape != losses.shape:
            raise ValidationError("loss_years must have one value per loss")
        if not np.isin(loss_years, years).all():
            raise ValidationError("every loss year must appear in years")
        object.__setattr__(self, "years", years.astype(np.intp))
        object.__setattr__(self, "subject_premium", premium)
        object.__setattr__(self, "loss_years", loss_years.astype(np.intp))
        object.__setattr__(self, "losses", losses)

    @classmethod
    def from_arrays(
        cls, years: ArrayLike, subject_premium: ArrayLike, loss_years: ArrayLike, losses: ArrayLike
    ) -> ExperienceData:
        return cls(
            np.asarray(years),
            np.asarray(subject_premium),
            np.asarray(loss_years),
            np.asarray(losses),
        )

    def year_position(self) -> IntArray:
        """Position of each loss's year within :attr:`years` (an index usable with bincount)."""
        lookup = {int(y): i for i, y in enumerate(self.years)}
        return np.array([lookup[int(y)] for y in self.loss_years], dtype=np.intp)
