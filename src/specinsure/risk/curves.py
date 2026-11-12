"""Exceedance probability curves (AEP and OEP)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..core.base import ValidationError
from ..core.types import ArrayLike, FloatArray
from ..core.validation import as_float_array


@dataclass(frozen=True)
class ExceedanceCurve:
    """Losses (descending) and their empirical annual exceedance probabilities.

    ``kind`` is ``"AEP"`` (built from annual aggregate losses) or ``"OEP"``
    (built from the largest single loss per year).
    """

    losses: FloatArray
    probabilities: FloatArray
    kind: Literal["AEP", "OEP"]

    @classmethod
    def from_annual(cls, annual_values: ArrayLike, kind: Literal["AEP", "OEP"]) -> ExceedanceCurve:
        if kind not in ("AEP", "OEP"):
            raise ValidationError(f"kind must be 'AEP' or 'OEP', got {kind!r}")
        values = np.sort(as_float_array("annual_values", annual_values))[::-1]
        if values.size == 0:
            raise ValidationError("annual_values must not be empty")
        probs = np.arange(1, values.size + 1) / values.size
        return cls(losses=values, probabilities=probs, kind=kind)

    @property
    def return_periods(self) -> FloatArray:
        return 1 / self.probabilities

    def loss_at(self, return_period: float) -> float:
        """Loss at a return period, interpolated on the empirical curve."""
        if return_period < 1:
            raise ValidationError("return_period must be at least 1")
        # np.interp needs increasing x: use exceedance probability reversed
        return float(np.interp(1 / return_period, self.probabilities, self.losses))

    def exceedance_probability(self, loss: float) -> float:
        """Empirical annual probability that the loss is at least ``loss``."""
        return float(np.mean(self.losses >= loss))


def aep_curve(ylt, values: FloatArray | None = None) -> ExceedanceCurve:
    """Aggregate exceedance probability curve from a :class:`YearLossTable`."""
    return ExceedanceCurve.from_annual(ylt.annual_totals(values), "AEP")


def oep_curve(ylt, values: FloatArray | None = None) -> ExceedanceCurve:
    """Occurrence exceedance probability curve from a :class:`YearLossTable`."""
    return ExceedanceCurve.from_annual(ylt.annual_max(values), "OEP")
