"""Empirical distribution: resample observed or externally simulated values."""

from __future__ import annotations

import numpy as np

from ..core.base import ValidationError
from ..core.types import ArrayLike, FloatArray
from ..core.validation import as_float_array
from .base import Distribution


class Empirical(Distribution):
    """Discrete distribution placing equal weight on each observed value.

    Useful for bootstrapping historic (trended) losses or for re-using a vendor
    model's event losses as a severity distribution.
    """

    def __init__(self, values: ArrayLike) -> None:
        values = np.sort(as_float_array("values", values, non_negative=True))
        if values.size == 0:
            raise ValidationError("values must not be empty")
        self.values = values

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(self.values, size=size, replace=True)

    def mean(self) -> float:
        return float(self.values.mean())

    def cdf(self, x: ArrayLike) -> FloatArray:
        return (
            np.searchsorted(self.values, np.asarray(x, dtype=float), side="right")
            / self.values.size
        )

    def ppf(self, q: ArrayLike) -> FloatArray:
        return np.quantile(self.values, np.asarray(q, dtype=float), method="inverted_cdf")

    def limited_expected_value(self, limit: float) -> float:
        return float(np.minimum(self.values, limit).mean())

    def __repr__(self) -> str:
        return f"Empirical(n={self.values.size}, mean={self.mean():,.4g})"
