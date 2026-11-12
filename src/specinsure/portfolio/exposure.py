"""Exposure information: risk profiles by sum insured band."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..core.base import ValidationError
from ..core.types import ArrayLike, FloatArray
from ..core.validation import as_float_array


@dataclass(frozen=True)
class RiskProfile:
    """Property-style risk profile: sum insured (or PML) bands with premium per band.

    ``lower`` and ``upper`` are band bounds; ``premium`` is the subject premium
    written in each band. The representative sum insured of a band is its
    midpoint by default, or its upper bound (a more prudent choice for
    exposure rating, since larger risks reach higher layers).
    """

    lower: FloatArray
    upper: FloatArray
    premium: FloatArray
    representative: Literal["mid", "upper"] = "mid"

    def __post_init__(self) -> None:
        lower = as_float_array("lower", self.lower, non_negative=True)
        upper = as_float_array("upper", self.upper, non_negative=True)
        premium = as_float_array("premium", self.premium, non_negative=True)
        if not (lower.shape == upper.shape == premium.shape) or lower.size == 0:
            raise ValidationError("lower, upper and premium must be non-empty and the same length")
        if (upper <= lower).any():
            raise ValidationError("each band's upper bound must exceed its lower bound")
        if self.representative not in ("mid", "upper"):
            raise ValidationError("representative must be 'mid' or 'upper'")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "premium", premium)

    @classmethod
    def from_bands(
        cls, bands: ArrayLike, premium: ArrayLike, representative: Literal["mid", "upper"] = "mid"
    ) -> RiskProfile:
        """Build from band edges ``[e0, e1, ..., en]`` and ``n`` premiums."""
        edges = as_float_array("bands", bands, non_negative=True)
        return cls(edges[:-1], edges[1:], np.asarray(premium, dtype=float), representative)

    @property
    def sums_insured(self) -> FloatArray:
        """Representative sum insured of each band."""
        return (
            self.upper.copy() if self.representative == "upper" else (self.lower + self.upper) / 2
        )

    @property
    def total_premium(self) -> float:
        return float(self.premium.sum())
