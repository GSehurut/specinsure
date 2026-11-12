"""Finite mixtures, e.g. attritional plus large loss severity."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import optimize

from ..core.base import ValidationError
from ..core.types import ArrayLike, FloatArray
from .base import Distribution
from .scipy import as_distribution


class Mixture(Distribution):
    """Mixture with density sum_i w_i f_i(x). Weights must be positive and sum to 1."""

    def __init__(self, components: Sequence, weights: ArrayLike) -> None:
        self.components = [as_distribution(c) for c in components]
        w = np.asarray(weights, dtype=float)
        if w.shape != (len(self.components),) or w.size == 0:
            raise ValidationError("weights must have one entry per component")
        if (w <= 0).any() or not np.isclose(w.sum(), 1.0):
            raise ValidationError(f"weights must be positive and sum to 1, got {w.tolist()}")
        self.weights = w / w.sum()
        self.is_discrete = all(c.is_discrete for c in self.components)

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        counts = rng.multinomial(size, self.weights)
        parts = [c.sample(int(k), rng) for c, k in zip(self.components, counts, strict=False) if k]
        draws = np.concatenate(parts) if parts else np.empty(0)
        return rng.permutation(draws)

    def mean(self) -> float:
        return float(sum(w * c.mean() for w, c in zip(self.weights, self.components, strict=False)))

    def cdf(self, x: ArrayLike) -> FloatArray:
        return sum(w * c.cdf(x) for w, c in zip(self.weights, self.components, strict=False))

    def ppf(self, q: ArrayLike) -> FloatArray:
        # The mixture quantile always lies between the smallest and largest component
        # quantile at the same probability, which gives an exact root bracket.
        q_arr = np.atleast_1d(np.asarray(q, dtype=float))
        bounds = np.array([np.atleast_1d(c.ppf(q_arr)) for c in self.components])
        out = np.empty_like(q_arr)
        for i, qi in enumerate(q_arr):
            lo, hi = bounds[:, i].min(), bounds[:, i].max()
            if lo == hi or not np.isfinite(hi):
                out[i] = hi
            else:
                out[i] = optimize.brentq(lambda x, p=qi: float(self.cdf(x)) - p, lo, hi)
        return out if np.ndim(q) else out[0]

    def limited_expected_value(self, limit: float) -> float:
        return float(
            sum(
                w * c.limited_expected_value(limit)
                for w, c in zip(self.weights, self.components, strict=False)
            )
        )

    def __repr__(self) -> str:
        parts = ", ".join(
            f"{w:.3g}*{c!r}" for w, c in zip(self.weights, self.components, strict=False)
        )
        return f"Mixture({parts})"
