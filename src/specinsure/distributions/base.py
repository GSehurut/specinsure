"""The :class:`Distribution` interface used by simulation and pricing.

Loss distributions are assumed non-negative. Wrapping every source of randomness
(scipy, empirical data, mixtures) behind one small interface lets the simulation
engine and the pricing methods stay agnostic of where a distribution came from.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
from scipy import integrate

from ..core.types import ArrayLike, FloatArray
from ..core.validation import check_non_negative


class Distribution(ABC):
    """A univariate, non-negative distribution."""

    #: True for counting distributions (frequency); False for loss sizes.
    is_discrete: bool = False

    @abstractmethod
    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``size`` independent values using ``rng``."""

    @abstractmethod
    def mean(self) -> float:
        """Expected value (may be ``inf`` for very heavy tails)."""

    @abstractmethod
    def cdf(self, x: ArrayLike) -> FloatArray:
        """Cumulative distribution function F(x) = P(X <= x)."""

    @abstractmethod
    def ppf(self, q: ArrayLike) -> FloatArray:
        """Quantile function, the inverse of :meth:`cdf`."""

    def sf(self, x: ArrayLike) -> FloatArray:
        """Survival function S(x) = P(X > x)."""
        return 1.0 - self.cdf(x)

    def limited_expected_value(self, limit: float) -> float:
        """E[min(X, limit)], computed as the integral of S(x) from 0 to ``limit``.

        See Klugman, Panjer and Willmot, *Loss Models*, section 3.1.
        Subclasses override this with a closed form where one exists.
        """
        if limit < 0:
            check_non_negative("limit", limit)
        if limit == 0:
            return 0.0
        if math.isinf(limit):
            return self.mean()
        value, _ = integrate.quad(lambda x: float(self.sf(x)), 0.0, limit, limit=200)
        return float(value)

    def layer_expected_value(self, attachment: float, limit: float) -> float:
        """Expected loss to a ``limit`` xs ``attachment`` layer per ground-up loss.

        E[min(max(X - a, 0), l)] = E[X ^ (a + l)] - E[X ^ a].
        """
        check_non_negative("attachment", attachment)
        return self.limited_expected_value(attachment + limit) - self.limited_expected_value(
            attachment
        )
