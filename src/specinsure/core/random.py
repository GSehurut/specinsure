"""Random number streams.

Every stochastic function in specinsure takes an explicit seed or
``numpy.random.Generator``; nothing relies on global ``np.random`` state.
Independent components (for example each line of business) draw from their own
spawned stream, so adding a line does not change the simulated losses of the
others. That keeps comparisons between candidate portfolios on common random
numbers.
"""

from __future__ import annotations

import numpy as np

from .types import SeedLike


def make_rng(seed: SeedLike = None) -> np.random.Generator:
    """Return a ``Generator`` from a seed, ``SeedSequence`` or existing generator."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def spawn_rngs(seed: SeedLike, n: int) -> list[np.random.Generator]:
    """Return ``n`` statistically independent generators derived from ``seed``."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if isinstance(seed, np.random.Generator):
        return list(seed.spawn(n))
    if not isinstance(seed, np.random.SeedSequence):
        seed = np.random.SeedSequence(seed)
    return [np.random.default_rng(child) for child in seed.spawn(n)]
