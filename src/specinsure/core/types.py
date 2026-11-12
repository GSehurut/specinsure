"""Type aliases used across the package."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]
ArrayLike = npt.ArrayLike

#: Anything accepted as a source of randomness: ``None`` (fresh entropy), an int
#: seed, a ``SeedSequence`` or an existing ``Generator``.
SeedLike = int | np.random.SeedSequence | np.random.Generator | None
