"""Argument validation helpers. All raise :class:`ValidationError` with the argument name."""

from __future__ import annotations

import math

import numpy as np

from .base import ValidationError
from .types import ArrayLike, FloatArray, IntArray


def check_positive(name: str, value: float) -> float:
    value = float(value)
    if not value > 0 or not math.isfinite(value):
        raise ValidationError(f"{name} must be positive and finite, got {value}")
    return value


def check_non_negative(name: str, value: float) -> float:
    value = float(value)
    if not value >= 0 or not math.isfinite(value):
        raise ValidationError(f"{name} must be non-negative and finite, got {value}")
    return value


def check_fraction(
    name: str, value: float, *, allow_zero: bool = True, allow_one: bool = True
) -> float:
    """Check ``value`` lies in [0, 1], optionally excluding either end."""
    value = float(value)
    low_ok = value >= 0 if allow_zero else value > 0
    high_ok = value <= 1 if allow_one else value < 1
    if not (low_ok and high_ok):
        lo, hi = "[" if allow_zero else "(", "]" if allow_one else ")"
        raise ValidationError(f"{name} must be in {lo}0, 1{hi}, got {value}")
    return value


def check_level(level: float) -> float:
    """Check a confidence level lies strictly between 0 and 1 (e.g. 0.995)."""
    return check_fraction("level", level, allow_zero=False, allow_one=False)


def as_float_array(name: str, values: ArrayLike, *, non_negative: bool = False) -> FloatArray:
    """Convert to a 1-d float array, rejecting NaN/inf and (optionally) negatives."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValidationError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValidationError(f"{name} contains NaN or infinite values")
    if non_negative and arr.size and arr.min() < 0:
        raise ValidationError(f"{name} must be non-negative")
    return arr


def as_year_index(year_index: ArrayLike | None, n_losses: int) -> IntArray:
    """Validate a per-loss simulated-year index; ``None`` means all losses are in one year."""
    if year_index is None:
        return np.zeros(n_losses, dtype=np.intp)
    years = np.asarray(year_index)
    if years.shape != (n_losses,):
        raise ValidationError(
            f"year_index must have shape ({n_losses},) to match the losses, got {years.shape}"
        )
    if years.size and not np.issubdtype(years.dtype, np.integer):
        raise ValidationError("year_index must be integer valued")
    if years.size and years.min() < 0:
        raise ValidationError("year_index must be non-negative")
    return years.astype(np.intp, copy=False)
