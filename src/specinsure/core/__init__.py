"""Shared foundations: exceptions, types, validation and random number streams."""

from .base import ConvergenceError, SpecinsureError, ValidationError
from .random import make_rng, spawn_rngs

__all__ = ["ConvergenceError", "SpecinsureError", "ValidationError", "make_rng", "spawn_rngs"]
