"""Exception hierarchy shared by every specinsure module.

Catch :class:`SpecinsureError` to handle any error raised deliberately by the
library. :class:`ValidationError` subclasses :class:`ValueError` so existing
``except ValueError`` handlers keep working.
"""

from __future__ import annotations


class SpecinsureError(Exception):
    """Base class for errors raised by specinsure."""


class ValidationError(SpecinsureError, ValueError):
    """An argument is outside its valid domain (negative limit, level not in (0, 1), ...)."""


class ConvergenceError(SpecinsureError, RuntimeError):
    """A numerical routine (distribution fit, optimiser) failed to converge."""
