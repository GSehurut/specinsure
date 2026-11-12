"""Common output of the loss-cost rating methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LossCostEstimate:
    """A layer's expected loss expressed as a rate on subject premium.

    Every rating method (burning cost, exposure rating, blends) returns one,
    so estimates can be compared and credibility-weighted on the same basis.
    """

    method: str
    loss_cost_rate: float
    n_claims: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def expected_loss(self, subject_premium: float) -> float:
        """Expected layer loss for a prospective subject premium."""
        return self.loss_cost_rate * subject_premium
