"""Loss models that the simulation engine knows how to sample."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.base import ValidationError
from ..distributions.base import Distribution
from ..distributions.scipy import as_distribution


@dataclass(frozen=True)
class FrequencySeverityModel:
    """Collective risk model: N losses per year, each drawn independently from the severity.

    ``frequency`` and ``severity`` may be :class:`~specinsure.distributions.Distribution`
    objects or frozen ``scipy.stats`` distributions.
    """

    frequency: Distribution
    severity: Distribution
    name: str = "model"

    def __post_init__(self) -> None:
        frequency = as_distribution(self.frequency)
        severity = as_distribution(self.severity)
        if not frequency.is_discrete:
            raise ValidationError("frequency must be a discrete (counting) distribution")
        if severity.is_discrete:
            raise ValidationError("severity must be a continuous distribution")
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "severity", severity)

    def expected_annual_loss(self) -> float:
        """E[S] = E[N] E[X] (Wald's identity)."""
        return self.frequency.mean() * self.severity.mean()

    def expected_layer_loss(self, attachment: float, limit: float) -> float:
        """Expected annual loss to a per-occurrence layer with unlimited reinstatements.

        E[N] (E[X ^ (a + l)] - E[X ^ a]). A closed-form check on simulated layer losses.
        """
        return self.frequency.mean() * self.severity.layer_expected_value(attachment, limit)
