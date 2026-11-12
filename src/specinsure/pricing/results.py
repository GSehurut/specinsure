"""Output of simulation-based technical pricing."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .inputs import Loadings


@dataclass(frozen=True)
class PricingResult:
    """Technical price of a layer and the components that build it.

    ``technical_premium`` is the deposit (upfront) premium. Expected
    reinstatement premium income is ``technical_premium *
    expected_reinstatement_factor``.
    """

    expected_loss: float
    standard_deviation: float
    capital: float
    capital_charge: float
    expected_reinstatement_factor: float
    technical_premium: float
    limit: float | None
    loadings: Loadings
    n_years: int

    @property
    def rate_on_line(self) -> float | None:
        return None if not self.limit else self.technical_premium / self.limit

    @property
    def loss_on_line(self) -> float | None:
        return None if not self.limit else self.expected_loss / self.limit

    @property
    def expected_reinstatement_premium(self) -> float:
        return self.technical_premium * self.expected_reinstatement_factor

    def summary(self) -> dict[str, float | None]:
        out = {k: v for k, v in asdict(self).items() if k != "loadings"}
        out.update(
            rate_on_line=self.rate_on_line,
            loss_on_line=self.loss_on_line,
            expected_reinstatement_premium=self.expected_reinstatement_premium,
        )
        return out
