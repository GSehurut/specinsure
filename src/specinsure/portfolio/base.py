"""Lines of business and the portfolio they make up."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from ..core.base import ValidationError
from ..core.types import ArrayLike, SeedLike
from ..core.validation import check_fraction, check_non_negative, check_positive
from ..reinsurance.programme import Programme
from ..simulation.engine import simulate_portfolio
from ..simulation.models import FrequencySeverityModel
from .aggregation import PortfolioResults


@dataclass(frozen=True)
class LineOfBusiness:
    """One class of business in the plan.

    ``premium`` is the planned gross written premium, ``expense_ratio`` the
    acquisition plus internal expenses as a share of it. ``programme`` is the
    outwards reinsurance protecting this line and ``reinsurance_premium`` its
    fixed (deposit) cost. Reinstatement premiums are not deducted.
    """

    name: str
    model: FrequencySeverityModel
    premium: float
    expense_ratio: float = 0.0
    programme: Programme | None = None
    reinsurance_premium: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("line name must not be empty")
        check_positive("premium", self.premium)
        check_fraction("expense_ratio", self.expense_ratio, allow_one=False)
        check_non_negative("reinsurance_premium", self.reinsurance_premium)
        if self.model.name != self.name:
            object.__setattr__(self, "model", replace(self.model, name=self.name))


class Portfolio:
    """A set of lines of business simulated together."""

    def __init__(self, lines: Sequence[LineOfBusiness]) -> None:
        self.lines = tuple(lines)
        if not self.lines:
            raise ValidationError("a portfolio needs at least one line")
        names = [line.name for line in self.lines]
        if len(set(names)) != len(names):
            raise ValidationError(f"line names must be unique, got {names}")

    @property
    def names(self) -> list[str]:
        return [line.name for line in self.lines]

    def simulate(
        self,
        n_years: int,
        seed: SeedLike = None,
        *,
        correlation: ArrayLike | None = None,
        copula: Literal["gaussian", "t"] = "gaussian",
        dof: float = 4.0,
    ) -> PortfolioResults:
        """Simulate every line over the same years.

        See :func:`~specinsure.simulation.simulate_portfolio` for the dependency options.
        """
        simulation = simulate_portfolio(
            [line.model for line in self.lines],
            n_years,
            seed,
            correlation=correlation,
            copula=copula,
            dof=dof,
        )
        return PortfolioResults(self.lines, simulation)

    def __repr__(self) -> str:
        return f"Portfolio({', '.join(self.names)})"
