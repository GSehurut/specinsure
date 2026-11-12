"""Combine simulated lines into annual portfolio results."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from ..core.types import FloatArray
from ..simulation.results import SimulationResult

if TYPE_CHECKING:
    from .base import LineOfBusiness


@dataclass(frozen=True, eq=False)
class PortfolioResults:
    """Simulated annual losses and underwriting results for each line.

    All matrices have shape ``(n_years, n_lines)`` with columns in
    :attr:`names` order. Losses are positive; underwriting results are
    positive for profit.
    """

    lines: tuple[LineOfBusiness, ...]
    simulation: SimulationResult

    @property
    def names(self) -> list[str]:
        return [line.name for line in self.lines]

    @property
    def n_years(self) -> int:
        return self.simulation.n_years

    @cached_property
    def gross_losses(self) -> FloatArray:
        return np.column_stack([self.simulation[n].annual_totals() for n in self.names])

    @cached_property
    def ceded_losses(self) -> FloatArray:
        columns = []
        for line in self.lines:
            ylt = self.simulation[line.name]
            if line.programme is None:
                columns.append(np.zeros(self.n_years))
            else:
                columns.append(line.programme.apply_to(ylt).annual_ceded())
        return np.column_stack(columns)

    @property
    def net_losses(self) -> FloatArray:
        return self.gross_losses - self.ceded_losses

    def underwriting_results(self, *, net: bool = True) -> FloatArray:
        """Premium less expenses less losses (and reinsurance cost, when ``net``)."""
        premium = np.array([line.premium for line in self.lines])
        expenses = np.array([line.premium * line.expense_ratio for line in self.lines])
        result = premium - expenses
        if not net:
            return result - self.gross_losses
        ri_cost = np.array([line.reinsurance_premium for line in self.lines])
        return result - ri_cost - self.net_losses

    def total_result(self, *, net: bool = True) -> FloatArray:
        """Portfolio annual underwriting result."""
        return self.underwriting_results(net=net).sum(axis=1)
