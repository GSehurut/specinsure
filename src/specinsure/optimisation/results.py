"""Result objects returned by the optimisers."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.types import FloatArray


@dataclass(frozen=True)
class MixResult:
    """Optimal line weights from the mean-TVaR programme.

    ``weights`` scale each line's simulated results (1 = plan volume).
    ``expected_result`` is profit-positive; ``tvar`` and ``var`` are measured
    on losses (``-result``) at ``level``. ``at_lower`` / ``at_upper`` flag the
    lines whose bound is binding: the levers to discuss with underwriters.
    """

    weights: FloatArray
    expected_result: float
    tvar: float
    var: float
    level: float
    at_lower: FloatArray
    at_upper: FloatArray
    message: str

    @property
    def return_on_capital(self) -> float:
        """Expected result divided by TVaR capital (``nan`` when TVaR is not positive)."""
        return self.expected_result / self.tvar if self.tvar > 0 else float("nan")


@dataclass(frozen=True)
class ProgrammeEvaluation:
    """Cost and capital effect of one candidate reinsurance programme.

    ``margin`` = premium - expected recovery, the expected cost of the cover.
    ``capital`` is the risk measure of annual losses minus their mean
    (unexpected loss), so deterministic premium does not enter it.
    ``value`` = cost of capital x capital released - margin; positive values
    create value on a cost-of-capital basis.
    """

    name: str
    premium: float
    expected_recovery: float
    margin: float
    capital: float
    capital_released: float
    capital_cost_saved: float
    value: float
