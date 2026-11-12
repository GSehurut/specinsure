"""Standard treaty types: quota share, per-occurrence XoL and aggregate XoL."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..core.base import ValidationError
from ..core.types import FloatArray, IntArray
from ..core.validation import check_fraction, check_non_negative, check_positive
from .base import Treaty, allocate_pro_rata, cumulative_within_year


class QuotaShare(Treaty):
    """Cedes a fixed share ``cession`` of every loss, optionally capped per event.

    ``event_limit`` caps the ceded amount per loss, as is common on quota
    shares of catastrophe-exposed classes.
    """

    def __init__(
        self, cession: float, *, event_limit: float | None = None, name: str | None = None
    ):
        super().__init__(name)
        self.cession = check_fraction("cession", cession, allow_zero=False)
        self.event_limit = (
            None if event_limit is None else check_positive("event_limit", event_limit)
        )

    def _ceded(self, losses: FloatArray, years: IntArray) -> FloatArray:
        ceded = losses * self.cession
        if self.event_limit is not None:
            ceded = np.minimum(ceded, self.event_limit)
        return ceded

    def __repr__(self) -> str:
        cap = f", event_limit={self.event_limit:,.0f}" if self.event_limit else ""
        return f"QuotaShare({self.cession:.0%}{cap})"


class ExcessOfLoss(Treaty):
    """Per-occurrence excess of loss: ``limit`` xs ``attachment`` on each loss.

    Annual terms, applied in occurrence order within each simulated year:

    ``aad``
        Annual aggregate deductible: the first ``aad`` of layer losses in a
        year is retained.
    ``reinstatements``
        Number of reinstatements of the limit. The annual aggregate limit is
        ``limit * (1 + reinstatements)``. ``None`` means unlimited.
    ``reinstatement_rates``
        Premium rate for each reinstatement as a share of the deposit premium
        (1.0 = 100%), pro rata as to amount. One value applies to all
        reinstatements. Used by
        :func:`~specinsure.reinsurance.calculations.reinstatement_premiums`.
    ``aggregate_limit``
        Explicit annual aggregate limit, for contracts expressed that way
        rather than in reinstatements. Mutually exclusive with ``reinstatements``.
    """

    def __init__(
        self,
        *,
        limit: float,
        attachment: float,
        aad: float = 0.0,
        reinstatements: int | None = None,
        reinstatement_rates: float | Sequence[float] = 1.0,
        aggregate_limit: float | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        self.limit = check_positive("limit", limit)
        self.attachment = check_non_negative("attachment", attachment)
        self.aad = check_non_negative("aad", aad)
        if reinstatements is not None and aggregate_limit is not None:
            raise ValidationError("give either reinstatements or aggregate_limit, not both")
        if reinstatements is not None and (
            int(reinstatements) != reinstatements or reinstatements < 0
        ):
            raise ValidationError(
                f"reinstatements must be a non-negative integer, got {reinstatements}"
            )
        self.reinstatements = None if reinstatements is None else int(reinstatements)
        if aggregate_limit is not None:
            check_positive("aggregate_limit", aggregate_limit)
        self._aggregate_limit = aggregate_limit
        rates = np.atleast_1d(np.asarray(reinstatement_rates, dtype=float))
        if (rates < 0).any():
            raise ValidationError("reinstatement_rates must be non-negative")
        if self.reinstatements is not None and rates.size not in (1, self.reinstatements):
            raise ValidationError("give one reinstatement rate, or one per reinstatement")
        self.reinstatement_rates = rates

    @property
    def aggregate_limit(self) -> float:
        """Annual aggregate limit (``inf`` when unlimited)."""
        if self.reinstatements is not None:
            return self.limit * (1 + self.reinstatements)
        return float("inf") if self._aggregate_limit is None else float(self._aggregate_limit)

    def layer_losses(self, losses: FloatArray) -> FloatArray:
        """Loss to the layer before any annual terms: min(max(x - attachment, 0), limit)."""
        return np.clip(np.asarray(losses, dtype=float) - self.attachment, 0.0, self.limit)

    def _ceded(self, losses: FloatArray, years: IntArray) -> FloatArray:
        layer = self.layer_losses(losses)
        if self.aad == 0 and np.isinf(self.aggregate_limit):
            return layer
        cum_after = cumulative_within_year(layer, years)
        cum_before = cum_after - layer
        cap = self.aggregate_limit
        recovered_after = np.clip(cum_after - self.aad, 0.0, cap)
        recovered_before = np.clip(cum_before - self.aad, 0.0, cap)
        return recovered_after - recovered_before

    def __repr__(self) -> str:
        terms = []
        if self.aad:
            terms.append(f"aad={self.aad:,.0f}")
        if self.reinstatements is not None:
            terms.append(f"reinstatements={self.reinstatements}")
        elif self._aggregate_limit is not None:
            terms.append(f"aggregate_limit={self._aggregate_limit:,.0f}")
        extra = (", " + ", ".join(terms)) if terms else ""
        return f"ExcessOfLoss({self.limit:,.0f} xs {self.attachment:,.0f}{extra})"


class AggregateExcessOfLoss(Treaty):
    """Annual aggregate XoL (stop loss): ``limit`` xs ``attachment`` on each year's total.

    The ceded amount for a year is allocated back to that year's losses pro
    rata, so per-loss outputs still add up to the annual recovery.
    """

    def __init__(self, *, limit: float, attachment: float, name: str | None = None) -> None:
        super().__init__(name)
        self.limit = check_positive("limit", limit)
        self.attachment = check_non_negative("attachment", attachment)

    def _ceded(self, losses: FloatArray, years: IntArray) -> FloatArray:
        annual = np.bincount(years, weights=losses)
        annual_ceded = np.clip(annual - self.attachment, 0.0, self.limit)
        return allocate_pro_rata(losses, years, annual, annual_ceded)

    def __repr__(self) -> str:
        return f"AggregateExcessOfLoss({self.limit:,.0f} xs {self.attachment:,.0f})"
