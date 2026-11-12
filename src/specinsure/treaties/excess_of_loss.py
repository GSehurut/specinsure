"""Excess of loss treaties."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .base import ReinsuranceTreaty, _validate_layer


class ExcessOfLossTreaty(ReinsuranceTreaty):
    """Excess of loss (XoL) treaty: ``limit`` xs ``attachment``.

    Modes
    -----
    ``"per_occurrence"``
        Attachment and limit apply to each individual loss.
    ``"aggregate"``
        Attachment and limit apply to the *annual* total of losses. The ceded
        amount for each year is allocated back to that year's losses in
        proportion to their size. Pass ``year_index`` when the losses span many
        simulated years; without it, all losses are treated as one year.

    An optional ``aggregate_limit`` caps total annual recoveries on a
    per-occurrence layer (a simple way to model limited reinstatements).
    """

    def __init__(
        self,
        attachment: float,
        limit: float,
        mode: Literal["per_occurrence", "aggregate"] = "per_occurrence",
        aggregate_limit: float | None = None,
    ) -> None:
        _validate_layer(attachment, limit)
        if mode not in ("per_occurrence", "aggregate"):
            raise ValueError(f"Invalid mode: {mode!r}. Choose 'per_occurrence' or 'aggregate'.")
        if aggregate_limit is not None and aggregate_limit <= 0:
            raise ValueError(f"aggregate_limit must be > 0, got {aggregate_limit}")
        self.attachment = float(attachment)
        self.limit = float(limit)
        self.mode = mode
        self.aggregate_limit = aggregate_limit

    def apply(self, gross_losses: np.ndarray, year_index: np.ndarray | None = None) -> np.ndarray:
        gross_losses = np.asarray(gross_losses, dtype=float)
        years = _year_index_or_single(gross_losses, year_index)

        if self.mode == "per_occurrence":
            ceded = np.clip(gross_losses - self.attachment, 0.0, self.limit)
            if self.aggregate_limit is not None:
                ceded = _cap_annual(ceded, years, self.aggregate_limit)
            return ceded

        # Aggregate mode: apply the layer to each year's total, not the whole simulation
        annual_gross = np.bincount(years, weights=gross_losses)
        annual_ceded = np.clip(annual_gross - self.attachment, 0.0, self.limit)
        return _allocate(gross_losses, years, annual_gross, annual_ceded)

    def __repr__(self) -> str:
        return (
            f"ExcessOfLossTreaty({self.limit:,.0f} xs {self.attachment:,.0f}, mode={self.mode!r})"
        )


def _year_index_or_single(losses: np.ndarray, year_index: np.ndarray | None) -> np.ndarray:
    if year_index is None:
        return np.zeros(losses.shape[0], dtype=np.intp)
    years = np.asarray(year_index)
    if years.shape != losses.shape:
        raise ValueError("year_index must have the same shape as gross_losses")
    if years.size and years.min() < 0:
        raise ValueError("year_index must be non-negative")
    return years.astype(np.intp, copy=False)


def _allocate(
    losses: np.ndarray, years: np.ndarray, annual_gross: np.ndarray, annual_ceded: np.ndarray
) -> np.ndarray:
    """Spread each year's ceded amount over that year's losses pro rata."""
    safe_gross = np.where(annual_gross > 0, annual_gross, 1.0)
    ratio = np.where(annual_gross > 0, annual_ceded / safe_gross, 0.0)
    return losses * ratio[years]


def _cap_annual(ceded: np.ndarray, years: np.ndarray, cap: float) -> np.ndarray:
    annual = np.bincount(years, weights=ceded)
    return _allocate(ceded, years, annual, np.minimum(annual, cap))
