"""Base class shared by every reinsurance treaty."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ReinsuranceTreaty(ABC):
    """Abstract base class for reinsurance treaties.

    Treaties work on an array of individual (ground-up) losses. When the array
    holds losses from many simulated years, pass ``year_index`` (the year each
    loss belongs to) so that annual terms, such as aggregate limits, are applied
    year by year rather than across the whole simulation.
    """

    @abstractmethod
    def apply(self, gross_losses: np.ndarray, year_index: np.ndarray | None = None) -> np.ndarray:
        """Return the ceded (reinsured) amount for each loss.

        Parameters
        ----------
        gross_losses:
            Individual gross losses.
        year_index:
            Optional integer array, same length as ``gross_losses``, giving the
            simulated year of each loss. Treaties with annual terms need it;
            per-loss treaties ignore it.
        """

    def net_retained(
        self, gross_losses: np.ndarray, year_index: np.ndarray | None = None
    ) -> np.ndarray:
        """Return the retained amount for each loss (gross minus ceded)."""
        gross_losses = np.asarray(gross_losses, dtype=float)
        return gross_losses - self.apply(gross_losses, year_index)

    def summary(
        self, gross_losses: np.ndarray, year_index: np.ndarray | None = None
    ) -> dict[str, float]:
        """Mean and standard deviation of gross, ceded and net losses, plus reinsurer share."""
        gross_losses = np.asarray(gross_losses, dtype=float)
        if gross_losses.size == 0:
            raise ValueError("gross_losses is empty")
        ceded = self.apply(gross_losses, year_index)
        net = gross_losses - ceded
        mean_gross = float(np.mean(gross_losses))
        return {
            "mean_gross": mean_gross,
            "std_gross": float(np.std(gross_losses)),
            "mean_ceded": float(np.mean(ceded)),
            "std_ceded": float(np.std(ceded)),
            "mean_net": float(np.mean(net)),
            "std_net": float(np.std(net)),
            "reinsurer_share": float(np.mean(ceded) / mean_gross) if mean_gross > 0 else 0.0,
        }


def _validate_layer(attachment: float, limit: float) -> None:
    if attachment < 0:
        raise ValueError(f"attachment must be >= 0, got {attachment}")
    if limit <= 0:
        raise ValueError(f"limit must be > 0, got {limit}")
