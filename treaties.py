# specialtyre/treaties/base.py

from abc import ABC, abstractmethod
import numpy as np

class ReinsuranceTreaty(ABC):
    """
    Abstract base class for all reinsurance treaty types.
    """

    @abstractmethod
    def apply(self, gross_losses: np.ndarray) -> np.ndarray:
        """
        Return the reinsured (ceded) loss amounts.
        """
        pass

    def net_retained(self, gross_losses: np.ndarray) -> np.ndarray:
        """
        Return the insurer’s retained losses (gross - ceded).
        """
        return gross_losses - self.apply(gross_losses)

    def summary(self, gross_losses: np.ndarray) -> dict:
        """
        Return summary metrics: averages and reinsurer share.
        """
        reinsured = self.apply(gross_losses)
        retained = gross_losses - reinsured
        return {
            "Average Gross Loss": np.mean(gross_losses),
            "Average Reinsured Recovery": np.mean(reinsured),
            "Average Net Retention": np.mean(retained),
            "Reinsurer Share": np.mean(reinsured) / np.mean(gross_losses)
        }


class ExcessOfLossTreaty(ReinsuranceTreaty):
    """
    Per-occurrence Excess-of-Loss treaty: reinsurer pays (loss - attachment) up to limit.
    """
    def __init__(self, attachment: float, limit: float):
        self.attachment = attachment
        self.limit = limit

    def apply(self, gross_losses: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(gross_losses - self.attachment, 0), self.limit)

    def __repr__(self):
        return f"<ExcessOfLossTreaty: ${self.limit:,.0f} xs ${self.attachment:,.0f}>"


class QuotaShareTreaty(ReinsuranceTreaty):
    """
    Quota Share treaty: reinsurer takes a fixed % of all losses.
    """
    def __init__(self, quota: float):
        assert 0 <= quota <= 1, "Quota must be between 0 and 1"
        self.quota = quota

    def apply(self, gross_losses: np.ndarray) -> np.ndarray:
        return gross_losses * self.quota

    def __repr__(self):
        return f"<QuotaShareTreaty: {self.quota:.0%} quota>"


class AggregateStopLossTreaty(ReinsuranceTreaty):
    """
    Annual Aggregate Stop Loss: reinsurer pays total losses above attachment up to limit.
    Only applies across the entire portfolio (not per occurrence).
    """
    def __init__(self, attachment: float, limit: float):
        self.attachment = attachment
        self.limit = limit

    def apply(self, gross_losses: np.ndarray) -> np.ndarray:
        total_loss = gross_losses.sum()
        recovery = max(min(total_loss - self.attachment, self.limit), 0)
        ceded = np.zeros_like(gross_losses)
        if recovery > 0:
            # Spread recovery proportionally across all losses
            proportions = gross_losses / total_loss
            ceded = proportions * recovery
        return ceded

    def __repr__(self):
        return f"<AggregateStopLossTreaty: ${self.limit:,.0f} above ${self.attachment:,.0f}>"
