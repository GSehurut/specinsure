import numpy as np
from abc import ABC, abstractmethod

class ReinsuranceTreaty(ABC):
    """
    Abstract base class for reinsurance treaties.
    All treaty subclasses must implement the apply method.
    """

    @abstractmethod
    def apply(self, gross_losses: np.ndarray) -> np.ndarray:
        """
        Apply the treaty logic to the array of gross losses.

        Parameters:
            gross_losses (np.ndarray): Array of gross losses

        Returns:
            np.ndarray: Array of reinsurance recoveries (ceded losses)
        """
        pass

    def net_retained(self, gross_losses: np.ndarray) -> np.ndarray:
        """
        Calculate net retained losses after reinsurance recovery.

        Parameters:
            gross_losses (np.ndarray): Array of gross losses

        Returns:
            np.ndarray: Array of net retained losses
        """
        ceded = self.apply(gross_losses)
        return gross_losses - ceded

    def summary(self, gross_losses: np.ndarray) -> dict:
        """
        Compute summary statistics of gross, ceded, and net retained losses.

        Returns a dictionary with:
        - mean, std dev for gross, ceded, net retained
        - VaR 95%, 99% for gross losses
        - TVaR 95%, 99% for gross losses
        - reinsurer share (mean ceded / mean gross)
        """
        ceded = self.apply(gross_losses)
        net = gross_losses - ceded

        def var(losses, level):
            return np.percentile(losses, level * 100)

        def tvar(losses, level):
            threshold = var(losses, level)
            tail_losses = losses[losses >= threshold]
            return tail_losses.mean() if len(tail_losses) > 0 else 0

        summary_dict = {
            'Mean Gross Loss': np.mean(gross_losses),
            'Std Gross Loss': np.std(gross_losses),

            'Mean Ceded Loss': np.mean(ceded),
            'Std Ceded Loss': np.std(ceded),

            'Mean Net Retained': np.mean(net),
            'Std Net Retained': np.std(net),

            'VaR 95% Gross': var(gross_losses, 0.95),
            'VaR 99% Gross': var(gross_losses, 0.99),

            'TVaR 95% Gross': tvar(gross_losses, 0.95),
            'TVaR 99% Gross': tvar(gross_losses, 0.99),

            'Reinsurer Share': np.mean(ceded) / np.mean(gross_losses) if np.mean(gross_losses) > 0 else 0,
        }

        return summary_dict
