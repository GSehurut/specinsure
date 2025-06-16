from .base import ReinsuranceTreaty
import numpy as np
from typing import Literal

class ExcessOfLossTreaty(ReinsuranceTreaty):
    """
    Excess-of-Loss (XoL) reinsurance treaty implementation.

    This treaty can be applied in two modes:
    
    - 'per_occurrence': Applies attachment and limit to each individual loss event.
    - 'aggregate': Applies attachment and limit to the aggregate sum of all losses,
      then allocates ceded amount proportionally to individual losses.

    Parameters
    ----------
    attachment : float
        The attachment point (retention) of the treaty.
    limit : float
        The maximum limit of coverage the reinsurer will pay.
    mode : Literal["per_occurrence", "aggregate"], default="per_occurrence"
        The mode of the treaty application.
    """

    def __init__(self, attachment: float, limit: float, mode: Literal["per_occurrence", "aggregate"] = "per_occurrence"):
        self.attachment = attachment
        self.limit = limit
        self.mode = mode

    def apply(self, gross_losses: np.ndarray) -> np.ndarray:
        """
        Calculate the ceded losses based on the treaty parameters and mode.

        Parameters
        ----------
        gross_losses : np.ndarray
            Array of gross losses (per loss event).

        Returns
        -------
        np.ndarray
            Array of ceded losses paid by the reinsurer.

        Raises
        ------
        ValueError
            If the mode is not one of 'per_occurrence' or 'aggregate'.
        """
        if self.mode == "per_occurrence":
            # Calculate ceded amount per individual loss
            ceded = np.minimum(np.maximum(gross_losses - self.attachment, 0), self.limit)
            return ceded

        elif self.mode == "aggregate":
            # Calculate ceded amount on aggregate loss
            total_loss = gross_losses.sum()
            ceded_total = min(max(total_loss - self.attachment, 0), self.limit)
            # Allocate ceded total proportionally to each loss
            proportion = ceded_total / total_loss if total_loss > 0 else 0
            ceded = gross_losses * proportion
            return ceded

        else:
            raise ValueError(f"Invalid mode: {self.mode}. Choose 'per_occurrence' or 'aggregate'.")