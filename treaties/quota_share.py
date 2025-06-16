import numpy as np
from base import ReinsuranceTreaty

class QuotaShareTreaty(ReinsuranceTreaty):
    """
    Quota Share Treaty - cedes a fixed proportion of every loss.

    Parameters:
    -----------
    quota : float
        Proportion of losses ceded to reinsurer (0 < quota <= 1).
    """

    def __init__(self, quota: float):
        if quota <= 0 or quota > 1:
            raise ValueError("Quota must be between 0 (exclusive) and 1 (inclusive)")
        self.quota = quota

    def apply(self, gross_losses: np.ndarray) -> np.ndarray:
        """
        Calculate reinsurance recoveries as a fixed proportion of gross losses.
        """
        ceded = gross_losses * self.quota
        return ceded
