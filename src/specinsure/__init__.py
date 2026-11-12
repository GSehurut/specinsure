"""specinsure: simulate specialty insurance losses and reinsurance structures."""

from . import metrics
from .simulation import YearLossTable, simulate
from .treaties import ExcessOfLossTreaty, QuotaShareTreaty, ReinsuranceTreaty

__all__ = [
    "ExcessOfLossTreaty",
    "QuotaShareTreaty",
    "ReinsuranceTreaty",
    "YearLossTable",
    "metrics",
    "simulate",
]

__version__ = "0.1.0"
