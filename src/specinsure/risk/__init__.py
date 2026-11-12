"""Risk measures, return periods, exceedance curves and capital allocation."""

from .aggregation import diversification_benefit, euler_tvar_allocation
from .curves import ExceedanceCurve, aep_curve, oep_curve
from .measures import (
    aal,
    aal_confidence_interval,
    capital_requirement,
    standard_deviation,
    standard_error,
    tvar,
    var,
)
from .return_period import (
    STANDARD_RETURN_PERIODS,
    empirical_return_period,
    level_for_return_period,
    return_period_for_level,
    return_period_loss,
    return_period_table,
)

__all__ = [
    "STANDARD_RETURN_PERIODS",
    "ExceedanceCurve",
    "aal",
    "aal_confidence_interval",
    "aep_curve",
    "capital_requirement",
    "diversification_benefit",
    "empirical_return_period",
    "euler_tvar_allocation",
    "level_for_return_period",
    "oep_curve",
    "return_period_for_level",
    "return_period_loss",
    "return_period_table",
    "standard_deviation",
    "standard_error",
    "tvar",
    "var",
]
