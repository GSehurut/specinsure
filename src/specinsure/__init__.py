"""specinsure: Monte Carlo pricing, capital and portfolio analysis for specialty (re)insurance.

Subpackages
-----------
core           exceptions, types, validation, random streams
distributions  frequency/severity distributions, fitting
simulation     year loss tables, multi-line simulation with copulas
risk           VaR, TVaR, return periods, AEP/OEP, capital allocation
reinsurance    quota share, XoL (AAD, reinstatements), aggregate XoL, programmes
pricing        burning cost, exposure rating, credibility, technical premium
portfolio      lines of business, risk profiles, portfolio results
optimisation   mean-TVaR line mix, reinsurance structure evaluation
reporting      summary tables and plots

The most common entry points are re-exported here.
"""

from . import (
    core,
    distributions,
    optimisation,
    portfolio,
    pricing,
    reinsurance,
    reporting,
    risk,
    simulation,
)
from .portfolio import LineOfBusiness, Portfolio
from .reinsurance import AggregateExcessOfLoss, ExcessOfLoss, Programme, QuotaShare
from .simulation import FrequencySeverityModel, YearLossTable, simulate, simulate_portfolio

__all__ = [
    "AggregateExcessOfLoss",
    "ExcessOfLoss",
    "FrequencySeverityModel",
    "LineOfBusiness",
    "Portfolio",
    "Programme",
    "QuotaShare",
    "YearLossTable",
    "core",
    "distributions",
    "optimisation",
    "portfolio",
    "pricing",
    "reinsurance",
    "reporting",
    "risk",
    "simulate",
    "simulate_portfolio",
    "simulation",
]

__version__ = "0.2.0"
