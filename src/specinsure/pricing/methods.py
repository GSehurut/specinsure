"""Rating methods: burning cost, exposure rating, credibility blending and simulation pricing."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from ..core.base import ValidationError
from ..core.types import ArrayLike, FloatArray
from ..core.validation import check_fraction, check_non_negative
from ..portfolio.exposure import RiskProfile
from ..reinsurance.base import Treaty
from ..reinsurance.calculations import reinstatement_premium_factors
from ..reinsurance.treaties import ExcessOfLoss
from ..risk.measures import standard_deviation, tvar, var
from .adjustments import develop_losses, on_level_factors, trend_losses
from .base import LossCostEstimate
from .inputs import ExperienceData, Loadings
from .results import PricingResult

# --- Experience rating ----------------------------------------------------------------


def burning_cost(
    experience: ExperienceData,
    layer: Treaty,
    target_year: int,
    *,
    claims_inflation: float = 0.0,
    rate_changes: Mapping[int, float] | None = None,
    development_factors: Mapping[int, float] | None = None,
) -> LossCostEstimate:
    """Burning cost rate of ``layer`` on subject premium.

    Steps, in order: develop losses to ultimate, trend them to
    ``target_year``, apply the layer (annual terms year by year), and divide
    total layer losses by the on-levelled subject premium. The result is a
    volume-weighted average over the experience period.
    """
    developed = develop_losses(experience.losses, experience.loss_years, development_factors or {})
    trended = trend_losses(developed, experience.loss_years, target_year, claims_inflation)
    position = experience.year_position()
    ceded = layer.apply(trended, position)
    n_years = experience.years.size
    layer_by_year = np.bincount(position, weights=ceded, minlength=n_years)
    premium = experience.subject_premium * on_level_factors(
        experience.years, target_year, rate_changes or {}
    )
    if premium.sum() <= 0:
        raise ValidationError("on-levelled subject premium is zero")
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_by_year = np.where(premium > 0, layer_by_year / premium, np.nan)
    return LossCostEstimate(
        method="burning_cost",
        loss_cost_rate=float(layer_by_year.sum() / premium.sum()),
        n_claims=int(np.count_nonzero(ceded)),
        details={
            "years": experience.years.copy(),
            "layer_losses": layer_by_year,
            "on_level_premium": premium,
            "rate_by_year": rate_by_year,
        },
    )


# --- Exposure rating ------------------------------------------------------------------


class MBBEFDCurve:
    """MBBEFD exposure curve G(x) of Bernegger (1997).

    ``x`` is the loss as a share of the sum insured (or PML); ``G(x)`` is the
    share of expected loss below that point. Build from the parameters ``b``
    and ``g`` directly or from the one-parameter Swiss Re ``c`` family with
    :meth:`swiss_re` (c = 1.5, 2, 3, 4, 5 for increasingly large risks; c = 5
    is close to the Lloyd's industrial curve).
    """

    def __init__(self, b: float, g: float) -> None:
        if b <= 0 or g < 1:
            raise ValidationError(f"MBBEFD needs b > 0 and g >= 1, got b={b}, g={g}")
        self.b, self.g = float(b), float(g)

    @classmethod
    def swiss_re(cls, c: float) -> MBBEFDCurve:
        check_non_negative("c", c)
        return cls(b=math.exp(3.1 - 0.15 * (1 + c) * c), g=math.exp((0.78 + 0.12 * c) * c))

    def __call__(self, x: ArrayLike) -> FloatArray:
        x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
        b, g = self.b, self.g
        if math.isclose(g, 1.0):
            return x
        if math.isclose(b, 1.0):
            return np.log1p((g - 1) * x) / math.log(g)
        if math.isclose(b * g, 1.0):
            return (1 - b**x) / (1 - b)
        return np.log(((g - 1) * b + (1 - g * b) * b**x) / (1 - b)) / math.log(g * b)

    @property
    def total_loss_probability(self) -> float:
        """P(loss = sum insured) = 1 / g."""
        return 1 / self.g

    def __repr__(self) -> str:
        return f"MBBEFDCurve(b={self.b:.4g}, g={self.g:.4g})"


def exposure_rating(
    profile: RiskProfile, layer: ExcessOfLoss, curve: MBBEFDCurve, expected_loss_ratio: float
) -> LossCostEstimate:
    """Per-risk XoL loss cost from a risk profile and an exposure curve.

    For each band with representative sum insured SI and premium P:
    layer loss = P x ELR x (G(min((a + l) / SI, 1)) - G(min(a / SI, 1))).
    Annual terms on the layer (AAD, reinstatements) are ignored by this method.
    """
    check_fraction("expected_loss_ratio", expected_loss_ratio, allow_one=True)
    si = profile.sums_insured
    with np.errstate(divide="ignore"):
        lower = np.where(si > 0, layer.attachment / si, np.inf)
        upper = np.where(si > 0, (layer.attachment + layer.limit) / si, np.inf)
    share = curve(np.minimum(upper, 1.0)) - curve(np.minimum(lower, 1.0))
    layer_loss = profile.premium * expected_loss_ratio * share
    return LossCostEstimate(
        method="exposure_rating",
        loss_cost_rate=float(layer_loss.sum() / profile.total_premium),
        details={"layer_loss_by_band": layer_loss, "share_by_band": share},
    )


def blend(
    experience: LossCostEstimate, exposure: LossCostEstimate, credibility: float
) -> LossCostEstimate:
    """Credibility-weighted rate: Z x experience + (1 - Z) x exposure."""
    z = check_fraction("credibility", credibility)
    return LossCostEstimate(
        method="blend",
        loss_cost_rate=z * experience.loss_cost_rate + (1 - z) * exposure.loss_cost_rate,
        n_claims=experience.n_claims,
        details={"credibility": z, "experience": experience, "exposure": exposure},
    )


# --- Simulation pricing ---------------------------------------------------------------


def technical_price(ylt, layer: Treaty, loadings: Loadings | None = None) -> PricingResult:
    """Technical deposit premium for ``layer`` from a simulated year loss table.

    With income I = P (1 + E[f]), where f is the reinstatement premium factor
    (zero for treaties without paid reinstatements), the premium P solves

        I (1 - brokerage - expenses - profit) = E[L] + CoC x capital

    so the proportional loads apply to reinstatement premiums too. Capital is
    the stand-alone layer measure at ``loadings.capital_level`` minus E[L];
    for a layer inside a portfolio, marginal capital is the better measure
    (see :mod:`specinsure.optimisation`).
    """
    loadings = loadings or Loadings()
    annual = ylt.annual_totals(ylt.ceded(layer))
    expected = float(annual.mean())
    measure = tvar if loadings.capital_measure == "tvar" else var
    capital = max(0.0, measure(annual, loadings.capital_level) - expected)
    capital_charge = loadings.cost_of_capital * capital

    if isinstance(layer, ExcessOfLoss):
        factor = float(reinstatement_premium_factors(layer, annual).mean())
        limit: float | None = layer.limit
    else:
        factor, limit = 0.0, None

    premium = (expected + capital_charge) / ((1 + factor) * (1 - loadings.premium_loads))
    return PricingResult(
        expected_loss=expected,
        standard_deviation=standard_deviation(annual),
        capital=capital,
        capital_charge=capital_charge,
        expected_reinstatement_factor=factor,
        technical_premium=premium,
        limit=limit,
        loadings=loadings,
        n_years=ylt.n_years,
    )
