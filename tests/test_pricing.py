import numpy as np
import pytest

from specinsure.core import ValidationError
from specinsure.distributions import lognormal, poisson
from specinsure.portfolio import RiskProfile
from specinsure.pricing import (
    ExperienceData,
    Loadings,
    MBBEFDCurve,
    blend,
    burning_cost,
    credibility_weight,
    exposure_rating,
    on_level_factors,
    technical_price,
    trend_losses,
)
from specinsure.reinsurance import ExcessOfLoss, QuotaShare
from specinsure.simulation import FrequencySeverityModel, simulate


def test_trend_and_on_level():
    np.testing.assert_allclose(trend_losses([100.0], [2020], 2022, 0.1), [121.0])
    np.testing.assert_allclose(
        on_level_factors([2020, 2022], 2022, {2021: 0.1, 2022: -0.05}), [1.045, 1.0]
    )


def test_burning_cost_trends_losses_before_applying_layer():
    exp = ExperienceData.from_arrays(
        years=[2020, 2021],
        subject_premium=[100.0, 100.0],
        loss_years=[2020, 2021],
        losses=[90.0, 50.0],
    )
    layer = ExcessOfLoss(limit=50, attachment=100)
    untrended = burning_cost(exp, layer, target_year=2022)
    assert untrended.loss_cost_rate == 0.0
    trended = burning_cost(exp, layer, target_year=2022, claims_inflation=0.1)
    # 90 * 1.21 = 108.9 -> 8.9 into the layer; 50 * 1.1 = 55 stays below
    assert trended.loss_cost_rate == pytest.approx(8.9 / 200)
    assert trended.n_claims == 1


def test_burning_cost_applies_annual_terms_per_year():
    exp = ExperienceData.from_arrays(
        [2020, 2021], [10.0, 10.0], [2020, 2020, 2021], [10.0, 10.0, 10.0]
    )
    layer = ExcessOfLoss(limit=10, attachment=0, reinstatements=0)
    assert burning_cost(exp, layer, 2021).loss_cost_rate == pytest.approx(20 / 20)


def test_experience_data_validation():
    with pytest.raises(ValidationError):
        ExperienceData.from_arrays([2020], [1.0], [2019], [1.0])


def test_mbbefd_curve_properties():
    for c in (0.0, 1.5, 3.0, 5.0):
        curve = MBBEFDCurve.swiss_re(c)
        assert float(curve(0.0)) == pytest.approx(0.0, abs=1e-12)
        assert float(curve(1.0)) == pytest.approx(1.0)
        g = curve(np.linspace(0, 1, 101))
        assert (np.diff(g) >= -1e-12).all()
        assert (np.diff(g, 2) <= 1e-9).all()  # concave
    # larger c (bigger risks) puts less of the loss near total loss
    assert float(MBBEFDCurve.swiss_re(5)(0.2)) > float(MBBEFDCurve.swiss_re(2)(0.2))


def test_exposure_rating_full_layer_recovers_elr():
    profile = RiskProfile.from_bands([0, 1e6, 2e6], [100.0, 200.0])
    whole = ExcessOfLoss(limit=1e9, attachment=0)
    est = exposure_rating(profile, whole, MBBEFDCurve.swiss_re(3), expected_loss_ratio=0.6)
    assert est.loss_cost_rate == pytest.approx(0.6)
    xs = exposure_rating(
        profile, ExcessOfLoss(limit=1e6, attachment=5e5), MBBEFDCurve.swiss_re(3), 0.6
    )
    assert 0 < xs.loss_cost_rate < 0.6


def test_credibility_and_blend():
    assert credibility_weight(1082) == 1.0
    assert credibility_weight(270.5) == pytest.approx(0.5)
    a = burning_cost(
        ExperienceData.from_arrays([2020], [100.0], [2020], [30.0]),
        ExcessOfLoss(limit=50, attachment=0),
        2020,
    )
    b = exposure_rating(
        RiskProfile.from_bands([0, 1e6], [100.0]),
        ExcessOfLoss(limit=1e9, attachment=0),
        MBBEFDCurve.swiss_re(3),
        0.5,
    )
    assert blend(a, b, 0.25).loss_cost_rate == pytest.approx(0.25 * 0.3 + 0.75 * 0.5)


def test_technical_price_formula():
    model = FrequencySeverityModel(poisson(3.0), lognormal(2e6, 1.5))
    ylt = simulate(model, 50_000, seed=2)
    layer = ExcessOfLoss(limit=5e6, attachment=5e6, reinstatements=1, reinstatement_rates=1.0)
    loads = Loadings(brokerage=0.1, internal_expenses=0.05, cost_of_capital=0.1, capital_level=0.99)
    result = technical_price(ylt, layer, loads)
    income = result.technical_premium * (1 + result.expected_reinstatement_factor) * (1 - 0.15)
    assert income == pytest.approx(result.expected_loss + 0.1 * result.capital)
    assert 0 < result.expected_reinstatement_factor < 1
    assert result.rate_on_line == pytest.approx(result.technical_premium / 5e6)
    assert result.rate_on_line > result.loss_on_line
    assert "rate_on_line" in result.summary()


def test_technical_price_without_reinstatements():
    ylt = simulate(FrequencySeverityModel(poisson(2.0), lognormal(1e6, 1.0)), 5_000, seed=1)
    result = technical_price(ylt, QuotaShare(0.5))
    assert result.technical_premium == pytest.approx(result.expected_loss)
    assert result.rate_on_line is None


def test_loadings_validated():
    with pytest.raises(ValidationError):
        Loadings(brokerage=0.6, internal_expenses=0.5)
