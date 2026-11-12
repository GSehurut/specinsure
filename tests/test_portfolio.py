import numpy as np
import pytest

from specinsure.core import ValidationError
from specinsure.distributions import lognormal, poisson
from specinsure.portfolio import LineOfBusiness, Portfolio, RiskProfile
from specinsure.reinsurance import ExcessOfLoss, Programme
from specinsure.simulation import FrequencySeverityModel


def make_portfolio():
    prop = LineOfBusiness(
        "property",
        FrequencySeverityModel(poisson(4.0), lognormal(1e6, 2.0)),
        premium=6e6,
        expense_ratio=0.3,
        programme=Programme(ExcessOfLoss(limit=5e6, attachment=2e6)),
        reinsurance_premium=5e5,
    )
    marine = LineOfBusiness(
        "marine", FrequencySeverityModel(poisson(2.0), lognormal(8e5, 1.5)), premium=2.5e6
    )
    return Portfolio([prop, marine])


def test_line_renames_model_to_line_name():
    line = make_portfolio().lines[0]
    assert line.model.name == "property"


def test_portfolio_results_reconcile():
    results = make_portfolio().simulate(5_000, seed=4, correlation=[[1, 0.5], [0.5, 1]])
    assert results.gross_losses.shape == (5_000, 2)
    np.testing.assert_allclose(results.net_losses + results.ceded_losses, results.gross_losses)
    assert (results.ceded_losses[:, 1] == 0).all()
    gross = results.underwriting_results(net=False)
    net = results.underwriting_results(net=True)
    np.testing.assert_allclose(net[:, 0] - gross[:, 0], results.ceded_losses[:, 0] - 5e5)
    np.testing.assert_allclose(results.total_result(), net.sum(axis=1))


def test_portfolio_validation():
    line = make_portfolio().lines[1]
    with pytest.raises(ValidationError):
        Portfolio([line, line])
    with pytest.raises(ValidationError):
        LineOfBusiness("x", line.model, premium=0)


def test_risk_profile():
    p = RiskProfile.from_bands([0, 1e6, 5e6], [10.0, 20.0])
    np.testing.assert_allclose(p.sums_insured, [5e5, 3e6])
    assert p.total_premium == 30.0
    up = RiskProfile.from_bands([0, 1e6, 5e6], [10.0, 20.0], representative="upper")
    np.testing.assert_allclose(up.sums_insured, [1e6, 5e6])
    with pytest.raises(ValidationError):
        RiskProfile([1.0], [1.0], [1.0])
