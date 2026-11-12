import numpy as np
import pytest
from scipy import stats

from specinsure.core import ValidationError
from specinsure.distributions import lognormal, poisson
from specinsure.simulation import (
    FrequencySeverityModel,
    YearLossTable,
    simulate,
    simulate_portfolio,
)


@pytest.fixture
def model():
    return FrequencySeverityModel(poisson(5.0), lognormal(1e6, 1.5), name="property")


def test_model_rejects_swapped_distributions():
    with pytest.raises(ValidationError):
        FrequencySeverityModel(lognormal(1e6, 1.0), poisson(5.0))


def test_model_accepts_frozen_scipy():
    m = FrequencySeverityModel(stats.poisson(2), stats.lognorm(s=1, scale=1))
    assert m.expected_annual_loss() == pytest.approx(2 * np.exp(0.5))


def test_simulation_is_reproducible_and_consistent(model):
    a = simulate(model, 2_000, seed=1)
    b = simulate(model, 2_000, seed=1)
    np.testing.assert_array_equal(a.losses, b.losses)
    assert a.annual_totals().sum() == pytest.approx(a.losses.sum())
    assert a.annual_counts().sum() == a.n_losses
    assert (a.annual_max() <= a.annual_totals() + 1e-9).all()
    assert a.name == "property"


def test_simulated_mean_matches_compound_poisson(model):
    ylt = simulate(model, 200_000, seed=3)
    annual = ylt.annual_totals()
    se = annual.std() / np.sqrt(annual.size)
    assert abs(annual.mean() - model.expected_annual_loss()) < 4 * se


def test_simulated_layer_loss_matches_closed_form(model):
    from specinsure.reinsurance import ExcessOfLoss

    ylt = simulate(model, 200_000, seed=5)
    layer = ExcessOfLoss(limit=4e6, attachment=2e6)
    annual = ylt.annual_totals(ylt.ceded(layer))
    se = annual.std() / np.sqrt(annual.size)
    assert abs(annual.mean() - model.expected_layer_loss(2e6, 4e6)) < 4 * se


def test_year_loss_table_validates_inputs():
    with pytest.raises(ValidationError):
        YearLossTable(np.array([1.0]), np.array([3]), n_years=2)
    with pytest.raises(ValidationError):
        YearLossTable(np.array([-1.0]), np.array([0]), n_years=1)


def test_adding_a_line_does_not_change_existing_lines(model):
    other = FrequencySeverityModel(poisson(2.0), lognormal(5e5, 1.0), name="marine")
    one = simulate_portfolio([model], 1_000, seed=9)
    two = simulate_portfolio([model, other], 1_000, seed=9)
    np.testing.assert_array_equal(one["property"].losses, two["property"].losses)


def test_copula_keeps_marginals_and_induces_dependence(model):
    other = FrequencySeverityModel(poisson(5.0), lognormal(1e6, 1.5), name="energy")
    indep = simulate_portfolio([model, other], 20_000, seed=11)
    corr = simulate_portfolio([model, other], 20_000, seed=11, correlation=[[1, 0.8], [0.8, 1]])
    for name in ("property", "energy"):
        np.testing.assert_allclose(
            np.sort(indep[name].annual_totals()), np.sort(corr[name].annual_totals())
        )
    rho_indep = stats.spearmanr(*indep.annual_matrix().T).statistic
    rho_corr = stats.spearmanr(*corr.annual_matrix().T).statistic
    assert abs(rho_indep) < 0.05
    assert rho_corr > 0.6
    assert corr.annual_total().shape == (20_000,)


def test_t_copula_runs_and_rejects_bad_correlation(model):
    other = FrequencySeverityModel(poisson(1.0), lognormal(1e6, 1.0), name="cyber")
    simulate_portfolio([model, other], 500, seed=1, correlation=[[1, 0.3], [0.3, 1]], copula="t")
    with pytest.raises(ValidationError):
        simulate_portfolio([model, other], 500, seed=1, correlation=[[1, 2], [2, 1]])


def test_portfolio_names_must_be_unique(model):
    with pytest.raises(ValidationError):
        simulate_portfolio([model, model], 10, seed=1)
