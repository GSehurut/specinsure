import numpy as np
import pytest

from specinsure.core import ConvergenceError, ValidationError
from specinsure.distributions import lognormal, poisson
from specinsure.optimisation import efficient_frontier, evaluate_programmes, optimise_mix, xol_grid
from specinsure.reinsurance import ExcessOfLoss, Programme
from specinsure.risk import tvar
from specinsure.simulation import FrequencySeverityModel, simulate


def test_two_line_hand_case():
    # Line A always earns 1. Line B earns 3 except in the worst 10% of scenarios, where it loses 7.
    n = 100
    a = np.ones(n)
    b = np.where(np.arange(n) < 10, -7.0, 3.0)
    r = np.column_stack([a, b])
    # Max return with TVaR_90 <= 0: TVaR = -(w_a) + 7 w_b <= 0
    res = optimise_mix(r, 0.9, tvar_budget=0.0)
    np.testing.assert_allclose(res.weights, [1.0, 1 / 7], atol=1e-7)
    assert res.tvar == pytest.approx(0.0, abs=1e-9)
    assert res.at_upper.tolist() == [True, False]


def test_lp_tvar_matches_empirical_tvar(rng):
    r = rng.normal([1.0, 2.0, 0.5], [1.0, 4.0, 0.5], size=(2_000, 3))
    res = optimise_mix(r, 0.95, target_return=1.5)
    assert res.expected_result == pytest.approx(1.5, abs=1e-6)
    assert res.tvar == pytest.approx(tvar(-(r @ res.weights), 0.95), rel=1e-6)


def test_frontier_is_monotone(rng):
    r = rng.normal([1.0, 2.0, 0.5], [1.0, 4.0, 0.5], size=(1_000, 3))
    frontier = efficient_frontier(r, 0.99, n_points=6)
    returns = [p.expected_result for p in frontier]
    risks = [p.tvar for p in frontier]
    assert returns == sorted(returns)
    assert all(b >= a - 1e-6 for a, b in zip(risks, risks[1:], strict=False))


def test_infeasible_and_bad_arguments():
    r = np.column_stack([np.ones(10), np.ones(10)])
    with pytest.raises(ConvergenceError):
        optimise_mix(r, 0.9, target_return=100.0)
    with pytest.raises(ValidationError):
        optimise_mix(r, 0.9)
    with pytest.raises(ValidationError):
        optimise_mix(r, 0.9, target_return=1.0, bounds=(1.0, 0.0))


def test_evaluate_programmes_identifies_value_and_baseline():
    ylt = simulate(FrequencySeverityModel(poisson(3.0), lognormal(2e6, 2.0)), 20_000, seed=8)
    candidates = xol_grid([5e6, 10e6], [10e6])
    assert list(candidates) == ["10,000,000 xs 5,000,000", "10,000,000 xs 10,000,000"]

    def at_cost(result):  # priced at expected recovery: pure capital relief
        return float(result.annual_ceded().mean())

    evals = evaluate_programmes(ylt, candidates, at_cost, level=0.995, cost_of_capital=0.1)
    names = [e.name for e in evals]
    assert "no reinsurance" in names
    best = evals[0]
    assert best.name != "no reinsurance"
    assert best.margin == pytest.approx(0.0)
    assert best.capital_released > 0
    assert best.value == pytest.approx(0.1 * best.capital_released)


def test_evaluate_programmes_with_fixed_premiums_and_background():
    ylt = simulate(FrequencySeverityModel(poisson(3.0), lognormal(2e6, 2.0)), 5_000, seed=8)
    prog = {"layer": Programme(ExcessOfLoss(limit=1e7, attachment=5e6))}
    background = np.random.default_rng(0).gamma(2.0, 5e6, size=5_000)
    stand_alone = evaluate_programmes(ylt, prog, {"layer": 1e9})
    diversified = evaluate_programmes(ylt, prog, {"layer": 1e9}, other_losses=background)
    assert stand_alone[0].name == "no reinsurance"  # absurd premium destroys value
    layer_sa = next(e for e in stand_alone if e.name == "layer")
    layer_div = next(e for e in diversified if e.name == "layer")
    assert layer_sa.capital_released != layer_div.capital_released
    with pytest.raises(ValidationError):
        evaluate_programmes(ylt, prog, {"layer": 1.0}, other_losses=np.zeros(3))
