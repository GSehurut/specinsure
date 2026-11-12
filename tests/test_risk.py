import numpy as np
import pytest

from specinsure.core import ValidationError
from specinsure.risk import (
    ExceedanceCurve,
    aal,
    aal_confidence_interval,
    capital_requirement,
    diversification_benefit,
    empirical_return_period,
    euler_tvar_allocation,
    level_for_return_period,
    return_period_for_level,
    return_period_loss,
    return_period_table,
    standard_error,
    tvar,
    var,
)

X = np.arange(1, 1001, dtype=float)


def test_basic_measures():
    assert aal(X) == pytest.approx(500.5)
    assert var(X, 0.99) == pytest.approx(np.quantile(X, 0.99))
    assert standard_error(X) == pytest.approx(X.std(ddof=1) / np.sqrt(1000))
    lo, hi = aal_confidence_interval(X, 0.95)
    assert lo < 500.5 < hi


def test_tvar_is_mean_of_worst_tail():
    # worst 1% of 1..1000 is 991..1000
    assert tvar(X, 0.99) == pytest.approx(995.5)
    # ceil(1000 * 0.005) = 5 scenarios, robust to float error in 1 - level
    assert tvar(X, 0.995) == pytest.approx(998.0)


def test_tvar_with_ties_uses_exact_count():
    x = np.array([0.0] * 98 + [10.0, 10.0])
    assert tvar(x, 0.99) == 10.0
    assert tvar(x, 0.97) == pytest.approx(20 / 3)


def test_return_periods():
    assert level_for_return_period(200) == pytest.approx(0.995)
    assert return_period_for_level(0.995) == pytest.approx(200)
    assert return_period_loss(X, 10) == pytest.approx(np.quantile(X, 0.9))
    assert empirical_return_period(X, 1000) == pytest.approx(1000)
    assert empirical_return_period(X, 5000) == float("inf")
    assert list(return_period_table(X, (10, 100))) == [10, 100]
    with pytest.raises(ValidationError):
        return_period_loss(X, 1)


def test_capital_requirement():
    assert capital_requirement(X) == pytest.approx(var(X, 0.995))
    assert capital_requirement(X, relative_to_mean=True) == pytest.approx(var(X, 0.995) - 500.5)


def test_exceedance_curve():
    curve = ExceedanceCurve.from_annual(X, "AEP")
    assert curve.losses[0] == 1000 and curve.probabilities[-1] == pytest.approx(1.0)
    assert curve.loss_at(100) == pytest.approx(991.0)
    assert curve.exceedance_probability(991) == pytest.approx(0.01)
    with pytest.raises(ValidationError):
        ExceedanceCurve.from_annual(X, "XEP")


def test_euler_allocation_adds_up_to_portfolio_tvar(rng):
    m = rng.lognormal(0, 1, size=(10_000, 3))
    alloc = euler_tvar_allocation(m, 0.99)
    assert alloc.sum() == pytest.approx(tvar(m.sum(axis=1), 0.99))
    assert (alloc <= np.array([tvar(m[:, j], 0.99) for j in range(3)]) + 1e-9).all()


def test_tvar_diversification_is_non_negative(rng):
    m = rng.lognormal(0, 1, size=(5_000, 2))
    assert diversification_benefit(m, lambda x: tvar(x, 0.99)) >= 0
