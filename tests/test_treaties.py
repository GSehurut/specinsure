import numpy as np
import pytest
from scipy import stats

from specinsure import ExcessOfLossTreaty, QuotaShareTreaty, metrics, simulate


def test_xol_per_occurrence():
    treaty = ExcessOfLossTreaty(attachment=2.0, limit=5.0)
    ceded = treaty.apply(np.array([1.0, 3.0, 10.0]))
    np.testing.assert_allclose(ceded, [0.0, 1.0, 5.0])


def test_net_plus_ceded_equals_gross():
    losses = np.array([0.5, 4.0, 12.0])
    treaty = ExcessOfLossTreaty(1.0, 5.0)
    np.testing.assert_allclose(treaty.apply(losses) + treaty.net_retained(losses), losses)


def test_aggregate_mode_applies_per_year_not_across_years():
    # Two years, each with total 6. A 5 xs 4 aggregate layer cedes 2 per year.
    losses = np.array([3.0, 3.0, 1.0, 5.0])
    years = np.array([0, 0, 1, 1])
    treaty = ExcessOfLossTreaty(attachment=4.0, limit=5.0, mode="aggregate")
    ceded = treaty.apply(losses, years)
    annual = np.bincount(years, weights=ceded)
    np.testing.assert_allclose(annual, [2.0, 2.0])


def test_aggregate_limit_caps_annual_recoveries():
    losses = np.array([10.0, 10.0, 10.0])
    years = np.array([0, 0, 0])
    treaty = ExcessOfLossTreaty(attachment=0.0, limit=5.0, aggregate_limit=8.0)
    assert treaty.apply(losses, years).sum() == pytest.approx(8.0)


def test_quota_share():
    np.testing.assert_allclose(QuotaShareTreaty(0.25).apply(np.array([4.0, 8.0])), [1.0, 2.0])


@pytest.mark.parametrize("kwargs", [dict(attachment=-1, limit=1), dict(attachment=0, limit=0)])
def test_invalid_layer_rejected(kwargs):
    with pytest.raises(ValueError):
        ExcessOfLossTreaty(**kwargs)


def test_invalid_quota_rejected():
    with pytest.raises(ValueError):
        QuotaShareTreaty(1.5)


def test_year_index_shape_checked():
    with pytest.raises(ValueError):
        ExcessOfLossTreaty(1, 1).apply(np.ones(3), np.zeros(2, dtype=int))


def test_simulation_is_reproducible_and_aggregates_correctly():
    freq, sev = stats.poisson(5), stats.lognorm(s=1.0, scale=1e6)
    a = simulate(freq, sev, n_years=1000, seed=1)
    b = simulate(freq, sev, n_years=1000, seed=1)
    np.testing.assert_array_equal(a.losses, b.losses)
    assert a.annual_totals().sum() == pytest.approx(a.losses.sum())
    assert a.annual_totals().shape == (1000,)
    assert (a.annual_max() <= a.annual_totals() + 1e-9).all()


def test_metrics():
    x = np.arange(1, 1001, dtype=float)
    assert metrics.aal(x) == pytest.approx(500.5)
    assert metrics.return_period_loss(x, 10) == pytest.approx(np.quantile(x, 0.9))
    assert metrics.tvar(x, 0.99) >= metrics.var(x, 0.99)
    levels, probs = metrics.exceedance_curve(x)
    assert levels[0] == 1000 and probs[-1] == pytest.approx(1.0)
