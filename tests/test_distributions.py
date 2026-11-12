import math

import numpy as np
import pytest
from scipy import stats

from specinsure.core import ConvergenceError, ValidationError
from specinsure.distributions import (
    Empirical,
    Mixture,
    ScipyDistribution,
    compare_fits,
    fit,
    lognormal,
    negative_binomial,
    pareto,
    poisson,
)


def lognormal_lev(mu, sigma, u):
    """Closed-form E[min(X, u)] for a lognormal (Loss Models, appendix A)."""
    z = (math.log(u) - mu) / sigma
    return math.exp(mu + sigma**2 / 2) * stats.norm.cdf(z - sigma) + u * (1 - stats.norm.cdf(z))


def test_lognormal_from_moments():
    d = lognormal(mean=1e6, cv=2.0)
    assert d.mean() == pytest.approx(1e6)
    assert math.sqrt(d.frozen.var()) / d.mean() == pytest.approx(2.0)


def test_negative_binomial_from_moments():
    d = negative_binomial(mean=10, variance=25)
    assert d.mean() == pytest.approx(10)
    assert d.frozen.var() == pytest.approx(25)
    assert d.is_discrete
    with pytest.raises(ValidationError):
        negative_binomial(mean=10, variance=5)


def test_limited_expected_value_matches_closed_form():
    mu, sigma = 13.0, 1.5
    d = ScipyDistribution(stats.lognorm(s=sigma, scale=math.exp(mu)))
    for u in (1e5, 1e6, 1e7):
        assert d.limited_expected_value(u) == pytest.approx(lognormal_lev(mu, sigma, u), rel=1e-6)
    assert d.limited_expected_value(math.inf) == pytest.approx(d.mean())


def test_layer_expected_value_pareto_closed_form():
    # Single-parameter Pareto, alpha = 2, threshold = 1: E[X ^ u] = 2 - 1/u for u >= 1
    d = pareto(alpha=2.0, threshold=1.0)
    assert d.layer_expected_value(attachment=2.0, limit=3.0) == pytest.approx(
        (2 - 1 / 5) - (2 - 1 / 2)
    )


def test_discrete_limited_expected_value():
    d = poisson(3.0)
    expected = sum(min(k, 2.5) * d.frozen.pmf(k) for k in range(60))
    assert d.limited_expected_value(2.5) == pytest.approx(expected)


def test_empirical(rng):
    d = Empirical([1.0, 2.0, 3.0, 10.0])
    assert d.mean() == 4.0
    assert d.limited_expected_value(2.5) == pytest.approx((1 + 2 + 2.5 + 2.5) / 4)
    assert float(d.cdf(2.0)) == 0.5
    assert set(d.sample(100, rng)) <= {1.0, 2.0, 3.0, 10.0}
    with pytest.raises(ValidationError):
        Empirical([])


def test_mixture_moments_cdf_and_quantile(rng):
    small, large = lognormal(1e4, 1.0), pareto(1.8, 1e6)
    m = Mixture([small, large], [0.95, 0.05])
    assert m.mean() == pytest.approx(0.95 * small.mean() + 0.05 * large.mean())
    q = m.ppf(0.99)
    assert float(m.cdf(q)) == pytest.approx(0.99, abs=1e-8)
    assert m.limited_expected_value(1e6) == pytest.approx(
        0.95 * small.limited_expected_value(1e6) + 0.05 * large.limited_expected_value(1e6)
    )
    draws = m.sample(200_000, rng)
    assert np.mean(draws >= 1e6) == pytest.approx(0.05, abs=0.003)


def test_mixture_weights_validated():
    with pytest.raises(ValidationError):
        Mixture([poisson(1), poisson(2)], [0.5, 0.6])


def test_fit_recovers_lognormal_parameters(rng):
    data = stats.lognorm(s=1.2, scale=5e5).rvs(size=5_000, random_state=rng)
    result = fit(data, "lognorm")
    s, loc, scale = result.params
    assert loc == 0.0
    assert s == pytest.approx(1.2, rel=0.05)
    assert scale == pytest.approx(5e5, rel=0.05)
    assert result.n_params == 2


def test_compare_fits_ranks_true_family_first(rng):
    data = stats.lognorm(s=1.0, scale=1e5).rvs(size=3_000, random_state=rng)
    results = compare_fits(data, ("expon", "lognorm", "gamma"))
    assert results[0].family == "lognorm"
    assert [r.aic for r in results] == sorted(r.aic for r in results)


def test_fit_rejects_unknown_family():
    with pytest.raises(ValidationError):
        fit([1.0, 2.0, 3.0], "not_a_distribution")
    with pytest.raises(ConvergenceError):
        compare_fits([1.0, 2.0], ())
