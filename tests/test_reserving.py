"""Reserving tests, benchmarked on the Taylor and Ashe (1983) triangle used by Mack (1993)
and England and Verrall (2002). Published values are from those papers and the R
ChainLadder package (MackChainLadder(GenIns))."""

import numpy as np
import pytest

from specinsure.core import ValidationError
from specinsure.reserving import (
    DevelopmentPattern,
    Triangle,
    bornhuetter_ferguson,
    cape_cod,
    chain_ladder,
    development_factors,
    expected_loss_ratio,
    exponential_tail,
    mack_chain_ladder,
    odp_bootstrap,
)

TAYLOR_ASHE = [
    [357848, 1124788, 1735330, 2218270, 2745596, 3319994, 3466336, 3606286, 3833515, 3901463],
    [352118, 1236139, 2170033, 3353322, 3799067, 4120063, 4647867, 4914039, 5339085],
    [290507, 1292306, 2218525, 3235179, 3985995, 4132918, 4628910, 4909315],
    [310608, 1418858, 2195047, 3757447, 4029929, 4381982, 4588268],
    [443160, 1136350, 2128333, 2897821, 3402672, 3873311],
    [396132, 1333217, 2180715, 2985752, 3691712],
    [440832, 1288463, 2419861, 3483130],
    [359480, 1421128, 2864498],
    [376686, 1363294],
    [344014],
]


def to_array(rows):
    out = np.full((len(rows), max(map(len, rows))), np.nan)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


@pytest.fixture
def ta():
    return Triangle.from_cumulative(to_array(TAYLOR_ASHE), origins=range(1, 11))


# --- Triangle -----------------------------------------------------------------------


def test_triangle_views(ta):
    assert ta.shape == (10, 10)
    np.testing.assert_array_equal(ta.latest_index, np.arange(9, -1, -1))
    assert ta.latest[0] == 3901463 and ta.latest[-1] == 344014
    inc = ta.incremental()
    np.testing.assert_allclose(np.nansum(inc, axis=1), ta.latest)
    assert ta.link_ratios()[0, 0] == pytest.approx(1124788 / 357848)


def test_from_incremental_round_trip(ta):
    again = Triangle.from_incremental(ta.incremental(), ta.origins, ta.ages)
    np.testing.assert_allclose(again.values, ta.values)


def test_from_claims_long_format():
    # Two transactions in the same cell are summed; origin 2021 is observed at age 1 only.
    t = Triangle.from_claims(
        origin=[2020, 2020, 2020, 2021],
        age=[1, 1, 2, 1],
        amount=[60.0, 40.0, 50.0, 80.0],
    )
    np.testing.assert_allclose(t.values, [[100.0, 150.0], [80.0, np.nan]])
    np.testing.assert_array_equal(t.origins, [2020, 2021])


@pytest.mark.parametrize(
    "values",
    [
        [[1.0, np.nan, 3.0], [1.0, 2.0, np.nan]],  # gap inside a row
        [[1.0, np.nan], [1.0, 2.0]],  # later origin more developed
        [[1.0]],  # single age
    ],
)
def test_triangle_shape_validation(values):
    with pytest.raises(ValidationError):
        Triangle.from_cumulative(np.array(values))


def test_triangle_is_immutable(ta):
    with pytest.raises(ValueError):
        ta.values[0, 0] = 1.0


# --- Development --------------------------------------------------------------------


def test_volume_weighted_factors(ta):
    f = development_factors(ta).factors
    np.testing.assert_allclose(
        f,
        [3.4906, 1.7473, 1.4574, 1.1739, 1.1038, 1.0863, 1.0539, 1.0766, 1.0177],
        atol=5e-5,
    )
    simple = development_factors(ta, average="simple").factors
    assert simple[0] == pytest.approx(np.nanmean(ta.link_ratios()[:, 0]))
    recent = development_factors(ta, n_periods=1).factors
    assert recent[0] == pytest.approx(1363294 / 376686)


def test_pattern_cumulative_factors():
    p = DevelopmentPattern(np.array([2.0, 1.5]), tail=1.1)
    np.testing.assert_allclose(p.cdf_to_ultimate, [3.3, 1.65, 1.1])
    np.testing.assert_allclose(p.percent_developed, 1 / np.array([3.3, 1.65, 1.1]))


def test_exponential_tail_recovers_geometric_decay():
    k = np.arange(6)
    factors = 1 + 0.5 * 0.5**k
    tail = exponential_tail(DevelopmentPattern(factors), max_ages=200)
    expected = np.prod(1 + 0.5 * 0.5 ** np.arange(6, 206))
    assert tail == pytest.approx(expected)
    with pytest.raises(ValidationError):
        exponential_tail(DevelopmentPattern(np.array([1.0, 1.0, 1.0])))


# --- Deterministic methods ---------------------------------------------------------


def test_chain_ladder_matches_published(ta):
    cl = chain_ladder(ta)
    assert cl.total_reserve == pytest.approx(18_680_856, abs=1)
    assert cl.reserve[0] == 0
    assert cl.summary()[-1]["reserve"] == pytest.approx(cl.total_reserve)


def test_chain_ladder_with_tail(ta):
    pattern = development_factors(ta, tail=1.05)
    cl = chain_ladder(ta, pattern)
    np.testing.assert_allclose(cl.ultimate, chain_ladder(ta).ultimate * 1.05)


def test_bf_equals_chain_ladder_when_prior_matches(ta):
    cl = chain_ladder(ta)
    premium = np.full(10, 1e7)
    bf = bornhuetter_ferguson(ta, premium, cl.ultimate / premium)
    np.testing.assert_allclose(bf.ultimate, cl.ultimate)


def test_bf_formula_and_fully_developed_origin(ta):
    premium = np.full(10, 8e6)
    bf = bornhuetter_ferguson(ta, premium, 0.65)
    cdf = bf.details["cdf"]
    np.testing.assert_allclose(bf.reserve, premium * 0.65 * (1 - 1 / cdf))
    assert bf.reserve[0] == 0
    np.testing.assert_allclose(bf.loss_ratios(premium), bf.ultimate / premium)


def test_cape_cod_loss_ratio(ta):
    premium = np.linspace(9e6, 11e6, 10)
    cc = cape_cod(ta, premium)
    cdf = cc.details["cdf"]
    assert cc.details["elr"] == pytest.approx(ta.latest.sum() / (premium / cdf).sum())
    np.testing.assert_allclose(cc.reserve, premium * cc.details["elr"] * (1 - 1 / cdf))


def test_expected_loss_ratio(ta):
    elr = expected_loss_ratio(ta, np.full(10, 1e7), 0.7)
    np.testing.assert_allclose(elr.ultimate, 7e6)


def test_premium_and_pattern_lengths_validated(ta):
    with pytest.raises(ValidationError):
        bornhuetter_ferguson(ta, np.ones(3), 0.6)
    with pytest.raises(ValidationError):
        chain_ladder(ta, DevelopmentPattern(np.ones(3)))


# --- Mack ---------------------------------------------------------------------------


def test_mack_matches_published(ta):
    m = mack_chain_ladder(ta)
    assert m.total_reserve == pytest.approx(18_680_856, abs=1)
    assert m.total_standard_error == pytest.approx(2_447_095, abs=1)
    np.testing.assert_allclose(
        m.standard_error[1:],
        [75_535, 121_699, 133_549, 261_406, 411_010, 558_317, 875_328, 971_258, 1_363_155],
        atol=1,
    )
    np.testing.assert_allclose(
        m.standard_error**2, m.process_error**2 + m.parameter_error**2, rtol=1e-12
    )
    assert m.total_cv == pytest.approx(0.131, abs=5e-4)
    rows = m.summary()
    assert rows[-1]["standard_error"] == pytest.approx(m.total_standard_error)


def test_mack_total_exceeds_root_sum_of_squares(ta):
    m = mack_chain_ladder(ta)
    assert m.total_standard_error > np.sqrt(np.sum(m.standard_error**2))


def test_mack_loglinear_extrapolation(ta):
    m = mack_chain_ladder(ta, sigma_extrapolation="loglinear")
    assert np.all(np.isfinite(m.sigma2))
    assert m.total_standard_error == pytest.approx(2_447_095, rel=0.01)


def test_mack_needs_enough_data():
    with pytest.raises(ValidationError):
        mack_chain_ladder(Triangle.from_cumulative(np.array([[1.0, 2.0], [1.0, np.nan]])))


# --- ODP bootstrap ------------------------------------------------------------------


def test_bootstrap_matches_england_verrall(ta):
    b = odp_bootstrap(ta, n_sims=20_000, seed=7)
    assert b.scale == pytest.approx(52_601, abs=1)
    assert b.total.mean() == pytest.approx(18_680_856, rel=0.02)
    # England & Verrall (2002) report an ODP prediction error of about 16% for this triangle
    assert b.total.std() / 18_680_856 == pytest.approx(0.16, abs=0.01)
    assert b.reserves.shape == (20_000, 10)
    assert np.all(b.reserves[:, 0] == 0)
    assert b.percentile(0.75) > np.median(b.total)
    assert b.total_tvar(0.995) > b.percentile(0.995)


def test_bootstrap_is_reproducible(ta):
    a = odp_bootstrap(ta, n_sims=200, seed=1)
    b = odp_bootstrap(ta, n_sims=200, seed=1)
    np.testing.assert_array_equal(a.reserves, b.reserves)


def test_bootstrap_rejects_tiny_triangle():
    small = Triangle.from_cumulative(np.array([[1.0, 2.0], [1.0, np.nan]]))
    with pytest.raises(ValidationError):
        odp_bootstrap(small, 10, seed=1)
