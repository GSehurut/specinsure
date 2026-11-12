import numpy as np
import pytest

from specinsure.core import ValidationError
from specinsure.reinsurance import (
    AggregateExcessOfLoss,
    ExcessOfLoss,
    Programme,
    QuotaShare,
    loss_on_line,
    payback_years,
    rate_on_line,
    reinstatement_premium_factors,
    reinstatement_premiums,
)


def test_xol_per_occurrence():
    ceded = ExcessOfLoss(limit=5, attachment=2).apply([1.0, 3.0, 10.0])
    np.testing.assert_allclose(ceded, [0.0, 1.0, 5.0])


def test_xol_limit_and_attachment_are_keyword_only():
    with pytest.raises(TypeError):
        ExcessOfLoss(5, 2)


def test_net_plus_ceded_equals_gross():
    losses = np.array([0.5, 4.0, 12.0])
    t = ExcessOfLoss(limit=5, attachment=1)
    np.testing.assert_allclose(t.apply(losses) + t.net(losses), losses)


def test_reinstatements_cap_annual_recoveries_in_occurrence_order():
    # 10 xs 0 with one reinstatement: annual cap 20, so the third full loss recovers nothing
    t = ExcessOfLoss(limit=10, attachment=0, reinstatements=1)
    ceded = t.apply([10.0, 10.0, 10.0, 10.0], [0, 0, 0, 1])
    np.testing.assert_allclose(ceded, [10.0, 10.0, 0.0, 10.0])
    assert t.aggregate_limit == 20


def test_aad_retains_first_layer_losses_each_year():
    t = ExcessOfLoss(limit=10, attachment=0, aad=15)
    ceded = t.apply([10.0, 10.0, 10.0, 8.0], [0, 0, 0, 1])
    np.testing.assert_allclose(ceded, [0.0, 5.0, 10.0, 0.0])


def test_aggregate_limit_works_with_unsorted_years():
    t = ExcessOfLoss(limit=5, attachment=0, aggregate_limit=8)
    ceded = t.apply([10.0, 10.0, 10.0, 10.0], [1, 0, 1, 0])
    np.testing.assert_allclose(np.bincount([1, 0, 1, 0], weights=ceded), [8.0, 8.0])
    np.testing.assert_allclose(ceded, [5.0, 5.0, 3.0, 3.0])


def test_reinstatements_and_aggregate_limit_are_exclusive():
    with pytest.raises(ValidationError):
        ExcessOfLoss(limit=1, attachment=0, reinstatements=1, aggregate_limit=2)


@pytest.mark.parametrize("kwargs", [dict(attachment=-1, limit=1), dict(attachment=0, limit=0)])
def test_invalid_layer_rejected(kwargs):
    with pytest.raises(ValidationError):
        ExcessOfLoss(**kwargs)


def test_aggregate_xol_applies_per_year():
    losses, years = np.array([3.0, 3.0, 1.0, 5.0]), np.array([0, 0, 1, 1])
    ceded = AggregateExcessOfLoss(limit=5, attachment=4).apply(losses, years)
    np.testing.assert_allclose(np.bincount(years, weights=ceded), [2.0, 2.0])


def test_quota_share_with_event_limit():
    np.testing.assert_allclose(QuotaShare(0.25).apply([4.0, 8.0]), [1.0, 2.0])
    np.testing.assert_allclose(QuotaShare(0.5, event_limit=3).apply([4.0, 10.0]), [2.0, 3.0])
    with pytest.raises(ValidationError):
        QuotaShare(1.5)


def test_programme_inures_quota_share_before_xol_tower():
    qs = QuotaShare(0.5, name="QS")
    low = ExcessOfLoss(limit=10, attachment=10, name="10x10")
    high = ExcessOfLoss(limit=20, attachment=20, name="20x20")
    result = Programme(qs, [low, high]).apply([100.0, 30.0], [0, 1], n_years=2)
    # retained after QS: 50, 15
    np.testing.assert_allclose(result.ceded_by_treaty["QS"], [50.0, 15.0])
    np.testing.assert_allclose(result.ceded_by_treaty["10x10"], [10.0, 5.0])
    np.testing.assert_allclose(result.ceded_by_treaty["20x20"], [20.0, 0.0])
    np.testing.assert_allclose(result.net, [20.0, 10.0])
    np.testing.assert_allclose(result.annual_ceded(), [80.0, 20.0])


def test_programme_rejects_overlapping_layers_and_duplicate_names():
    a = ExcessOfLoss(limit=10, attachment=0, name="a")
    b = ExcessOfLoss(limit=10, attachment=0, name="b")
    with pytest.raises(ValidationError):
        Programme([a, b]).apply([10.0])
    with pytest.raises(ValidationError):
        Programme(a, ExcessOfLoss(limit=1, attachment=0, name="a"))


def test_market_metrics():
    assert rate_on_line(2.0, 10.0) == pytest.approx(0.2)
    assert loss_on_line(1.0, 10.0) == pytest.approx(0.1)
    assert payback_years(2.0, 10.0) == pytest.approx(5.0)


def test_reinstatement_premiums_pro_rata_as_to_amount():
    t = ExcessOfLoss(limit=10, attachment=0, reinstatements=2, reinstatement_rates=[1.0, 0.5])
    # recoveries of 5 use half of reinstatement 1; 15 uses all of 1 and half of 2 at 50%
    np.testing.assert_allclose(
        reinstatement_premium_factors(t, [0.0, 5.0, 15.0, 30.0]), [0, 0.5, 1.25, 1.5]
    )
    np.testing.assert_allclose(reinstatement_premiums(t, [15.0], deposit_premium=4.0), [5.0])
    free = ExcessOfLoss(limit=10, attachment=0)
    np.testing.assert_allclose(reinstatement_premium_factors(free, [15.0]), [0.0])
