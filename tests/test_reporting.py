import numpy as np
import pytest

from specinsure.distributions import lognormal, poisson
from specinsure.reinsurance import ExcessOfLoss, Programme, QuotaShare
from specinsure.reporting import format_table, layer_summary, risk_summary
from specinsure.risk import aep_curve, oep_curve
from specinsure.simulation import FrequencySeverityModel, simulate


@pytest.fixture
def ylt():
    return simulate(FrequencySeverityModel(poisson(3.0), lognormal(1e6, 2.0)), 5_000, seed=3)


def test_risk_summary_keys(ylt):
    s = risk_summary(ylt.annual_totals())
    assert {"aal", "aal_standard_error", "var_0.995", "tvar_0.99", "1_in_200"} <= set(s)
    assert s["tvar_0.995"] >= s["var_0.995"]


def test_layer_summary(ylt):
    layer = ExcessOfLoss(limit=2e6, attachment=1e6, reinstatements=1, name="2x1")
    prog = Programme(QuotaShare(0.2, name="QS"), layer)
    rows = layer_summary(prog, prog.apply_to(ylt))
    assert [r["treaty"] for r in rows] == ["QS", "2x1"]
    assert 0 < rows[1]["probability_of_exhaustion"] <= rows[1]["probability_of_recovery"] <= 1
    assert rows[1]["loss_on_line"] == pytest.approx(rows[1]["expected_recovery"] / 2e6)
    text = format_table(rows)
    assert "2x1" in text and "expected_recovery" in text


def test_format_table_empty():
    assert format_table([]) == ""


def test_plot_exceedance_curves(ylt):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from specinsure.reporting import plot_exceedance_curves

    ax = plot_exceedance_curves([aep_curve(ylt), oep_curve(ylt)])
    assert len(ax.lines) == 2
    assert ax.get_xscale() == "log"
    np.testing.assert_allclose(ax.lines[0].get_ydata(), aep_curve(ylt).losses)
