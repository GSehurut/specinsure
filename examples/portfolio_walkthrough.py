"""End-to-end walkthrough: simulate a specialty portfolio, price an XoL layer,
compare reinsurance structures and optimise the line mix.

Run with:  python examples/portfolio_walkthrough.py
All data is synthetic.
"""

import numpy as np

from specinsure import ExcessOfLoss, FrequencySeverityModel, LineOfBusiness, Portfolio, Programme
from specinsure.distributions import lognormal, negative_binomial, pareto, poisson
from specinsure.optimisation import evaluate_programmes, optimise_mix, xol_grid
from specinsure.pricing import Loadings, technical_price
from specinsure.reporting import format_table, layer_summary, risk_summary
from specinsure.risk import euler_tvar_allocation, tvar

N_YEARS, SEED = 50_000, 2026

# 1. Portfolio: three classes with a Student t copula between annual results
portfolio = Portfolio(
    [
        LineOfBusiness(
            "property_dnf",
            FrequencySeverityModel(negative_binomial(6, 12), lognormal(1.5e6, 2.5)),
            premium=14e6,
            expense_ratio=0.32,
        ),
        LineOfBusiness(
            "marine_hull",
            FrequencySeverityModel(poisson(8), lognormal(4e5, 1.8)),
            premium=5e6,
            expense_ratio=0.28,
        ),
        LineOfBusiness(
            "energy",
            FrequencySeverityModel(poisson(0.6), pareto(alpha=1.6, threshold=2.5e6)),
            premium=6e6,
            expense_ratio=0.30,
        ),
    ]
)
correlation = [[1.0, 0.3, 0.4], [0.3, 1.0, 0.2], [0.4, 0.2, 1.0]]
results = portfolio.simulate(N_YEARS, SEED, correlation=correlation, copula="t", dof=5)

print(f"Simulated {N_YEARS:,} years, seed {SEED}\n")
rows = [{"line": n, **risk_summary(results.gross_losses[:, j])} for j, n in enumerate(results.names)]
cols = ["line", "aal", "aal_standard_error", "var_0.995", "tvar_0.99"]
print(format_table(rows, cols, ",.0f"), "\n")

# 2. Price a per-risk XoL on property, one reinstatement at 100%
prop = results.simulation["property_dnf"]
layer = ExcessOfLoss(limit=10e6, attachment=5e6, reinstatements=1, name="10x5")
loads = Loadings(brokerage=0.10, internal_expenses=0.05, cost_of_capital=0.10, capital_level=0.99)
price = technical_price(prop, layer, loads)
print("Technical price for 10m xs 5m (1 @ 100%):")
for key in ("expected_loss", "capital", "technical_premium", "rate_on_line", "loss_on_line"):
    print(f"  {key:>18}: {price.summary()[key]:,.4g}")
print(format_table(layer_summary(Programme(layer), Programme(layer).apply_to(prop))), "\n")

# 3. Reinsurance structures, marginal capital measured on the whole portfolio
others = results.gross_losses[:, 1:].sum(axis=1)
candidates = xol_grid(attachments=[2.5e6, 5e6, 10e6], limits=[5e6, 10e6, 20e6], reinstatements=1)


def market_price(programme_result):
    # Illustrative market pricing: expected loss loaded to a 55% technical loss ratio
    return float(programme_result.annual_ceded().mean()) / 0.55


evals = evaluate_programmes(prop, candidates, market_price, other_losses=others, cost_of_capital=0.10)
print("Reinsurance structures ranked by value (CoC 10%, 1-in-200 portfolio VaR):")
print(format_table([e.__dict__ for e in evals[:5]], float_format=",.0f"), "\n")

# 4. Line mix: maximise expected profit for no more TVaR than the current plan
r = results.underwriting_results(net=False)
plan_tvar = tvar(-r.sum(axis=1), 0.99)
best = optimise_mix(r, 0.99, tvar_budget=plan_tvar, bounds=(0.5, 1.5))
alloc = euler_tvar_allocation(-r * best.weights, 0.99)
print(f"Plan TVaR99 {plan_tvar:,.0f}; optimised expected profit {best.expected_result:,.0f}")
print(
    format_table(
        [
            {"line": n, "weight": f"{w:.3f}", "allocated_tvar": a}
            for n, w, a in zip(results.names, best.weights, alloc)
        ],
        float_format=",.2f",
    )
)
print(f"Plan expected profit {r.mean(axis=0).sum():,.0f}; binding upper bounds: "
      f"{[n for n, b in zip(results.names, best.at_upper) if b]}")
assert np.isfinite(best.expected_result)
