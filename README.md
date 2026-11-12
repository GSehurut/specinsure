# specinsure

[![CI](https://github.com/GSehurut/specinsure/actions/workflows/ci.yml/badge.svg)](https://github.com/GSehurut/specinsure/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A transparent, well tested Python toolkit for specialty insurance and reinsurance analytics in the London Market style: simulate losses, structure and price reinsurance, measure capital, and optimise the portfolio.

It is aimed at actuaries, analysts and students who want Monte Carlo methods they can read, check against textbook formulas, and extend.

## What it does

| Area | Highlights |
| --- | --- |
| **Distributions** | Any `scipy.stats` distribution, actuarial parametrisations (lognormal from mean/CV, negative binomial from mean/variance, Pareto above a threshold), empirical and mixture distributions, MLE fitting with AIC/BIC comparison, closed-form layer expected values |
| **Simulation** | Frequency-severity year loss tables; multi-line portfolios with Gaussian or Student t copula dependency (Iman-Conover), independent random streams per line |
| **Reinsurance** | Quota share (with event limit), per-occurrence XoL with AAD, reinstatements and aggregate limits, aggregate XoL / stop loss, inuring programmes, ROL / LOL, reinstatement premiums |
| **Risk & capital** | AAL with Monte Carlo standard error, VaR, TVaR, 1-in-N losses, AEP / OEP curves, SCR-style capital, Euler (co-TVaR) capital allocation, diversification benefit |
| **Pricing** | Burning cost with trend, on-levelling and development; MBBEFD exposure rating; credibility blending; technical premium with expenses, cost of capital and reinstatement premium offset |
| **Reserving** | Triangles from arrays or transaction data, volume / simple / recent-period factors, exponential tail, chain ladder, Bornhuetter-Ferguson, Cape Cod, Mack standard errors, ODP bootstrap reserve distribution |
| **Optimisation** | Mean-TVaR line mix (Rockafellar-Uryasev LP), efficient frontier, reinsurance structure evaluation on cost of capital released |
| **Reporting** | Risk and layer summary tables, AEP / OEP plots (optional matplotlib) |

## Quick start

```sh
pip install git+https://github.com/GSehurut/specinsure.git
```

```python
from specinsure import ExcessOfLoss, FrequencySeverityModel, simulate
from specinsure.distributions import lognormal, negative_binomial
from specinsure.pricing import Loadings, technical_price
from specinsure.risk import aal, return_period_loss

# 50,000 years of an over-dispersed property book
model = FrequencySeverityModel(negative_binomial(mean=6, variance=12), lognormal(mean=1.5e6, cv=2.5))
ylt = simulate(model, n_years=50_000, seed=42)

# 10m xs 5m per risk, one reinstatement at 100%
layer = ExcessOfLoss(limit=10e6, attachment=5e6, reinstatements=1)
net_annual = ylt.annual_totals(ylt.losses - ylt.ceded(layer))

print(f"Gross AAL:       {aal(ylt.annual_totals()):,.0f}")
print(f"Net 1-in-200:    {return_period_loss(net_annual, 200):,.0f}")
print(f"Gross OEP 1-in-100: {return_period_loss(ylt.annual_max(), 100):,.0f}")

price = technical_price(ylt, layer, Loadings(brokerage=0.10, cost_of_capital=0.10))
print(f"Technical ROL:   {price.rate_on_line:.1%}  (LOL {price.loss_on_line:.1%})")
```

Reserving works on claims triangles:

```python
import numpy as np
from specinsure.reserving import Triangle, bornhuetter_ferguson, mack_chain_ladder, odp_bootstrap

triangle = Triangle.from_claims(origin=uw_year, age=dev_year, amount=incremental_paid)
mack = mack_chain_ladder(triangle)
print(f"Reserve {mack.total_reserve:,.0f}, Mack s.e. {mack.total_standard_error:,.0f}")

bf = bornhuetter_ferguson(triangle, premium=premium_by_year, loss_ratio=0.65)
boot = odp_bootstrap(triangle, n_sims=10_000, seed=1)
print(f"75th percentile reserve {boot.percentile(0.75):,.0f}")
```

[`examples/portfolio_walkthrough.py`](examples/portfolio_walkthrough.py) goes end to end: a three-class portfolio with a t copula, layer pricing, ranking reinsurance structures by value, and optimising the line mix. The notebook in [`examples/`](examples/) compares severity distributions.

## Conventions

- **Losses are positive** in risk measures (`var`, `tvar`, ...). Underwriting results are positive for profit; pass `-result` to a risk measure.
- **Annual terms are applied per simulated year**, in occurrence order within the year.
- **1-in-N = VaR at 1 - 1/N**: the 1-in-200 loss is VaR 99.5%.
- **TVaR** is the mean of the worst `ceil(n (1 - level))` simulations.
- **Every stochastic function takes a seed**; nothing uses global `np.random` state.

## Package layout

```
specinsure/
├── core/           exceptions, types, validation, random streams
├── distributions/  base, scipy adapters, empirical, mixture, fitting
├── simulation/     models, sampling (copulas), engine, results (YearLossTable)
├── risk/           measures, return periods, exceedance curves, aggregation / allocation
├── reinsurance/    treaties, programmes, layer calculations
├── pricing/        inputs, adjustments, rating methods, results
├── reserving/      triangles, development, CL / BF / Cape Cod, Mack, ODP bootstrap
├── portfolio/      lines of business, exposure profiles, portfolio results
├── optimisation/   line mix LP, efficient frontier, reinsurance evaluation
└── reporting/      summary tables, plots
```

Dependencies point down the list: `core` depends on nothing, and `optimisation` and `reporting` sit on top. See [CONTRIBUTING.md](CONTRIBUTING.md#architecture) for where new code belongs.

## Contributing

Contributions are very welcome, from fixing a typo to adding a reserving module. Start with [CONTRIBUTING.md](CONTRIBUTING.md) and the [roadmap](ROADMAP.md), or pick an issue labelled [`good first issue`](https://github.com/GSehurut/specinsure/labels/good%20first%20issue).

## References

- Klugman, Panjer and Willmot, *Loss Models: From Data to Decisions*, Wiley.
- Kaas, Goovaerts, Dhaene and Denuit, *Modern Actuarial Risk Theory*, Springer.
- Bernegger (1997), "The Swiss Re exposure curves and the MBBEFD distribution class", *ASTIN Bulletin* 27(1).
- Rockafellar and Uryasev (2000), "Optimization of conditional value-at-risk", *Journal of Risk* 2(3).
- Iman and Conover (1982), "A distribution-free approach to inducing rank correlation among input variables".
- Mack (1993), "Distribution-free calculation of the standard error of chain ladder reserve estimates", *ASTIN Bulletin* 23(2).
- England and Verrall (2002), "Stochastic claims reserving in general insurance", *British Actuarial Journal* 8(3).
- Tasche (1999), "Risk contributions and performance measurement".
- [Monte Carlo loss simulation for specialty insurance](https://gsehurut.github.io/blog/2024/03/20/loss-simulation-monte-carlo/), the blog post this project grew from.

## Licence

MIT. See [LICENSE](LICENSE).
