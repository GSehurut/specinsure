# specinsure

[![CI](https://github.com/GSehurut/specinsure/actions/workflows/ci.yml/badge.svg)](https://github.com/GSehurut/specinsure/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A small, well tested Python toolkit for simulating specialty insurance losses and the reinsurance structures that protect them.

It is aimed at actuaries, analysts and students who want to explore how excess of loss and quota share programmes change a loss portfolio, using transparent Monte Carlo methods rather than a black box.

## Features

- **Frequency and severity simulation** into a year loss table, using any `scipy.stats` distribution
- **Treaties:** per-occurrence and annual aggregate excess of loss (with optional aggregate limit), and quota share
- **Annual terms applied correctly,** year by year across a simulation
- **Risk metrics:** average annual loss, VaR, TVaR, return period losses, AEP and OEP curves

## Quick start

```sh
pip install git+https://github.com/GSehurut/specinsure.git
```

```python
from scipy import stats
from specinsure import ExcessOfLossTreaty, simulate, metrics

# 50,000 years: negative binomial frequency, lognormal severity
ylt = simulate(
    frequency=stats.nbinom(n=12, p=0.5),
    severity=stats.lognorm(s=1.2, scale=1_000_000),
    n_years=50_000,
    seed=42,
)

layer = ExcessOfLossTreaty(attachment=2_000_000, limit=5_000_000)
ceded = ylt.ceded(layer)

gross_annual = ylt.annual_totals()
net_annual = ylt.annual_totals(ylt.losses - ceded)

print(f"AAL gross: {metrics.aal(gross_annual):,.0f}")
print(f"1-in-200 net: {metrics.return_period_loss(net_annual, 200):,.0f}")
print(f"OEP 1-in-100 gross: {metrics.return_period_loss(ylt.annual_max(), 100):,.0f}")
```

The [examples](examples/) folder has a notebook comparing severity distributions and the effect of an XoL layer.

## Contributing

Contributions are very welcome, from fixing a typo to adding a new treaty type. Start with [CONTRIBUTING.md](CONTRIBUTING.md) and the [roadmap](ROADMAP.md), or pick an issue labelled [`good first issue`](https://github.com/GSehurut/specinsure/labels/good%20first%20issue).

## Background reading

The methods follow standard actuarial references:

- Klugman, Panjer and Willmot, *Loss Models: From Data to Decisions*, Wiley.
- Kaas, Goovaerts, Dhaene and Denuit, *Modern Actuarial Risk Theory*, Springer.
- [Monte Carlo loss simulation for specialty insurance](https://gsehurut.github.io/blog/2024/03/20/loss-simulation-monte-carlo/), the blog post this project grew from.

## Licence

MIT. See [LICENSE](LICENSE).
