# Changelog

## Unreleased

### Added

- `reserving`: `Triangle` (from cumulative, incremental or long-format claims), development factor selection and exponential tail, chain ladder, expected loss ratio, Bornhuetter-Ferguson, Cape Cod, Mack standard errors (validated against Mack 1993 on the Taylor-Ashe triangle) and the ODP bootstrap (England and Verrall 2002).

## 0.2.0 (unreleased)

A restructure into layered subpackages and a large expansion of scope: pricing, capital and portfolio optimisation.

### Breaking changes

- Modules moved: `specinsure.metrics` is now `specinsure.risk`; `specinsure.treaties` is now `specinsure.reinsurance`; `specinsure.simulation` is a package.
- `ExcessOfLossTreaty` is now `ExcessOfLoss`, with `limit` and `attachment` **keyword-only** (market order "limit xs attachment" without positional ambiguity). `mode="aggregate"` is replaced by the separate `AggregateExcessOfLoss` class.
- `QuotaShareTreaty(quota)` is now `QuotaShare(cession)`; `ReinsuranceTreaty` is now `Treaty`.
- `simulate(frequency, severity, n_years, seed)` is now `simulate(FrequencySeverityModel(frequency, severity), n_years, seed)`.
- `tvar` is the mean of the worst `ceil(n (1 - level))` outcomes (previously all values at or above the VaR quantile), which handles ties consistently.
- Requires numpy >= 1.25.

### Added

- `distributions`: actuarial parametrisations, `Empirical`, `Mixture`, MLE `fit` / `compare_fits`, limited and layer expected values.
- `simulation`: `simulate_portfolio` with Gaussian / t copulas, independent random streams per line.
- `reinsurance`: AAD, reinstatements and reinstatement premiums, quota share event limit, inuring `Programme`, ROL / LOL / payback.
- `risk`: AAL standard error and confidence interval, `ExceedanceCurve`, SCR-style capital, Euler TVaR allocation, diversification benefit.
- `pricing`: burning cost, MBBEFD exposure rating, credibility, technical premium.
- `portfolio`: `LineOfBusiness`, `Portfolio`, `RiskProfile`.
- `optimisation`: mean-TVaR line mix LP, efficient frontier, reinsurance structure evaluation.
- `reporting`: summary tables and AEP / OEP plots (`pip install "specinsure[plot]"`).

## 0.1.0

- Frequency-severity simulation, XoL and quota share treaties, basic risk metrics.
