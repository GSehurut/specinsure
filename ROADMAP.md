# Roadmap

Where specinsure is heading. Items marked **good first issue** are well scoped for new contributors; open an issue or comment on an existing one before starting.

## 0.3: reserving and capital

- [ ] **`reserving/` package:** triangles (paid / incurred, by underwriting year), chain ladder, Bornhuetter-Ferguson, Cape Cod
- [ ] **Reserve uncertainty:** Mack standard error and ODP bootstrap (England & Verrall), one-year (Merz-Wüthrich) view for reserve risk
- [ ] **Premium and reserve risk combined** in the capital view, with reinsurer credit risk on recoveries
- [ ] **Ultimate (uSCR) and one-year capital** helpers, with the Lloyd's 35% economic capital uplift as an option
- [ ] **Seed stability check** for tail quantiles: rerun with several seeds and report the spread **(good first issue)**

## 0.4: richer simulation and treaties

- [ ] **Event-based / catastrophe simulation:** import vendor ELTs and YLTs, event IDs shared across lines, hours clauses
- [ ] **Parameter uncertainty:** simulate distribution parameters as well as losses
- [ ] **Inflation and exposure trend** applied within the simulation horizon
- [ ] **Surplus treaty** with lines and retention (needs sums insured per loss)
- [ ] **Reinstatement premiums pro rata as to time** as well as amount
- [ ] **Index (stability) clauses** on XoL layers

## Pricing

- [ ] **ILF curves for casualty** exposure rating: Riebesell and mixed exponential **(good first issue: Riebesell)**
- [ ] **Rate change monitoring (RARC):** split rate change into pure rate, exposure, terms and limit/deductible effects
- [ ] **Swiss Re curve fitting:** fit MBBEFD parameters to a risk's loss experience

## Optimisation

- [ ] **Appetite constraints** in the line-mix LP: 1-in-200 OEP caps and RDS limits per class
- [ ] **Joint optimisation** of line mix and quota share cession
- [ ] **Return on allocated capital** report per line from Euler allocation **(good first issue)**

## Reporting and documentation

- [ ] Example notebook: comparing two reinsurance structures with plots **(good first issue)**
- [ ] Example notebook: OEP versus AEP explained
- [ ] Documentation site (MkDocs) with API reference
- [ ] `__repr__` for `LineOfBusiness` and `PortfolioResults` **(good first issue)**

## Done

- [x] 0.2: restructured into `core`, `distributions`, `simulation`, `risk`, `reinsurance`, `pricing`, `portfolio`, `optimisation`, `reporting`
- [x] Reinstatements, AAD and aggregate limits on XoL; aggregate XoL; inuring programmes with per-treaty recoveries
- [x] Dependence between lines with Gaussian and Student t copulas
- [x] AAL standard error and confidence interval; AEP / OEP plot helper
- [x] Burning cost, MBBEFD exposure rating, credibility, technical premium
- [x] Mean-TVaR line mix optimisation, efficient frontier, reinsurance structure evaluation
- [x] 0.1: package structure, tests and CI; aggregate XoL applied per simulated year
