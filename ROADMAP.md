# Roadmap

Where specinsure is heading. Items marked **good first issue** are well scoped for new contributors; open an issue or comment on an existing one before starting.

## 0.2: richer treaties

- [ ] **Reinstatements** for XoL layers: number of reinstatements and reinstatement premium rates
- [ ] **Surplus treaty** with lines and retention
- [ ] **Stacked programmes:** apply several layers in order and report recoveries per layer
- [ ] **Annual aggregate deductible (AAD)** on XoL layers
- [ ] `__repr__` and `summary()` for programmes **(good first issue)**

## 0.3: better simulation

- [ ] **Parameter uncertainty:** simulate distribution parameters as well as losses
- [ ] **Inflation and exposure trend** applied to severities **(good first issue)**
- [ ] **Dependence between lines** using Gaussian and Student t copulas
- [ ] **Event based simulation** for catastrophe style year loss tables

## Metrics and reporting

- [ ] Standard error and confidence interval for AAL **(good first issue)**
- [ ] Seed stability check for tail quantiles (rerun with several seeds and report the spread)
- [ ] Plot helpers for AEP / OEP curves (optional matplotlib dependency) **(good first issue)**
- [ ] Expected reinsurance recovery and burning cost reports per layer

## Documentation

- [ ] Example: comparing two reinsurance structures for the same portfolio **(good first issue)**
- [ ] Example: OEP versus AEP explained with plots
- [ ] Documentation site (MkDocs) with API reference

## Done

- [x] Package structure, tests and CI
- [x] Aggregate XoL applied per simulated year (previously summed across all years)
- [x] Frequency and severity simulation into a year loss table
