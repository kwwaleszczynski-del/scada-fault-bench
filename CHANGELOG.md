# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [semantic versioning](https://semver.org/).

## [0.1.0] - 2026-11-12

First release.

### Added

- Deterministic fault injection with known ground truth on tables of the `scada-schema` format: 12 fault types (8 for wind, 8 for PV), five sizes each, truth with start, end, device and size.
- One interface for detectors (`fit` and `score`), alarm rules shared by all methods (threshold at a common alarm budget on healthy calibration data), and an evaluation with event and row metrics, a placebo (chance recall from the untouched test block) and the excess recall over it, detection delay, false alarm hours per year, AUROC and average precision of calibration ranks, recall by fault size and block-bootstrap confidence intervals over four-week blocks.
- Methods are scored only on the fault types whose channel they read (`scope`).
- Simple baselines (rules, `scada-schema` quality control, robust z-score, EWMA, CUSUM, Isolation Forest) and reference methods from other repositories of the portfolio (`wind-thermal-nbm`, `res-asset-kpi`, `pv-anomaly-autoencoder`, `pv-outage-classifier`).
- A part with real labelled faults (CARE to Compare, farm A).
- Command line (`list`, `prepare`, `inject`, `run`, `summarise`), README and project page generated from `results/summary.json`, tests, CI on Linux and Windows.

[0.1.0]: https://github.com/kwwaleszczynski-del/scada-fault-bench/releases/tag/v0.1.0
