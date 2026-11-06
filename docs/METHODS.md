# Methods

This page says what each method is given and what the adapter does. The authority on what a reference method does is the repository that owns it; the adapters only rename and rescale columns and turn the output into one non-negative score per row. Nothing is tuned for the benchmark. The table of methods with their descriptions is generated from the code in the README.

## Interface

`fit(train, meta)` gets a table in the `scada-schema` format (the training part of a split, which may contain gaps, real stops and real faults) and its metadata. `score(table, meta)` returns one number per row, in the order of the rows. Higher is more anomalous, `NaN` is "cannot be assessed" (PV at night, a turbine with no data). The harness sets the alarm threshold from held-out calibration weeks of the training part; a method with `fixed_threshold = True` is not calibrated (an alarm is a score above 0.5).

## Baselines

**Shared residuals.** `zscore`, `ewma`, `cusum` and `iforest` read the same matrix of standardised residuals (`scada_fault_bench/features.py`): power against the turbine's own binned power curve, wind speed against the other turbines and against the second anemometer, the nacelle position against the other turbines and against vane and absolute direction, the gearbox oil against a regression on load and ambient temperature, the one-hour variation of wind speed; for PV the performance ratio against irradiance and module temperature (also per sun position), the variation of irradiance, the module and the inverter temperature. Curves and regressions are fitted on the training weeks only; each residual is smoothed with a trailing one-hour median and standardised by median and median absolute deviation.

* `zscore`: the largest absolute residual, each divided by its 99.5 % quantile on the training weeks.
* `ewma`: exponentially weighted moving average (time constant about one day) of every residual clipped at 5 sigma.
* `cusum`: two-sided CUSUM of every residual clipped at 5 sigma, hourly values, allowance 1 sigma, memory 3 days (a finite memory so that slow seasonal offsets cannot saturate it).
* `iforest`: Isolation Forest (100 trees, 4096 samples) on the residual vector.
* `rules`: hand-written physical rules, no learning (see the description in the README).
* `scada_qc`: the row-level findings of `scada_schema.qc` (flatline, spikes, out of range, units, power curve, wind direction, energy counter, energy against power) on the channels that the fault models touch.

## Reference methods from the portfolio

| Name | Repository | Entry point | Adapter |
|---|---|---|---|
| `nbm_rc` | [wind-thermal-nbm](https://github.com/kwwaleszczynski-del/wind-thermal-nbm) | `RCModel` | per turbine, gearbox oil; power as a share of rated power times 2050 kW and rotor speed as a share of its 99th percentile times 15 rpm (the constants that repository was written for); operating rows only for the fit; standardised residual, trailing one-hour median |
| `nbm_gbm` | wind-thermal-nbm | `GBMModel` | the same |
| `power_curve_kpi` | [res-asset-kpi](https://github.com/kwwaleszczynski-del/res-asset-kpi) (the method of [wind-power-curve-kelmarsh](https://github.com/kwwaleszczynski-del/wind-power-curve-kelmarsh)) | `bin_power_curve`, `curve_power` | per turbine on the training weeks; residual of power against the curve at the measured wind speed |
| `pv_autoencoder` | [pv-anomaly-autoencoder](https://github.com/kwwaleszczynski-del/pv-anomaly-autoencoder) | `AutoencoderDetector` | per inverter, default configuration, whole-window rows only, 20 % of the training days for early stopping; reconstruction error (mse), trailing one-hour median |
| `pv_outage_expected` | [pv-outage-classifier](https://github.com/kwwaleszczynski-del/pv-outage-classifier) | `features.compute`, `detectors.score("expected")` | the table is turned into the long layout of that repository (met station as the plant, inverters below); its calibration is taken from the table that is scored, as it is there; shortfall only |
| `pv_outage_iforest` | pv-outage-classifier | `detectors.score("iforest")` | the same; the forest is fitted on the table that is scored, as it is there |

The two PV methods of `pv-outage-classifier` calibrate on the table they score. That is how the repository uses them, and here it makes them partly absorb a fault that lasts a large part of the table; it is not a flaw of the benchmark, but it is a property of the numbers.

## Not wrapped, and why

* `wind-wake-aware-verification` and `wind-bayes-twin` work on daily farm-level statistics with wake models (py_wake, a layout, neighbouring turbines); they do not produce a score per time stamp of one device from a SCADA table alone.
* `pv-string-iv-faults` diagnoses 14-day windows of per-MPPT voltage and current against peers; none of the open PV tables here has MPPT channels.
* `pv-shade-fingerprint` finds fixed obstructions as a pattern in sun position over a whole record; it is a diagnosis, not a score per time stamp, and it needs about a year of clear-sky samples.
* `res-asset-kpi` also offers data-quality flags (stuck, out of range); the same checks are part of `scada_qc`.

## Adding a method

Write a class as in the README, run it with `--plugin package.module:Class`, and add the class to `scada_fault_bench/baselines.py` or `portfolio.py` with `@register` if it should be part of the standard set. A method that needs an optional dependency checks it in `available()` so that the benchmark skips it, and says why, where the dependency is missing.
