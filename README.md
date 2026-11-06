# scada-fault-bench

An open benchmark for fault detection in wind and PV SCADA data. Faults with a known position in time, device and size are injected into clean stretches of open datasets; any detector that reads a table of the [`scada-schema`](https://github.com/kwwaleszczynski-del/scada-schema) format is scored by one evaluation, with confidence intervals. A second part scores the same methods on real, labelled faults of CARE to Compare.

Not on PyPI yet. A DOI is assigned by Zenodo when the first release is archived. Live results: [kwwaleszczynski-del.github.io/scada-fault-bench](https://kwwaleszczynski-del.github.io/scada-fault-bench/).

**What this is not.** The injected faults are a signal-level model, not recorded failures. A method that does well here is not shown to be better in general, and a method that does badly may be badly matched to these models rather than bad. The tables describe this benchmark, these plants and these fault models, and nothing else. See [Limitations](#limitations).

## What was run

12 methods (6 simple baselines and 6 reference methods called from other repositories of this portfolio) on 7 datasets: 3 wind farms and 4 PV systems. 2,205 faults were injected: 956 into the wind farms and 1,249 into the PV systems, 12 fault types in all (8 for wind, 8 for PV), each at 5 sizes. 4 test blocks per dataset, the rest of the year (or two years) is training data. The real-fault part has 12 labelled anomalies and 10 normal datasets of CARE farm A. Every number below is produced by `scada-fault-bench run` and `summarise` and written to `results/summary.json`; this README is rendered from it and a test fails when they differ.

### Wind farms

All fault types and datasets added. Values in brackets are 95 % block-bootstrap intervals. The highest F1 is that of `ewma`; the methods whose F1 interval overlaps its interval are `ewma`, `cusum`, `iforest`, so the data do not separate them.

| Method | F1 | Recall | Precision | Delay (h) | False alarms (h/yr) | AUROC | AP |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Robust z-score (`zscore`) | 0.29 (0.27-0.32) | 0.94 (0.92-0.96) | 0.17 (0.15-0.19) | 54 (47-60) | 142 (107-183) | 0.68 (0.67-0.69) | 0.29 (0.26-0.33) |
| EWMA chart (`ewma`) | 0.40 (0.35-0.46) | 0.67 (0.63-0.71) | 0.29 (0.24-0.35) | 82 (75-91) | 200 (131-285) | 0.76 (0.74-0.77) | 0.37 (0.33-0.42) |
| CUSUM chart (`cusum`) | 0.39 (0.34-0.46) | 0.54 (0.50-0.58) | 0.31 (0.25-0.40) | 102 (92-112) | 244 (162-344) | 0.76 (0.74-0.78) | 0.39 (0.34-0.44) |
| Isolation Forest (`iforest`) | 0.33 (0.30-0.36) | 0.85 (0.82-0.88) | 0.20 (0.18-0.23) | 70 (63-78) | 177 (130-228) | 0.69 (0.68-0.70) | 0.18 (0.15-0.20) |
| Expert rules (`rules`) | 0.22 (0.19-0.26) | 0.37 (0.33-0.40) | 0.16 (0.13-0.21) | 53 (42-66) | 97 (47-160) | 0.52 (0.52-0.53) | 0.08 (0.07-0.09) |
| scada-schema QC (`scada_qc`) | 0.25 (0.23-0.28) | 0.66 (0.62-0.71) | 0.16 (0.14-0.18) | 102 (90-114) | 113 (71-164) | 0.51 (0.51-0.51) | 0.07 (0.06-0.08) |
| Thermal model (RC) (`nbm_rc`) | 0.19 (0.17-0.22) | 0.49 (0.45-0.54) | 0.12 (0.10-0.14) | 112 (99-126) | 123 (84-168) | 0.52 (0.51-0.54) | 0.11 (0.10-0.12) |
| Thermal model (GBM) (`nbm_gbm`) | 0.18 (0.16-0.20) | 0.51 (0.46-0.57) | 0.11 (0.09-0.12) | 110 (95-124) | 104 (63-148) | 0.56 (0.55-0.57) | 0.13 (0.12-0.15) |
| Power curve (method of bins) (`power_curve_kpi`) | 0.24 (0.21-0.27) | 0.62 (0.57-0.67) | 0.15 (0.13-0.17) | 101 (89-113) | 125 (82-173) | 0.59 (0.58-0.60) | 0.10 (0.09-0.11) |

### PV systems

The highest F1 is that of `iforest`; the methods whose F1 interval overlaps its interval are `zscore`, `iforest`, `rules`, `pv_autoencoder`, so the data do not separate them.

| Method | F1 | Recall | Precision | Delay (h) | False alarms (h/yr) | AUROC | AP |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Robust z-score (`zscore`) | 0.51 (0.48-0.55) | 0.61 (0.58-0.64) | 0.44 (0.40-0.49) | 48 (42-54) | 70 (52-91) | 0.59 (0.58-0.60) | 0.30 (0.28-0.32) |
| EWMA chart (`ewma`) | 0.42 (0.38-0.45) | 0.34 (0.31-0.37) | 0.53 (0.44-0.63) | 52 (44-60) | 58 (40-78) | 0.65 (0.64-0.66) | 0.33 (0.31-0.36) |
| CUSUM chart (`cusum`) | 0.41 (0.37-0.44) | 0.33 (0.29-0.37) | 0.53 (0.45-0.62) | 50 (42-59) | 122 (86-162) | 0.63 (0.62-0.64) | 0.32 (0.30-0.34) |
| Isolation Forest (`iforest`) | 0.53 (0.50-0.57) | 0.55 (0.51-0.58) | 0.52 (0.46-0.58) | 36 (31-42) | 67 (49-86) | 0.65 (0.64-0.66) | 0.31 (0.30-0.33) |
| Expert rules (`rules`) | 0.46 (0.43-0.50) | 0.39 (0.36-0.42) | 0.57 (0.48-0.70) | 13 (10-18) | 22 (11-35) | 0.54 (0.53-0.54) | 0.23 (0.22-0.24) |
| scada-schema QC (`scada_qc`) | 0.46 (0.43-0.48) | 0.67 (0.63-0.72) | 0.35 (0.32-0.37) | 35 (30-40) | 443 (361-520) | 0.51 (0.51-0.52) | 0.18 (0.17-0.19) |
| Autoencoder (PV) (`pv_autoencoder`) | 0.52 (0.49-0.55) | 0.47 (0.43-0.50) | 0.59 (0.53-0.66) | 31 (25-38) | 46 (31-63) | 0.58 (0.58-0.59) | 0.28 (0.26-0.29) |
| Shortfall vs PVWatts model (`pv_outage_expected`) | 0.44 (0.41-0.48) | 0.36 (0.33-0.40) | 0.57 (0.49-0.68) | 31 (26-38) | 48 (28-70) | 0.57 (0.57-0.58) | 0.24 (0.23-0.26) |
| Outage features + Isolation Forest (`pv_outage_iforest`) | 0.43 (0.39-0.47) | 0.36 (0.33-0.39) | 0.53 (0.42-0.66) | 34 (27-40) | 50 (31-71) | 0.62 (0.61-0.62) | 0.26 (0.24-0.27) |

### Recall by fault type

Share of injected events that were detected, all sizes and datasets of the profile added. Recall against the size of the fault (the sensitivity curve) is on the [project page](https://kwwaleszczynski-del.github.io/scada-fault-bench/).

Wind:

| Method | anemometer drift | nacelle direction offset | derating | power loss | temperature drift | flatline | unit jump | clock shift |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `zscore` | 0.92 | 0.96 | 0.82 | 0.96 | 0.94 | 1.00 | 0.97 | 0.95 |
| `ewma` | 0.93 | 0.87 | 0.66 | 0.76 | 0.64 | 0.51 | 0.56 | 0.42 |
| `cusum` | 0.87 | 0.86 | 0.43 | 0.65 | 0.57 | 0.31 | 0.43 | 0.19 |
| `iforest` | 0.92 | 0.94 | 0.74 | 0.86 | 0.84 | 0.98 | 0.57 | 0.95 |
| `rules` | 0.34 | 0.28 | 0.09 | 0.18 | 0.12 | 0.88 | 0.93 | 0.12 |
| `scada_qc` | 0.59 | 0.62 | 0.82 | 0.76 | 0.68 | 1.00 | 0.21 | 0.62 |
| `nbm_rc` | 0.34 | 0.44 | 0.31 | 0.55 | 0.72 | 0.10 | 1.00 | 0.46 |
| `nbm_gbm` | 0.37 | 0.45 | 0.38 | 0.56 | 0.68 | 0.15 | 1.00 | 0.50 |
| `power_curve_kpi` | 0.75 | 0.58 | 0.76 | 0.78 | 0.68 | 0.65 | 0.22 | 0.57 |

PV:

| Method | temperature drift | flatline | unit jump | clock shift | string loss | shading | inverter outage | pyranometer drift |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `zscore` | 0.60 | 0.74 | 1.00 | 0.81 | 0.33 | 0.31 | 0.56 | 0.57 |
| `ewma` | 0.34 | 0.41 | 0.64 | 0.19 | 0.29 | 0.11 | 0.49 | 0.29 |
| `cusum` | 0.31 | 0.38 | 0.58 | 0.21 | 0.30 | 0.09 | 0.50 | 0.29 |
| `iforest` | 0.50 | 0.65 | 0.51 | 0.89 | 0.31 | 0.28 | 0.85 | 0.37 |
| `rules` | 0.11 | 0.62 | 0.82 | 0.44 | 0.04 | 0.04 | 0.96 | 0.10 |
| `scada_qc` | 0.75 | 0.97 | 0.71 | 0.72 | 0.61 | 0.56 | 0.36 | 0.72 |
| `pv_autoencoder` | 0.26 | 0.59 | 0.93 | 0.68 | 0.13 | 0.15 | 0.81 | 0.20 |
| `pv_outage_expected` | 0.18 | 0.48 | 0.04 | 0.78 | 0.23 | 0.23 | 0.75 | 0.15 |
| `pv_outage_iforest` | 0.26 | 0.54 | 0.21 | 0.53 | 0.18 | 0.13 | 0.81 | 0.18 |

### Real faults: CARE to Compare, farm A

The same methods, trained on the first year of each turbine and scored on the prediction section; the truth is the labelled anomaly. These are single-turbine tables, so the residuals that compare a turbine with its neighbours do not exist and the methods that need neighbours work with less. 12 anomalies and 10 normal datasets are too few to rank methods; the intervals show how little this part can say.

| Method | Recall | Precision | Delay (h) | False alarms (h/yr) | AUROC | AP |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Robust z-score (`zscore`) | 0.58 (0.27-0.86) | 0.61 (0.33-0.83) | 152 (97-224) | 17 (7-28) | 0.52 (0.47-0.59) | 0.09 (0.03-0.20) |
| EWMA chart (`ewma`) | 0.33 (0.08-0.62) | 0.86 (0.50-1.00) | 133 (6-346) | 13 (0-44) | 0.68 (0.52-0.85) | 0.29 (0.03-0.53) |
| CUSUM chart (`cusum`) | 0.08 (0.00-0.29) | 0.50 (0.00-1.00) | 341 (341-341) | 6 (0-19) | 0.81 (0.66-0.98) | 0.32 (0.04-0.79) |
| Isolation Forest (`iforest`) | 0.67 (0.38-0.91) | 0.71 (0.44-0.92) | 166 (66-313) | 8 (2-15) | 0.47 (0.39-0.58) | 0.09 (0.03-0.19) |
| Expert rules (`rules`) | 0.00 (0.00-0.00) | n/a | n/a | 0 (0-0) | 0.50 (0.50-0.50) | 0.09 (0.02-0.18) |
| scada-schema QC (`scada_qc`) | 0.75 (0.50-1.00) | 0.87 (0.71-0.97) | 134 (80-186) | 19 (1-45) | 0.53 (0.50-0.58) | 0.12 (0.03-0.22) |
| Thermal model (RC) (`nbm_rc`) | 0.08 (0.00-0.25) | 0.56 (0.14-1.00) | 248 (248-248) | 50 (0-102) | 0.57 (0.45-0.74) | 0.15 (0.03-0.34) |
| Thermal model (GBM) (`nbm_gbm`) | 0.08 (0.00-0.25) | 0.50 (0.00-0.88) | 387 (387-387) | 55 (3-120) | 0.51 (0.36-0.69) | 0.11 (0.03-0.27) |
| Power curve (method of bins) (`power_curve_kpi`) | 0.50 (0.22-0.80) | 0.61 (0.32-0.93) | 136 (77-205) | 56 (2-143) | 0.44 (0.38-0.50) | 0.08 (0.02-0.17) |

## The faults

Every fault is a change of the logged signals of one device. The start, the device and the size of an event are drawn from a seeded generator (the same table, type and seed give the same events), events of one device keep 2 days apart. The truth of an event starts at the first row where the change exceeds a tolerance and ends at the last such row, leaving out the rows where an alarm is neither right nor wrong (night, a known stop, missing data); an event that would show in fewer than 3 hours of data is redrawn. Where the change would move other channels by simple arithmetic (power, current and energy, the energy meter counter) they are changed together.

| Type | Profile | What is injected | Five levels |
| :--- | :--- | :--- | :--- |
| `anemometer_drift` | wind | nacelle anemometer gain drifts linearly to +-m % (sign drawn per event); the second anemometer is untouched | 3, 6, 9, 12, 15 % (size) |
| `nacelle_dir_offset` | wind | nacelle position sensor offset by +-m degrees (step); the vane and the absolute direction are untouched | 3, 6, 10, 15, 25 deg (size) |
| `derating` | wind | active power capped at (1 - m %) of rated power; power, current and energy move together | 5, 10, 20, 30, 40 % of rated (size) |
| `power_loss` | wind | active power reads (1 - m %) of the true value at every power level (underperformance) | 2, 4, 6, 10, 15 % (size) |
| `thermal_drift` | wind+pv | temperature channel drifts linearly to +m K (gearbox oil of a turbine, heat sink of an inverter) | 2, 4, 6, 10, 15 K (size) |
| `flatline` | wind+pv | sensor stuck at its first value of the window (wind speed of a turbine, plane-of-array irradiance of a met station) | 3, 6, 12, 24, 48 h (duration) |
| `unit_jump` | wind+pv | temperature logged in degF instead of degC (gearbox oil of a turbine, module temperature of a met station) | 6, 12, 24, 48, 96 h (duration) |
| `clock_shift` | wind+pv | all channels of one device are delayed by m minutes against the others (logger clock error) | 10, 30, 60, 120, 360 min (size) |
| `string_loss` | pv | a share of the strings of an inverter is lost: AC and DC power and current scale by (1 - m %) | 2, 5, 10, 20, 35 % (size) |
| `shading` | pv | a fixed obstacle in the west shades the plant at low sun in the afternoon: power scales by (1 - m %) when the sun is below 30 deg and west of south-west | 5, 10, 20, 35, 50 % (size) |
| `inverter_outage` | pv | inverter produces nothing for the duration of the window (power and currents zero) | 2, 6, 12, 24, 72 h (duration) |
| `pyranometer_drift` | pv | plane-of-array irradiance gain drifts linearly to +-m % (soiling, ageing; sign drawn per event) | 2, 4, 6, 10, 15 % (size) |

## The methods

| Name | Profile | Family | Threshold | Description |
| :--- | :--- | :--- | :--- | :--- |
| `zscore` | wind+pv | baseline | calibrated | largest absolute residual of the shared features (one-hour median), each divided by its 99.5 % quantile on the training weeks so that no single heavy-tailed feature decides the score |
| `ewma` | wind+pv | baseline | calibrated | exponentially weighted moving average of every residual clipped at 5 sigma (time constant about one day), largest standardised value |
| `cusum` | wind+pv | baseline | calibrated | two-sided CUSUM of every residual clipped at 5 sigma, on hourly values (allowance 1 sigma, memory 3 days), largest statistic |
| `iforest` | wind+pv | baseline | calibrated | Isolation Forest (100 trees, 4096 samples) on the vector of standardised residuals; missing residuals set to 0 |
| `rules` | wind+pv | baseline | fixed | fixed physical rules, no learning: the two anemometers differ by more than 1.5 m/s, a wind speed with zero variation over an hour while the turbine produces, gearbox oil outside -20 to 100 degC, vane + nacelle - absolute direction beyond 25 deg, power above 110 % of rated; PV: irradiance frozen in daylight, inverter below 2 % of rated in sun above 300 W/m2, module temperature outside -30 to 90 degC, power above 110 % of rated |
| `scada_qc` | wind+pv | baseline | fixed | rows flagged by the row-level checks of scada_schema.qc (flatline, spikes, out of range, units, power curve, wind direction, energy counter, energy against power) |
| `nbm_rc` | wind | portfolio | calibrated | wind-thermal-nbm RCModel: first-order thermal model of the gearbox oil temperature from load, rotor speed, nacelle and ambient temperature; the standardised residual is the score Source: wind-thermal-nbm: wind_thermal_nbm.thermal.RCModel. |
| `nbm_gbm` | wind | portfolio | calibrated | wind-thermal-nbm GBMModel: gradient boosting on the same inputs and their trailing means (1-12 h) for the gearbox oil temperature; the standardised residual is the score Source: wind-thermal-nbm: wind_thermal_nbm.ml.GBMModel. |
| `power_curve_kpi` | wind | portfolio | calibrated | res-asset-kpi bin_power_curve per turbine on the training weeks; the standardised residual of power against the curve at the measured wind speed is the score Source: res-asset-kpi: res_asset_kpi.wind.bin_power_curve, curve_power (the method of wind-power-curve-kelmarsh). |
| `pv_autoencoder` | pv | portfolio | calibrated | pv-anomaly-autoencoder AutoencoderDetector per inverter: windows of 8 steps of power, performance ratio, irradiance, clear-sky index, temperatures and sun position; the reconstruction error (mse), smoothed over one hour, is the score; trained for at most 100 epochs on 80 % of the training days Source: pv-anomaly-autoencoder: pv_anomaly.autoencoder.AutoencoderDetector, pv_anomaly.features.build_features. |
| `pv_outage_expected` | pv | portfolio | calibrated | pv-outage-classifier detectors.score('expected'): shortfall of inverter power against a PVWatts-style model driven by the met station's irradiance and module temperature, calibrated per bin on the table that is scored (as in that repository); shortfall only Source: pv-outage-classifier: pv_outage.features.compute, pv_outage.detectors.score('expected'). |
| `pv_outage_iforest` | pv | portfolio | calibrated | pv-outage-classifier detectors.score('iforest'): Isolation Forest on that repository's step features (residuals, level, sun position, ...), fitted on the table that is scored, as in that repository Source: pv-outage-classifier: pv_outage.features.compute, pv_outage.detectors.score('iforest'). |

The other repositories of the portfolio were looked at and are not wrapped, for reasons that are properties of their inputs and not of their quality: `wind-wake-aware-verification` and `wind-bayes-twin` work on daily farm-level statistics with wake models, `pv-string-iv-faults` needs per-MPPT voltage and current channels that none of these datasets has, `pv-shade-fingerprint` produces a one-off diagnosis of a whole record and not a score per time stamp. Details in [docs/METHODS.md](docs/METHODS.md).

## The datasets

| Name | Dataset | Profile | Period (UTC) | Injected events |
| :--- | :--- | :--- | :--- | ---: |
| `kelmarsh_2019` | Kelmarsh 2019 | wind | 2019-01-01 to 2020-01-01 | 320 |
| `penmanshiel_2020` | Penmanshiel 2020 | wind | 2020-01-01 to 2021-01-01 | 320 |
| `hill_of_towie_2021` | Hill of Towie 2021 | wind | 2021-01-01 to 2022-01-01 | 316 |
| `pvdaq_34` | PVDAQ system 34 | pv | 2016-01-01 to 2018-01-01 | 318 |
| `pvdaq_1276` | PVDAQ system 1276 | pv | 2016-01-01 to 2018-01-01 | 308 |
| `pvdaq_1278` | PVDAQ system 1278 | pv | 2016-01-01 to 2018-01-01 | 307 |
| `pvdaq_35` | PVDAQ system 35 | pv | 2016-01-01 to 2018-01-01 | 316 |

The tables are built from the original open files with the loaders of `scada-schema` (`scada-fault-bench prepare`); nothing is redistributed. Kelmarsh and Penmanshiel are Cubico (CC BY 4.0), Hill of Towie is RES (CC BY 4.0), PVDAQ is NREL (CC BY 4.0). CARE to Compare is CC BY-SA 4.0, which also applies to datasets derived from it.

## The evaluation

For every dataset and fold the table is cut into a test block (91 days for wind farms, 150 for PV systems, which have one inverter or met station to carry all events), a margin of 7 days and the training part. A method is fitted on the training part without its calibration weeks (one week in four) and scored on the calibration weeks; the alarm threshold is the 0.5 % upper tail of those scores, so every calibrated method has the same nominal false alarm budget. Rule-based methods have a fixed threshold, and their false alarm rate is a result. Alarms closer than 6 hours are merged and alarms with less than 0.5 hours of flagged rows are dropped.

An event is detected when an alarm covers any row between its first affected row and 1 day after its last. The delay is the time from the first affected row to the first alarm row. An alarm is true when it overlaps such a window, ignored when it lies only on rows where an alarm is neither right nor wrong (known stops, night, real outages of the open data), and false otherwise. A wind alarm belongs to a turbine; a PV alarm belongs to the plant, because one met station and one or two inverters do not let a plant-level fault be attributed to a device. False alarms are counted in hours per year of healthy time. AUROC and average precision score the raw scores of affected rows against healthy rows without a threshold.

Confidence intervals come from a bootstrap over time blocks: the unit of resampling is a week of a test block of one dataset and fold, with all devices, fault types and methods of that week drawn together, 1000 resamples. They describe the sampling of weeks, devices and events inside this benchmark and nothing beyond it.

## Use it

```bash
pip install -e .                 # installs scada-schema from PyPI
scada-fault-bench list
scada-fault-bench prepare --datasets kelmarsh_2019 pvdaq_34 --fetch      # builds data/<name>.parquet with scada-schema
scada-fault-bench prepare --datasets care_A --fetch                      # the 22 CARE datasets of farm A (real faults)
scada-fault-bench run --datasets kelmarsh_2019 --folds 0 1 --out results
scada-fault-bench summarise --out results
```

A method is a class with `fit(train, meta)` and `score(table, meta)`; the score is one non-negative number per row of the table (higher means more anomalous, `NaN` means not assessable):

```python
import numpy as np
from scada_fault_bench.base import Detector

class MyMethod(Detector):
    name = "my_method"
    title = "My method"
    description = "what it does"
    profiles = ("wind",)

    def fit(self, train, meta):
        self.mean = train["p_ac"].mean()
        return self

    def score(self, table, meta):
        return np.abs(table["p_ac"].to_numpy() - self.mean)
```

```bash
scada-fault-bench run --datasets kelmarsh_2019 --plugin my_package.my_module:MyMethod --out results
```

`scada-fault-bench inject --dataset kelmarsh_2019 --type derating --out out/derating` writes one injected test block, its truth and the row labels, for use with any other tool. The reference methods from other repositories are optional dependencies that are installed from GitHub (none is on PyPI); a method whose repository cannot be imported is skipped and listed as unavailable in `results/summary.json`:

```bash
pip install git+https://github.com/kwwaleszczynski-del/wind-thermal-nbm git+https://github.com/kwwaleszczynski-del/pv-anomaly-autoencoder git+https://github.com/kwwaleszczynski-del/pv-outage-classifier
```

## Limitations

* The faults are injected into recorded signals. Physical faults change more channels, and in ways that depend on the machine; the models here favour detectors that look at the channel that was changed. Nothing here says how a method finds a real fault except the CARE part, which is small.
* Faults of one type are drawn from one simple model with five sizes. The sensitivity curves describe that model.
* The open data are not guaranteed to be healthy. Real stops that the status log shows, night, and real inverter outages of PVDAQ are removed from the count of false alarms; real faults that are not visible in them are not, so alarms on them are counted as false. This is conservative.
* PV plants are one met station with one or two inverters: there is no string level and no device-level attribution. The PV results are plant-level.
* Calibrated methods share a false alarm budget on a calibration sample from the same year; a method whose scores drift with the seasons will exceed it, and the false alarm column shows by how much.
* The reference methods are called through adapters that rename and rescale columns. A weak result can come from the adapter or from a method that was written for other data; the repositories that own the methods are the authority on what they do.
* 1000 bootstrap resamples of weeks are not independent of the fault draws: another seed gives other events. Differences within the intervals should not be read as differences.
* The file formats and the fault models are at version 0.1.0; results of different versions are not comparable.

## Reproduce

`results/` holds the shards and `summary.json` of the run in this repository; `python scripts/render_readme.py` rebuilds this file and `python scripts/build_pages.py` the page from them. The tests in `tests/test_docs.py` compare both with the summary, and `tests/test_data.py` (marked `data`) repeats a small part of the run on the real tables.

## Citation and licence

Code: MIT. Cite the repository (`CITATION.cff`) and the datasets above. `scada-fault-bench` is built on `scada-schema` 0.1.0, whose loaders read the open datasets.
