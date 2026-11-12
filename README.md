# scada-fault-bench

An open benchmark for fault detection in wind and PV SCADA data. Faults with a known position in time, device and size are injected into blocks cut from open datasets (the data are not selected for health: known stops, night and real outages are left out of the count of false alarms, nothing else); any detector that reads a table of the [`scada-schema`](https://github.com/kwwaleszczynski-del/scada-schema) format is scored by one evaluation, with confidence intervals. A second part scores the same methods on real, labelled faults of CARE to Compare.

On PyPI: `pip install scada-fault-bench`. DOI (all versions): [10.5281/zenodo.24164281](https://doi.org/10.5281/zenodo.24164281); version 0.1.0: [10.5281/zenodo.24164282](https://doi.org/10.5281/zenodo.24164282). Live results: [kwwaleszczynski-del.github.io/scada-fault-bench](https://kwwaleszczynski-del.github.io/scada-fault-bench/).

**How to read the numbers.** An alarm that happens to fall on the affected rows of an event is not a detection of the fault. Every table therefore shows the *chance* level next to recall (the recall that the same method reaches on the untouched test block, matched to the same events) and the *excess recall* over it; the other main columns (row F1, AUROC and average precision on affected against healthy rows, false alarm hours) do not depend on where alarms fall by luck. Methods marked with an asterisk are scored only on the fault types whose changed channel they read (column *Fault types scored* in [The methods](#the-methods)), so their rows cannot be compared with rows of methods scored on all types.

**What this is not.** The injected faults are a signal-level model, not recorded failures. A method that does well here is not shown to be better in general, and a method that does badly may be badly matched to these models rather than bad. The tables describe this benchmark, these plants and these fault models, and nothing else. See [Limitations](#limitations).

## What was run

12 methods (6 simple baselines and 6 reference methods called from other repositories of this portfolio) on 7 datasets: 3 wind farms and 4 PV systems. 2,213 faults were injected: 956 into the wind farms and 1,257 into the PV systems, 12 fault types in all (8 for wind, 8 for PV), each at 5 sizes. 4 test blocks per dataset; the rest of the year (or two years), including the weeks after a test block, is training data. 1 injection seed per fold for the wind farms and 2 for the PV systems (a PV plant has one inverter or met station to carry all events of a block, so it gets fewer events per seed). The real-fault part has 12 labelled anomalies and 10 normal datasets of CARE farm A. Every number below is produced by `scada-fault-bench run` and `summarise` and written to `results/summary.json`; this README is rendered from it and a test fails when they differ.

### Wind farms

All datasets and the fault types each method is scored on, added. Values in brackets are 95 % block-bootstrap intervals. The excess recall has an interval above zero for: `zscore`, `ewma`, `cusum`, `iforest`, `rules`, `scada_qc`, `nbm_rc`, `nbm_gbm`, `power_curve_kpi`. The highest AUROC is that of `nbm_gbm`; the methods whose AUROC interval overlaps its interval are `nbm_gbm` (the intervals describe the sampling of weeks and events in this benchmark, so a method outside that list is not shown to be worse on other data, and one inside it is not shown to be equal).

| Method | Excess recall | Recall | Chance | Precision | Row F1 | AUROC | AP | Delay (h) | False alarms (h/yr) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Robust z-score (`zscore`) | 0.23 (0.20-0.26) | 0.91 (0.89-0.93) | 0.68 | 0.14 (0.12-0.16) | 0.37 (0.33-0.42) | 0.68 (0.66-0.69) | 0.31 (0.26-0.36) | 49 (43-56) | 180 (117-256) |
| EWMA chart (`ewma`) | 0.38 (0.33-0.44) | 0.70 (0.65-0.75) | 0.32 | 0.23 (0.19-0.29) | 0.39 (0.32-0.46) | 0.75 (0.73-0.78) | 0.32 (0.23-0.44) | 81 (73-90) | 265 (144-411) |
| CUSUM chart (`cusum`) | 0.42 (0.36-0.48) | 0.57 (0.51-0.63) | 0.16 | 0.27 (0.20-0.38) | 0.39 (0.32-0.47) | 0.77 (0.74-0.79) | 0.35 (0.27-0.43) | 97 (88-108) | 327 (175-527) |
| Isolation Forest (`iforest`) | 0.28 (0.23-0.32) | 0.82 (0.77-0.86) | 0.55 | 0.19 (0.16-0.23) | 0.28 (0.24-0.32) | 0.70 (0.68-0.71) | 0.21 (0.17-0.27) | 68 (60-76) | 185 (105-288) |
| Expert rules (`rules`) | 0.26 (0.23-0.28) | 0.35 (0.30-0.39) | 0.09 | 0.15 (0.11-0.25) | 0.09 (0.08-0.11) | 0.62 (0.59-0.65) | 0.13 (0.11-0.15) | 45 (32-56) | 97 (26-180) |
| scada-schema QC (`scada_qc`) | 0.22 (0.20-0.25) | 0.53 (0.47-0.60) | 0.31 | 0.12 (0.10-0.15) | 0.06 (0.05-0.08) | 0.60 (0.55-0.64) | 0.09 (0.07-0.12) | 93 (81-105) | 113 (56-181) |
| Thermal model (RC) (`nbm_rc`) * | 0.53 (0.47-0.59) | 0.87 (0.81-0.92) | 0.35 | 0.13 (0.11-0.17) | 0.30 (0.25-0.34) | 0.72 (0.69-0.75) | 0.27 (0.23-0.32) | 77 (62-92) | 171 (117-230) |
| Thermal model (GBM) (`nbm_gbm`) * | 0.51 (0.44-0.58) | 0.88 (0.83-0.93) | 0.37 | 0.12 (0.10-0.15) | 0.34 (0.29-0.40) | 0.85 (0.82-0.87) | 0.39 (0.32-0.47) | 75 (61-91) | 133 (81-193) |
| Power curve (method of bins) (`power_curve_kpi`) * | 0.24 (0.19-0.30) | 0.69 (0.61-0.77) | 0.45 | 0.14 (0.11-0.19) | 0.16 (0.13-0.19) | 0.68 (0.66-0.70) | 0.12 (0.10-0.16) | 80 (68-93) | 160 (83-242) |

### PV systems

The excess recall has an interval above zero for: `zscore`, `ewma`, `cusum`, `iforest`, `rules`, `scada_qc`, `pv_autoencoder`, `pv_outage_expected`, `pv_outage_iforest`. The highest AUROC is that of `iforest`; the methods whose AUROC interval overlaps its interval are `ewma`, `cusum`, `iforest`, `pv_outage_iforest` (the intervals describe the sampling of weeks and events in this benchmark, so a method outside that list is not shown to be worse on other data, and one inside it is not shown to be equal).

| Method | Excess recall | Recall | Chance | Precision | Row F1 | AUROC | AP | Delay (h) | False alarms (h/yr) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Robust z-score (`zscore`) | 0.30 (0.27-0.34) | 0.65 (0.61-0.69) | 0.35 | 0.36 (0.33-0.40) | 0.22 (0.20-0.25) | 0.59 (0.58-0.60) | 0.30 (0.28-0.32) | 40 (35-44) | 173 (84-316) |
| EWMA chart (`ewma`) | 0.32 (0.27-0.36) | 0.48 (0.43-0.52) | 0.16 | 0.53 (0.43-0.66) | 0.24 (0.21-0.27) | 0.66 (0.64-0.67) | 0.34 (0.30-0.40) | 49 (42-56) | 175 (83-319) |
| CUSUM chart (`cusum`) | 0.30 (0.26-0.33) | 0.43 (0.38-0.49) | 0.14 | 0.46 (0.38-0.56) | 0.27 (0.24-0.30) | 0.66 (0.64-0.67) | 0.32 (0.28-0.38) | 48 (41-56) | 317 (164-544) |
| Isolation Forest (`iforest`) | 0.37 (0.33-0.42) | 0.63 (0.57-0.67) | 0.25 | 0.45 (0.40-0.52) | 0.22 (0.20-0.25) | 0.66 (0.65-0.67) | 0.32 (0.28-0.38) | 38 (34-43) | 214 (87-409) |
| Expert rules (`rules`) | 0.32 (0.29-0.35) | 0.36 (0.33-0.40) | 0.05 | 0.58 (0.46-0.76) | 0.13 (0.11-0.15) | 0.54 (0.53-0.55) | 0.24 (0.22-0.25) | 10 (6-14) | 85 (14-195) |
| scada-schema QC (`scada_qc`) | 0.14 (0.12-0.17) | 0.48 (0.42-0.55) | 0.34 | 0.20 (0.19-0.22) | 0.12 (0.10-0.13) | 0.52 (0.51-0.53) | 0.19 (0.18-0.20) | 34 (29-40) | 443 (317-577) |
| Autoencoder (PV) (`pv_autoencoder`) | 0.31 (0.28-0.35) | 0.58 (0.54-0.63) | 0.27 | 0.38 (0.34-0.43) | 0.19 (0.17-0.21) | 0.60 (0.59-0.61) | 0.29 (0.27-0.31) | 37 (31-42) | 173 (118-249) |
| Shortfall vs PVWatts model (`pv_outage_expected`) * | 0.31 (0.27-0.36) | 0.59 (0.53-0.66) | 0.28 | 0.37 (0.32-0.41) | 0.20 (0.18-0.22) | 0.62 (0.61-0.63) | 0.26 (0.23-0.31) | 30 (26-35) | 212 (110-374) |
| Outage features + Isolation Forest (`pv_outage_iforest`) * | 0.24 (0.20-0.28) | 0.44 (0.40-0.49) | 0.20 | 0.45 (0.35-0.56) | 0.19 (0.17-0.22) | 0.66 (0.64-0.67) | 0.29 (0.26-0.32) | 36 (31-42) | 143 (83-228) |

### Excess recall by fault type

Share of injected events that were detected, minus the share that the same alarms hit by chance, all sizes and datasets of the profile added; a dash is a type the method is not scored on. The sensitivity curve (excess recall against the size of the fault) is on the [project page](https://kwwaleszczynski-del.github.io/scada-fault-bench/).

Wind:

| Method | anemometer drift | nacelle direction offset | derating | power loss | temperature drift | flatline | unit jump | clock shift |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `zscore` | 0.13 | 0.06 | 0.11 | 0.01 | 0.04 | 0.81 | 0.65 | 0.07 |
| `ewma` | 0.58 | 0.47 | 0.42 | 0.28 | 0.29 | 0.47 | 0.47 | 0.08 |
| `cusum` | 0.71 | 0.66 | 0.37 | 0.47 | 0.42 | 0.27 | 0.36 | 0.07 |
| `iforest` | 0.29 | 0.24 | 0.19 | 0.04 | 0.07 | 0.86 | 0.30 | 0.23 |
| `rules` | 0.19 | 0.13 | 0.00 | 0.00 | 0.00 | 0.85 | 0.88 | 0.01 |
| `scada_qc` | 0.03 | 0.01 | 0.62 | 0.01 | 0.00 | 0.95 | 0.03 | 0.17 |
| `nbm_rc` | - | - | - | - | 0.15 | - | 0.90 | - |
| `nbm_gbm` | - | - | - | - | 0.16 | - | 0.87 | - |
| `power_curve_kpi` | 0.20 | - | 0.29 | 0.11 | - | 0.59 | - | 0.02 |

PV:

| Method | temperature drift | flatline | unit jump | clock shift | string loss | shading | inverter outage | pyranometer drift |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `zscore` | 0.04 | 0.55 | 0.88 | 0.46 | -0.01 | 0.04 | 0.47 | 0.06 |
| `ewma` | 0.27 | 0.39 | 0.73 | 0.15 | 0.24 | -0.04 | 0.66 | 0.17 |
| `cusum` | 0.24 | 0.37 | 0.65 | 0.24 | 0.23 | -0.01 | 0.57 | 0.11 |
| `iforest` | 0.14 | 0.59 | 0.56 | 0.59 | 0.16 | 0.16 | 0.73 | 0.07 |
| `rules` | 0.00 | 0.52 | 0.77 | 0.38 | -0.02 | 0.00 | 0.93 | 0.00 |
| `scada_qc` | 0.01 | 0.77 | 0.35 | 0.08 | 0.00 | 0.00 | -0.03 | 0.00 |
| `pv_autoencoder` | 0.00 | 0.59 | 0.89 | 0.44 | -0.05 | -0.02 | 0.74 | -0.01 |
| `pv_outage_expected` | - | 0.37 | - | 0.39 | 0.16 | 0.21 | 0.76 | -0.01 |
| `pv_outage_iforest` | - | 0.35 | - | 0.24 | 0.04 | 0.02 | 0.78 | 0.02 |

### Real faults: CARE to Compare, farm A

The same methods, trained on the first year of each turbine and scored on the prediction section; the truth is every row of the labelled anomaly that has a power reading (the turbine is usually in a downtime state inside an event, which is part of the fault and not a reason to ignore those rows). The chance level is the recall that the alarms of the method reach when they are shifted in time at random over the prediction section. These are single-turbine tables, so the residuals that compare a turbine with its neighbours do not exist and the methods that need neighbours work with less. 12 anomalies and 10 normal datasets are too few to rank methods; the intervals show how little this part can say. Precision is not reported: only the normal datasets contribute healthy time with no fault in it, and that is not a precision.

| Method | Excess recall | Recall | Chance | AUROC | AP | Delay (h) | False alarms (h/yr) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Robust z-score (`zscore`) | 0.09 (-0.05-0.20) | 0.83 (0.58-1.00) | 0.75 | 0.53 (0.49-0.57) | 0.44 (0.24-0.66) | 99 (48-162) | 30 (14-49) |
| EWMA chart (`ewma`) | -0.02 (-0.21-0.11) | 0.25 (0.00-0.53) | 0.27 | 0.65 (0.56-0.75) | 0.56 (0.32-0.77) | n/a | 16 (0-58) |
| CUSUM chart (`cusum`) | -0.12 (-0.30-0.00) | 0.00 (0.00-0.00) | 0.12 | 0.66 (0.54-0.78) | 0.56 (0.31-0.80) | n/a | 27 (0-68) |
| Isolation Forest (`iforest`) | -0.04 (-0.24-0.10) | 0.50 (0.22-0.79) | 0.54 | 0.54 (0.49-0.59) | 0.45 (0.25-0.67) | 165 (37-363) | 9 (1-23) |
| Expert rules (`rules`) | 0.00 (0.00-0.00) | 0.00 (0.00-0.00) | 0.00 | 0.50 (0.50-0.50) | 0.41 (0.22-0.63) | n/a | 0 (0-0) |
| scada-schema QC (`scada_qc`) | 0.06 (-0.09-0.18) | 0.67 (0.38-0.92) | 0.60 | 0.48 (0.34-0.59) | 0.42 (0.25-0.65) | 129 (75-187) | 19 (2-42) |
| Thermal model (RC) (`nbm_rc`) | -0.03 (-0.17-0.08) | 0.17 (0.00-0.40) | 0.20 | 0.57 (0.52-0.62) | 0.51 (0.27-0.72) | n/a | 59 (0-147) |
| Thermal model (GBM) (`nbm_gbm`) | -0.01 (-0.18-0.12) | 0.25 (0.00-0.50) | 0.26 | 0.54 (0.47-0.60) | 0.49 (0.25-0.70) | n/a | 63 (0-156) |
| Power curve (method of bins) (`power_curve_kpi`) | 0.03 (-0.09-0.13) | 0.67 (0.38-0.92) | 0.64 | 0.54 (0.47-0.60) | 0.44 (0.23-0.66) | 117 (66-172) | 64 (3-166) |

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

| Name | Profile | Family | Threshold | Fault types scored | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `zscore` | wind+pv | baseline | calibrated | all | largest absolute residual of the shared features (one-hour median), each divided by its 99.5 % quantile on the training weeks so that no single heavy-tailed feature decides the score |
| `ewma` | wind+pv | baseline | calibrated | all | exponentially weighted moving average of every residual clipped at 5 sigma (time constant about one day), largest standardised value |
| `cusum` | wind+pv | baseline | calibrated | all | two-sided CUSUM of every residual clipped at 5 sigma, on hourly values (allowance 1 sigma, memory 3 days), largest statistic |
| `iforest` | wind+pv | baseline | calibrated | all | Isolation Forest (100 trees, 4096 samples) on the vector of standardised residuals; missing residuals set to 0 |
| `rules` | wind+pv | baseline | fixed | all | fixed physical rules, no learning: the two anemometers differ by more than 1.5 m/s, a wind speed with zero variation over an hour while the turbine produces, gearbox oil outside -20 to 100 degC, vane + nacelle - absolute direction beyond 25 deg, power above 110 % of rated; PV: irradiance frozen in daylight, inverter below 2 % of rated in sun above 300 W/m2, module temperature outside -30 to 90 degC, power above 110 % of rated |
| `scada_qc` | wind+pv | baseline | fixed | all | rows flagged by the row-level checks of scada_schema.qc (flatline, spikes, out of range, units, power curve, wind direction, energy counter, energy against power) |
| `nbm_rc` | wind | portfolio | calibrated | temperature drift, unit jump | wind-thermal-nbm RCModel: first-order thermal model of the gearbox oil temperature from load, rotor speed, nacelle and ambient temperature; the standardised residual is the score Source: wind-thermal-nbm: wind_thermal_nbm.thermal.RCModel. |
| `nbm_gbm` | wind | portfolio | calibrated | temperature drift, unit jump | wind-thermal-nbm GBMModel: gradient boosting on the same inputs and their trailing means (1-12 h) for the gearbox oil temperature; the standardised residual is the score Source: wind-thermal-nbm: wind_thermal_nbm.ml.GBMModel. |
| `power_curve_kpi` | wind | portfolio | calibrated | anemometer drift, derating, power loss, flatline, clock shift | res-asset-kpi bin_power_curve per turbine on the training weeks; the standardised residual of power against the curve at the measured wind speed is the score Source: res-asset-kpi: res_asset_kpi.wind.bin_power_curve, curve_power (the method of wind-power-curve-kelmarsh). |
| `pv_autoencoder` | pv | portfolio | calibrated | all | pv-anomaly-autoencoder AutoencoderDetector per inverter: windows of 8 steps of power, performance ratio, irradiance, clear-sky index, temperatures and sun position; the reconstruction error (mse), smoothed over one hour, is the score; trained for at most 100 epochs on 80 % of the training days Source: pv-anomaly-autoencoder: pv_anomaly.autoencoder.AutoencoderDetector, pv_anomaly.features.build_features. |
| `pv_outage_expected` | pv | portfolio | calibrated | string loss, shading, inverter outage, pyranometer drift, flatline, clock shift | pv-outage-classifier detectors.score('expected'): shortfall of inverter power against a PVWatts-style model driven by the met station's irradiance and module temperature, calibrated per bin on the table that is scored (as in that repository); shortfall only Source: pv-outage-classifier: pv_outage.features.compute, pv_outage.detectors.score('expected'). |
| `pv_outage_iforest` | pv | portfolio | calibrated | string loss, shading, inverter outage, pyranometer drift, flatline, clock shift | pv-outage-classifier detectors.score('iforest'): Isolation Forest on that repository's step features (residuals, level, sun position, ...), fitted on the table that is scored, as in that repository Source: pv-outage-classifier: pv_outage.features.compute, pv_outage.detectors.score('iforest'). |

The other repositories of the portfolio were looked at and are not wrapped, for reasons that are properties of their inputs and not of their quality: `wind-wake-aware-verification` and `wind-bayes-twin` work on daily farm-level statistics with wake models, `pv-string-iv-faults` needs per-MPPT voltage and current channels that none of these datasets has, `pv-shade-fingerprint` produces a one-off diagnosis of a whole record and not a score per time stamp. Details in [docs/METHODS.md](docs/METHODS.md).

## The datasets

| Name | Dataset | Profile | Period (UTC) | Injected events |
| :--- | :--- | :--- | :--- | ---: |
| `kelmarsh_2019` | Kelmarsh 2019 | wind | 2019-01-01 to 2020-01-01 | 320 |
| `penmanshiel_2020` | Penmanshiel 2020 | wind | 2020-01-01 to 2021-01-01 | 320 |
| `hill_of_towie_2021` | Hill of Towie 2021 | wind | 2021-01-01 to 2022-01-01 | 316 |
| `pvdaq_34` | PVDAQ system 34 | pv | 2016-01-01 to 2018-01-01 | 318 |
| `pvdaq_1276` | PVDAQ system 1276 | pv | 2016-01-01 to 2018-01-01 | 308 |
| `pvdaq_1278` | PVDAQ system 1278 | pv | 2016-01-01 to 2018-01-01 | 317 |
| `pvdaq_35` | PVDAQ system 35 | pv | 2016-01-01 to 2018-01-01 | 314 |

The tables are built from the original open files with the loaders of `scada-schema` (`scada-fault-bench prepare`); nothing is redistributed. Kelmarsh and Penmanshiel are Cubico (CC BY 4.0), Hill of Towie is RES (CC BY 4.0), PVDAQ is NREL (CC BY 4.0). CARE to Compare is CC BY-SA 4.0, which also applies to datasets derived from it.

## The evaluation

For every dataset and fold the table is cut into a test block (91 days for wind farms, 150 for PV systems, which have one inverter or met station to carry all events), a margin of 7 days and the training part. A method is fitted on the training part without its calibration weeks (one week in four) and scored on the calibration weeks; the alarm threshold is the lowest one at which the merged alarms (below) cover at most 1 % of the healthy rows of the calibration weeks, so every calibrated method works at about the same operating point (the same hours of alarm on healthy data). Rule-based methods have a fixed threshold, and their false alarm rate is a result. Alarms closer than 6 hours are merged and alarms with less than 0.5 hours of flagged rows are dropped.

An event is detected when an alarm of its device covers at least one affected row of it (a row where the injected change is above its tolerance and an alarm can be right or wrong). The delay is the time from the first affected row to the first alarm on an affected row. An alarm is true when it covers an affected row, ignored when it lies only on rows where an alarm is neither right nor wrong (known stops, night, real outages of the open data), and false otherwise. The *chance* recall of a method is the share of the events that the alarms of the same method on the untouched test block would have hit; the *excess* is the difference. A wind alarm belongs to a turbine; a PV alarm belongs to the plant, because one met station and one or two inverters do not let a plant-level fault be attributed to a device. False alarms are counted in hours per year of healthy time. AUROC and average precision compare the ranks of the scores of affected and of healthy rows, where the rank of a score is its position among the scores of the healthy calibration rows of the same fold; they do not use the alarm threshold, but they do depend on how finely a score separates values near the healthy range, and a score with few distinct values (the rules) has a coarse rank.

Confidence intervals come from a bootstrap over time blocks: the unit of resampling is 4 consecutive weeks of a test block of one dataset and fold (longer than most events), with all devices, fault types and methods of those weeks drawn together, 1000 resamples; for the CARE part a block is one dataset. A mean delay over fewer than five detected events is not shown. They describe the sampling of weeks, devices and events inside this benchmark and nothing beyond it.

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

`scada-fault-bench inject --dataset kelmarsh_2019 --type derating --out out/derating` writes one injected test block, its truth and the row labels, for use with any other tool. `power_curve_kpi` needs `res-asset-kpi`, which `scada-schema` installs from PyPI. The other reference methods are optional dependencies that are installed from GitHub (they are not on PyPI), and `pv_autoencoder` also needs `torch`; a method whose repository cannot be imported is skipped and listed as unavailable in `results/summary.json`. The full run in this repository used all of them:

```bash
pip install torch   # for pv_autoencoder
pip install git+https://github.com/kwwaleszczynski-del/wind-thermal-nbm git+https://github.com/kwwaleszczynski-del/pv-anomaly-autoencoder git+https://github.com/kwwaleszczynski-del/pv-outage-classifier
```

## Limitations

* The faults are injected into recorded signals. Physical faults change more channels, and in ways that depend on the machine; the models here favour detectors that look at the channel that was changed. Nothing here says how a method finds a real fault except the CARE part, which is small.
* Faults of one type are drawn from one simple model with five sizes. The sensitivity curves describe that model.
* The open data are not guaranteed to be healthy. Real stops that the status log shows, night, and real inverter outages of PVDAQ are removed from the count of false alarms; real faults that are not visible in them are not, so alarms on them are counted as false. This is conservative.
* PV plants are one met station with one or two inverters: there is no string level and no device-level attribution. The PV results are plant-level.
* Calibrated methods share a false alarm budget on a calibration sample from the same year; a method whose scores drift with the seasons will exceed it, and the false alarm column shows by how much.
* An alarm on an affected row counts as a detection whatever its cause, so a method with many alarms also "detects" events by coincidence. The chance column and the excess recall show how much that is; the sensitivity curves show the excess and are not always monotone in the size of the fault. A method whose excess interval contains zero has not been shown to detect these faults.
* The simple baselines read a residual matrix that was built with the fault models in view: roughly one residual for each kind of fault (power against the curve, wind speed against neighbours and the second anemometer, direction against neighbours, gearbox temperature, frozen signals; for PV the ratio against irradiance, sun position, irradiance variation, temperatures). That makes them well suited to these faults by construction. `scada_qc` is run on the channels that the fault models touch, because it would otherwise flag quantised channels the faults do not concern. Nothing was tuned on the results, but the design was not blind to the faults.
* `nbm_rc`, `nbm_gbm` and `power_curve_kpi` are models taken from their repositories, not the alarm methods of those repositories (which average residuals over days and chart them with a zero-false-alarm limit), and they are scored only on the fault types in their channels; `pv_outage_*` likewise. Their overall rows say nothing about the other fault types.
* The calibrated methods share an operating point on healthy calibration data, but their scores have different autocorrelation, so the hours of alarm on the untouched test block still differ from method to method; the false alarm column shows it. A method whose score is too persistent to meet the budget gets the highest threshold of the search and may raise almost no alarm.
* The rule that hides real PV outages of the open data from the false alarm count (a whole calendar day on which at least four fifths of the sunny rows have zero power, or no power reading) is not the one-hour zero-power rule of the `rules` method; short real outages are counted as false alarms for every method.
* The reference methods are called through adapters that rename and rescale columns. A weak result can come from the adapter or from a method that was written for other data; the repositories that own the methods are the authority on what they do.
* 1000 bootstrap resamples of weeks are not independent of the fault draws: another seed gives other events. Differences within the intervals should not be read as differences.
* The file formats and the fault models are at version 0.1.0; results of different versions are not comparable.

## Reproduce

The run in this repository is `scada-fault-bench prepare` (all datasets, `--fetch` the first time), then `scada-fault-bench run --out results` (all datasets, all folds, 1 injection seed for wind and 2 for PV, which are the defaults) and `scada-fault-bench summarise --out results`, with one process per dataset to save time. `results/` holds the shards and `summary.json` of that run; `python scripts/render_readme.py` rebuilds this file and `python scripts/build_pages.py` the page from them. The tests in `tests/test_docs.py` compare both with the summary, and `tests/test_data.py` (marked `data`) repeats a small part of the run on the real tables.

## Citation and licence

Code: MIT. Cite the repository (`CITATION.cff`) and the datasets above. `scada-fault-bench` is built on `scada-schema` 0.1.1, whose loaders read the open datasets.
