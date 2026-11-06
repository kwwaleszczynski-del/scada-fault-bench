# scada-fault-bench

An open benchmark for fault detection in wind and PV SCADA data. Faults with a known position in time, device and size are injected into clean stretches of open datasets; any detector that reads a table of the [`scada-schema`](https://github.com/kwwaleszczynski-del/scada-schema) format is scored by one evaluation, with confidence intervals. A second part scores the same methods on real, labelled faults of CARE to Compare.

Not on PyPI yet. A DOI is assigned by Zenodo when the first release is archived. Live results: [kwwaleszczynski-del.github.io/scada-fault-bench](https://kwwaleszczynski-del.github.io/scada-fault-bench/).

**What this is not.** The injected faults are a signal-level model, not recorded failures. A method that does well here is not shown to be better in general, and a method that does badly may be badly matched to these models rather than bad. The tables describe this benchmark, these plants and these fault models, and nothing else. See [Limitations](#limitations).

## What was run

{{n_methods}} methods ({{n_baselines}} simple baselines and {{n_portfolio}} reference methods called from other repositories of this portfolio) on {{n_datasets}} datasets: {{n_wind_datasets}} wind farms and {{n_pv_datasets}} PV systems. {{n_events}} faults were injected: {{events_wind}} into the wind farms and {{events_pv}} into the PV systems, {{n_types}} fault types in all ({{n_wind_types}} for wind, {{n_pv_types}} for PV), each at {{n_levels}} sizes. {{n_folds}} test blocks per dataset, the rest of the year (or two years) is training data. The real-fault part has {{n_real_anomalies}} labelled anomalies and {{n_real_normal}} normal datasets of CARE farm A. Every number below is produced by `scada-fault-bench run` and `summarise` and written to `results/summary.json`; this README is rendered from it and a test fails when they differ.

### Wind farms

All fault types and datasets added. Values in brackets are 95 % block-bootstrap intervals. {{top_wind}}

{{table:overall_wind}}

### PV systems

{{top_pv}}

{{table:overall_pv}}

### Recall by fault type

Share of injected events that were detected, all sizes and datasets of the profile added. Recall against the size of the fault (the sensitivity curve) is on the [project page](https://kwwaleszczynski-del.github.io/scada-fault-bench/).

Wind:

{{table:types_wind}}

PV:

{{table:types_pv}}

### Real faults: CARE to Compare, farm A

The same methods, trained on the first year of each turbine and scored on the prediction section; the truth is the labelled anomaly. These are single-turbine tables, so the residuals that compare a turbine with its neighbours do not exist and the methods that need neighbours work with less. {{n_real_anomalies}} anomalies and {{n_real_normal}} normal datasets are too few to rank methods; the intervals show how little this part can say.

{{table:real}}

## The faults

Every fault is a change of the logged signals of one device. The start, the device and the size of an event are drawn from a seeded generator (the same table, type and seed give the same events), events of one device keep {{guard_days}} days apart. The truth of an event starts at the first row where the change exceeds a tolerance and ends at the last such row, leaving out the rows where an alarm is neither right nor wrong (night, a known stop, missing data); an event that would show in fewer than {{min_affected_h}} hours of data is redrawn. Where the change would move other channels by simple arithmetic (power, current and energy, the energy meter counter) they are changed together.

{{table:faults}}

## The methods

{{table:methods}}

The other repositories of the portfolio were looked at and are not wrapped, for reasons that are properties of their inputs and not of their quality: `wind-wake-aware-verification` and `wind-bayes-twin` work on daily farm-level statistics with wake models, `pv-string-iv-faults` needs per-MPPT voltage and current channels that none of these datasets has, `pv-shade-fingerprint` produces a one-off diagnosis of a whole record and not a score per time stamp. Details in [docs/METHODS.md](docs/METHODS.md).

## The datasets

{{table:datasets}}

The tables are built from the original open files with the loaders of `scada-schema` (`scada-fault-bench prepare`); nothing is redistributed. Kelmarsh and Penmanshiel are Cubico (CC BY 4.0), Hill of Towie is RES (CC BY 4.0), PVDAQ is NREL (CC BY 4.0). CARE to Compare is CC BY-SA 4.0, which also applies to datasets derived from it.

## The evaluation

For every dataset and fold the table is cut into a test block ({{test_days_wind}} days for wind farms, {{test_days_pv}} for PV systems, which have one inverter or met station to carry all events), a margin of {{buffer_days}} days and the training part. A method is fitted on the training part without its calibration weeks (one week in four) and scored on the calibration weeks; the alarm threshold is the {{cal_pct}} % upper tail of those scores, so every calibrated method has the same nominal false alarm budget. Rule-based methods have a fixed threshold, and their false alarm rate is a result. Alarms closer than {{merge_gap_h}} hours are merged and alarms with less than {{min_alarm_h}} hours of flagged rows are dropped.

An event is detected when an alarm covers any row between its first affected row and {{tolerance_days}} day after its last. The delay is the time from the first affected row to the first alarm row. An alarm is true when it overlaps such a window, ignored when it lies only on rows where an alarm is neither right nor wrong (known stops, night, real outages of the open data), and false otherwise. A wind alarm belongs to a turbine; a PV alarm belongs to the plant, because one met station and one or two inverters do not let a plant-level fault be attributed to a device. False alarms are counted in hours per year of healthy time. AUROC and average precision score the raw scores of affected rows against healthy rows without a threshold.

Confidence intervals come from a bootstrap over time blocks: the unit of resampling is a week of a test block of one dataset and fold, with all devices, fault types and methods of that week drawn together, {{n_boot}} resamples. They describe the sampling of weeks, devices and events inside this benchmark and nothing beyond it.

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
* {{n_boot}} bootstrap resamples of weeks are not independent of the fault draws: another seed gives other events. Differences within the intervals should not be read as differences.
* The file formats and the fault models are at version {{version}}; results of different versions are not comparable.

## Reproduce

`results/` holds the shards and `summary.json` of the run in this repository; `python scripts/render_readme.py` rebuilds this file and `python scripts/build_pages.py` the page from them. The tests in `tests/test_docs.py` compare both with the summary, and `tests/test_data.py` (marked `data`) repeats a small part of the run on the real tables.

## Citation and licence

Code: MIT. Cite the repository (`CITATION.cff`) and the datasets above. `scada-fault-bench` is built on `scada-schema` {{scada_schema_version}}, whose loaders read the open datasets.
