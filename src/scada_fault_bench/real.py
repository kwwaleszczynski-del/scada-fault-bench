"""The part of the benchmark that is not injected: real, labelled faults of CARE to Compare, wind farm A.

CARE to Compare (M. Gück et al., Zenodo 10.5281/zenodo.15846963, CC BY-SA 4.0) has 22 datasets of farm A, each one turbine with
about a year of training data and a prediction section; 12 of them contain a labelled anomaly (``event_label``), 10 are labelled
normal. Time stamps are anonymised (shifted), which is irrelevant here: no method uses the calendar. The turbines of the datasets
are different turbines of one farm, but every dataset is a single table, so there is no fleet reference: the residuals that
compare a turbine with its neighbours are missing and the methods work with what one turbine has.

Protocol, the same as for injected faults: fit on the training section without the calibration weeks, set the threshold on
the calibration weeks, score the prediction section. Truth: the rows of the labelled anomaly. Grey rows: rows whose status
is not normal operation, and one day after the event. Everything else of a prediction section is healthy; the 10 normal
datasets are pure false-alarm tests. Events are not graded by size (``level`` is 1).

The dataset name in the results is ``care_A``; the ``fold`` of a block is the CARE ``event_id``.

Usage:
    prepare_care("data", fetch=True)        # writes data/care_A_<event_id>.parquet and .events.parquet
    run_care(RunConfig(out="results", data="data"))
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scada_schema import read_parquet, write_parquet

from scada_fault_bench.base import METHODS, calibrate
from scada_fault_bench.data import CAL_PERIOD_DAYS, CAL_WEEK_DAYS, data_dir
from scada_fault_bench.evaluate import evaluate
from scada_fault_bench.faults import TRUTH_COLUMNS
from scada_fault_bench.features import step_minutes
from scada_fault_bench.scenario import EVENT_TAIL_DAYS, GREY, NEG, POS, Scenario

log = logging.getLogger(__name__)

__all__ = ["REAL_NAME", "care_events", "prepare_care", "run_care"]

REAL_NAME = "care_A"
FARM = "A"


def _paths(event_id: int, data: str | Path | None) -> tuple[Path, Path]:
    d = data_dir(data)
    return d / f"{REAL_NAME}_{event_id}.parquet", d / f"{REAL_NAME}_{event_id}.events.parquet"


def care_events(data: str | Path | None = None) -> list[int]:
    """Event ids of the prepared CARE datasets, sorted."""
    d = data_dir(data)
    return sorted(int(p.stem.split("_")[-1]) for p in d.glob(f"{REAL_NAME}_*.parquet") if not p.stem.endswith(".events") and p.stem.split("_")[-1].isdigit())


def prepare_care(data: str | Path | None = None, fetch: bool = False, raw: str | Path | None = None) -> list[Path]:
    """Write every dataset of CARE farm A as a canonical table (with ``x_train_test``) and its event log."""
    from scada_schema.loaders import care
    from scada_schema.loaders._paths import raw_dir

    folder = raw_dir("care", raw)
    if fetch:
        care.fetch(folder, {FARM: list(range(96))})  # the zip holds 95 datasets over three farms; ids that are not farm A are simply not matched
    ids = [int(i) for i in care.list_datasets(folder, FARM)["event_id"]]
    out = []
    for e in ids:
        loaded = care.load(folder, farm=FARM, event_id=e)
        t, ev = _paths(e, data)
        write_parquet(loaded.table, loaded.meta, t)
        loaded.events.to_parquet(ev)
        out.append(t)
    log.info("wrote %d CARE datasets of farm %s to %s", len(out), FARM, t.parent)
    return out


def _scenario(df: pd.DataFrame, meta, ev: pd.DataFrame, event_id: int) -> Scenario:
    step = pd.Timedelta(meta.interval)
    test = df[df["x_train_test"] == "prediction"].reset_index(drop=True)
    labels = np.full(len(test), NEG, dtype=np.uint8)
    grey = (test["iec_category"].fillna("no_info") != "full_performance").to_numpy() | test["p_ac"].isna().to_numpy()
    labels[grey] = GREY
    rows = []
    anomalies = ev[ev["status"] == "anomaly"]
    for r in anomalies.itertuples():
        inside = ((test["timestamp"] >= r.start) & (test["timestamp"] < r.end)).to_numpy()
        tail = ((test["timestamp"] >= r.end) & (test["timestamp"] < r.end + pd.Timedelta(days=EVENT_TAIL_DAYS))).to_numpy()
        labels[tail] = GREY
        labels[inside & ~grey] = POS
        labels[inside & grey] = GREY
        rows.append({"fault_id": f"care_A-{event_id}", "type": "real_fault", "device_id": r.device_id, "level": 1, "magnitude": float("nan"), "unit": "",
                     "start": r.start, "end": r.end, "window_start": r.start, "window_end": r.end, "n_affected": int((labels[inside] == POS).sum()), "params": "{}"})
    truth = pd.DataFrame(rows, columns=TRUTH_COLUMNS)
    for c in ("start", "end", "window_start", "window_end"):
        truth[c] = pd.to_datetime(truth[c], utc=True).dt.as_unit("ns")
    t_lo = test["timestamp"].min().floor(step)
    t_hi = test["timestamp"].max().floor(step) + step
    return Scenario(REAL_NAME, event_id, "real_fault" if len(truth) else "real_normal", 0, meta, test, truth, labels, t_lo, t_hi)


def run_care(cfg, methods: list[str]) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[dict]]:
    """All prepared CARE datasets scored by the given methods: blocks, events, truth tables and a run log."""
    blocks, events, truths, runs = [], [], [], []
    for e in care_events(cfg.data):
        tpath, epath = _paths(e, cfg.data)
        df, meta = read_parquet(tpath)
        ev = pd.read_parquet(epath)
        step = step_minutes(meta)
        sc = _scenario(df, meta, ev, e)
        if len(sc.truth):
            tt = sc.truth.copy()
            tt.insert(0, "seed", 0)
            tt.insert(0, "fold", e)
            tt.insert(0, "dataset", REAL_NAME)
            truths.append(tt)
        train = df[df["x_train_test"] == "train"].reset_index(drop=True)
        t0 = train["timestamp"].min()
        week = ((train["timestamp"] - t0) // pd.Timedelta(days=CAL_WEEK_DAYS)).to_numpy()
        cal = (week % (CAL_PERIOD_DAYS // CAL_WEEK_DAYS)) == (CAL_PERIOD_DAYS // CAL_WEEK_DAYS - 1)
        train_fit, train_cal = train[~cal].reset_index(drop=True), train[cal].reset_index(drop=True)
        grey_cal = (train_cal["iec_category"].fillna("no_info") != "full_performance").to_numpy() | train_cal["p_ac"].isna().to_numpy()
        for name in methods:
            cls = METHODS[name]
            rec = {"dataset": REAL_NAME, "fold": e, "method": name, "error": ""}
            try:
                m = cls().fit(train_fit, meta)
                s_cal = np.asarray(m.score(train_cal, meta), dtype="float64")
                thr = calibrate(s_cal[~grey_cal], fixed=cls.fixed_threshold)
                rec["threshold"] = thr
                s = np.asarray(m.score(sc.table, meta), dtype="float64")
                sums, evs = evaluate(sc, s, thr, step)
                sums.insert(0, "seed", 0)
                sums.insert(0, "ftype", sc.ftype)
                sums.insert(0, "fold", e)
                sums.insert(0, "dataset", REAL_NAME)
                sums.insert(0, "method", name)
                blocks.append(sums)
                if len(evs):
                    evs.insert(0, "seed", 0)
                    evs.insert(0, "ftype", sc.ftype)
                    evs.insert(0, "fold", e)
                    evs.insert(0, "dataset", REAL_NAME)
                    evs.insert(0, "method", name)
                    events.append(evs)
            except Exception as ex:  # the failure is recorded and shown, the benchmark goes on
                log.exception("method %s failed on CARE event %s", name, e)
                rec["error"] = f"{type(ex).__name__}: {ex}"
            runs.append(rec)
        log.info("care_A event %s (%s) done", e, sc.ftype)
    return blocks, events, truths, runs
