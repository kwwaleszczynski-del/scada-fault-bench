"""The benchmark run: datasets x folds x fault types x methods, written to ``results/``.

For every dataset and fold the table is cut into a training part and a test block (see :mod:`scada_fault_bench.data`);
the test block is injected once per fault type (and once not at all, the *null* scenario, which measures false alarms on
untouched data). Every applicable method is fitted on the training weeks that are not held out, its alarm threshold is set
on the held-out calibration weeks, and its scores on every scenario are evaluated (:mod:`scada_fault_bench.evaluate`).

Files written for a dataset ``D`` (Parquet, in ``<out>/shards``): ``D.blocks`` (the weekly sums), ``D.events`` (one row per
injected event and method), ``D.truth`` (the injected faults), ``D.runs`` (fit and scoring times, thresholds, failures).

Usage:
    run_dataset("kelmarsh_2019", RunConfig(out="results", folds=(0, 1, 2, 3)))
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scada_schema.model import DatasetMeta

from scada_fault_bench import baselines, portfolio  # noqa: F401 - register the methods
from scada_fault_bench.base import METHODS, calibrate_budget
from scada_fault_bench.data import DATASETS, Split, load_dataset, make_split
from scada_fault_bench.evaluate import alarms_of, evaluate
from scada_fault_bench.faults import types_for
from scada_fault_bench.features import step_minutes
from scada_fault_bench.scenario import Scenario, grey_mask, make_scenario

log = logging.getLogger(__name__)

__all__ = ["RunConfig", "applicable_methods", "run_dataset", "run_split"]


@dataclass
class RunConfig:
    out: str | Path = "results"
    data: str | Path | None = None
    folds: tuple[int, ...] = (0, 1, 2, 3)
    seeds: int | None = None  # injection seeds per fold: 1 for wind farms, 2 for PV plants (fewer events per plant) unless given
    repeats: int | None = None  # events per level: 2 for wind farms, 1 for PV plants (their test block carries every event of one inverter)
    methods: tuple[str, ...] | None = None  # None: every available method
    types: tuple[str, ...] | None = None  # None: every fault type of the profile
    extra: dict = field(default_factory=dict)


def applicable_methods(profile: str, names: tuple[str, ...] | None = None) -> tuple[list[str], dict[str, str]]:
    """Methods that apply to a profile and can run here, and the reasons the others cannot."""
    ok, why = [], {}
    for n, cls in METHODS.items():
        if names is not None and n not in names:
            continue
        if profile not in cls.profiles:
            continue
        avail, reason = cls.available()
        if avail:
            ok.append(n)
        else:
            why[n] = reason
    return ok, why


def _tag(df: pd.DataFrame, **cols) -> pd.DataFrame:
    for i, (k, v) in enumerate(cols.items()):
        df.insert(i, k, v)
    return df


def run_split(split: Split, cfg: RunConfig, methods: list[str]) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[dict]]:
    """All scenarios of one fold scored by the given methods: blocks, events, truth tables and a run log."""
    meta = split.meta
    prof = meta.profile
    step = step_minutes(meta)
    types = [t for t in (cfg.types or types_for(prof)) if t in types_for(prof)]
    grey_test = grey_mask(split.test, meta)
    scenarios: list[Scenario] = [make_scenario(split, None, 0, grey=grey_test)]
    for seed in range(cfg.seeds or (1 if prof == "wind" else 2)):
        for t in types:
            scenarios.append(make_scenario(split, t, seed, repeats=cfg.repeats or (2 if prof == "wind" else 1), grey=grey_test))
    blocks, events, truths, runs = [], [], [], []
    for sc in scenarios:
        if len(sc.truth):
            truths.append(_tag(sc.truth.copy(), dataset=split.dataset, fold=split.fold, seed=sc.seed))
    train_fit = split.train[~split.cal].reset_index(drop=True)
    train_cal = split.train[split.cal].reset_index(drop=True)
    grey_cal = grey_mask(train_cal, meta)
    for name in methods:
        cls = METHODS[name]
        rec = {"dataset": split.dataset, "fold": split.fold, "method": name, "error": ""}
        try:
            t0 = time.perf_counter()
            m = cls().fit(train_fit, meta)
            rec["fit_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            s_cal = np.asarray(m.score(train_cal, meta), dtype="float64")
            if len(s_cal) != len(train_cal):
                raise ValueError(f"{name}: score() returned {len(s_cal)} values for {len(train_cal)} rows")
            thr = calibrate_budget(s_cal, train_cal["device_id"].to_numpy(), train_cal["timestamp"].to_numpy(), grey_cal, step, fixed=cls.fixed_threshold)
            rec["threshold"] = thr
            healthy_cal = s_cal[~grey_cal]
            rec["cal_alarm_share"] = float(np.mean(healthy_cal[np.isfinite(healthy_cal)] > thr)) if np.isfinite(healthy_cal).any() else float("nan")
            scope = cls.scope
            null_alarms = None
            for sc in scenarios:  # the null scenario comes first: its alarms are the placebo of every other scenario
                if sc.ftype is not None and scope is not None and sc.ftype not in scope:
                    continue
                s = np.asarray(m.score(sc.table, meta), dtype="float64")
                if len(s) != len(sc.table):
                    raise ValueError(f"{name}: score() returned {len(s)} values for {len(sc.table)} rows")
                if sc.ftype is None:
                    null_alarms = alarms_of(sc, s, thr, step)
                sums, ev = evaluate(sc, s, thr, step, cal_scores=healthy_cal, placebo=[null_alarms])
                _tag(sums, method=name, dataset=split.dataset, fold=split.fold, ftype=sc.ftype or "null", seed=sc.seed)
                blocks.append(sums)
                if len(ev):
                    events.append(_tag(ev, method=name, dataset=split.dataset, fold=split.fold, ftype=sc.ftype, seed=sc.seed))
            rec["score_s"] = time.perf_counter() - t0
        except Exception as e:  # a failing method must not stop the benchmark; the failure is recorded and shown
            log.exception("method %s failed on %s fold %s", name, split.dataset, split.fold)
            rec["error"] = f"{type(e).__name__}: {e}"
        runs.append(rec)
        log.info("%s fold %s %s: %s", split.dataset, split.fold, name, rec["error"] or f"fit {rec['fit_s']:.1f}s score {rec['score_s']:.1f}s thr {rec['threshold']:.3g}")
    return blocks, events, truths, runs


def _write(shard: Path, stem: str, blocks, events, truth, runs, unavailable) -> None:
    pd.concat(blocks, ignore_index=True).to_parquet(shard / f"{stem}.blocks.parquet")
    (pd.concat(events, ignore_index=True) if events else pd.DataFrame()).to_parquet(shard / f"{stem}.events.parquet")
    (pd.concat(truth, ignore_index=True) if truth else pd.DataFrame()).to_parquet(shard / f"{stem}.truth.parquet")
    r = pd.DataFrame(runs)
    r["unavailable"] = "; ".join(f"{n}: {w}" for n, w in unavailable.items())
    r.to_parquet(shard / f"{stem}.runs.parquet")


def run_dataset(name: str, cfg: RunConfig) -> Path:
    """Run every fold of one dataset and write its shard files (one set per fold, so that an interrupted run resumes: a fold whose
    files exist is skipped); returns the shard folder."""
    from scada_fault_bench.real import REAL_NAME, run_care

    profile = "wind" if name == REAL_NAME else DATASETS[name].profile
    methods, unavailable = applicable_methods(profile, cfg.methods)
    for n, why in unavailable.items():
        log.warning("method %s not available: %s", n, why)
    shard = Path(cfg.out) / "shards"
    shard.mkdir(parents=True, exist_ok=True)
    if name == REAL_NAME:
        if not (shard / f"{name}.runs.parquet").exists():
            b, e, t, r = run_care(cfg, methods)
            _write(shard, name, b, e, t, r, unavailable)
        return shard
    df, meta = load_dataset(name, cfg.data)
    for fold in cfg.folds:
        stem = f"{name}.f{fold}"
        if (shard / f"{stem}.runs.parquet").exists():  # written last: the fold is complete
            log.info("%s fold %s already done", name, fold)
            continue
        split = make_split(df, meta, fold, name=name)
        b, e, t, r = run_split(split, cfg, methods)
        _write(shard, stem, b, e, t, r, unavailable)
    return shard


def meta_of(name: str, data=None) -> DatasetMeta:
    return load_dataset(name, data)[1]
