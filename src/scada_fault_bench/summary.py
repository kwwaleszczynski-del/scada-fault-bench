"""From the shard files of a run to ``results/summary.json``: every number of the README and of the project page.

Selections (each with point values and 95 % block-bootstrap intervals, see :mod:`scada_fault_bench.stats`):

* ``cells``     method x dataset x fault type;
* ``datasets``  method x dataset, all fault types and the null scenario together;
* ``types``     method x profile x fault type, datasets of the profile added (also gives the sensitivity curve, recall by level);
* ``overall``   method x profile, everything added.

"Added" means the weekly sums of the selection are added before a metric is computed (a pooled metric, not a mean of means);
the block of resampling is the week of a test block of one dataset and fold. Fault types and datasets with more events or
more healthy time weigh more.

Usage:
    summary = summarise("results")        # reads results/shards/*.parquet
    write_summary("results")              # writes results/summary.json
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from scada_fault_bench import __version__, baselines, portfolio  # noqa: F401 - the methods register themselves
from scada_fault_bench.base import CAL_QUANTILE, FIXED_THRESHOLD, MERGE_GAP_HOURS, METHODS, MIN_ALARM_HOURS
from scada_fault_bench.data import BUFFER_DAYS, DATASETS, TEST_DAYS
from scada_fault_bench.evaluate import COLUMNS, TOLERANCE_DAYS
from scada_fault_bench.faults import FAULT_TYPES, GUARD_DAYS, LEVEL_COUNT, MIN_AFFECTED_HOURS
from scada_fault_bench.real import REAL_NAME
from scada_fault_bench.stats import METRICS, N_BOOT, bootstrap_metrics

__all__ = ["load_shards", "summarise", "write_summary"]

SUMS = COLUMNS


def load_shards(out: str | Path) -> dict[str, pd.DataFrame]:
    shard = Path(out) / "shards"
    kinds = {}
    for kind in ("blocks", "events", "truth", "runs"):
        files = sorted(shard.glob(f"*.{kind}.parquet"))
        frames = [pd.read_parquet(f) for f in files]
        frames = [f for f in frames if len(f)]
        kinds[kind] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if kinds["blocks"].empty:
        raise FileNotFoundError(f"no shards in {shard}; run the benchmark first")
    return kinds


def _entry(V: np.ndarray, n_boot: int, seed: int) -> dict:
    point, lo, hi = bootstrap_metrics(V, n_boot, seed)
    return {k: {"value": _f(point[k]), "lo": _f(lo[k]), "hi": _f(hi[k])} for k in METRICS}


def _f(x: float):
    return None if x is None or not np.isfinite(x) else float(x)


def _blocks_matrix(df: pd.DataFrame) -> np.ndarray:
    """One row per (dataset, fold, week) block of the selection, summed over methods' seeds and fault types of the selection."""
    g = df.groupby(["dataset", "fold", "week"], sort=True)[SUMS].sum()
    return g.to_numpy()


def summarise(out: str | Path, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    sh = load_shards(out)
    blocks, truth, runs = sh["blocks"], sh["truth"], sh["runs"]
    methods = [m for m in METHODS if m in set(blocks["method"])]
    datasets = [d for d in DATASETS if d in set(blocks["dataset"])]
    types_present = [t for t in FAULT_TYPES if t in set(blocks["ftype"])]
    profile_of = {d: DATASETS[d].profile for d in datasets} | {REAL_NAME: "wind"}
    blocks = blocks.assign(profile=blocks["dataset"].map(profile_of))
    real = blocks[blocks["dataset"] == REAL_NAME]
    blocks = blocks[blocks["dataset"] != REAL_NAME]
    S: dict = {"cells": [], "datasets": [], "types": [], "overall": [], "real": []}
    for m in methods:
        br = real[real["method"] == m]
        if not br.empty:
            S["real"].append({"method": m, "dataset": REAL_NAME, **_entry(_blocks_matrix(br), n_boot, seed)})
    for m in methods:
        bm = blocks[blocks["method"] == m]
        for prof in ("wind", "pv"):
            bp = bm[bm["profile"] == prof]
            if bp.empty:
                continue
            S["overall"].append({"method": m, "profile": prof, **_entry(_blocks_matrix(bp), n_boot, seed)})
            for t in types_present:
                if prof not in FAULT_TYPES[t].profiles():
                    continue
                bt = bp[bp["ftype"] == t]
                if bt.empty:
                    continue
                S["types"].append({"method": m, "profile": prof, "type": t, **_entry(_blocks_matrix(bt), n_boot, seed)})
        for d in datasets:
            bd = bm[bm["dataset"] == d]
            if bd.empty:
                continue
            S["datasets"].append({"method": m, "dataset": d, **_entry(_blocks_matrix(bd), n_boot, seed)})
            for t in types_present:
                bt = bd[bd["ftype"] == t]
                if not bt.empty:
                    S["cells"].append({"method": m, "dataset": d, "type": t, **_entry(_blocks_matrix(bt), n_boot, seed)})
    null = blocks[blocks["ftype"] == "null"].groupby(["method", "dataset"])[["neg_rows", "fa_rows", "al_fp", "al_ign"]].sum().reset_index()
    null["fa_h_per_year"] = null["fa_rows"] / null["neg_rows"].where(null["neg_rows"] > 0) * 8766.0
    S["null"] = [{"method": r.method, "dataset": r.dataset, "fa_h_per_year": _f(r.fa_h_per_year), "false_alarms": int(r.al_fp)} for r in null.itertuples()]

    ev_counts = []
    if len(truth):
        tt = truth[truth["dataset"] != REAL_NAME].groupby(["dataset", "type", "level"]).size().reset_index(name="n")
        ev_counts = [{"dataset": r.dataset, "type": r.type, "level": int(r.level), "n": int(r.n)} for r in tt.itertuples()]
    S["events"] = ev_counts
    S["real_events"] = []
    if len(truth):
        rt = truth[truth["dataset"] == REAL_NAME]
        n_norm = int(real.loc[real["ftype"] == "real_normal", "fold"].nunique()) if len(real) else 0
        S["real_events"] = [{"anomalies": int(len(rt)), "normal_datasets": n_norm}]
    S["levels"] = {t: {"levels": list(FAULT_TYPES[t].levels), "unit": FAULT_TYPES[t].unit, "kind": FAULT_TYPES[t].level_kind} for t in types_present}
    S["methods"] = {m: {"title": METHODS[m].title, "description": METHODS[m].description, "family": METHODS[m].family, "source": METHODS[m].source,
                        "profiles": list(METHODS[m].profiles), "fixed_threshold": METHODS[m].fixed_threshold} for m in methods}
    S["fault_types"] = {t: {"summary": FAULT_TYPES[t].summary, "profiles": list(FAULT_TYPES[t].profiles()), "unit": FAULT_TYPES[t].unit,
                            "levels": list(FAULT_TYPES[t].levels), "level_kind": FAULT_TYPES[t].level_kind} for t in types_present}
    S["datasets_info"] = {d: {"title": DATASETS[d].title, "profile": DATASETS[d].profile, "start": DATASETS[d].start, "end": DATASETS[d].end} for d in datasets}
    run_rows = []
    if len(runs):
        g = runs.groupby(["method", "dataset"]).agg(fit_s=("fit_s", "mean"), score_s=("score_s", "mean"), threshold=("threshold", "mean"),
                                                    cal_alarm_share=("cal_alarm_share", "mean"),
                                                    errors=("error", lambda s: "; ".join(sorted({x for x in s if x})))).reset_index()
        run_rows = [{"method": r.method, "dataset": r.dataset, "fit_s": _f(r.fit_s), "score_s": _f(r.score_s), "threshold": _f(r.threshold),
                     "cal_alarm_share": _f(r.cal_alarm_share), "errors": r.errors} for r in g.itertuples()]
    S["runs"] = run_rows
    S["order"] = {"methods": methods, "types": types_present, "datasets": datasets}
    folds = sorted(int(x) for x in blocks["fold"].unique())
    S["config"] = {
        "version": __version__, "n_boot": n_boot, "bootstrap_seed": seed, "folds": folds, "seeds": int(blocks["seed"].nunique()), "levels": LEVEL_COUNT,
        "test_days": {"wind": TEST_DAYS, "pv": 150}, "buffer_days": BUFFER_DAYS, "cal_quantile": CAL_QUANTILE, "fixed_threshold": FIXED_THRESHOLD, "merge_gap_hours": MERGE_GAP_HOURS,
        "min_alarm_hours": MIN_ALARM_HOURS, "tolerance_days": TOLERANCE_DAYS, "guard_days": GUARD_DAYS, "min_affected_hours": MIN_AFFECTED_HOURS,
        "n_events": int((truth["dataset"] != REAL_NAME).sum()) if len(truth) else 0, "python": platform.python_version(), "scada_schema": _version("scada_schema"), "pandas": pd.__version__, "numpy": np.__version__,
    }
    return S


def _version(mod: str) -> str:
    try:
        return __import__(mod).__version__
    except Exception:
        return "unknown"


def write_summary(out: str | Path, n_boot: int = N_BOOT, seed: int = 0) -> Path:
    S = summarise(out, n_boot, seed)
    path = Path(out) / "summary.json"
    path.write_text(json.dumps(S, indent=1, sort_keys=True), encoding="utf-8")
    return path
