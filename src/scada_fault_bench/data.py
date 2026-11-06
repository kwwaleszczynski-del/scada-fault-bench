"""The datasets of the benchmark and how a table is cut into a training part and a test block.

The benchmark reads *canonical* tables of the ``scada-schema`` library (self-describing Parquet). They are not stored in this
repository: ``scada-fault-bench prepare`` builds them from the original open files with the loaders of ``scada-schema``
(``--fetch`` downloads the originals first), or you can point ``SCADA_FAULT_BENCH_DATA`` at a folder that already holds
``<name>.parquet`` files written by ``scada-schema load``.

Injected datasets (one table = one plant):

=================  =======  ===========================================  =============
name               profile  source                                       period used
=================  =======  ===========================================  =============
kelmarsh_2019      wind     Kelmarsh wind farm, 6 turbines (CC BY 4.0)   2019
penmanshiel_2020   wind     Penmanshiel, 14 turbines (CC BY 4.0)         2020
hill_of_towie_2021 wind     Hill of Towie, 21 turbines (CC BY 4.0)       2021
pvdaq_34 ...       pv       NREL PVDAQ systems 34, 1276, 1278, 35        2016-2017
=================  =======  ===========================================  =============

Usage:
    spec = DATASETS["kelmarsh_2019"]
    df, meta = load_dataset("kelmarsh_2019", data_dir="data")
    split = make_split(df, meta, fold=1)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scada_schema import read_meta, read_parquet
from scada_schema.model import DatasetMeta

log = logging.getLogger(__name__)

__all__ = ["BUFFER_DAYS", "DATASETS", "TEST_DAYS", "DatasetSpec", "Split", "data_dir", "test_block", "load_dataset", "make_split", "needed_columns"]

TEST_DAYS = 91  # wind; the PV datasets carry their own test_days
BUFFER_DAYS = 7
CAL_PERIOD_DAYS = 28  # one week in four of the training part is held out to calibrate alarm thresholds
CAL_WEEK_DAYS = 7


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    profile: str
    start: str  # inclusive, UTC
    end: str  # exclusive, UTC
    title: str
    test_days: int = 91  # length of a test block; PV plants have one inverter and one met station to carry all events, so a longer block


DATASETS: dict[str, DatasetSpec] = {s.name: s for s in [
    DatasetSpec("kelmarsh_2019", "wind", "2019-01-01", "2020-01-01", "Kelmarsh 2019"),
    DatasetSpec("penmanshiel_2020", "wind", "2020-01-01", "2021-01-01", "Penmanshiel 2020"),
    DatasetSpec("hill_of_towie_2021", "wind", "2021-01-01", "2022-01-01", "Hill of Towie 2021"),
    DatasetSpec("pvdaq_34", "pv", "2016-01-01", "2018-01-01", "PVDAQ system 34", 150),
    DatasetSpec("pvdaq_1276", "pv", "2016-01-01", "2018-01-01", "PVDAQ system 1276", 150),
    DatasetSpec("pvdaq_1278", "pv", "2016-01-01", "2018-01-01", "PVDAQ system 1278", 150),
    DatasetSpec("pvdaq_35", "pv", "2016-01-01", "2018-01-01", "PVDAQ system 35", 150),
]}

# the signals any method or fault may use; everything else is dropped on load to keep the tables small
WIND_COLUMNS = ["p_ac", "e_export", "e_export_counter", "i_ac", "wind_speed", "wind_speed_2", "wind_dir", "wind_dir_rel", "nacelle_dir",
                "rotor_speed", "pitch", "t_amb", "t_nacelle", "t_gearbox_oil", "t_gen_winding", "t_gen_bearing", "t_main_bearing",
                "shutdown_s", "status_code", "iec_category"]
PV_COLUMNS = ["p_ac", "p_dc", "i_ac", "i_dc", "v_dc", "t_amb", "t_inverter", "t_module", "poa", "wind_speed", "status_code"]
STATS = ("min", "max", "std")


def needed_columns(profile: str) -> list[str]:
    base = WIND_COLUMNS if profile == "wind" else PV_COLUMNS
    return [*base, *[f"{b}__{s}" for b in base for s in STATS]]


def data_dir(explicit: str | Path | None = None) -> Path:
    """Folder with the canonical tables: ``explicit``, ``$SCADA_FAULT_BENCH_DATA`` or ``./data``."""
    if explicit is not None:
        return Path(explicit)
    return Path(os.environ.get("SCADA_FAULT_BENCH_DATA") or "data")


def dataset_path(name: str, data: str | Path | None = None) -> Path:
    return data_dir(data) / f"{name}.parquet"


def load_dataset(name: str, data: str | Path | None = None) -> tuple[pd.DataFrame, DatasetMeta]:
    """Read one benchmark dataset, restricted to its period and to the signals the benchmark uses."""
    spec = DATASETS[name]
    path = dataset_path(name, data)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `scada-fault-bench prepare` (see README) or set SCADA_FAULT_BENCH_DATA")
    meta = read_meta(path)
    import pyarrow.parquet as pq

    present = set(pq.read_schema(path).names)
    cols = [c for c in needed_columns(spec.profile) if c in present]
    df, meta = read_parquet(path, columns=cols)
    t = df["timestamp"]
    df = df[(t >= pd.Timestamp(spec.start, tz="UTC")) & (t < pd.Timestamp(spec.end, tz="UTC"))].reset_index(drop=True)
    df = df.dropna(axis=1, how="all")
    df = df.sort_values(["device_id", "timestamp"], kind="stable").reset_index(drop=True)
    return df, meta


@dataclass
class Split:
    """A table cut for one fold: the training part (with a calibration subset) and the test block.

    ``train`` holds every row outside the test block and a ``BUFFER_DAYS`` margin around it. ``cal`` marks the
    calibration weeks inside ``train`` (one week in four); methods are fitted on ``train[~cal]`` and the alarm threshold
    is set on ``train[cal]``. ``test`` is the clean test block, to be injected with faults.
    """

    dataset: str
    fold: int
    meta: DatasetMeta
    train: pd.DataFrame
    cal: np.ndarray
    test: pd.DataFrame
    t_lo: pd.Timestamp
    t_hi: pd.Timestamp


def test_block(spec: DatasetSpec, fold: int, n_folds: int = 4, test_days: int | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Start and end of the test block of ``fold``: blocks of ``test_days`` days (91 for wind, 150 for PV) spread evenly over the period."""
    test_days = test_days or spec.test_days
    t0, t1 = pd.Timestamp(spec.start, tz="UTC"), pd.Timestamp(spec.end, tz="UTC")
    room = (t1 - t0) - pd.Timedelta(days=test_days)
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in 0..{n_folds - 1}")
    lo = t0 + room * fold / (n_folds - 1)
    lo = lo.floor("D")
    return lo, lo + pd.Timedelta(days=test_days)


def make_split(df: pd.DataFrame, meta: DatasetMeta, fold: int, name: str | None = None, n_folds: int = 4, spec: DatasetSpec | None = None,
               test_days: int | None = None) -> Split:
    """Cut ``df`` for ``fold``; ``spec`` and ``test_days`` are only overridden by tests on short synthetic tables."""
    name = name or meta.name
    spec = spec or DATASETS[name]
    t_lo, t_hi = test_block(spec, fold, n_folds, test_days or spec.test_days)
    buf = pd.Timedelta(days=BUFFER_DAYS)
    t = df["timestamp"]
    test = df[(t >= t_lo) & (t < t_hi)].reset_index(drop=True)
    train = df[(t < t_lo - buf) | (t >= t_hi + buf)].reset_index(drop=True)
    t0 = pd.Timestamp(spec.start, tz="UTC")
    week = ((train["timestamp"] - t0) // pd.Timedelta(days=CAL_WEEK_DAYS)).to_numpy()
    cal = (week % (CAL_PERIOD_DAYS // CAL_WEEK_DAYS)) == (CAL_PERIOD_DAYS // CAL_WEEK_DAYS - 1)
    return Split(name, fold, meta, train, cal, test, t_lo, t_hi)
