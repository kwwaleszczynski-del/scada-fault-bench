"""A scenario: a test table with injected faults, its truth and a label for every row.

Row labels (the unit of the time-based scores and of the false alarm rate):

* ``POS`` (1): the injected change at this row exceeds the tolerance of the fault type and the row is not grey.
* ``GREY`` (2): neither right nor wrong to alarm: the device is not operating or reports nothing (known stop in the
  status log, ``shutdown_s``, missing power or wind speed; for PV the sun is down, plane-of-array irradiance below
  ``DAYLIGHT_WM2``), or the row is inside the window of an event but unaffected, or within ``EVENT_TAIL_DAYS`` after it.
* ``NEG`` (0): everything else. Real faults and stops that the open data may contain and that the status log does not
  show are *not* removed: alarms on them count as false alarms, which is conservative.

The *group* of a device is the unit that an alarm is attributed to: the device itself for a wind farm, the whole plant
for a PV system (with one meteorological station and one or two inverters, a plant-level fault cannot be attributed to a
device from these data).

Usage:
    sc = make_scenario(split, "derating", seed=0)
    sc.table, sc.truth, sc.labels
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scada_schema.model import DatasetMeta

from scada_fault_bench.data import Split
from scada_fault_bench.faults import TRUTH_COLUMNS, inject_type

__all__ = ["DAYLIGHT_WM2", "EVENT_TAIL_DAYS", "GREY", "NEG", "POS", "Scenario", "group_ids", "grey_mask", "make_scenario"]

NEG, POS, GREY = 0, 1, 2
DAYLIGHT_WM2 = 20.0
EVENT_TAIL_DAYS = 1.0
NONOPERATING = {"forced_outage", "requested", "scheduled_maintenance", "planned_corrective", "environmental", "electrical",
                "technical_standby", "no_info"}


def group_ids(df: pd.DataFrame, meta: DatasetMeta) -> np.ndarray:
    """The alarm group of every row: the device for wind, the whole plant for PV."""
    if meta.profile == "wind":
        return df["device_id"].to_numpy(dtype=object)
    return np.full(len(df), "plant", dtype=object)


def grey_mask(df: pd.DataFrame, meta: DatasetMeta) -> np.ndarray:
    """Rows where an alarm is neither right nor wrong, from the (uninjected) table alone."""
    g = np.zeros(len(df), dtype=bool)
    if meta.profile == "wind":
        g |= df["p_ac"].isna().to_numpy() if "p_ac" in df else True
        if "wind_speed" in df:
            g |= df["wind_speed"].isna().to_numpy()
        if "iec_category" in df:
            g |= df["iec_category"].isin(NONOPERATING).to_numpy()
        if "shutdown_s" in df:
            g |= (df["shutdown_s"].fillna(0) > 0).to_numpy()
    else:
        met = [d.id for d in meta.devices if d.type == "met"]
        poa = df.loc[df["device_id"].isin(met), ["timestamp", "poa"]].drop_duplicates("timestamp").set_index("timestamp")["poa"]
        day = df["timestamp"].map(poa).to_numpy(dtype="float64") >= DAYLIGHT_WM2
        g |= ~np.nan_to_num(day, nan=0).astype(bool)
        # an inverter that reports nothing, or zero power in strong sun, is a real outage of the open data: neither right nor wrong to alarm on
        sun = df["timestamp"].map(poa).to_numpy(dtype="float64")
        inv = df["device_id"].isin([d.id for d in meta.devices if d.type == "inverter"]).to_numpy()
        rated = df["device_id"].map({d.id: (d.rated_kw or d.capacity_kwp or np.nan) for d in meta.devices}).to_numpy(dtype="float64")
        p = df["p_ac"].to_numpy(dtype="float64")
        down = inv & ((np.isnan(p) & (sun >= DAYLIGHT_WM2)) | ((p < 0.02 * rated) & (sun > 300)))
        g |= df["timestamp"].isin(df.loc[down, "timestamp"]).to_numpy()
    return g


@dataclass
class Scenario:
    dataset: str
    fold: int
    ftype: str | None  # None: the null scenario, no injection
    seed: int
    meta: DatasetMeta
    table: pd.DataFrame
    truth: pd.DataFrame
    labels: np.ndarray  # uint8 per row of table: NEG, POS, GREY
    t_lo: pd.Timestamp
    t_hi: pd.Timestamp

    @property
    def name(self) -> str:
        return f"{self.dataset}/f{self.fold}/{self.ftype or 'null'}/s{self.seed}"


def build_labels(table: pd.DataFrame, meta: DatasetMeta, truth: pd.DataFrame, affected: np.ndarray, grey: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """Row labels and the truth table restricted to what can be observed.

    A row is positive only where the injected change is above its tolerance *and* the row is not grey: a change in the
    logged temperature of a PV inverter at night, or of a turbine that is stopped, is not something a method can be asked
    to find. The start and end of every event are therefore the first and last positive row (``n_affected`` counts them),
    and an event without any positive row is dropped from the truth.
    """
    lab = np.zeros(len(table), dtype=np.uint8)
    lab[grey] = GREY
    pos = affected & ~grey
    kept = []
    if len(truth):
        ts = table["timestamp"]
        tsn = ts.to_numpy()
        dev = table["device_id"].to_numpy(dtype=object)
        tail = pd.Timedelta(days=EVENT_TAIL_DAYS)
        step = pd.Timedelta(meta.interval)
        for r in truth.itertuples():
            on_dev = dev == r.device_id
            lab[on_dev & ((ts >= r.window_start) & (ts < r.end + tail)).to_numpy()] = GREY
            inside = np.flatnonzero(on_dev & pos & ((ts >= r.start) & (ts < r.end)).to_numpy())
            if len(inside):
                kept.append({**r._asdict(), "start": pd.Timestamp(tsn[inside[0]]), "end": pd.Timestamp(tsn[inside[-1]]) + step, "n_affected": len(inside)})
    lab[pos] = POS
    out = pd.DataFrame(kept, columns=["Index", *truth.columns]).drop(columns="Index") if kept else truth.iloc[:0]
    for c in ("start", "end", "window_start", "window_end"):
        out[c] = pd.to_datetime(out[c], utc=True).dt.as_unit("ns")
    return lab, out.reset_index(drop=True)


def make_scenario(split: Split, ftype: str | None, seed: int, repeats: int = 2, grey: np.ndarray | None = None) -> Scenario:
    """Inject ``ftype`` into the test block of ``split`` (``None``: no injection) and label the rows."""
    base = split.test
    grey = grey_mask(base, split.meta) if grey is None else grey
    if ftype is None:
        table, truth, affected = base, pd.DataFrame(columns=TRUTH_COLUMNS), np.zeros(len(base), dtype=bool)
    else:
        table, truth, affected = inject_type(base, split.meta, ftype, seed=seed * 1000 + split.fold, t_lo=split.t_lo, t_hi=split.t_hi, repeats=repeats)
    labels, truth = build_labels(table, split.meta, truth, affected, grey)
    return Scenario(split.dataset, split.fold, ftype, seed, split.meta, table, truth, labels, split.t_lo, split.t_hi)
