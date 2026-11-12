"""The method interface: what a detector is, how its scores become alarms, and the registry of methods.

A **detector** is any object with

* ``fit(train, meta)``: learn from a table of the scada-schema format (the training part of a split; it may contain
  gaps, real stops and real faults, as any historical data does);
* ``score(table, meta) -> numpy array``: one non-negative anomaly score per row of ``table`` (higher = more anomalous,
  ``NaN`` = cannot be assessed, e.g. at night for PV). The array must be aligned with the rows of ``table``.

The harness, not the method, turns scores into alarms, so that all methods are held to the same rules:

1. **Threshold.** The method is fitted on the training weeks that are not held out, and scored on the held-out
   *calibration weeks*; the alarm threshold is the smallest one at which the merged alarms (rule 2) cover at most
   ``ALARM_BUDGET`` of the healthy rows of those weeks, pooled over devices, so that every calibrated method works at the
   same operating point (about the same hours of alarm on healthy data). Methods
   with ``fixed_threshold = True`` (hand-written rules, the quality control of scada-schema) are not calibrated: a score above
   ``FIXED_THRESHOLD`` is an alarm, and their false alarm rate is a result, not a setting.
2. **Alarm intervals.** Rows above the threshold that are less than ``MERGE_GAP_HOURS`` apart form one alarm (the rows
   between them belong to it); an alarm with fewer than ``MIN_ALARM_HOURS`` of flagged rows is dropped.

Usage:
    @register
    class MyMethod(Detector):
        name = "my_method"
        profiles = ("wind",)
        def fit(self, train, meta): ...
        def score(self, table, meta): return np.zeros(len(table))
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scada_schema.model import DatasetMeta

__all__ = ["ALARM_BUDGET", "CAL_QUANTILE", "FIXED_THRESHOLD", "MERGE_GAP_HOURS", "METHODS", "MIN_ALARM_HOURS", "Detector", "alarm_mask", "calibrate", "calibrate_budget", "register"]

CAL_QUANTILE = 0.995  # the raw-score quantile that calibrate() uses; the benchmark itself uses calibrate_budget()
ALARM_BUDGET = 0.01  # share of healthy rows that merged alarms may cover on the calibration weeks
FIXED_THRESHOLD = 0.5
MERGE_GAP_HOURS = 6.0
MIN_ALARM_HOURS = 0.5


class Detector:
    """Base class of a method. Subclasses set the class attributes and implement ``fit`` and ``score``."""

    name: str = ""
    title: str = ""
    description: str = ""
    profiles: tuple[str, ...] = ("wind", "pv")
    family: str = "baseline"  # "baseline" or "portfolio" (wraps another repository of the portfolio)
    source: str = ""  # for family == "portfolio": the repository and entry point that is called
    fixed_threshold: bool = False
    scope: tuple[str, ...] | None = None  # fault types whose injected change is in a channel the method reads; None: all types of the profile

    @classmethod
    def available(cls) -> tuple[bool, str]:
        """Whether the method can run here, and if not, why (a missing optional dependency)."""
        return True, ""

    def fit(self, train: pd.DataFrame, meta: DatasetMeta) -> Detector:
        raise NotImplementedError

    def score(self, table: pd.DataFrame, meta: DatasetMeta) -> np.ndarray:
        raise NotImplementedError


METHODS: dict[str, type[Detector]] = {}


def register(cls: type[Detector]) -> type[Detector]:
    if not cls.name or cls.name in METHODS:
        raise ValueError(f"method name missing or taken: {cls.name!r}")
    METHODS[cls.name] = cls
    return cls


def calibrate(scores: np.ndarray, fixed: bool = False, q: float = CAL_QUANTILE) -> float:
    """The alarm threshold: the ``q`` quantile of the finite calibration scores, or ``FIXED_THRESHOLD`` for a rule-based method."""
    if fixed:
        return FIXED_THRESHOLD
    s = scores[np.isfinite(scores)]
    if len(s) == 0:
        return float("inf")
    return max(float(np.quantile(s, q)), 1e-9)


def alarm_mask(flags: np.ndarray, step_minutes: float, merge_gap_hours: float = MERGE_GAP_HOURS, min_alarm_hours: float = MIN_ALARM_HOURS) -> np.ndarray:
    """Alarm rows of one regular series of boolean ``flags``: merge runs closer than ``merge_gap_hours`` (the gap rows
    become part of the alarm), then drop runs shorter than ``min_alarm_hours``.
    """
    f = np.asarray(flags, dtype=bool)
    n = len(f)
    if n == 0 or not f.any():
        return np.zeros(n, dtype=bool)
    gap = max(1, int(round(merge_gap_hours * 60.0 / step_minutes)))
    minlen = max(1, int(round(min_alarm_hours * 60.0 / step_minutes)))
    idx = np.flatnonzero(f)
    brk = np.flatnonzero(np.diff(idx) > gap)  # a gap of more than `gap` rows between flagged rows separates two alarms
    starts = np.r_[idx[0], idx[brk + 1]]
    ends = np.r_[idx[brk], idx[-1]]
    out = np.zeros(n, dtype=bool)
    for a, b in zip(starts, ends):
        if int(f[a:b + 1].sum()) >= minlen:
            out[a:b + 1] = True
    return out


def _segments(devices: np.ndarray, timestamps: np.ndarray, step_ns: int) -> list[tuple[int, int]]:
    """Runs of consecutive rows of one device on the interval grid (a change of device or a gap starts a new run)."""
    n = len(devices)
    if n == 0:
        return []
    brk = np.flatnonzero((devices[1:] != devices[:-1]) | (np.diff(timestamps) != step_ns)) + 1
    starts = np.r_[0, brk]
    ends = np.r_[brk, n]
    return list(zip(starts, ends))


def alarm_share(scores: np.ndarray, threshold: float, segments: list[tuple[int, int]], healthy: np.ndarray, step_minutes: float) -> float:
    """Share of the healthy rows that lie inside a merged alarm at ``threshold``."""
    inside = 0
    for a, b in segments:
        flags = np.nan_to_num(scores[a:b], nan=-np.inf) > threshold
        inside += int((alarm_mask(flags, step_minutes) & healthy[a:b]).sum())
    return inside / max(int(healthy.sum()), 1)


def calibrate_budget(scores: np.ndarray, devices: np.ndarray, timestamps: np.ndarray, grey: np.ndarray, step_minutes: float,
                     fixed: bool = False, budget: float = ALARM_BUDGET) -> float:
    """The smallest threshold at which merged alarms cover at most ``budget`` of the healthy (not grey) rows of the table.

    ``devices`` and ``timestamps`` (datetime64) give the row order; rows of one device on consecutive intervals are one series.
    A rule-based method (``fixed``) gets ``FIXED_THRESHOLD``. The share falls with the threshold, so the search is a bisection
    over a grid of quantiles of the finite scores.
    """
    if fixed:
        return FIXED_THRESHOLD
    s = np.asarray(scores, dtype="float64")
    healthy = ~np.asarray(grey, dtype=bool) & np.isfinite(s)
    if not healthy.any():
        return float("inf")
    step_ns = int(round(step_minutes * 60e9))
    segs = _segments(np.asarray(devices), pd.DatetimeIndex(timestamps).as_unit("ns").asi8, step_ns)
    cand = np.unique(np.quantile(s[healthy], 1.0 - np.geomspace(1e-5, 0.3, 60)))
    lo, hi = 0, len(cand) - 1
    if alarm_share(s, cand[hi], segs, ~np.asarray(grey, dtype=bool), step_minutes) > budget:
        return float(cand[hi])  # even the highest candidate gives more alarm than the budget (autocorrelated scores)
    while lo < hi:
        mid = (lo + hi) // 2
        if alarm_share(s, cand[mid], segs, ~np.asarray(grey, dtype=bool), step_minutes) <= budget:
            hi = mid
        else:
            lo = mid + 1
    return max(float(cand[lo]), 1e-9)
