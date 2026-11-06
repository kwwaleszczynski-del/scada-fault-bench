"""The method interface: what a detector is, how its scores become alarms, and the registry of methods.

A **detector** is any object with

* ``fit(train, meta)``: learn from a table of the scada-schema format (the training part of a split; it may contain
  gaps, real stops and real faults, as any historical data does);
* ``score(table, meta) -> numpy array``: one non-negative anomaly score per row of ``table`` (higher = more anomalous,
  ``NaN`` = cannot be assessed, e.g. at night for PV). The array must be aligned with the rows of ``table``.

The harness, not the method, turns scores into alarms, so that all methods are held to the same rules:

1. **Threshold.** The method is fitted on the training weeks that are not held out, and scored on the held-out
   *calibration weeks*; the alarm threshold is the ``CAL_QUANTILE`` quantile of those scores (pooled over devices). Methods
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

__all__ = ["CAL_QUANTILE", "FIXED_THRESHOLD", "MERGE_GAP_HOURS", "METHODS", "MIN_ALARM_HOURS", "Detector", "alarm_mask", "calibrate", "register"]

CAL_QUANTILE = 0.995
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
