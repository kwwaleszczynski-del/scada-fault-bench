"""Scoring one method on one scenario: alarms, matching to the injected events, and the additive sums behind every metric.

Everything the benchmark reports is a function of *sums* that can be added over weeks, so that a bootstrap over time
blocks only has to re-add them (see :mod:`scada_fault_bench.stats`). For one scenario the function :func:`evaluate`
returns, per week of the test block, the vector ``COLUMNS``:

``ev_n``, ``ev_det``, ``ev_delay_h``   events starting in the week, how many were detected, summed delay of those in hours;
``ev_n_l1`` .. ``ev_det_l5``           the same by fault level 1 (smallest) to 5;
``al_tp``, ``al_fp``, ``al_ign``       alarms starting in the week: overlapping an event, on healthy rows, only on grey rows;
``neg_rows``, ``fa_rows``              rows labelled healthy, and of those the rows inside an alarm (the false alarm rate);
``row_tp``, ``row_fp``, ``row_fn``     row-level counts of alarm against affected rows;
``hp_*``, ``hn_*``                     histograms of the score divided by the threshold on affected and on healthy rows (AUROC, AP).

Matching rules. A group is a device (wind) or the whole plant (PV, see :mod:`scada_fault_bench.scenario`). An event is
*detected* when an alarm of its group covers any row between the first affected row and ``TOLERANCE_DAYS`` after the last;
the *delay* is the time from the first affected row to the first alarm row there (0 when an alarm is already running).
An alarm is a *true* alarm when it overlaps such a window, *ignored* when all its rows are grey, otherwise *false*.

Usage:
    sums, events = evaluate(scenario, scores, threshold, step_minutes=10)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scada_fault_bench.base import alarm_mask
from scada_fault_bench.faults import LEVEL_COUNT
from scada_fault_bench.scenario import GREY, NEG, POS, Scenario, group_ids

__all__ = ["COLUMNS", "HIST_BINS", "TOLERANCE_DAYS", "evaluate", "hist_edges"]

TOLERANCE_DAYS = 1.0
HIST_INNER = 60
HIST_BINS = HIST_INNER + 2
SCALARS = ["ev_n", "ev_det", "ev_delay_h", *[f"ev_n_l{i}" for i in range(1, LEVEL_COUNT + 1)], *[f"ev_det_l{i}" for i in range(1, LEVEL_COUNT + 1)],
           "al_tp", "al_fp", "al_ign", "neg_rows", "fa_rows", "row_tp", "row_fp", "row_fn"]
COLUMNS = [*SCALARS, *[f"hp_{i}" for i in range(HIST_BINS)], *[f"hn_{i}" for i in range(HIST_BINS)]]
_IDX = {c: i for i, c in enumerate(COLUMNS)}


def hist_edges() -> np.ndarray:
    """Bin edges of the score divided by the alarm threshold: below 0.1, 60 geometric bins up to 100, above."""
    return np.geomspace(0.1, 100.0, HIST_INNER + 1)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if not mask.any():
        return []
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _grid(sc: Scenario, scores: np.ndarray, step: pd.Timedelta):
    """Scores and labels on a regular (group x step) grid of the test block; groups of PV are merged to one plant series."""
    gid = group_ids(sc.table, sc.meta)
    groups = sorted(set(gid))
    gix = np.searchsorted(groups, gid)
    n_steps = int(np.ceil((sc.t_hi - sc.t_lo) / step))
    t = ((sc.table["timestamp"] - sc.t_lo) // step).to_numpy()
    ok = (t >= 0) & (t < n_steps)
    S = np.full((len(groups), n_steps), np.nan)
    pri = np.full((len(groups), n_steps), -1, dtype=np.int8)
    np.fmax.at(S, (gix[ok], t[ok]), scores[ok])
    rank = np.array([0, 2, 1], dtype=np.int8)[sc.labels]  # NEG 0, POS 2, GREY 1: a plant row is positive if any device row is
    np.maximum.at(pri, (gix[ok], t[ok]), rank[ok])
    L = np.where(pri == 0, NEG, np.where(pri == 2, POS, GREY)).astype(np.uint8)
    return groups, S, L


def evaluate(sc: Scenario, scores: np.ndarray, threshold: float, step_minutes: float, fixed: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sums per week of the test block (rows: weeks, columns ``COLUMNS``) and one row per injected event."""
    step = pd.Timedelta(minutes=step_minutes)
    groups, S, L = _grid(sc, scores, step)
    n_steps = S.shape[1]
    per_week = int(round(7 * 24 * 60 / step_minutes))
    n_weeks = int(np.ceil(n_steps / per_week))
    out = np.zeros((n_weeks, len(COLUMNS)))
    tol = int(round(TOLERANCE_DAYS * 24 * 60 / step_minutes))
    alarms = np.zeros_like(L, dtype=bool)
    for g in range(len(groups)):
        flags = np.nan_to_num(S[g], nan=-np.inf) > threshold
        alarms[g] = alarm_mask(flags, step_minutes)

    ev_rows = []
    windows: dict[int, list[tuple[int, int]]] = {g: [] for g in range(len(groups))}
    for r in sc.truth.itertuples():
        g = groups.index("plant") if sc.meta.profile == "pv" else groups.index(r.device_id)
        t0 = int((r.start - sc.t_lo) // step)
        t1 = int(np.ceil((r.end - sc.t_lo) / step))
        lo, hi = max(t0, 0), min(t1 + tol, n_steps)
        windows[g].append((lo, hi))
        seg = alarms[g, lo:hi]
        det = bool(seg.any())
        delay = 0.0
        if det:
            first = int(np.argmax(seg))
            delay = 0.0 if (lo + first <= t0) else (first + lo - t0) * step_minutes / 60.0
        w = min(max(t0 // per_week, 0), n_weeks - 1)
        out[w, _IDX["ev_n"]] += 1
        out[w, _IDX[f"ev_n_l{r.level}"]] += 1
        if det:
            out[w, _IDX["ev_det"]] += 1
            out[w, _IDX[f"ev_det_l{r.level}"]] += 1
            out[w, _IDX["ev_delay_h"]] += delay
        ev_rows.append({"fault_id": r.fault_id, "type": r.type, "device_id": r.device_id, "level": r.level, "magnitude": r.magnitude,
                        "week": w, "detected": det, "delay_h": delay if det else np.nan})

    for g in range(len(groups)):
        for a, b in _runs(alarms[g]):
            w = min(a // per_week, n_weeks - 1)
            if any(a < hi and b > lo for lo, hi in windows[g]):
                out[w, _IDX["al_tp"]] += 1
            elif (L[g, a:b] == GREY).all():
                out[w, _IDX["al_ign"]] += 1
            else:
                out[w, _IDX["al_fp"]] += 1

    wk = np.arange(n_steps) // per_week
    edges = hist_edges()
    for g in range(len(groups)):
        lab, al = L[g], alarms[g]
        neg, pos = lab == NEG, lab == POS
        np.add.at(out[:, _IDX["neg_rows"]], wk[neg], 1)
        np.add.at(out[:, _IDX["fa_rows"]], wk[neg & al], 1)
        np.add.at(out[:, _IDX["row_tp"]], wk[pos & al], 1)
        np.add.at(out[:, _IDX["row_fp"]], wk[neg & al], 1)
        np.add.at(out[:, _IDX["row_fn"]], wk[pos & ~al], 1)
        r = np.nan_to_num(S[g], nan=0.0) / threshold
        b = np.digitize(r, edges)  # 0 .. HIST_BINS-1
        for sel, name in ((pos, "hp"), (neg, "hn")):
            if sel.any():
                flat = np.bincount(wk[sel] * HIST_BINS + b[sel], minlength=n_weeks * HIST_BINS).reshape(n_weeks, HIST_BINS)
                out[:, _IDX[f"{name}_0"]:_IDX[f"{name}_0"] + HIST_BINS] += flat
    sums = pd.DataFrame(out, columns=COLUMNS)
    sums.insert(0, "week", np.arange(n_weeks))
    return sums, pd.DataFrame(ev_rows)
