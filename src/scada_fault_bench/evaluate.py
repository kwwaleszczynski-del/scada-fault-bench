"""Scoring one method on one scenario: alarms, matching to the injected events, a placebo, and the additive sums behind every metric.

Everything the benchmark reports is a function of *sums* that can be added over weeks, so that a bootstrap over time
blocks only has to re-add them (see :mod:`scada_fault_bench.stats`). For one scenario the function :func:`evaluate`
returns, per week of the test block, the vector ``COLUMNS``:

``ev_n``, ``ev_det``, ``ev_delay_h``   events starting in the week, how many were detected, summed delay of those in hours;
``ev_n_l1`` .. ``ev_det_l5``           the same by fault level 1 (smallest) to 5;
``pl_det``, ``pl_det_l1`` .. ``l5``    expected number of detections *by chance* (the placebo, see below);
``al_tp``, ``al_fp``, ``al_ign``       alarms starting in the week: covering an affected row, only on healthy rows, only on grey rows;
``neg_rows``, ``fa_rows``              rows labelled healthy, and of those the rows inside an alarm (the false alarm rate);
``row_tp``, ``row_fp``, ``row_fn``     row-level counts of alarm against affected rows;
``hp_*``, ``hn_*``                     histograms of the *calibration rank* of the score on affected and on healthy rows (AUROC, AP).

Matching rules. A group is a device (wind) or the whole plant (PV, see :mod:`scada_fault_bench.scenario`). An event is
*detected* when an alarm of its group covers at least one *affected* row of the event (a row where the injected change is
above its tolerance and an alarm can be right or wrong); the *delay* is the time from the first affected row to the first alarm
on an affected row. An alarm is a true alarm when it covers an affected row, ignored when all its rows are grey, and false otherwise.

The **placebo** answers: how often would these alarms hit the affected rows of the event if nothing had been injected? For an
injected scenario it is the alarms of the *same method on the untouched test block* (the null scenario) matched to the
events of the scenario; for the real faults of CARE it is the average over circular shifts of the alarms of the method itself.
The benchmark reports recall next to the placebo recall and the difference (the *excess* over chance).

The score is turned into a rank before it is histogrammed: ``p`` is the mid-rank share of healthy calibration rows with a score not above it,
``t = -log10(1 - p)`` (at most 4); AUROC and average precision are computed from the counts of ``t`` bins, which can be added over folds.

Usage:
    sums, events = evaluate(scenario, scores, threshold, step_minutes=10, cal_scores=cal, placebo=[null_alarms])
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scada_fault_bench.base import alarm_mask
from scada_fault_bench.faults import LEVEL_COUNT
from scada_fault_bench.scenario import GREY, NEG, POS, Scenario, group_ids

__all__ = ["COLUMNS", "HIST_BINS", "alarms_of", "evaluate", "hist_edges", "rank_transform"]

HIST_INNER = 80
HIST_BINS = HIST_INNER + 2
T_MAX = 4.0
SCALARS = ["ev_n", "ev_det", "ev_delay_h", *[f"ev_n_l{i}" for i in range(1, LEVEL_COUNT + 1)], *[f"ev_det_l{i}" for i in range(1, LEVEL_COUNT + 1)],
           "pl_det", *[f"pl_det_l{i}" for i in range(1, LEVEL_COUNT + 1)],
           "al_tp", "al_fp", "al_ign", "neg_rows", "fa_rows", "row_tp", "row_fp", "row_fn"]
COLUMNS = [*SCALARS, *[f"hp_{i}" for i in range(HIST_BINS)], *[f"hn_{i}" for i in range(HIST_BINS)]]
_IDX = {c: i for i, c in enumerate(COLUMNS)}


def hist_edges() -> np.ndarray:
    """Bin edges of the rank variable ``t`` (0 to ``T_MAX``): ``HIST_INNER`` equal bins plus one below and one above."""
    return np.linspace(0.0, T_MAX, HIST_INNER + 1)


def rank_transform(scores: np.ndarray, cal_sorted: np.ndarray) -> np.ndarray:
    """``t = -log10(1 - p)`` with ``p`` the mid-rank share of the sorted healthy calibration scores not above the score (NaN counts as lowest)."""
    s = np.nan_to_num(np.asarray(scores, dtype="float64"), nan=-np.inf)
    n = max(len(cal_sorted), 1)
    p = (np.searchsorted(cal_sorted, s, side="left") + np.searchsorted(cal_sorted, s, side="right")) / (2.0 * n)  # mid-rank: ties get their average
    return np.minimum(-np.log10(np.maximum(1.0 - p, 10.0 ** -T_MAX)), T_MAX)


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


def alarms_of(sc: Scenario, scores: np.ndarray, threshold: float, step_minutes: float) -> np.ndarray:
    """Merged alarms of every group on the regular grid of the test block (boolean, groups x steps)."""
    _, S, _ = _grid(sc, np.asarray(scores, dtype="float64"), pd.Timedelta(minutes=step_minutes))
    return np.array([alarm_mask(np.nan_to_num(S[g], nan=-np.inf) > threshold, step_minutes) for g in range(S.shape[0])])


def evaluate(sc: Scenario, scores: np.ndarray, threshold: float, step_minutes: float, cal_scores: np.ndarray | None = None,
             placebo: list[np.ndarray] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sums per week of the test block (rows: weeks, columns ``COLUMNS``) and one row per injected event.

    ``cal_scores``: scores of the healthy calibration rows (the reference of the rank transform; default: the healthy rows of this
    scenario, which is only meant for tests). ``placebo``: alarm grids (as :func:`alarms_of`) of alarms that are independent of
    the injection; the placebo detection of an event is the share of them that cover an affected row of it.
    """
    step = pd.Timedelta(minutes=step_minutes)
    scores = np.asarray(scores, dtype="float64")
    groups, S, L = _grid(sc, scores, step)
    n_steps = S.shape[1]
    per_week = int(round(7 * 24 * 60 / step_minutes))
    n_weeks = int(np.ceil(n_steps / per_week))
    out = np.zeros((n_weeks, len(COLUMNS)))
    alarms = np.array([alarm_mask(np.nan_to_num(S[g], nan=-np.inf) > threshold, step_minutes) for g in range(len(groups))])
    if cal_scores is None:
        cal_scores = scores[sc.labels == NEG]
    cal = np.asarray(cal_scores, dtype="float64")
    cal_sorted = np.sort(cal[np.isfinite(cal)])

    ev_rows = []
    for r in sc.truth.itertuples():
        g = groups.index("plant") if sc.meta.profile == "pv" else groups.index(r.device_id)
        t0 = max(int((r.start - sc.t_lo) // step), 0)
        t1 = min(int(np.ceil((r.end - sc.t_lo) / step)), n_steps)
        affected = L[g, t0:t1] == POS
        hit = np.flatnonzero(alarms[g, t0:t1] & affected)
        det = len(hit) > 0
        delay = float(hit[0]) * step_minutes / 60.0 if det else 0.0
        chance = float(np.mean([bool((p[g, t0:t1] & affected).any()) for p in placebo])) if placebo else 0.0
        w = min(max(t0 // per_week, 0), n_weeks - 1)
        out[w, _IDX["ev_n"]] += 1
        out[w, _IDX[f"ev_n_l{r.level}"]] += 1
        out[w, _IDX["pl_det"]] += chance
        out[w, _IDX[f"pl_det_l{r.level}"]] += chance
        if det:
            out[w, _IDX["ev_det"]] += 1
            out[w, _IDX[f"ev_det_l{r.level}"]] += 1
            out[w, _IDX["ev_delay_h"]] += delay
        ev_rows.append({"fault_id": r.fault_id, "type": r.type, "device_id": r.device_id, "level": r.level, "magnitude": r.magnitude,
                        "week": w, "detected": det, "chance": chance, "delay_h": delay if det else np.nan})

    for g in range(len(groups)):
        for a, b in _runs(alarms[g]):
            w = min(a // per_week, n_weeks - 1)
            seg = L[g, a:b]
            if (seg == POS).any():
                out[w, _IDX["al_tp"]] += 1
            elif (seg == GREY).all():
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
        b = np.digitize(rank_transform(S[g], cal_sorted), edges)  # 0 .. HIST_BINS-1
        for sel, name in ((pos, "hp"), (neg, "hn")):
            if sel.any():
                flat = np.bincount(wk[sel] * HIST_BINS + b[sel], minlength=n_weeks * HIST_BINS).reshape(n_weeks, HIST_BINS)
                out[:, _IDX[f"{name}_0"]:_IDX[f"{name}_0"] + HIST_BINS] += flat
    sums = pd.DataFrame(out, columns=COLUMNS)
    sums.insert(0, "week", np.arange(n_weeks))
    return sums, pd.DataFrame(ev_rows)
