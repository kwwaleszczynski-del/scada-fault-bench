"""Metrics from sums, and block-bootstrap confidence intervals.

The unit of resampling is a *week of a test block* (key: dataset, fold, week): all devices, fault types and methods of that
week are drawn together, so that the dependence between devices that see the same weather, and between methods that see
the same data, is kept. A confidence interval is the 2.5 % and 97.5 % quantile of the metric over ``N_BOOT`` resamples
of the weeks of the selection; the point value is the metric on the original sums. Intervals only describe the sampling of
weeks, devices and events inside this benchmark; they say nothing about how a method behaves on other plants.

Metrics (``METRICS``): precision, recall and F1 of events and alarms (``precision``, ``recall``, ``f1``), the same on rows
(``row_precision``, ``row_recall``, ``row_f1``), mean detection delay in hours, false alarm hours per year of healthy time,
AUROC and average precision of the raw score on affected against healthy rows, and recall by fault level
(``recall_l1`` .. ``recall_l5``).

Usage:
    point, lo, hi = bootstrap_metrics(V, n_boot=1000, seed=0)     # V: (n_blocks, len(COLUMNS)) array of sums
"""

from __future__ import annotations

import numpy as np

from scada_fault_bench.evaluate import COLUMNS, HIST_BINS
from scada_fault_bench.faults import LEVEL_COUNT

__all__ = ["HOURS_PER_YEAR", "METRICS", "N_BOOT", "bootstrap_metrics", "metrics_from_sums"]

N_BOOT = 1000
HOURS_PER_YEAR = 8766.0
IDX = {c: i for i, c in enumerate(COLUMNS)}
METRICS = ["precision", "recall", "f1", "delay_h", "fa_h_per_year", "auroc", "ap", "row_precision", "row_recall", "row_f1",
           *[f"recall_l{i}" for i in range(1, LEVEL_COUNT + 1)]]
HP = slice(IDX["hp_0"], IDX["hp_0"] + HIST_BINS)
HN = slice(IDX["hn_0"], IDX["hn_0"] + HIST_BINS)


def _div(a, b):
    a, b = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)


def _f1(p, r):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((p + r) > 0, 2 * p * r / (p + r), np.where(np.isnan(p) | np.isnan(r), np.nan, 0.0))


def metrics_from_sums(T: np.ndarray) -> dict[str, np.ndarray]:
    """Metrics from sums ``T`` of shape (m,) or (B, m); each value has the shape of the leading dimension."""
    T = np.atleast_2d(T)
    c = lambda name: T[:, IDX[name]]  # noqa: E731
    out: dict[str, np.ndarray] = {}
    prec = _div(c("al_tp"), c("al_tp") + c("al_fp"))
    rec = _div(c("ev_det"), c("ev_n"))
    out["precision"], out["recall"], out["f1"] = prec, rec, _f1(prec, rec)
    out["delay_h"] = _div(c("ev_delay_h"), c("ev_det"))
    out["fa_h_per_year"] = _div(c("fa_rows"), c("neg_rows")) * HOURS_PER_YEAR
    rp, rr = _div(c("row_tp"), c("row_tp") + c("row_fp")), _div(c("row_tp"), c("row_tp") + c("row_fn"))
    out["row_precision"], out["row_recall"], out["row_f1"] = rp, rr, _f1(rp, rr)
    P, N = T[:, HP], T[:, HN]
    pt, nt = P.sum(axis=1), N.sum(axis=1)
    below = np.cumsum(N, axis=1) - N
    out["auroc"] = _div((P * (below + 0.5 * N)).sum(axis=1), pt * nt)
    Pd, Nd = P[:, ::-1], N[:, ::-1]
    tp, fp = np.cumsum(Pd, axis=1), np.cumsum(Nd, axis=1)
    precision_at = np.where((tp + fp) > 0, tp / np.where((tp + fp) > 0, tp + fp, 1.0), 1.0)
    out["ap"] = _div((Pd * precision_at).sum(axis=1), pt)
    for i in range(1, LEVEL_COUNT + 1):
        out[f"recall_l{i}"] = _div(c(f"ev_det_l{i}"), c(f"ev_n_l{i}"))
    return out


def bootstrap_metrics(V: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> tuple[dict, dict, dict]:
    """Point values and 95 % intervals of every metric. ``V`` has one row per block (a week of a test block)."""
    V = np.asarray(V, dtype="float64")
    point = {k: float(v[0]) for k, v in metrics_from_sums(V.sum(axis=0)).items()}
    K = len(V)
    if K < 2:
        nan = {k: float("nan") for k in METRICS}
        return point, nan, dict(nan)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(K, np.full(K, 1.0 / K), size=n_boot)
    boot = metrics_from_sums(counts @ V)
    lo = {k: float(np.nanquantile(v, 0.025)) if np.isfinite(v).any() else float("nan") for k, v in boot.items()}
    hi = {k: float(np.nanquantile(v, 0.975)) if np.isfinite(v).any() else float("nan") for k, v in boot.items()}
    return point, lo, hi
