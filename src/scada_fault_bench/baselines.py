"""Simple reference methods: fixed rules, the quality control of scada-schema, and four statistics on shared residuals.

============  ================================================================================  ==========
name          what it does                                                                      threshold
============  ================================================================================  ==========
rules         hand-written physical rules (redundant sensors, frozen signals, limits)           fixed
scada_qc      the row-level findings of ``scada_schema.qc`` (flatline, spikes, units, ...)       fixed
zscore        largest absolute standardised residual                                            calibrated
ewma          exponentially weighted moving average of the residuals (about one day)            calibrated
cusum         two-sided CUSUM of the residuals (hourly, allowance 1 sigma, memory 3 days)      calibrated
iforest       Isolation Forest on the residual vector                                           calibrated
============  ================================================================================  ==========

The last four read the matrix ``Z`` of :mod:`scada_fault_bench.features`; they are fitted on the training weeks only.

Usage:
    m = METHODS["cusum"]().fit(train, meta)
    s = m.score(table, meta)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scada_schema.model import DatasetMeta
from scipy.signal import lfilter
from sklearn.ensemble import IsolationForest

from scada_fault_bench.base import Detector, register
from scada_fault_bench.features import FeatureModel, circ_diff, rated_of, step_minutes

CLIP = 5.0  # residuals are clipped at +-5 sigma before they are accumulated, so that a single outlier cannot saturate a chart

__all__ = ["CUSUM", "EWMA", "IsolationForestMethod", "QualityControl", "Rules", "ZScore", "apply_series"]


def apply_series(table: pd.DataFrame, meta: DatasetMeta, Z: pd.DataFrame, fn) -> np.ndarray:
    """Apply ``fn(matrix) -> vector`` to the residual matrix of every device (wind) or of the plant's unique times (PV)."""
    out = np.full(len(table), np.nan)
    M = Z.to_numpy(dtype="float64")
    if meta.profile == "wind":
        ids = table["device_id"].to_numpy()
        start = 0
        for i in range(1, len(ids) + 1):
            if i == len(ids) or ids[i] != ids[start]:
                out[start:i] = fn(M[start:i])
                start = i
        return out
    ts = table["timestamp"].to_numpy()
    first = pd.Series(np.arange(len(table))).groupby(ts).first()
    order = np.argsort(first.index.to_numpy())
    rows = first.to_numpy()[order]
    s = pd.Series(fn(M[rows]), index=pd.DatetimeIndex(first.index.to_numpy()[order]))
    return s.reindex(pd.DatetimeIndex(ts)).to_numpy()


class _ResidualMethod(Detector):
    def fit(self, train, meta):
        self.fm = FeatureModel(meta).fit(train)
        self._fit_z(self.fm.transform(train), meta)
        return self

    def _fit_z(self, Z: pd.DataFrame, meta: DatasetMeta) -> None:  # pragma: no cover - overridden where needed
        pass

    tail = False  # True: features divided by their training tail quantile (see FeatureModel.transform)

    def score(self, table, meta):
        return apply_series(table, meta, self.fm.transform(table, tail=self.tail), self._series(meta))

    def _series(self, meta: DatasetMeta):
        raise NotImplementedError


@register
class ZScore(_ResidualMethod):
    name = "zscore"
    title = "Robust z-score"
    description = ("largest absolute residual of the shared features (one-hour median), each divided by its 99.5 % quantile on the training weeks "
                   "so that no single heavy-tailed feature decides the score")
    tail = True

    def _series(self, meta):
        def fn(M):
            with np.errstate(all="ignore"):
                a = np.abs(M)
                out = np.where(np.isfinite(a).any(axis=1), np.nanmax(np.where(np.isfinite(a), a, -np.inf), axis=1), np.nan)
            return out
        return fn


@register
class EWMA(_ResidualMethod):
    name = "ewma"
    title = "EWMA chart"
    description = "exponentially weighted moving average of every residual clipped at 5 sigma (time constant about one day), largest standardised value"

    def _series(self, meta):
        per_day = 24 * 60 / step_minutes(meta)
        lam = 2.0 / (per_day + 1.0)
        sd = np.sqrt(lam / (2.0 - lam))

        def fn(M):
            ok = np.isfinite(M).any(axis=1)
            x = np.nan_to_num(M, nan=0.0)
            e = lfilter([lam], [1.0, -(1.0 - lam)], np.clip(x, -CLIP, CLIP), axis=0)
            return np.where(ok, np.abs(e).max(axis=1) / sd, np.nan)
        return fn


def _cusum(x: np.ndarray, k: float, memory: int) -> np.ndarray:
    """Two-sided CUSUM with a finite memory of ``memory`` steps: the largest sum of (x - k) over any window that ends now, per side.

    With the running sum C of (x - k), the statistic is C minus the smallest C within the memory; it never exceeds
    ``memory`` times the largest input and forgets a shift after ``memory`` steps, so it cannot saturate on slow seasonal offsets.
    """
    out = np.zeros(len(x))
    for sign in (1.0, -1.0):
        c = np.cumsum(sign * x - k)
        low = pd.Series(np.r_[0.0, c]).rolling(memory + 1, min_periods=1).min().to_numpy()[1:]
        out = np.maximum(out, c - low)
    return out


@register
class CUSUM(_ResidualMethod):
    name = "cusum"
    title = "CUSUM chart"
    description = "two-sided CUSUM of every residual clipped at 5 sigma, on hourly values (allowance 1 sigma, memory 3 days), largest statistic"
    K, MEMORY = 1.0, 72

    def _series(self, meta):
        h = max(1, int(round(60.0 / step_minutes(meta))))

        def fn(M):
            n = len(M)
            ok = np.isfinite(M).any(axis=1)
            hourly = np.clip(np.nan_to_num(M[::h], nan=0.0), -CLIP, CLIP)
            s = np.max([_cusum(hourly[:, j], self.K, self.MEMORY) for j in range(M.shape[1])], axis=0)
            return np.where(ok, np.repeat(s, h)[:n], np.nan)
        return fn


@register
class IsolationForestMethod(_ResidualMethod):
    name = "iforest"
    title = "Isolation Forest"
    description = "Isolation Forest (100 trees, 4096 samples) on the vector of standardised residuals; missing residuals set to 0"

    def _fit_z(self, Z, meta):
        X = Z.to_numpy(dtype="float64")
        X = X[np.isfinite(X).any(axis=1)]
        rng = np.random.default_rng(0)
        if len(X) > 100_000:
            X = X[rng.choice(len(X), 100_000, replace=False)]
        self.model = IsolationForest(n_estimators=100, max_samples=min(4096, len(X)), random_state=0, n_jobs=1).fit(np.nan_to_num(X, nan=0.0))

    def score(self, table, meta):
        M = self.fm.transform(table).to_numpy(dtype="float64")
        ok = np.isfinite(M).any(axis=1)
        out = np.full(len(table), np.nan)
        if ok.any():
            out[ok] = -self.model.score_samples(np.nan_to_num(M[ok], nan=0.0))
        return np.where(ok, out, np.nan)

    def _series(self, meta):  # pragma: no cover - score() is overridden
        raise NotImplementedError


def _rolling_share(flag: np.ndarray, n: int) -> np.ndarray:
    """Share of the last ``n`` rows (trailing) in which ``flag`` holds, NaN while fewer than ``n`` rows are known."""
    return pd.Series(flag.astype("float64")).rolling(n, min_periods=n).mean().to_numpy()


@register
class Rules(Detector):
    name = "rules"
    title = "Expert rules"
    description = ("fixed physical rules, no learning: the two anemometers differ by more than 1.5 m/s, a wind speed with zero variation over an hour while the "
                   "turbine produces, gearbox oil outside -20 to 100 degC, vane + nacelle - absolute direction beyond 25 deg, power above 110 % of rated; "
                   "PV: irradiance frozen in daylight, inverter below 2 % of rated in sun above 300 W/m2, module temperature outside -30 to 90 degC, power above 110 % of rated")
    fixed_threshold = True

    def fit(self, train, meta):
        return self

    def score(self, table, meta):
        n = len(table)
        hour = max(2, int(round(60.0 / step_minutes(meta))))
        out = np.full(n, np.nan)
        if meta.profile == "wind":
            ids = table["device_id"].to_numpy()
            start = 0
            for i in range(1, n + 1):
                if i == n or ids[i] != ids[start]:
                    out[start:i] = self._wind(table.iloc[start:i], meta, ids[start], hour)
                    start = i
            return out
        return self._pv(table, meta, hour)

    @staticmethod
    def _col(g: pd.DataFrame, c: str) -> np.ndarray | None:
        return g[c].to_numpy(dtype="float64") if c in g and g[c].notna().any() else None

    def _wind(self, g, meta, dev, hour):
        rated = rated_of(meta, dev, g["p_ac"])
        ws, p = self._col(g, "wind_speed"), self._col(g, "p_ac")
        rules = []
        ws2 = self._col(g, "wind_speed_2")
        if ws2 is not None and ws is not None:
            rules.append(np.abs(ws - ws2) > 1.5)
        if ws is not None and p is not None:
            sd = pd.Series(ws).rolling(hour, min_periods=hour).std().to_numpy()
            rules.append((sd == 0) & (p > 0.1 * rated))
        oil = self._col(g, "t_gearbox_oil")
        if oil is not None:
            rules.append((oil > 100) | (oil < -20))
        rel, nac, wd = self._col(g, "wind_dir_rel"), self._col(g, "nacelle_dir"), self._col(g, "wind_dir")
        if rel is not None and nac is not None and wd is not None and ws is not None:
            rules.append((np.abs(circ_diff(rel + nac, wd)) > 25) & (ws > 4))
        if p is not None:
            rules.append(p > 1.1 * rated)
        known = np.isfinite(p) if p is not None else np.ones(len(g), dtype=bool)
        share = np.max([_rolling_share(r, hour) for r in rules], axis=0) if rules else np.zeros(len(g))
        return np.where(known, share, np.nan)

    def _pv(self, table, meta, hour):
        met = [d.id for d in meta.devices if d.type == "met"]
        inv = [d.id for d in meta.devices if d.type == "inverter"]
        m = table[table["device_id"].isin(met)].drop_duplicates("timestamp").set_index("timestamp").sort_index()
        idx = m.index
        poa = m["poa"].to_numpy(dtype="float64") if "poa" in m else np.full(len(m), np.nan)
        rules = []
        sd = pd.Series(poa).rolling(hour, min_periods=hour).std().to_numpy()
        rules.append((sd == 0) & (poa > 100))
        if "t_module" in m:
            tm = m["t_module"].to_numpy(dtype="float64")
            rules.append((tm > 90) | (tm < -30))
        for d in inv:
            g = table[table["device_id"] == d].drop_duplicates("timestamp").set_index("timestamp").reindex(idx)
            rated = rated_of(meta, d, g["p_ac"])
            p = g["p_ac"].to_numpy(dtype="float64")
            rules.append((p < 0.02 * rated) & (poa > 300))
            rules.append(p > 1.1 * rated)
        share = np.max([_rolling_share(r, hour) for r in rules], axis=0)
        s = pd.Series(np.where(np.isfinite(poa) & (poa >= 20), share, np.nan), index=idx)
        return s.reindex(pd.DatetimeIndex(table["timestamp"])).to_numpy()


@register
class QualityControl(Detector):
    name = "scada_qc"
    title = "scada-schema QC"
    description = "rows flagged by the row-level checks of scada_schema.qc (flatline, spikes, out of range, units, power curve, wind direction, energy counter, energy against power)"
    fixed_threshold = True
    CHECKS = ("flatline", "spikes", "out_of_range", "units", "power_curve", "wind_direction", "energy_counter", "energy_power")
    # the channels that the fault models of the benchmark touch or depend on; quantised or irrelevant channels (DC voltage, a met station's wind
    # speed) would otherwise fill the false alarm count of a method that was written to inspect every channel
    SIGNALS = {"wind": ("p_ac", "e_export", "e_export_counter", "i_ac", "wind_speed", "wind_speed_2", "wind_dir", "wind_dir_rel", "nacelle_dir", "t_gearbox_oil"),
               "pv": ("p_ac", "p_dc", "i_dc", "poa", "t_module", "t_inverter")}

    def fit(self, train, meta):
        return self

    def score(self, table, meta):
        from scada_schema import qc

        keep = ["timestamp", "device_id", *[c for c in table.columns if c.split("__")[0] in self.SIGNALS[meta.profile]]]
        report = qc.run(table[keep].reset_index(drop=True), meta, checks=list(self.CHECKS))
        flag = np.zeros(len(table), dtype=bool)
        for f in report.findings:
            if f.rows is not None and len(f.rows):
                flag[f.rows] = True
        return flag.astype("float64")
