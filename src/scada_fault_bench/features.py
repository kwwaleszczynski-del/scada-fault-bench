"""Standardised residual features that the simple baselines share.

Every baseline (robust z-score, EWMA, CUSUM, Isolation Forest) looks at the *same* matrix ``Z`` of residuals, so that they
differ in the statistic and not in the information they get. The residuals are what an analyst would build first:

wind (per turbine, 10 min):
    ``pc_res``    power minus the turbine's own binned power curve, as a share of rated power;
    ``ws_fleet``  wind speed minus the median of the other turbines at the same time;
    ``ws_ws2``    wind speed minus the second anemometer (where there is one);
    ``yaw_fleet`` nacelle position minus the circular mean of the other turbines (wind above 4 m/s);
    ``yaw_cons``  vane + nacelle - absolute direction (where the three exist);
    ``t_res``     gearbox oil temperature minus a regression on load and ambient (or nacelle) temperature;
    ``ws_flat``   log of the one-hour standard deviation of wind speed (a frozen sensor has none).
pv (plant level, 15 min, daylight only):
    ``pr_res``    normalised AC power against a curve of plane-of-array irradiance with a module temperature correction;
    ``pr_tod``    the same residual minus its median in the same sun position (azimuth, elevation bin);
    ``poa_flat``  log of the one-hour standard deviation of irradiance;
    ``t_mod_res`` module temperature against ambient and irradiance;
    ``t_inv_res`` inverter temperature against ambient and load.

All curves and regressions are fitted on the training weeks only. Each residual is smoothed with a trailing one-hour
median and standardised by its median and median absolute deviation on the training weeks (per device for wind).

Usage:
    fm = FeatureModel(meta).fit(train)
    Z = fm.transform(table)          # DataFrame aligned with the rows of table, columns fm.names
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scada_schema.model import DatasetMeta

from scada_fault_bench.solar import solar_position

__all__ = ["FeatureModel", "circ_diff", "rated_of", "step_minutes", "trailing_median"]

MAD_SCALE = 1.4826
TAIL_QUANTILE = 0.995


def step_minutes(meta: DatasetMeta) -> float:
    return pd.Timedelta(meta.interval).total_seconds() / 60.0


def rated_of(meta: DatasetMeta, device: str, series: pd.Series | None = None) -> float:
    for d in meta.devices:
        if d.id == device and (d.rated_kw or d.capacity_kwp):
            return float(d.rated_kw or d.capacity_kwp)
    if series is not None and series.notna().any():
        return float(np.nanpercentile(series.to_numpy(dtype="float64"), 99.5))
    return 1.0


def circ_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed angle ``a - b`` in degrees, wrapped to [-180, 180)."""
    return (np.asarray(a, dtype="float64") - np.asarray(b, dtype="float64") + 180.0) % 360.0 - 180.0


def trailing_median(x: np.ndarray, n: int) -> np.ndarray:
    """Trailing median over ``n`` rows (at least ``n // 2`` finite values)."""
    return pd.Series(x).rolling(n, min_periods=max(1, n // 2)).median().to_numpy()


def _slices(ids: np.ndarray) -> list[tuple[str, int, int]]:
    out, start = [], 0
    for i in range(1, len(ids) + 1):
        if i == len(ids) or ids[i] != ids[start]:
            out.append((ids[start], start, i))
            start = i
    return out


def _robust_scale(x: np.ndarray) -> tuple[float, float]:
    v = x[np.isfinite(x)]
    if len(v) < 20:
        return 0.0, 1.0
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * MAD_SCALE
    return med, mad if mad > 1e-9 else max(float(np.std(v)), 1e-6)


class FeatureModel:
    """Fit on the training table, transform any table of the same plant. Pick the profile from ``meta``."""

    def __init__(self, meta: DatasetMeta):
        self.meta = meta
        self.profile = meta.profile
        self.step = step_minutes(meta)
        self.hour = max(2, int(round(60.0 / self.step)))
        self.names: list[str] = []
        self.fitted = False

    # ------------------------------------------------------------------------------------------ fit
    def fit(self, train: pd.DataFrame) -> FeatureModel:
        if self.profile == "wind":
            self._fit_wind(train)
        else:
            self._fit_pv(train)
        raw = self._raw(train)
        self.names = list(raw.columns)
        sm = self._smooth(raw, train)
        self.scale: dict[tuple[str, str], tuple[float, float]] = {}
        self.tail: dict[tuple[str, str], float] = {}
        for key, rows in self._group_rows(train).items():
            for c in sm.columns:
                med, scale = _robust_scale(sm[c].to_numpy()[rows])
                self.scale[(key, c)] = (med, scale)
                z = np.abs((sm[c].to_numpy()[rows] - med) / scale)
                z = z[np.isfinite(z)]
                self.tail[(key, c)] = max(float(np.quantile(z, TAIL_QUANTILE)), 1.0) if len(z) >= 20 else 1.0
        self.fitted = True
        return self

    def transform(self, table: pd.DataFrame, tail: bool = False) -> pd.DataFrame:
        """Standardised features (median 0, MAD-based sigma 1 on the training weeks); with ``tail=True`` each is divided by
        the ``TAIL_QUANTILE`` quantile of its absolute value on the training weeks, so that 1 means a rare value of that feature."""
        if not self.fitted:
            raise RuntimeError("fit first")
        sm = self._smooth(self._raw(table), table)
        z = np.full(sm.shape, np.nan)
        for key, rows in self._group_rows(table).items():
            for j, c in enumerate(sm.columns):
                med, scale = self.scale.get((key, c), (0.0, 1.0))
                z[rows, j] = (sm[c].to_numpy()[rows] - med) / scale
                if tail:
                    z[rows, j] /= self.tail.get((key, c), 1.0)
        return pd.DataFrame(z, columns=sm.columns, index=table.index)

    def _group_rows(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        if self.profile == "wind":
            return {dev: np.arange(a, b) for dev, a, b in _slices(df["device_id"].to_numpy())}
        return {"plant": np.arange(len(df))}

    def _smooth(self, raw: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        out = raw.copy()
        for _dev, rows in self._group_rows(df).items():
            if self.profile == "wind":
                for c in raw.columns:
                    out.iloc[rows, out.columns.get_loc(c)] = trailing_median(raw[c].to_numpy()[rows], self.hour)
            else:
                # PV features are per timestamp and repeated for every device row: smooth on the unique times
                ts = df["timestamp"].to_numpy()
                first = pd.Series(np.arange(len(df))).groupby(ts).first().to_numpy()
                order = np.argsort(ts[first])
                first = first[order]
                for c in raw.columns:
                    sm = trailing_median(raw[c].to_numpy()[first], self.hour)
                    s = pd.Series(sm, index=pd.DatetimeIndex(ts[first]))
                    out[c] = s.reindex(pd.DatetimeIndex(ts)).to_numpy()
        return out

    # ------------------------------------------------------------------------------------------ raw features
    def _raw(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._raw_wind(df) if self.profile == "wind" else self._raw_pv(df)

    # wind ------------------------------------------------------------------------------------------
    def _fit_wind(self, train: pd.DataFrame) -> None:
        self.rated = {}
        self.curve: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.thermal: dict[str, np.ndarray] = {}
        self.tref = "t_amb" if "t_amb" in train else ("t_nacelle" if "t_nacelle" in train else None)
        for dev, a, b in _slices(train["device_id"].to_numpy()):
            g = train.iloc[a:b]
            rated = rated_of(self.meta, dev, g["p_ac"])
            self.rated[dev] = rated
            ws, p = g["wind_speed"].to_numpy(dtype="float64"), g["p_ac"].to_numpy(dtype="float64")
            ok = np.isfinite(ws) & np.isfinite(p) & (ws >= 2) & (ws <= 25)
            edges = np.arange(2.0, 25.5, 0.5)
            idx = np.digitize(ws[ok], edges)
            centres, med = [], []
            for k in np.unique(idx):
                m = idx == k
                if m.sum() >= 10:
                    centres.append(float(np.mean(ws[ok][m])))
                    med.append(float(np.median(p[ok][m])))
            self.curve[dev] = (np.array(centres), np.array(med)) if len(centres) >= 3 else (np.array([2.0, 25.0]), np.array([0.0, rated]))
            if "t_gearbox_oil" in g and self.tref:
                self.thermal[dev] = self._fit_thermal(g, rated)

    def _thermal_design(self, g: pd.DataFrame, rated: float) -> tuple[np.ndarray, np.ndarray]:
        load = g["p_ac"].to_numpy(dtype="float64") / rated
        t = g[self.tref].to_numpy(dtype="float64")
        X = np.column_stack([np.ones(len(g)), load, load**2, t, load * t])
        return X, load

    def _fit_thermal(self, g: pd.DataFrame, rated: float) -> np.ndarray:
        X, load = self._thermal_design(g, rated)
        y = g["t_gearbox_oil"].to_numpy(dtype="float64")
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y) & (load > 0.05)
        if ok.sum() < 200:
            return np.zeros(X.shape[1])
        beta = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
        for _ in range(2):  # trimmed refit: drop gross outliers (stops, sensor errors)
            r = y - X @ beta
            mad = np.nanmedian(np.abs(r[ok] - np.nanmedian(r[ok]))) * MAD_SCALE
            keep = ok & (np.abs(r) < 3 * max(mad, 0.1))
            if keep.sum() < 200:
                break
            beta = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
        return beta

    def _others(self, df: pd.DataFrame, col: str, circular: bool = False) -> np.ndarray:
        """Median (circular mean) of the other devices' ``col`` at the same time, aligned with ``df``'s rows; NaN with fewer than 3 others."""
        piv = df.pivot_table(index="timestamp", columns="device_id", values=col, aggfunc="first")
        arr = piv.to_numpy(dtype="float64")
        row_t = piv.index.get_indexer(pd.DatetimeIndex(df["timestamp"]))
        out = np.full(len(df), np.nan)
        if arr.shape[1] < 4:  # fewer than three others: no fleet reference (a single-turbine table)
            return out
        cols = list(piv.columns)
        devs = df["device_id"].to_numpy()
        for j, dev in enumerate(cols):
            others = np.delete(arr, j, axis=1)
            n_ok = np.isfinite(others).sum(axis=1)
            if circular:
                r = np.radians(others)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    val = np.degrees(np.arctan2(np.nanmean(np.sin(r), axis=1), np.nanmean(np.cos(r), axis=1)))
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    val = np.nanmedian(others, axis=1)
            val = np.where(n_ok >= 3, val, np.nan)
            rows = np.flatnonzero(devs == dev)
            out[rows] = val[row_t[rows]]
        return out

    def _raw_wind(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        f: dict[str, np.ndarray] = {}
        ws = df["wind_speed"].to_numpy(dtype="float64")
        p = df["p_ac"].to_numpy(dtype="float64")
        dev = df["device_id"].to_numpy()
        pc = np.full(n, np.nan)
        thermal = np.full(n, np.nan)
        for d, a, b in _slices(dev):
            rated = self.rated.get(d, rated_of(self.meta, d, df["p_ac"].iloc[a:b]))
            cw, cp = self.curve.get(d, (np.array([2.0, 25.0]), np.array([0.0, rated])))
            w = ws[a:b]
            exp = np.interp(w, cw, cp)
            pc[a:b] = np.where((w >= 2) & (w <= 25), (p[a:b] - exp) / rated, np.nan)
            if d in self.thermal and "t_gearbox_oil" in df and self.tref in df:
                X, load = self._thermal_design(df.iloc[a:b], rated)
                y = df["t_gearbox_oil"].to_numpy(dtype="float64")[a:b]
                thermal[a:b] = np.where(load > 0.05, y - X @ self.thermal[d], np.nan)
        f["pc_res"] = pc
        f["ws_fleet"] = ws - self._others(df, "wind_speed")
        if "wind_speed_2" in df and df["wind_speed_2"].notna().any():
            f["ws_ws2"] = ws - df["wind_speed_2"].to_numpy(dtype="float64")
        if "nacelle_dir" in df:
            nac = df["nacelle_dir"].to_numpy(dtype="float64")
            yaw = circ_diff(nac, self._others(df, "nacelle_dir", circular=True))
            f["yaw_fleet"] = np.where(ws > 4, yaw, np.nan)
            if {"wind_dir", "wind_dir_rel"} <= set(df.columns) and df["wind_dir"].notna().any():
                cons = circ_diff(df["wind_dir_rel"].to_numpy(dtype="float64") + nac, df["wind_dir"].to_numpy(dtype="float64"))
                f["yaw_cons"] = np.where(ws > 4, cons, np.nan)
        if np.isfinite(thermal).any():
            f["t_res"] = thermal
        std = np.full(n, np.nan)
        for _d, a, b in _slices(dev):
            std[a:b] = pd.Series(ws[a:b]).rolling(self.hour, min_periods=self.hour).std().to_numpy()
        f["ws_flat"] = np.where(ws > 2, np.log(std + 0.01), np.nan)
        return pd.DataFrame(f, index=df.index)

    # pv --------------------------------------------------------------------------------------------
    def _plant(self, df: pd.DataFrame) -> pd.DataFrame:
        """One row per timestamp: plane-of-array irradiance, temperatures and the mean normalised inverter power."""
        met = [d.id for d in self.meta.devices if d.type == "met"]
        inv = [d.id for d in self.meta.devices if d.type == "inverter"]
        m = df[df["device_id"].isin(met)].drop_duplicates("timestamp").set_index("timestamp")
        out = pd.DataFrame(index=pd.DatetimeIndex(sorted(df["timestamp"].unique())))
        for c in ("poa", "t_amb", "t_module"):
            out[c] = m[c].reindex(out.index).astype("float64") if c in m else np.nan
        pn, ti = [], []
        for d in inv:
            g = df[df["device_id"] == d].drop_duplicates("timestamp").set_index("timestamp")
            if len(g) == 0:
                continue
            rated = rated_of(self.meta, d, g["p_ac"])
            pn.append(g["p_ac"].reindex(out.index).astype("float64") / rated)
            if "t_inverter" in g:
                ti.append(g["t_inverter"].reindex(out.index).astype("float64"))
        out["pn"] = pd.concat(pn, axis=1).mean(axis=1) if pn else np.nan
        out["t_inv"] = pd.concat(ti, axis=1).mean(axis=1) if ti else np.nan
        lat = next((d.lat for d in self.meta.devices if d.lat is not None), None)
        lon = next((d.lon for d in self.meta.devices if d.lon is not None), None)
        if lat is not None and lon is not None:
            elev, az = solar_position(out.index + pd.Timedelta(minutes=self.step / 2), lat, lon)
            out["elev"], out["azim"] = elev, az
        return out

    def _fit_pv(self, train: pd.DataFrame) -> None:
        P = self._plant(train)
        day = (P["poa"] >= 50) & P["pn"].notna()
        edges = np.arange(50.0, 1400.0, 50.0)
        idx = np.digitize(P.loc[day, "poa"].to_numpy(), edges)
        cw, cp = [], []
        for k in np.unique(idx):
            m = idx == k
            if m.sum() >= 10:
                cw.append(float(P.loc[day, "poa"].to_numpy()[m].mean()))
                cp.append(float(np.median(P.loc[day, "pn"].to_numpy()[m])))
        self.pv_curve = (np.array(cw), np.array(cp)) if len(cw) >= 3 else (np.array([50.0, 1000.0]), np.array([0.05, 0.9]))
        base = np.interp(P["poa"].to_numpy(), *self.pv_curve)
        ratio = P["pn"].to_numpy() / base - 1.0
        hi = day.to_numpy() & (P["poa"].to_numpy() > 400) & P["t_module"].notna().to_numpy()
        dt = P["t_module"].to_numpy()[hi] - 25.0
        self.gamma = float(np.dot(dt, ratio[hi]) / np.dot(dt, dt)) if hi.sum() > 50 and np.dot(dt, dt) > 0 else 0.0
        self.tod_bins: dict[int, float] = {}
        res = self._pr_res(P)
        if "elev" in P:
            key = self._sun_bins(P)
            s = pd.Series(res).groupby(key).median()
            self.tod_bins = {int(k): float(v) for k, v in s.items() if np.isfinite(v) and k >= 0}
        self.reg_mod = self._fit_ols(P, "t_module", ["t_amb", "poa"], day.to_numpy())
        self.reg_inv = self._fit_ols(P, "t_inv", ["t_amb", "pn"], day.to_numpy())

    @staticmethod
    def _fit_ols(P: pd.DataFrame, y: str, xs: list[str], mask: np.ndarray) -> np.ndarray | None:
        if y not in P or P[y].notna().sum() < 100 or any(P[x].notna().sum() < 100 for x in xs):
            return None
        X = np.column_stack([np.ones(len(P)), *[P[x].to_numpy(dtype="float64") for x in xs]])
        yy = P[y].to_numpy(dtype="float64")
        ok = mask & np.isfinite(X).all(axis=1) & np.isfinite(yy)
        if ok.sum() < 100:
            return None
        return np.linalg.lstsq(X[ok], yy[ok], rcond=None)[0]

    def _sun_bins(self, P: pd.DataFrame) -> np.ndarray:
        """Integer code of the sun position bin (15 deg azimuth x 5 deg elevation); -1 when the sun is down."""
        el, az = P["elev"].to_numpy(), P["azim"].to_numpy()
        code = (az // 15).astype(int) * 100 + (np.clip(el, 0, None) // 5).astype(int)
        return np.where(el > 10, code, -1)

    def _pr_res(self, P: pd.DataFrame) -> np.ndarray:
        base = np.interp(P["poa"].to_numpy(), *self.pv_curve)
        corr = 1.0 + self.gamma * (np.nan_to_num(P["t_module"].to_numpy(), nan=25.0) - 25.0)
        res = P["pn"].to_numpy() / (base * corr) - 1.0
        return np.where(P["poa"].to_numpy() >= 150, res, np.nan)

    def _raw_pv(self, df: pd.DataFrame) -> pd.DataFrame:
        P = self._plant(df)
        day = (P["poa"] >= 50).to_numpy()
        pr = self._pr_res(P)
        f: dict[str, np.ndarray] = {"pr_res": pr}
        if "elev" in P and self.tod_bins:
            keys = self._sun_bins(P)
            f["pr_tod"] = pr - np.array([self.tod_bins.get(int(k), np.nan) for k in keys])
        poa = P["poa"].to_numpy()
        std = pd.Series(poa).rolling(self.hour, min_periods=self.hour).std().to_numpy()
        f["poa_flat"] = np.where(poa >= 100, np.log(std + 1.0), np.nan)
        if self.reg_mod is not None:
            X = np.column_stack([np.ones(len(P)), P["t_amb"].to_numpy(), P["poa"].to_numpy()])
            f["t_mod_res"] = np.where(day, P["t_module"].to_numpy() - X @ self.reg_mod, np.nan)
        if self.reg_inv is not None:
            X = np.column_stack([np.ones(len(P)), P["t_amb"].to_numpy(), P["pn"].to_numpy()])
            f["t_inv_res"] = np.where(day, P["t_inv"].to_numpy() - X @ self.reg_inv, np.nan)
        feats = pd.DataFrame(f, index=P.index)
        rows = P.index.get_indexer(pd.DatetimeIndex(df["timestamp"]))
        return feats.iloc[rows].set_index(df.index)
