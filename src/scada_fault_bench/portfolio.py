"""Reference methods from other repositories of the portfolio, called through their public code (nothing is copied).

======================  ==========================================  ===================================================================
name                    repository                                  what is called
======================  ==========================================  ===================================================================
nbm_rc                  wind-thermal-nbm                            ``RCModel``: first-order thermal model of the gearbox oil
nbm_gbm                 wind-thermal-nbm                            ``GBMModel``: gradient boosting on load, speed and lagged means
power_curve_kpi         res-asset-kpi (wind-power-curve-kelmarsh)   ``bin_power_curve`` and ``curve_power`` (method of bins)
pv_autoencoder          pv-anomaly-autoencoder                      ``AutoencoderDetector``: windowed autoencoder on power and weather
pv_outage_expected      pv-outage-classifier                        ``detectors.score("expected")``: shortfall against a PVWatts model
pv_outage_iforest       pv-outage-classifier                        ``detectors.score("iforest")``: Isolation Forest on its features
======================  ==========================================  ===================================================================

Each class checks in ``available()`` that its repository can be imported, so the benchmark still runs, with fewer methods,
where it cannot. The other repositories of the portfolio were looked at and are *not* wrapped, for stated reasons
(``docs/METHODS.md``): they work on farm-level daily statistics (wind-wake-aware-verification, wind-bayes-twin), on
per-MPPT channels that these datasets do not have (pv-string-iv-faults) or give a one-off diagnosis of a whole record
(pv-shade-fingerprint).

Adapters only rename and rescale columns (a method written for one turbine type gets power as a share of rated power
times that type's rating), and turn the method's output into a non-negative score. Nothing is tuned here.

Usage:
    METHODS["nbm_rc"]().fit(train, meta).score(table, meta)
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
from scada_schema.model import DatasetMeta

from scada_fault_bench.base import Detector, register
from scada_fault_bench.features import _robust_scale, _slices, rated_of, step_minutes, trailing_median

__all__ = ["NBMGradientBoosting", "NBMThermal", "PVAutoencoder", "PVOutageExpected", "PVOutageIForest", "PowerCurveKPI"]

NBM_RATED_KW, NBM_RATED_RPM = 2050.0, 15.0  # the constants wind-thermal-nbm was written for (Senvion MM92)


def _import(name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(name)
        return True, ""
    except Exception as e:  # missing package or a missing dependency of it
        return False, f"{name} cannot be imported ({type(e).__name__}: {e}); install the repository from GitHub (see README)"


# --------------------------------------------------------------------------------------------- wind-thermal-nbm
class _NBM(Detector):
    family = "portfolio"
    profiles = ("wind",)
    target = "t_gearbox_oil"
    model_name = ""

    @classmethod
    def available(cls):
        return _import("wind_thermal_nbm")

    def _model(self, refs):  # pragma: no cover - overridden
        raise NotImplementedError

    @staticmethod
    def _frame(g: pd.DataFrame, rated: float, rpm_rated: float, target: str) -> pd.DataFrame:
        from wind_thermal_nbm.prep import clean

        t = pd.DatetimeIndex(g["timestamp"])
        grid = pd.date_range(t.min(), t.max(), freq="10min")
        f = pd.DataFrame({
            "power_kw": g["p_ac"].to_numpy(dtype="float64") / rated * NBM_RATED_KW,
            "rotor_rpm": g["rotor_speed"].to_numpy(dtype="float64") / rpm_rated * NBM_RATED_RPM,
            "nacelle_temp": g["t_nacelle"].to_numpy(dtype="float64") if "t_nacelle" in g else np.nan,
            "ambient_temp": g["t_amb"].to_numpy(dtype="float64") if "t_amb" in g else np.nan,
            target: g[target].to_numpy(dtype="float64"),
        }, index=t).reindex(grid)
        return clean(f)

    def fit(self, train, meta):
        from wind_thermal_nbm.prep import operating

        if step_minutes(meta) != 10.0:
            raise ValueError("wind-thermal-nbm works on 10-minute data")
        for c in ("rotor_speed", self.target):
            if c not in train or train[c].notna().sum() < 1000:
                raise ValueError(f"{c} is not available in {meta.name}")
        self.refs = tuple(r for r, c in (("nacelle_temp", "t_nacelle"), ("ambient_temp", "t_amb")) if c in train and train[c].notna().sum() > 1000)
        if not self.refs:
            raise ValueError("neither nacelle nor ambient temperature available")
        self.hour = 6
        self.models, self.scale, self.rated, self.rpm = {}, {}, {}, {}
        for dev, a, b in _slices(train["device_id"].to_numpy()):
            g = train.iloc[a:b]
            self.rated[dev] = rated_of(meta, dev, g["p_ac"])
            self.rpm[dev] = float(np.nanpercentile(g["rotor_speed"].to_numpy(dtype="float64"), 99)) or 1.0
            frame = self._frame(g, self.rated[dev], self.rpm[dev], self.target)
            op = operating(frame).to_numpy()
            model = self._model(self.refs).fit(frame, self.target, mask=op)
            res = (frame[self.target] - model.predict(frame)).to_numpy()
            self.models[dev] = model
            self.scale[dev] = _robust_scale(res[op])
        return self

    def score(self, table, meta):
        out = np.full(len(table), np.nan)
        for dev, a, b in _slices(table["device_id"].to_numpy()):
            if dev not in self.models:
                continue
            g = table.iloc[a:b]
            frame = self._frame(g, self.rated[dev], self.rpm[dev], self.target)
            res = (frame[self.target] - self.models[dev].predict(frame)).to_numpy()
            med, sc = self.scale[dev]
            z = np.abs((res - med) / sc)
            s = pd.Series(trailing_median(z, self.hour), index=frame.index)
            out[a:b] = s.reindex(pd.DatetimeIndex(g["timestamp"])).to_numpy()
        return out


@register
class NBMThermal(_NBM):
    name = "nbm_rc"
    title = "Thermal model (RC)"
    description = "wind-thermal-nbm RCModel: first-order thermal model of the gearbox oil temperature from load, rotor speed, nacelle and ambient temperature; the standardised residual is the score"
    source = "wind-thermal-nbm: wind_thermal_nbm.thermal.RCModel"
    scope = ("thermal_drift", "unit_jump")  # the gearbox oil temperature is the only channel this model reads as a target

    def _model(self, refs):
        from wind_thermal_nbm.thermal import RCModel

        return RCModel(refs=refs)


@register
class NBMGradientBoosting(_NBM):
    name = "nbm_gbm"
    title = "Thermal model (GBM)"
    description = "wind-thermal-nbm GBMModel: gradient boosting on the same inputs and their trailing means (1-12 h) for the gearbox oil temperature; the standardised residual is the score"
    source = "wind-thermal-nbm: wind_thermal_nbm.ml.GBMModel"
    scope = ("thermal_drift", "unit_jump")

    def _model(self, refs):
        from wind_thermal_nbm.ml import GBMModel

        return GBMModel(refs=refs)


# --------------------------------------------------------------------------------------------- res-asset-kpi
@register
class PowerCurveKPI(Detector):
    name = "power_curve_kpi"
    title = "Power curve (method of bins)"
    family = "portfolio"
    profiles = ("wind",)
    description = "res-asset-kpi bin_power_curve per turbine on the training weeks; the standardised residual of power against the curve at the measured wind speed is the score"
    source = "res-asset-kpi: res_asset_kpi.wind.bin_power_curve, curve_power (the method of wind-power-curve-kelmarsh)"
    scope = ("anemometer_drift", "derating", "power_loss", "flatline", "clock_shift")  # faults in power or wind speed

    @classmethod
    def available(cls):
        return _import("res_asset_kpi")

    def fit(self, train, meta):
        from res_asset_kpi.wind import bin_power_curve

        self.hour = max(2, int(round(60.0 / step_minutes(meta))))
        self.curve, self.rated, self.scale = {}, {}, {}
        for dev, a, b in _slices(train["device_id"].to_numpy()):
            g = train.iloc[a:b]
            rated = rated_of(meta, dev, g["p_ac"])
            ws, p = g["wind_speed"].to_numpy(dtype="float64"), g["p_ac"].to_numpy(dtype="float64")
            ok = np.isfinite(ws) & np.isfinite(p) & (ws >= 2) & (ws <= 25)
            c = bin_power_curve(ws[ok], p[ok])
            self.curve[dev], self.rated[dev] = (c["ws"].to_numpy(), c["power_kw"].to_numpy()), rated
            self.scale[dev] = _robust_scale(trailing_median(self._res(g, dev), self.hour))
        return self

    def _res(self, g: pd.DataFrame, dev: str) -> np.ndarray:
        from res_asset_kpi.wind import curve_power

        ws, p = g["wind_speed"].to_numpy(dtype="float64"), g["p_ac"].to_numpy(dtype="float64")
        exp = curve_power(*self.curve[dev], ws)
        return np.where((ws >= 2) & (ws <= 25), (p - exp) / self.rated[dev], np.nan)

    def score(self, table, meta):
        out = np.full(len(table), np.nan)
        for dev, a, b in _slices(table["device_id"].to_numpy()):
            if dev in self.curve:
                med, sc = self.scale[dev]
                out[a:b] = np.abs((trailing_median(self._res(table.iloc[a:b], dev), self.hour) - med) / sc)
        return out


# --------------------------------------------------------------------------------------------- PV helpers
def _pv_devices(meta: DatasetMeta):
    met = next((d for d in meta.devices if d.type == "met"), None)
    inv = [d for d in meta.devices if d.type == "inverter"]
    if met is None or not inv:
        raise ValueError("a PV table needs one met station and at least one inverter")
    lat = next((d.lat for d in meta.devices if d.lat is not None), None)
    lon = next((d.lon for d in meta.devices if d.lon is not None), None)
    if lat is None or lon is None:
        raise ValueError("the PV metadata has no location")
    return met, inv, lat, lon


def _series(table: pd.DataFrame, device: str, col: str, grid: pd.DatetimeIndex) -> pd.Series:
    g = table[table["device_id"] == device].drop_duplicates("timestamp").set_index("timestamp")
    return g[col].astype("float64").reindex(grid) if col in g else pd.Series(np.nan, index=grid)


def _grid(table: pd.DataFrame, meta: DatasetMeta) -> pd.DatetimeIndex:
    t = pd.DatetimeIndex(table["timestamp"])
    return pd.date_range(t.min(), t.max(), freq=meta.interval)


def _back(values: dict[str, pd.Series], table: pd.DataFrame) -> np.ndarray:
    """Scores per inverter on the time grid to the rows of the table: inverter rows get their own score, other rows NaN."""
    out = np.full(len(table), np.nan)
    ts = pd.DatetimeIndex(table["timestamp"])
    for dev, s in values.items():
        rows = np.flatnonzero((table["device_id"] == dev).to_numpy())
        out[rows] = s.reindex(ts[rows]).to_numpy()
    return out


# --------------------------------------------------------------------------------------------- pv-anomaly-autoencoder
@register
class PVAutoencoder(Detector):
    name = "pv_autoencoder"
    title = "Autoencoder (PV)"
    family = "portfolio"
    profiles = ("pv",)
    description = ("pv-anomaly-autoencoder AutoencoderDetector per inverter: windows of 8 steps of power, performance ratio, irradiance, clear-sky index, temperatures and sun position; "
                   "the reconstruction error (mse), smoothed over one hour, is the score; trained for at most 100 epochs on 80 % of the training days")
    source = "pv-anomaly-autoencoder: pv_anomaly.autoencoder.AutoencoderDetector, pv_anomaly.features.build_features"

    @classmethod
    def available(cls):
        ok, why = _import("pv_anomaly")
        return (ok, why) if not ok else _import("torch")

    def _feats(self, table, meta, dev, grid):
        from pv_anomaly.features import build_features

        met, _, _, _ = _pv_devices(meta)
        df = pd.DataFrame({
            "p_ac": _series(table, dev, "p_ac", grid), "poa": _series(table, met.id, "poa", grid),
            "t_mod": _series(table, met.id, "t_module", grid), "t_amb": _series(table, met.id, "t_amb", grid),
        })
        df["t_amb"] = df["t_amb"].fillna(df["t_mod"])
        return build_features(df, self.plants[dev])

    def fit(self, train, meta):
        import torch
        from pv_anomaly.autoencoder import AEConfig, AutoencoderDetector
        from pv_anomaly.features import FEATURES
        from pv_anomaly.plant import PlantConfig

        torch.set_num_threads(2)
        _, inv, lat, lon = _pv_devices(meta)
        grid = _grid(train, meta)
        self.plants, self.models = {}, {}
        for d in inv:
            rated = float(d.rated_kw or d.capacity_kwp)
            self.plants[d.id] = PlantConfig(latitude=lat, longitude=lon, tz="Etc/GMT+8", tilt=d.tilt_deg or 20.0, azimuth=d.azimuth_deg or 180.0,
                                            pdc0_kw=float(d.capacity_kwp or rated), pac0_kw=rated)
            f = self._feats(train, meta, d.id, grid)
            ok = f[list(FEATURES)].notna().all(axis=1)
            window = AEConfig().window
            full = ok.rolling(window, min_periods=window).min().fillna(0).astype(bool).to_numpy()  # whole window finite
            day = pd.DatetimeIndex(f.index).normalize()
            rng = np.random.default_rng(0)
            days = np.unique(day)
            val_days = set(rng.choice(days, size=int(round(0.2 * len(days))), replace=False))
            is_val = np.array([x in val_days for x in day])
            use = full & f["daylight"].to_numpy()
            f = f.copy()
            f[list(FEATURES)] = f[list(FEATURES)].fillna(0.0)
            det = AutoencoderDetector(AEConfig())
            det.fit(f, use & ~is_val, use & is_val)
            self.models[d.id] = det
        return self

    def score(self, table, meta):
        from pv_anomaly.features import FEATURES

        _, inv, _, _ = _pv_devices(meta)
        grid = _grid(table, meta)
        hour = max(2, int(round(60.0 / step_minutes(meta))))
        out = {}
        for d in inv:
            f = self._feats(table, meta, d.id, grid)
            bad = ~f[list(FEATURES)].notna().all(axis=1)
            bad_win = bad.rolling(8, min_periods=1).max().astype(bool).to_numpy()  # a window with a missing value cannot be scored
            f = f.copy()
            f[list(FEATURES)] = f[list(FEATURES)].fillna(0.0)
            s = self.models[d.id].score(f, kind="mse").to_numpy()
            s = np.where(f["daylight"].to_numpy() & ~bad_win, s, np.nan)
            out[d.id] = pd.Series(trailing_median(s, hour), index=grid)
        return _back(out, table)


# --------------------------------------------------------------------------------------------- pv-outage-classifier
class _PVOutage(Detector):
    family = "portfolio"
    profiles = ("pv",)
    method = ""

    @classmethod
    def available(cls):
        ok, why = _import("pv_outage")
        return (ok, why) if not ok else _import("pvlib")

    def fit(self, train, meta):
        _pv_devices(meta)  # checks that the table has a met station, an inverter and a location
        return self

    def score(self, table, meta):
        from pv_outage import detectors, features
        from pv_outage.schema import to_panel

        met, inv, lat, lon = _pv_devices(meta)
        m = table[table["device_id"] == met.id]
        parts = [pd.DataFrame({"timestamp": m["timestamp"], "device_id": "PLANT", "parent_id": "", "level": "plant",
                               "poa": m["poa"].to_numpy(), "t_mod": m["t_module"].to_numpy()})]
        for d in inv:
            g = table[table["device_id"] == d.id]
            cols = {"p_ac": g["p_ac"].to_numpy()}
            for src, dst in (("i_dc", "i_dc"), ("v_dc", "v_dc")):
                if src in g:
                    cols[dst] = g[src].to_numpy()
            parts.append(pd.DataFrame({"timestamp": g["timestamp"], "device_id": d.id, "parent_id": "PLANT", "level": "inverter", **cols}))
        long = pd.concat(parts, ignore_index=True)
        pm = {"lat": lat, "lon": lon, "tilt": next((d.tilt_deg for d in inv if d.tilt_deg), 20.0), "azimuth": next((d.azimuth_deg for d in inv if d.azimuth_deg), 180.0),
              "pac_kw": {d.id: float(d.rated_kw or d.capacity_kwp) for d in inv}}
        panel = to_panel(long, pm)
        f = features.compute(panel, calibration="poa")
        s = np.asarray(detectors.score(self.method, f), dtype="float64")
        s = np.where(np.isfinite(s), np.maximum(s, 0.0), np.nan)
        cols_ix = {dev: j for j, dev in enumerate(panel.devices.index)}
        out = {d.id: pd.Series(s[:, cols_ix[d.id]], index=panel.time) for d in inv}
        return _back(out, table)


@register
class PVOutageExpected(_PVOutage):
    name = "pv_outage_expected"
    title = "Shortfall vs PVWatts model"
    description = ("pv-outage-classifier detectors.score('expected'): shortfall of inverter power against a PVWatts-style model driven by the met station's irradiance and module "
                   "temperature, calibrated per bin on the table that is scored (as in that repository); shortfall only")
    source = "pv-outage-classifier: pv_outage.features.compute, pv_outage.detectors.score('expected')"
    scope = ("string_loss", "shading", "inverter_outage", "pyranometer_drift", "flatline", "clock_shift")  # faults in power or irradiance
    method = "expected"


@register
class PVOutageIForest(_PVOutage):
    name = "pv_outage_iforest"
    title = "Outage features + Isolation Forest"
    description = ("pv-outage-classifier detectors.score('iforest'): Isolation Forest on that repository's step features (residuals, level, sun position, ...), fitted on the "
                   "table that is scored, as in that repository")
    source = "pv-outage-classifier: pv_outage.features.compute, pv_outage.detectors.score('iforest')"
    scope = ("string_loss", "shading", "inverter_outage", "pyranometer_drift", "flatline", "clock_shift")
    method = "iforest"
