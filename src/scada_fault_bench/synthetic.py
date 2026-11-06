"""Small synthetic plants in the ``scada-schema`` format, for tests and for trying the benchmark without downloading data.

They are physically plausible, not realistic: correlated wind across turbines, a power curve, a thermal lag for the
gearbox oil, a clear-sky irradiance model for PV with cloud noise. Nothing in the published results comes from them.

Usage:
    df, meta = synthetic_wind(n_turbines=5, days=120, seed=0)
    df, meta = synthetic_pv(days=120, seed=0)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scada_schema import finalise
from scada_schema.model import DatasetMeta, Device

from scada_fault_bench.solar import solar_position

__all__ = ["synthetic_pv", "synthetic_wind"]

RATED_KW = 2000.0


def _power_curve(ws: np.ndarray) -> np.ndarray:
    x = np.clip((ws - 3.0) / (12.0 - 3.0), 0, 1)
    p = RATED_KW * x**3
    return np.where(ws > 25.0, 0.0, p)


def _ar1(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    e = rng.normal(size=n) * np.sqrt(1 - phi**2)
    x = np.empty(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def synthetic_wind(n_turbines: int = 5, days: int = 120, seed: int = 0, start: str = "2019-01-01") -> tuple[pd.DataFrame, DatasetMeta]:
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=days * 144, freq="10min", tz="UTC")
    n = len(t)
    farm_ws = np.clip(8.0 + 3.5 * _ar1(rng, n, 0.997), 0.0, 30.0)
    farm_dir = np.mod(220.0 + 60.0 * _ar1(rng, n, 0.998), 360.0)
    t_amb_farm = 10.0 + 6.0 * np.sin(2 * np.pi * (t.day_of_year.to_numpy() - 100) / 365) + 3.0 * np.sin(2 * np.pi * t.hour.to_numpy() / 24)
    frames, devices = [], []
    for k in range(n_turbines):
        dev = f"T{k + 1:02d}"
        ws = np.clip(farm_ws * (1 + 0.03 * rng.normal()) + 0.35 * rng.normal(size=n), 0.0, 35.0)
        ws2 = np.clip(ws * 1.01 + 0.15 * rng.normal(size=n), 0.0, 35.0)
        p = np.clip(_power_curve(ws) + 12.0 * rng.normal(size=n), 0.0, RATED_KW)
        wdir = np.mod(farm_dir + 3.0 * rng.normal(size=n), 360.0)
        nac = np.mod(wdir - 2.0 + 4.0 * rng.normal(size=n), 360.0)
        rel = np.mod(wdir - nac + 180.0, 360.0) - 180.0
        load = p / RATED_KW
        oil = np.empty(n)
        target = 30.0 + 35.0 * load + 0.4 * t_amb_farm
        oil[0] = target[0]
        for i in range(1, n):
            oil[i] = oil[i - 1] + (target[i] - oil[i - 1]) * 0.08
        oil += 0.15 * rng.normal(size=n)
        energy = p / 6.0
        frames.append(pd.DataFrame({
            "timestamp": t, "device_id": dev, "p_ac": p, "e_export": energy, "e_export_counter": np.cumsum(energy) + 1e5,
            "wind_speed": ws, "wind_speed_2": ws2, "wind_dir": wdir, "wind_dir_rel": rel, "nacelle_dir": nac,
            "rotor_speed": np.clip(15.0 * np.clip(ws / 10.0, 0, 1) ** 0.8, 0, 15.0) * (ws > 3), "pitch": np.where(ws > 12, (ws - 12) * 3.0, 0.0),
            "t_amb": t_amb_farm + 0.3 * rng.normal(size=n), "t_nacelle": t_amb_farm + 8.0 * load + 5.0, "t_gearbox_oil": oil,
            "i_ac": p * 0.6,
        }))
        devices.append(Device(id=dev, type="turbine", rated_kw=RATED_KW))
    df = pd.concat(frames, ignore_index=True)
    meta = DatasetMeta(name="synthetic_wind", profile="wind", interval="10min", devices=devices)
    return finalise(df, meta), meta


def synthetic_pv(days: int = 120, seed: int = 0, start: str = "2017-03-01", inverters: int = 1) -> tuple[pd.DataFrame, DatasetMeta]:
    rng = np.random.default_rng(seed)
    lat, lon = 36.2, -115.2
    t = pd.date_range(start, periods=days * 96, freq="15min", tz="UTC")
    n = len(t)
    elev, _ = solar_position(t + pd.Timedelta(minutes=7.5), lat, lon)
    clear = np.where(elev > 0, 1000.0 * np.sin(np.radians(np.clip(elev, 0, 90))) ** 1.15, 0.0)
    cloud = np.clip(1.0 - 0.35 * np.clip(_ar1(rng, n, 0.9) + 0.3, 0, None), 0.15, 1.0)
    poa = np.clip(clear * cloud + 4.0 * rng.normal(size=n) * (clear > 5), 0.0, 1300.0)
    t_amb = 20.0 + 12.0 * np.sin(2 * np.pi * (t.day_of_year.to_numpy() - 110) / 365) + 6.0 * np.sin(2 * np.pi * (t.hour.to_numpy() - 8) / 24)
    t_mod = t_amb + 0.03 * poa
    rated = 100.0
    frames = [pd.DataFrame({"timestamp": t, "device_id": "MET", "poa": poa, "t_amb": t_amb, "t_module": t_mod})]
    devices = [Device(id="MET", type="met", lat=lat, lon=lon)]
    for k in range(inverters):
        dev = "INV" if inverters == 1 else f"INV{k + 1}"
        p = np.clip(rated * 0.9 * poa / 1000.0 * (1 - 0.004 * (t_mod - 25.0)) + 0.4 * rng.normal(size=n) * (poa > 5), 0.0, rated)
        pdc = p / 0.97
        energy = p / 4.0
        frames.append(pd.DataFrame({
            "timestamp": t, "device_id": dev, "p_ac": p, "p_dc": pdc, "i_dc": pdc * 1000.0 / 600.0, "v_dc": 600.0 + 0 * p,
            "t_inverter": t_amb + 12.0 * p / rated, "e_export": energy, "e_export_counter": np.cumsum(energy),
        }))
        devices.append(Device(id=dev, type="inverter", rated_kw=rated, capacity_kwp=110.0, lat=lat, lon=lon))
    df = pd.concat(frames, ignore_index=True)
    meta = DatasetMeta(name="synthetic_pv", profile="pv", interval="15min", devices=devices)
    return finalise(df, meta), meta
