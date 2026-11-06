"""Sun position (NOAA approximation, Spencer series) in plain numpy.

Used to place the PV shading fault in the part of the day where a fixed obstacle would cast a shadow, without needing pvlib.
The accuracy is a fraction of a degree, which is far finer than the fault needs; ``tests/test_solar.py`` compares it with
pvlib when pvlib is installed.

Usage:
    elev, azim = solar_position(pd.date_range("2017-06-21", periods=96, freq="15min", tz="UTC"), lat=36.2, lon=-115.2)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["solar_position"]


def solar_position(times: pd.DatetimeIndex, lat: float, lon: float) -> tuple[np.ndarray, np.ndarray]:
    """Apparent elevation above the horizon and azimuth (clockwise from north) in degrees, for UTC ``times``."""
    t = pd.DatetimeIndex(times)
    if t.tz is None:
        raise ValueError("times must be timezone-aware (UTC)")
    t = t.tz_convert("UTC")
    doy = t.day_of_year.to_numpy(dtype=float)
    hour = t.hour.to_numpy(dtype=float) + t.minute.to_numpy() / 60.0 + t.second.to_numpy() / 3600.0
    g = 2.0 * np.pi / 365.0 * (doy - 1.0 + (hour - 12.0) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
            - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = hour * 60.0 + eqtime + 4.0 * lon  # true solar time in minutes (times are UTC)
    ha = np.radians(tst / 4.0 - 180.0)
    la = np.radians(lat)
    cos_zen = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)
    elev = 90.0 - np.degrees(np.arccos(np.clip(cos_zen, -1.0, 1.0)))
    az = np.degrees(np.arctan2(np.sin(ha), np.cos(ha) * np.sin(la) - np.tan(decl) * np.cos(la))) + 180.0
    return elev, np.mod(az, 360.0)
