"""Fault injection with known ground truth, on tables in the ``scada-schema`` format.

A *fault type* is a small model of one thing that goes wrong with a wind turbine or a PV plant in the data (a sensor
drifts, a channel freezes, a clock is wrong, a string is lost, ...). :func:`inject_type` puts five *levels* of it, with
``repeats`` events per level, into a table: the device, the start and the size of every event are drawn from a seeded
generator, so the same ``(table, type, seed)`` always gives the same faults. The result is the modified table and the
truth, one row per event.

What the truth contains
-----------------------
``start`` and ``end`` of an event are the first and the last row where the injected change is larger than the tolerance
of the type (``end`` is exclusive: last such row plus one interval); :mod:`scada_fault_bench.scenario` then narrows them to
the rows where an alarm can be right or wrong (not at night, not during a known stop). ``window_start`` and ``window_end`` are the nominal
window that was drawn. The tolerance keeps the truth honest: a ramp that is still zero at its start, or a derating in a
calm week, does not count as a fault that a method should have found. An event whose effect would show in fewer than
``MIN_AFFECTED_HOURS`` hours of data (for the duration-graded types: in less than half of its window) is redrawn.

What the faults are, and are not
--------------------------------
Every fault is a change of the logged signals of one device (a *signal-level* model). Where a physical fault would also
move other channels (a real derating changes rotor speed and pitch, a real string loss changes the DC voltage) only the
channels that follow from simple arithmetic are changed together: power, current and energy move together and the
energy meter counter is corrected, so that the energy checks of the quality control stay consistent. This is a
simplification that favours detectors based on one or two channels over those that use the full physical picture. The
README says so, and the results must be read with it.

Usage:
    out, truth, affected = inject_type(table, meta, "derating", seed=3, t_lo=t0, t_hi=t1)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scada_schema.model import DatasetMeta

from scada_fault_bench.solar import solar_position

log = logging.getLogger(__name__)

__all__ = ["FAULT_TYPES", "LEVEL_COUNT", "MIN_AFFECTED_HOURS", "TRUTH_COLUMNS", "FaultType", "inject_type", "types_for"]

LEVEL_COUNT = 5
MIN_AFFECTED_HOURS = 3.0
GUARD_DAYS = 2.0
TRUTH_COLUMNS = ["fault_id", "type", "device_id", "level", "magnitude", "unit", "start", "end", "window_start", "window_end", "n_affected", "params"]
KEYS = ("timestamp", "device_id")


# ------------------------------------------------------------------------------------------------------ helpers
def family(df: pd.DataFrame, base: str) -> list[str]:
    """A signal column and its statistics columns (``base__min``, ``base__max``, ``base__std``)."""
    return [c for c in df.columns if c == base or c.startswith(base + "__")]


def _interval_minutes(meta: DatasetMeta) -> float:
    return pd.Timedelta(meta.interval).total_seconds() / 60.0


def _set(w: pd.DataFrame, col: str, values: np.ndarray) -> None:
    w[col] = values.astype(w[col].dtype, copy=False)


def _scale_cols(w: pd.DataFrame, base: str, factor: np.ndarray) -> None:
    for c in family(w, base):
        _set(w, c, w[c].to_numpy(dtype="float64") * factor)


def _ramp(w: pd.DataFrame) -> np.ndarray:
    t = w["timestamp"].astype("int64").to_numpy().astype("float64")
    span = t[-1] - t[0]
    return (t - t[0]) / span if span > 0 else np.ones(len(t))


def _rated_kw(meta: DatasetMeta, device: str, series: pd.Series) -> float:
    for d in meta.devices:
        if d.id == device:
            if d.rated_kw:
                return float(d.rated_kw)
            if d.capacity_kwp:
                return float(d.capacity_kwp)
    v = series.dropna()
    return float(np.nanpercentile(v, 99.5)) if len(v) else 1.0


@dataclass
class Ctx:
    """What an injector may look at besides the window itself."""

    meta: DatasetMeta
    device: str
    rated_kw: float
    interval_min: float
    rng: np.random.Generator
    dev: pd.DataFrame  # the device's whole table, read-only
    lat: float | None = None
    lon: float | None = None
    params: dict = field(default_factory=dict)


def _scale_power(w: pd.DataFrame, ratio: np.ndarray, ctx: Ctx) -> np.ndarray:
    """Scale power, current and energy of the window by ``ratio`` (per row) and return the energy lost per row in kWh.

    ``p_ac``, ``p_dc``, ``i_ac``, ``i_dc`` (with their statistics) and the interval energy ``e_export`` follow the ratio.
    The cumulative meter reading is corrected afterwards by the caller from the returned loss.
    """
    hours = ctx.interval_min / 60.0
    p0 = w["p_ac"].to_numpy(dtype="float64") if "p_ac" in w else None
    loss = np.zeros(len(w))
    if "e_export" in w.columns:
        e0 = w["e_export"].to_numpy(dtype="float64")
        new = e0 * ratio
        loss = np.nan_to_num(e0 - new)
        _set(w, "e_export", new)
    elif p0 is not None:
        loss = np.nan_to_num(p0 * (1.0 - ratio)) * hours
    for base in ("p_ac", "p_dc", "i_ac", "i_dc"):
        if base in w.columns:
            _scale_cols(w, base, ratio)
    return loss


def _ratio_from_power(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(old > 0, new / old, 1.0)
    return np.where(np.isfinite(r), r, 1.0)


# ------------------------------------------------------------------------------------------------------ injectors
# each takes (w, ctx, magnitude) with w the copy of the device's rows in the window; it changes w and returns
# (affected mask, energy loss per row or None)

def _anemometer_drift(w, ctx, m):
    sign = ctx.params.setdefault("sign", int(ctx.rng.choice([-1, 1])))
    old = w["wind_speed"].to_numpy(dtype="float64")
    gain = 1.0 + sign * m / 100.0 * _ramp(w)
    _scale_cols(w, "wind_speed", gain)
    return np.nan_to_num(np.abs(w["wind_speed"].to_numpy(dtype="float64") - old)) > 0.1, None


def _nacelle_dir_offset(w, ctx, m):
    sign = ctx.params.setdefault("sign", int(ctx.rng.choice([-1, 1])))
    old = w["nacelle_dir"].to_numpy(dtype="float64")
    _set(w, "nacelle_dir", np.mod(old + sign * m, 360.0))
    return np.isfinite(old) & (m > 0.5), None


def _derating(w, ctx, m):
    old = w["p_ac"].to_numpy(dtype="float64")
    cap = (1.0 - m / 100.0) * ctx.rated_kw
    new = np.where(old > cap, cap, old)
    ratio = _ratio_from_power(old, new)
    loss = _scale_power(w, ratio, ctx)
    cap_cols = [c for c in w.columns if c in ("p_ac__max", "p_ac__min")]
    for c in cap_cols:
        _set(w, c, np.minimum(w[c].to_numpy(dtype="float64"), cap))
    return np.nan_to_num(old - new) > 0.005 * ctx.rated_kw, loss


def _power_loss(w, ctx, m):
    old = w["p_ac"].to_numpy(dtype="float64")
    ratio = np.full(len(w), 1.0 - m / 100.0)
    loss = _scale_power(w, ratio, ctx)
    return np.nan_to_num(old * m / 100.0) > 0.005 * ctx.rated_kw, loss


def _flatline_base(ctx) -> str:
    return "wind_speed" if ctx.meta.profile == "wind" else "poa"


def _flatline(w, ctx, hours):
    base = _flatline_base(ctx)
    old = w[base].to_numpy(dtype="float64")
    valid = np.flatnonzero(np.isfinite(old))
    if len(valid) == 0:
        return np.zeros(len(w), dtype=bool), None
    hold = old[valid[0]]
    _set(w, base, np.full(len(w), hold))
    for c in family(w, base):
        if c.endswith("__std"):
            _set(w, c, np.zeros(len(w)))
        elif c.endswith("__min") or c.endswith("__max"):
            _set(w, c, np.full(len(w), hold))
    tol = 0.05 if base == "wind_speed" else 5.0
    return np.nan_to_num(np.abs(old - hold)) > tol, None


def _unit_jump(w, ctx, hours):
    base = "t_gearbox_oil" if ctx.meta.profile == "wind" else "t_module"
    old = w[base].to_numpy(dtype="float64")
    for c in family(w, base):
        if c.endswith("__std"):
            _set(w, c, w[c].to_numpy(dtype="float64") * 1.8)
        else:
            _set(w, c, w[c].to_numpy(dtype="float64") * 1.8 + 32.0)  # degC logged as degF
    return np.nan_to_num(np.abs(w[base].to_numpy(dtype="float64") - old)) > 0.5, None


def _clock_shift(w, ctx, minutes):
    k = max(1, round(minutes / ctx.interval_min))
    step = pd.Timedelta(minutes=ctx.interval_min)
    t0, t1 = w["timestamp"].iloc[0], w["timestamp"].iloc[-1]
    grid = pd.date_range(t0 - k * step, t1, freq=step)
    cols = [c for c in w.columns if c not in KEYS]
    src = ctx.dev.set_index("timestamp")[cols].reindex(grid)
    shifted = src.shift(k).reindex(pd.DatetimeIndex(w["timestamp"]))
    probe = "p_ac" if "p_ac" in cols else cols[0]
    old = w[probe].to_numpy(dtype="float64")
    for c in cols:
        w[c] = shifted[c].to_numpy(dtype=w[c].dtype if w[c].dtype != object else object)
    new = w[probe].to_numpy(dtype="float64")
    tol = 0.02 * ctx.rated_kw if probe == "p_ac" else 1e-6
    both = np.isfinite(old) & np.isfinite(new)
    return (both & (np.abs(new - old) > tol)) | (np.isfinite(old) ^ np.isfinite(new)), None


def _thermal_drift(w, ctx, dk):
    base = "t_gearbox_oil" if ctx.meta.profile == "wind" else "t_inverter"
    add = dk * _ramp(w)
    for c in family(w, base):
        if not c.endswith("__std"):
            _set(w, c, w[c].to_numpy(dtype="float64") + add)
    return np.nan_to_num(np.abs(add)) > 0.5, None


def _string_loss(w, ctx, pct):
    old = w["p_ac"].to_numpy(dtype="float64")
    ratio = np.full(len(w), 1.0 - pct / 100.0)
    loss = _scale_power(w, ratio, ctx)
    return np.nan_to_num(old * pct / 100.0) > 0.005 * ctx.rated_kw, loss


def _shading(w, ctx, pct):
    if ctx.lat is None or ctx.lon is None:
        raise ValueError("shading needs lat/lon of the plant in the metadata")
    mid = pd.DatetimeIndex(w["timestamp"]) + pd.Timedelta(minutes=ctx.interval_min / 2)
    elev, azim = solar_position(mid, ctx.lat, ctx.lon)
    shaded = (elev > 0) & (elev < 30.0) & (azim > 200.0)  # a fixed obstacle in the west, low sun in the afternoon
    old = w["p_ac"].to_numpy(dtype="float64")
    ratio = np.where(shaded, 1.0 - pct / 100.0, 1.0)
    loss = _scale_power(w, ratio, ctx)
    return shaded & (np.nan_to_num(old * pct / 100.0) > 0.005 * ctx.rated_kw), loss


def _inverter_outage(w, ctx, hours):
    old = w["p_ac"].to_numpy(dtype="float64")
    loss = _scale_power(w, np.zeros(len(w)), ctx)
    return np.nan_to_num(old) > 0.005 * ctx.rated_kw, loss


def _pyranometer_drift(w, ctx, m):
    sign = ctx.params.setdefault("sign", int(ctx.rng.choice([-1, 1])))
    old = w["poa"].to_numpy(dtype="float64")
    gain = 1.0 + sign * m / 100.0 * _ramp(w)
    for base in ("poa", "ghi"):
        if base in w.columns:
            _scale_cols(w, base, gain)
    return np.nan_to_num(np.abs(w["poa"].to_numpy(dtype="float64") - old)) > 5.0, None


@dataclass(frozen=True)
class FaultType:
    """One kind of injected fault: where it applies, how big its five levels are and how it changes the table."""

    name: str
    summary: str
    device_type: dict  # profile -> device type
    requires: dict  # profile -> columns the device needs
    unit: str
    levels: tuple
    level_kind: str  # "size" (the level is the size of the change) or "duration" (the level is the duration, hours)
    duration_days: float  # window length of a size-graded fault in wind farms (ignored for duration-graded)
    fn: Callable
    ramp: bool = False
    pv_days: float = 7.0  # the same for PV plants, where one inverter or met station must carry all events of the block

    def profiles(self) -> tuple[str, ...]:
        return tuple(self.device_type)

    def window(self, level_value: float, profile: str = "wind") -> pd.Timedelta:
        if self.level_kind == "duration":
            return pd.Timedelta(hours=level_value)
        return pd.Timedelta(days=self.duration_days if profile == "wind" else self.pv_days)


_W, _P = "wind", "pv"
FAULT_TYPES: dict[str, FaultType] = {f.name: f for f in [
    FaultType("anemometer_drift", "nacelle anemometer gain drifts linearly to +-m % (sign drawn per event); the second anemometer is untouched",
              {_W: "turbine"}, {_W: ("wind_speed",)}, "%", (3, 6, 9, 12, 15), "size", 14, _anemometer_drift, True),
    FaultType("nacelle_dir_offset", "nacelle position sensor offset by +-m degrees (step); the vane and the absolute direction are untouched",
              {_W: "turbine"}, {_W: ("nacelle_dir",)}, "deg", (3, 6, 10, 15, 25), "size", 14, _nacelle_dir_offset),
    FaultType("derating", "active power capped at (1 - m %) of rated power; power, current and energy move together",
              {_W: "turbine"}, {_W: ("p_ac",)}, "% of rated", (5, 10, 20, 30, 40), "size", 14, _derating),
    FaultType("power_loss", "active power reads (1 - m %) of the true value at every power level (underperformance)",
              {_W: "turbine"}, {_W: ("p_ac",)}, "%", (2, 4, 6, 10, 15), "size", 21, _power_loss),
    FaultType("thermal_drift", "temperature channel drifts linearly to +m K (gearbox oil of a turbine, heat sink of an inverter)",
              {_W: "turbine", _P: "inverter"}, {_W: ("t_gearbox_oil",), _P: ("t_inverter",)}, "K", (2, 4, 6, 10, 15), "size", 21, _thermal_drift, True, 14.0),
    FaultType("flatline", "sensor stuck at its first value of the window (wind speed of a turbine, plane-of-array irradiance of a met station)",
              {_W: "turbine", _P: "met"}, {_W: ("wind_speed",), _P: ("poa",)}, "h", (3, 6, 12, 24, 48), "duration", 0, _flatline),
    FaultType("unit_jump", "temperature logged in degF instead of degC (gearbox oil of a turbine, module temperature of a met station)",
              {_W: "turbine", _P: "met"}, {_W: ("t_gearbox_oil",), _P: ("t_module",)}, "h", (6, 12, 24, 48, 96), "duration", 0, _unit_jump),
    FaultType("clock_shift", "all channels of one device are delayed by m minutes against the others (logger clock error)",
              {_W: "turbine", _P: "inverter"}, {_W: ("p_ac", "wind_speed"), _P: ("p_ac",)}, "min", (10, 30, 60, 120, 360), "size", 14, _clock_shift),
    FaultType("string_loss", "a share of the strings of an inverter is lost: AC and DC power and current scale by (1 - m %)",
              {_P: "inverter"}, {_P: ("p_ac",)}, "%", (2, 5, 10, 20, 35), "size", 14, _string_loss),
    FaultType("shading", "a fixed obstacle in the west shades the plant at low sun in the afternoon: power scales by (1 - m %) when the sun is below 30 deg and west of south-west",
              {_P: "inverter"}, {_P: ("p_ac",)}, "%", (5, 10, 20, 35, 50), "size", 14, _shading),
    FaultType("inverter_outage", "inverter produces nothing for the duration of the window (power and currents zero)",
              {_P: "inverter"}, {_P: ("p_ac",)}, "h", (2, 6, 12, 24, 72), "duration", 0, _inverter_outage),
    FaultType("pyranometer_drift", "plane-of-array irradiance gain drifts linearly to +-m % (soiling, ageing; sign drawn per event)",
              {_P: "met"}, {_P: ("poa",)}, "%", (2, 4, 6, 10, 15), "size", 21, _pyranometer_drift, True, 14.0),
]}


def types_for(profile: str) -> list[str]:
    """Names of the fault types that exist for a profile (``wind`` or ``pv``), in registry order."""
    return [n for n, f in FAULT_TYPES.items() if profile in f.profiles()]


# ------------------------------------------------------------------------------------------------------ placement
def _device_slices(df: pd.DataFrame) -> dict[str, tuple[int, int]]:
    ids = df["device_id"].to_numpy()
    out: dict[str, tuple[int, int]] = {}
    start = 0
    for i in range(1, len(ids) + 1):
        if i == len(ids) or ids[i] != ids[start]:
            out[ids[start]] = (start, i)
            start = i
    return out


def _is_sorted_by_device(df: pd.DataFrame) -> bool:
    d = df["device_id"].to_numpy()
    return len(set(d)) == 1 + int(np.count_nonzero(d[1:] != d[:-1]))


def _eligible_devices(df: pd.DataFrame, meta: DatasetMeta, ft: FaultType, slices) -> list[str]:
    dtype = ft.device_type[meta.profile]
    need = ft.requires[meta.profile]
    ids = [d.id for d in meta.devices if d.type == dtype and d.id in slices]
    out = []
    for dev in ids:
        a, b = slices[dev]
        if all(c in df.columns and df[c].iloc[a:b].notna().mean() > 0.5 for c in need):
            out.append(dev)
    return out


def _lat_lon(meta: DatasetMeta) -> tuple[float | None, float | None]:
    for d in meta.devices:
        if d.lat is not None and d.lon is not None:
            return d.lat, d.lon
    return None, None


def _free_intervals(lo: pd.Timestamp, hi: pd.Timestamp, win: pd.Timedelta, guard: pd.Timedelta,
                    taken: list[tuple[pd.Timestamp, pd.Timestamp]]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Start times in ``[lo, hi]`` for which a window of length ``win`` keeps ``guard`` away from every taken window."""
    blocked = sorted((a - guard - win, b + guard) for a, b in taken)
    free, cur = [], lo
    for f0, f1 in blocked:
        if f0 > cur:
            free.append((cur, min(f0, hi)))
        cur = max(cur, f1)
        if cur > hi:
            break
    if cur <= hi:
        free.append((cur, hi))
    return [(a, b) for a, b in free if b >= a]


def _draw_start(rng: np.random.Generator, free: list[tuple[pd.Timestamp, pd.Timestamp]], step: pd.Timedelta) -> pd.Timestamp:
    """A start on the interval grid, uniform over the free intervals."""
    sizes = np.array([int((b - a) / step) + 1 for a, b in free])
    k = int(rng.integers(int(sizes.sum())))
    i = int(np.searchsorted(np.cumsum(sizes), k, side="right"))
    off = k - (int(np.cumsum(sizes)[i - 1]) if i else 0)
    return (free[i][0] + step * off).ceil(step)


def inject_type(df: pd.DataFrame, meta: DatasetMeta, ftype: str, *, seed: int, t_lo: pd.Timestamp, t_hi: pd.Timestamp,
                repeats: int = 2, max_tries: int = 100) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Inject ``LEVEL_COUNT`` levels x ``repeats`` events of ``ftype`` between ``t_lo`` and ``t_hi``.

    Returns ``(table, truth, affected)``: a modified copy of ``df`` (the input is not changed), the truth table
    (columns ``TRUTH_COLUMNS``) and a boolean array over the rows of the table that is true where an injected change
    exceeds its tolerance. Events of one device do not come closer than ``GUARD_DAYS`` days and stay a day inside
    the interval. Raises ``ValueError`` when the type does not apply to the profile or no eligible device exists.
    """
    ft = FAULT_TYPES[ftype]
    if meta.profile not in ft.profiles():
        raise ValueError(f"{ftype} does not apply to the {meta.profile} profile")
    if not _is_sorted_by_device(df):
        raise ValueError("the table must be sorted by (device_id, timestamp)")
    rng = np.random.default_rng([seed, abs(hash_str(ftype)) % (2**31)])
    out = df.copy()
    slices = _device_slices(out)
    devices = _eligible_devices(out, meta, ft, slices)
    if not devices:
        raise ValueError(f"no device of {meta.name} has the columns {ft.requires[meta.profile]} for {ftype}")
    lat, lon = _lat_lon(meta)
    step = pd.Timedelta(meta.interval)
    interval_min = _interval_minutes(meta)
    guard = pd.Timedelta(days=GUARD_DAYS)
    taken: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {d: [] for d in devices}
    affected_rows = np.zeros(len(out), dtype=bool)
    rows = []
    jobs = [(lv, level_value) for lv, level_value in enumerate(ft.levels, start=1) for _ in range(repeats)]
    order = rng.permutation(len(jobs))
    order = sorted(order, key=lambda j: -ft.window(jobs[j][1], meta.profile))  # the longest windows first: they are the hardest to fit; ties stay in random order
    n_done = 0
    for j in order:
        level, value = jobs[j]
        win = ft.window(value, meta.profile)
        # a size-graded event must show in at least MIN_AFFECTED_HOURS hours of data, a duration-graded one in half of its window
        min_rows = int(np.ceil((0.5 * win / step) if ft.level_kind == "duration" else MIN_AFFECTED_HOURS * 60.0 / interval_min))
        placed = False
        lo_t = t_lo + pd.Timedelta(days=1)
        hi_t = t_hi - pd.Timedelta(days=1) - win
        if hi_t <= lo_t:
            raise ValueError("the test interval is too short for the fault window")
        for _ in range(max_tries):
            free = {d: _free_intervals(lo_t, hi_t, win, guard, taken[d]) for d in devices}
            free = {d: f for d, f in free.items() if f}
            if not free:
                break  # no room left on any device
            dev = list(free)[int(rng.integers(len(free)))]
            start = _draw_start(rng, free[dev], step)
            end = start + win
            a, b = slices[dev]
            ts = out["timestamp"].iloc[a:b]
            i0 = a + int(ts.searchsorted(start, side="left"))
            i1 = a + int(ts.searchsorted(end, side="left"))
            if i1 - i0 < max(2, int(0.5 * win / step)):
                continue
            w = out.iloc[i0:i1].copy()
            dev_df = out.iloc[a:b]
            ctx = Ctx(meta, dev, _rated_kw(meta, dev, out["p_ac"].iloc[a:b] if "p_ac" in out else pd.Series(dtype=float)),
                      interval_min, rng, dev_df, lat, lon)
            affected, loss = ft.fn(w, ctx, float(value))
            if int(affected.sum()) < min_rows:
                continue
            for c in w.columns:
                if c not in KEYS:
                    out.iloc[i0:i1, out.columns.get_loc(c)] = w[c].to_numpy()
            if loss is not None and "e_export_counter" in out.columns and np.any(loss):
                cum = np.cumsum(loss)
                col = out.columns.get_loc("e_export_counter")
                out.iloc[i0:i1, col] = out["e_export_counter"].iloc[i0:i1].to_numpy() - cum
                if i1 < b:
                    out.iloc[i1:b, col] = out["e_export_counter"].iloc[i1:b].to_numpy() - cum[-1]
            idx = np.flatnonzero(affected)
            affected_rows[i0:i1] |= affected
            t = w["timestamp"].to_numpy()
            rows.append({
                "type": ftype, "device_id": dev, "level": level, "magnitude": float(value), "unit": ft.unit,
                "start": pd.Timestamp(t[idx[0]]), "end": pd.Timestamp(t[idx[-1]]) + step,
                "window_start": start, "window_end": end, "n_affected": int(affected.sum()),
                "params": json.dumps(ctx.params, sort_keys=True),
            })
            taken[dev].append((start, end))
            placed = True
            n_done += 1
            break
        if not placed:
            log.warning("%s level %s: no admissible placement (no room left or the effect is too small); event skipped", ftype, level)
    truth = pd.DataFrame(rows, columns=[c for c in TRUTH_COLUMNS if c != "fault_id"])
    truth = truth.sort_values(["start", "device_id"]).reset_index(drop=True)
    truth.insert(0, "fault_id", [f"{ftype}-{seed}-{i:02d}" for i in range(len(truth))])
    for c in ("start", "end", "window_start", "window_end"):
        truth[c] = pd.to_datetime(truth[c], utc=True).dt.as_unit("ns")
    return out, truth, affected_rows


def hash_str(s: str) -> int:
    """A stable hash of a string (Python's ``hash`` changes between runs)."""
    h = 2166136261
    for ch in s.encode():
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h
