import numpy as np
import pandas as pd
import pytest
from conftest import window
from scada_schema import validate

from scada_fault_bench.faults import FAULT_TYPES, LEVEL_COUNT, MIN_AFFECTED_HOURS, TRUTH_COLUMNS, inject_type, types_for
from scada_fault_bench.solar import solar_position

WIND_TYPES = types_for("wind")
PV_TYPES = types_for("pv")


def run(df, meta, ftype, seed=0, repeats=1):
    lo, hi = window(df)
    return inject_type(df, meta, ftype, seed=seed, t_lo=lo, t_hi=hi, repeats=repeats)


def sel_window(out, r):
    return ((out["device_id"] == r.device_id) & (out["timestamp"] >= r.window_start) & (out["timestamp"] < r.window_end)).to_numpy()


def f64(s):
    return s.to_numpy(dtype="float64")


def test_registry_covers_eight_types_per_profile_and_twelve_in_all():
    assert len(WIND_TYPES) == 8 and len(PV_TYPES) == 8 and len(FAULT_TYPES) == 12
    for f in FAULT_TYPES.values():
        assert len(f.levels) == LEVEL_COUNT and list(f.levels) == sorted(f.levels)


@pytest.mark.parametrize("ftype", WIND_TYPES)
def test_wind_faults_are_deterministic_and_leave_the_input_alone(wind, ftype):
    df, meta = wind
    before = df.copy()
    a, ta, ma = run(df, meta, ftype, seed=3)
    b, tb, mb = run(df, meta, ftype, seed=3)
    pd.testing.assert_frame_equal(df, before)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ta, tb)
    assert (ma == mb).all()
    _, tc, _ = run(df, meta, ftype, seed=4)
    assert not ta[["device_id", "start"]].equals(tc[["device_id", "start"]])


@pytest.mark.parametrize("ftype", PV_TYPES)
def test_pv_faults_are_deterministic(pv, ftype):
    df, meta = pv
    a, ta, _ = run(df, meta, ftype, seed=2)
    b, tb, _ = run(df, meta, ftype, seed=2)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ta, tb)


@pytest.mark.parametrize("ftype", WIND_TYPES)
def test_truth_matches_what_changed_in_the_table(wind, ftype):
    """Affected rows lie inside their event, nothing outside the windows changed (except the cumulative counter)."""
    df, meta = wind
    out, truth, aff = run(df, meta, ftype, seed=5, repeats=2)
    assert list(truth.columns) == TRUTH_COLUMNS
    assert len(truth) == LEVEL_COUNT * 2
    assert truth["fault_id"].is_unique
    step = pd.Timedelta(meta.interval)
    ts = out["timestamp"]
    for r in truth.itertuples():
        m = (out["device_id"] == r.device_id).to_numpy()
        inside = m & (ts >= r.start).to_numpy() & (ts < r.end).to_numpy()
        assert aff[inside].sum() == r.n_affected
        if FAULT_TYPES[ftype].level_kind == "size":
            assert r.n_affected * step >= pd.Timedelta(hours=MIN_AFFECTED_HOURS)
        else:
            assert r.n_affected * step >= 0.5 * (r.window_end - r.window_start)
        assert r.window_start <= r.start and r.end <= r.window_end + step
    cols = [c for c in df.columns if c not in ("timestamp", "device_id", "e_export_counter")]
    changed = np.zeros(len(df), dtype=bool)
    for c in cols:
        a, b = df[c].to_numpy(), out[c].to_numpy()
        changed |= (a != b) & ~(pd.isna(a) & pd.isna(b))
    in_window = np.zeros(len(df), dtype=bool)
    for r in truth.itertuples():
        in_window |= sel_window(out, r)
    assert not (changed & ~in_window).any()
    assert not (aff & ~changed).any()


@pytest.mark.parametrize("ftype", WIND_TYPES)
def test_injected_wind_tables_still_validate(wind, ftype):
    df, meta = wind
    out, _, _ = run(df, meta, ftype, seed=1)
    assert not validate(out, meta).errors


@pytest.mark.parametrize("ftype", PV_TYPES)
def test_injected_pv_tables_still_validate(pv, ftype):
    df, meta = pv
    out, _, _ = run(df, meta, ftype, seed=1)
    assert not validate(out, meta).errors


def test_events_of_one_device_keep_their_distance(wind):
    df, meta = wind
    _, truth, _ = run(df, meta, "derating", seed=7, repeats=3)
    for _, g in truth.groupby("device_id"):
        g = g.sort_values("window_start")
        gaps = g["window_start"].iloc[1:].to_numpy() - g["window_end"].iloc[:-1].to_numpy()
        assert (gaps >= pd.Timedelta(days=2).to_timedelta64()).all()


def test_derating_caps_power_and_keeps_energy_consistent(wind):
    df, meta = wind
    out, truth, aff = run(df, meta, "derating", seed=2)
    for r in truth.itertuples():
        sel = sel_window(out, r)
        cap = (1 - r.magnitude / 100) * 2000.0
        assert out.loc[sel, "p_ac"].max() <= cap + 1e-3
        assert np.allclose(f64(out.loc[sel, "e_export"]), f64(out.loc[sel, "p_ac"]) / 6.0, atol=0.05)
    for _, g in out.groupby("device_id"):
        assert np.allclose(g["e_export_counter"].diff().iloc[1:].to_numpy(dtype="float64"), g["e_export"].iloc[1:].to_numpy(dtype="float64"), atol=0.5)


def test_power_loss_and_string_loss_scale_power_by_the_level(wind, pv):
    df, meta = wind
    out, truth, aff = run(df, meta, "power_loss", seed=2)
    r = truth.iloc[0]
    sel = sel_window(out, r) & aff
    assert np.allclose(f64(out.loc[sel, "p_ac"]) / f64(df.loc[sel, "p_ac"]), 1 - r.magnitude / 100, atol=1e-4)
    d2, m2 = pv
    out, truth, aff = run(d2, m2, "string_loss", seed=2)
    r = truth.iloc[0]
    sel = sel_window(out, r) & aff
    assert np.allclose(f64(out.loc[sel, "p_dc"]) / f64(d2.loc[sel, "p_dc"]), 1 - r.magnitude / 100, atol=1e-4)


def test_flatline_holds_one_value_and_the_duration_is_the_level(wind):
    df, meta = wind
    out, truth, aff = run(df, meta, "flatline", seed=2)
    for r in truth.itertuples():
        sel = sel_window(out, r)
        assert out.loc[sel, "wind_speed"].nunique() == 1
        assert out.loc[sel, "wind_speed_2"].nunique() > 1
        assert r.window_end - r.window_start == pd.Timedelta(hours=r.magnitude)


def test_unit_jump_is_degrees_fahrenheit(wind):
    df, meta = wind
    out, truth, _ = run(df, meta, "unit_jump", seed=2)
    r = truth.iloc[0]
    sel = sel_window(out, r)
    assert np.allclose(f64(out.loc[sel, "t_gearbox_oil"]), f64(df.loc[sel, "t_gearbox_oil"]) * 1.8 + 32, atol=1e-3)


def test_clock_shift_delays_every_channel_by_the_shift(wind):
    df, meta = wind
    out, truth, _ = run(df, meta, "clock_shift", seed=2)
    r = truth[truth["level"] == 3].iloc[0]
    k = int(r.magnitude // 10)
    d = df[df["device_id"] == r.device_id].set_index("timestamp")
    o = out[out["device_id"] == r.device_id].set_index("timestamp")
    t = o.loc[r.window_start + pd.Timedelta(hours=2):r.window_end - pd.Timedelta(hours=2)].index
    for c in ("p_ac", "wind_speed"):
        assert np.allclose(f64(o.loc[t, c]), f64(d[c].shift(k).loc[t]), atol=1e-3)


def test_nacelle_offset_wraps_and_thermal_drift_ramps(wind):
    df, meta = wind
    out, truth, _ = run(df, meta, "nacelle_dir_offset", seed=2)
    assert out["nacelle_dir"].between(0, 360).all()
    r = truth.iloc[0]
    sel = sel_window(out, r)
    d = (f64(out.loc[sel, "nacelle_dir"]) - f64(df.loc[sel, "nacelle_dir"]) + 180) % 360 - 180
    assert np.allclose(np.abs(d), r.magnitude, atol=1e-2)
    out, truth, _ = run(df, meta, "thermal_drift", seed=2)
    r = truth.iloc[0]
    sel = sel_window(out, r)
    dt = f64(out.loc[sel, "t_gearbox_oil"]) - f64(df.loc[sel, "t_gearbox_oil"])
    assert dt[0] == pytest.approx(0, abs=1e-3) and dt[-1] == pytest.approx(r.magnitude, abs=0.05) and (np.diff(dt) >= -1e-3).all()


def test_pv_outage_shading_and_pyranometer(pv):
    df, meta = pv
    out, truth, aff = run(df, meta, "inverter_outage", seed=2)
    for r in truth.itertuples():
        assert (out.loc[sel_window(out, r), ["p_ac", "p_dc", "i_dc"]].to_numpy(dtype="float64") == 0).all()
    out, truth, aff = run(df, meta, "shading", seed=2)
    ts = out.loc[aff, "timestamp"]
    elev, azim = solar_position(pd.DatetimeIndex(ts + pd.Timedelta(minutes=7.5)), meta.devices[0].lat, meta.devices[0].lon)
    assert (elev < 30).all() and (azim > 200).all()
    out, truth, aff = run(df, meta, "pyranometer_drift", seed=2)
    assert set(out.loc[aff, "device_id"]) == {"MET"}


def test_unsorted_table_and_wrong_profile_are_refused(wind):
    df, meta = wind
    lo, hi = window(df)
    with pytest.raises(ValueError, match="does not apply"):
        inject_type(df, meta, "string_loss", seed=0, t_lo=lo, t_hi=hi)
    with pytest.raises(ValueError, match="sorted"):
        inject_type(df.sample(frac=1.0, random_state=0), meta, "derating", seed=0, t_lo=lo, t_hi=hi)
    with pytest.raises(ValueError, match="too short"):
        inject_type(df, meta, "derating", seed=0, t_lo=lo, t_hi=lo + pd.Timedelta(days=5))


def test_solar_position_matches_pvlib_when_available():
    pvlib = pytest.importorskip("pvlib")
    t = pd.date_range("2017-03-20", "2017-09-22", freq="37min", tz="UTC")
    elev, azim = solar_position(t, 36.2, -115.2)
    ref = pvlib.solarposition.get_solarposition(t, 36.2, -115.2, method="nrel_numpy")
    up = ref["apparent_elevation"].to_numpy() > 5
    assert np.abs(elev - ref["apparent_elevation"].to_numpy())[up].max() < 0.7
    assert np.abs((azim - ref["azimuth"].to_numpy() + 180) % 360 - 180)[up].max() < 2.0
