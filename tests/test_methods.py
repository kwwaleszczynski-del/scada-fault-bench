import numpy as np
import pandas as pd
import pytest
from scada_schema import write_parquet

import scada_fault_bench.portfolio  # noqa: F401 - registers the portfolio methods
from scada_fault_bench import baselines  # noqa: F401
from scada_fault_bench.base import METHODS, Detector, register
from scada_fault_bench.baselines import _cusum
from scada_fault_bench.data import DatasetSpec, make_split
from scada_fault_bench.faults import inject_type
from scada_fault_bench.features import FeatureModel, circ_diff, trailing_median
from scada_fault_bench.scenario import grey_mask

BASELINES = ["rules", "scada_qc", "zscore", "ewma", "cusum", "iforest"]


def cut(df, meta, profile):
    start = "2019-01-01" if profile == "wind" else "2017-03-01"
    spec = DatasetSpec("syn", profile, start, (pd.Timestamp(start) + pd.Timedelta(days=120)).strftime("%Y-%m-%d"), "syn", 40)
    return make_split(df, meta, 1, name="syn", n_folds=3, spec=spec)


def test_registry_has_the_baselines_and_the_portfolio_methods():
    assert set(BASELINES) <= set(METHODS)
    for name in ("nbm_rc", "nbm_gbm", "power_curve_kpi", "pv_autoencoder", "pv_outage_expected", "pv_outage_iforest"):
        assert name in METHODS and METHODS[name].family == "portfolio" and METHODS[name].source
    assert {m.family for m in METHODS.values()} == {"baseline", "portfolio"}
    for m in METHODS.values():
        assert m.title and m.description and m.profiles


def test_duplicate_or_nameless_methods_are_refused():
    class Dup(Detector):
        name = "zscore"

    with pytest.raises(ValueError):
        register(Dup)

    class Nameless(Detector):
        pass

    with pytest.raises(ValueError):
        register(Nameless)


@pytest.mark.parametrize("name", BASELINES)
def test_wind_baselines_return_one_nonnegative_score_per_row_and_are_deterministic(wind, name):
    df, meta = wind
    sp = cut(df, meta, "wind")
    m1 = METHODS[name]().fit(sp.train, meta)
    s1 = m1.score(sp.test, meta)
    s2 = METHODS[name]().fit(sp.train, meta).score(sp.test, meta)
    assert s1.shape == (len(sp.test),) and np.isfinite(s1).sum() > 0.5 * len(s1)
    assert np.nanmin(s1) >= 0
    np.testing.assert_allclose(s1, s2, equal_nan=True)


@pytest.mark.parametrize("name", BASELINES)
def test_pv_baselines_return_one_score_per_row(pv, name):
    df, meta = pv
    sp = cut(df, meta, "pv")
    s = METHODS[name]().fit(sp.train, meta).score(sp.test, meta)
    assert s.shape == (len(sp.test),) and np.isfinite(s).any() and np.nanmin(s) >= 0


def test_a_score_does_not_depend_on_the_order_of_columns_or_on_extra_columns(wind):
    df, meta = wind
    sp = cut(df, meta, "wind")
    m = METHODS["zscore"]().fit(sp.train, meta)
    a = m.score(sp.test, meta)
    b = m.score(sp.test[sp.test.columns[::-1]].assign(x_extra=1.0), meta)
    np.testing.assert_allclose(a, b, equal_nan=True)


def test_baselines_see_a_strong_injected_fault_that_a_healthy_table_does_not_show(wind):
    df, meta = wind
    sp = cut(df, meta, "wind")
    out, truth, aff = inject_type(sp.test, meta, "power_loss", seed=0, t_lo=sp.t_lo, t_hi=sp.t_hi, repeats=1)
    big = truth[truth["level"] == 5].iloc[0]
    sel = ((out["device_id"] == big.device_id) & (out["timestamp"] >= big.start) & (out["timestamp"] < big.end)).to_numpy()
    for name in ("zscore", "ewma", "cusum"):
        m = METHODS[name]().fit(sp.train, meta)
        clean, hit = m.score(sp.test, meta), m.score(out, meta)
        assert np.nanmedian(hit[sel & aff]) > 1.5 * np.nanmedian(clean[sel & aff]), name


def test_rules_flag_a_frozen_anemometer_and_a_degf_temperature(wind):
    df, meta = wind
    sp = cut(df, meta, "wind")
    m = METHODS["rules"]().fit(sp.train, meta)
    for ftype in ("flatline", "unit_jump"):
        out, truth, aff = inject_type(sp.test, meta, ftype, seed=1, t_lo=sp.t_lo, t_hi=sp.t_hi, repeats=1)
        s = m.score(out, meta)
        long = truth[truth["level"] == 5].iloc[0]
        sel = ((out["device_id"] == long.device_id) & (out["timestamp"] >= long.start) & (out["timestamp"] < long.end)).to_numpy()
        # the oil rule (above 100 degC) only fires when the oil is warm enough that degF exceeds it; the frozen sensor is caught almost always
        assert np.nanmean(s[sel & aff] > 0.5) > (0.5 if ftype == "flatline" else 0.01), ftype
        assert np.nanmean(m.score(sp.test, meta)[sel] > 0.5) < 0.005


def test_scada_qc_marks_rows_and_flags_a_frozen_sensor(wind):
    df, meta = wind
    sp = cut(df, meta, "wind")
    out, truth, aff = inject_type(sp.test, meta, "flatline", seed=1, t_lo=sp.t_lo, t_hi=sp.t_hi, repeats=1)
    s = METHODS["scada_qc"]().fit(sp.train, meta).score(out, meta)
    assert set(np.unique(s)) <= {0.0, 1.0} and s[aff].mean() > 0.3


def test_features_are_standardised_on_the_training_part_and_pv_features_exist_by_day(wind, pv):
    df, meta = wind
    sp = cut(df, meta, "wind")
    fm = FeatureModel(meta).fit(sp.train)
    Z = fm.transform(sp.train)
    assert set(fm.names) >= {"pc_res", "ws_fleet", "ws_ws2", "yaw_fleet", "yaw_cons", "t_res", "ws_flat"}
    assert abs(np.nanmedian(Z["pc_res"])) < 0.5
    Zt = fm.transform(sp.train, tail=True)
    assert np.nanquantile(np.abs(Zt["pc_res"]), 0.995) == pytest.approx(1.0, abs=0.25)
    d2, m2 = pv
    sp2 = cut(d2, m2, "pv")
    f2 = FeatureModel(m2).fit(sp2.train)
    assert {"pr_res", "poa_flat", "t_mod_res", "t_inv_res"} <= set(f2.names)
    Z2 = f2.transform(sp2.test)
    assert Z2["pr_res"].notna().sum() > 0 and len(Z2) == len(sp2.test)


def test_circular_helpers_and_trailing_median():
    assert circ_diff(np.array([359.0, 1.0, 90.0]), np.array([1.0, 359.0, 270.0])).tolist() == [-2.0, 2.0, -180.0]
    x = np.array([1.0, 1.0, 100.0, 1.0, 1.0, 1.0])
    t = trailing_median(x, 3)
    assert t[2] == 1.0 and t[5] == 1.0 and np.isfinite(t).all()


def test_the_finite_memory_cusum_responds_to_a_shift_and_forgets_it():
    x = np.r_[np.zeros(50), np.full(30, 3.0), np.zeros(200)]
    s = _cusum(x, 1.0, 72)
    assert s[:50].max() == 0 and s[79] == pytest.approx(60.0) and s[-1] == 0 and s.max() == pytest.approx(60.0)
    down = _cusum(-x, 1.0, 72)
    np.testing.assert_allclose(s, down)  # two-sided


def test_grey_rows_cover_missing_power_and_night(wind, pv):
    df, meta = wind
    g = grey_mask(df, meta)
    assert g.mean() < 0.05
    d2, m2 = pv
    g2 = grey_mask(d2, m2)
    assert 0.3 < g2.mean() < 0.7  # roughly the night


def test_a_portfolio_method_that_cannot_be_imported_reports_why(monkeypatch):
    import importlib

    real = importlib.import_module

    def fake(name, *a, **k):
        if name == "wind_thermal_nbm":
            raise ModuleNotFoundError("No module named 'wind_thermal_nbm'")
        return real(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", fake)
    ok, why = METHODS["nbm_rc"].available()
    assert not ok and "wind_thermal_nbm" in why and "GitHub" in why


@pytest.mark.portfolio
@pytest.mark.parametrize("name", ["nbm_rc", "nbm_gbm", "power_curve_kpi"])
def test_wind_portfolio_methods_run_on_a_synthetic_farm(wind, name):
    ok, why = METHODS[name].available()
    if not ok:
        pytest.skip(why)
    df, meta = wind
    sp = cut(df, meta, "wind")
    s = METHODS[name]().fit(sp.train, meta).score(sp.test, meta)
    assert s.shape == (len(sp.test),) and np.isfinite(s).sum() > 0.3 * len(s) and np.nanmin(s) >= 0


@pytest.mark.portfolio
@pytest.mark.parametrize("name", ["pv_outage_expected", "pv_outage_iforest", "pv_autoencoder"])
def test_pv_portfolio_methods_run_on_a_synthetic_plant(pv, name):
    ok, why = METHODS[name].available()
    if not ok:
        pytest.skip(why)
    df, meta = pv
    sp = cut(df, meta, "pv")
    s = METHODS[name]().fit(sp.train, meta).score(sp.test, meta)
    assert s.shape == (len(sp.test),) and np.isfinite(s).any() and np.nanmin(s) >= 0


def test_write_parquet_roundtrip_of_a_synthetic_table(wind, tmp_path):
    df, meta = wind
    p = write_parquet(df, meta, tmp_path / "w.parquet")
    assert p.exists()
