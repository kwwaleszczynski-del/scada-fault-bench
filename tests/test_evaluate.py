import numpy as np
import pandas as pd
import pytest
from conftest import window

from scada_fault_bench.base import MERGE_GAP_HOURS, MIN_ALARM_HOURS, alarm_mask, calibrate
from scada_fault_bench.data import DatasetSpec, make_split
from scada_fault_bench.evaluate import COLUMNS, HIST_BINS, alarms_of, evaluate, hist_edges
from scada_fault_bench.scenario import GREY, NEG, POS, group_ids, make_scenario
from scada_fault_bench.stats import METRICS, bootstrap_metrics, metrics_from_sums

IDX = {c: i for i, c in enumerate(COLUMNS)}


def spec(profile, days):
    start = "2019-01-01" if profile == "wind" else "2017-03-01"  # the starts of synthetic_wind and synthetic_pv
    return DatasetSpec("syn", profile, start, (pd.Timestamp(start) + pd.Timedelta(days=days)).strftime("%Y-%m-%d"), "syn", days // 3)


@pytest.fixture(scope="module")
def wind_scenario(wind):
    df, meta = wind
    split = make_split(df, meta, 1, name="syn", n_folds=3, spec=spec("wind", 120))
    return make_scenario(split, "power_loss", 0, repeats=1), split


def test_alarm_mask_merges_close_runs_and_drops_short_ones():
    step = 10.0
    f = np.zeros(400, dtype=bool)
    f[10:20] = True
    f[25:30] = True  # 5 rows = 50 min from the first run: merged
    f[200] = True  # a single row: dropped
    f[300:303] = True  # 3 rows = 30 min: kept (the minimum is half an hour)
    f[380:382] = True  # 2 rows, more than six hours from the others: dropped
    m = alarm_mask(f, step)
    assert m[10:30].all() and not m[200] and m[300:303].all() and not m[380:382].any()
    assert MIN_ALARM_HOURS == 0.5 and MERGE_GAP_HOURS == 6.0
    assert not alarm_mask(np.zeros(5, dtype=bool), step).any() and len(alarm_mask(np.array([], dtype=bool), step)) == 0


def test_calibrate_is_a_quantile_or_fixed():
    s = np.r_[np.arange(1000, dtype=float), np.nan]
    assert calibrate(s) == pytest.approx(np.quantile(np.arange(1000), 0.995))
    assert calibrate(s, fixed=True) == 0.5
    assert calibrate(np.array([np.nan])) == float("inf")


def test_labels_are_pos_inside_events_grey_around_them_and_neg_elsewhere(wind_scenario):
    sc, _ = wind_scenario
    lab = sc.labels
    assert set(np.unique(lab)) <= {NEG, POS, GREY}
    assert (lab == POS).sum() == sc.truth["n_affected"].sum()
    for r in sc.truth.itertuples():
        near = (sc.table["device_id"] == r.device_id) & (sc.table["timestamp"] >= r.start) & (sc.table["timestamp"] < r.end)
        assert (lab[near.to_numpy()] != NEG).all()
    assert (lab == NEG).sum() > 0.3 * len(lab)


def test_a_perfect_score_detects_every_event_with_zero_delay_and_full_auc(wind_scenario):
    sc, _ = wind_scenario
    score = np.where(sc.labels == POS, 10.0, 0.0)
    sums, ev = evaluate(sc, score, 1.0, 10.0)
    t = sums[COLUMNS].sum().to_numpy()
    m = {k: float(v[0]) for k, v in metrics_from_sums(t).items()}
    assert len(ev) == len(sc.truth) and ev["detected"].all()
    assert m["recall"] == 1.0 and m["precision"] == 1.0 and m["fa_h_per_year"] == 0.0 and m["auroc"] == 1.0
    assert ev["delay_h"].max() == 0 and (ev["chance"] == 0).all()  # no placebo given; the alarm covers the first affected row
    assert m["row_recall"] > 0.5 and m["row_precision"] == 1.0


def test_a_silent_method_misses_everything_and_never_raises_a_false_alarm(wind_scenario):
    sc, _ = wind_scenario
    sums, ev = evaluate(sc, np.zeros(len(sc.table)), 1.0, 10.0)
    t = sums[COLUMNS].sum().to_numpy()
    m = {k: float(v[0]) for k, v in metrics_from_sums(t).items()}
    assert m["recall"] == 0.0 and np.isnan(m["precision"]) and m["fa_h_per_year"] == 0.0 and m["auroc"] == 0.5


def test_constant_alarm_on_one_device_counts_false_alarm_hours_exactly(wind, wind_scenario):
    sc, _ = wind_scenario
    dev = sc.table["device_id"].iloc[0]
    score = np.where(sc.table["device_id"] == dev, 10.0, 0.0)
    sums, ev = evaluate(sc, score, 1.0, 10.0)
    fa_rows = sums["fa_rows"].sum()
    neg_dev = ((sc.labels == NEG) & (sc.table["device_id"] == dev).to_numpy()).sum()
    assert fa_rows == neg_dev
    assert sums["neg_rows"].sum() == (sc.labels == NEG).sum()
    assert sums["al_tp"].sum() + sums["al_fp"].sum() + sums["al_ign"].sum() >= 1


def test_the_histograms_hold_every_affected_and_healthy_row_once(wind_scenario):
    sc, _ = wind_scenario
    rng = np.random.default_rng(0)
    sums, _ = evaluate(sc, rng.random(len(sc.table)) * 3, 1.0, 10.0)
    hp = sums[[f"hp_{i}" for i in range(HIST_BINS)]].to_numpy().sum()
    hn = sums[[f"hn_{i}" for i in range(HIST_BINS)]].to_numpy().sum()
    assert hp == (sc.labels == POS).sum() and hn == (sc.labels == NEG).sum()
    assert len(hist_edges()) == HIST_BINS - 1


def test_pv_rows_of_one_plant_are_merged_into_one_group(pv):
    df, meta = pv
    split = make_split(df, meta, 1, name="syn", n_folds=3, spec=spec("pv", 120))
    sc = make_scenario(split, "string_loss", 0, repeats=1)
    assert set(group_ids(sc.table, meta)) == {"plant"}
    score = np.where(sc.labels == POS, 5.0, 0.0)
    sums, ev = evaluate(sc, score, 1.0, 15.0)
    assert ev["detected"].all()


def test_metrics_and_bootstrap_are_deterministic_and_bracket_the_point_value():
    rng = np.random.default_rng(1)
    V = np.zeros((40, len(COLUMNS)))
    V[:, IDX["ev_n"]] = 2
    V[:, IDX["ev_det"]] = rng.binomial(2, 0.7, 40)
    V[:, IDX["al_tp"]] = V[:, IDX["ev_det"]]
    V[:, IDX["al_fp"]] = rng.poisson(0.5, 40)
    V[:, IDX["neg_rows"]] = 1000
    V[:, IDX["fa_rows"]] = rng.poisson(5, 40)
    p1, lo1, hi1 = bootstrap_metrics(V, 500, seed=3)
    p2, lo2, hi2 = bootstrap_metrics(V, 500, seed=3)
    assert repr((p1, lo1, hi1)) == repr((p2, lo2, hi2))  # repr: NaN != NaN
    for k in ("recall", "precision", "fa_h_per_year"):
        assert lo1[k] <= p1[k] <= hi1[k]
    assert set(METRICS) <= set(p1)
    assert p1["recall"] == pytest.approx(V[:, IDX["ev_det"]].sum() / V[:, IDX["ev_n"]].sum())
    assert 0 < hi1["recall"] - lo1["recall"] < 0.4
    one = bootstrap_metrics(V[:1], 10)
    assert np.isnan(one[1]["recall"])


def test_f1_is_the_harmonic_mean_of_precision_and_recall():
    t = np.zeros(len(COLUMNS))
    t[IDX["ev_n"]], t[IDX["ev_det"]], t[IDX["al_tp"]], t[IDX["al_fp"]] = 10, 6, 6, 18
    m = {k: float(v[0]) for k, v in metrics_from_sums(t).items()}
    assert m["precision"] == pytest.approx(0.25) and m["recall"] == pytest.approx(0.6)
    assert m["f1"] == pytest.approx(2 * 0.25 * 0.6 / 0.85)


def test_auc_from_histograms_matches_the_rank_statistic():
    rng = np.random.default_rng(2)
    pos, neg = rng.normal(1.0, 1, 300), rng.normal(0.0, 1, 600)
    edges = np.linspace(-6, 7, HIST_BINS - 1)
    t = np.zeros(len(COLUMNS))
    t[IDX["hp_0"]:IDX["hp_0"] + HIST_BINS] = np.bincount(np.digitize(pos, edges), minlength=HIST_BINS)
    t[IDX["hn_0"]:IDX["hn_0"] + HIST_BINS] = np.bincount(np.digitize(neg, edges), minlength=HIST_BINS)
    auc = float(metrics_from_sums(t)["auroc"][0])
    exact = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    assert auc == pytest.approx(exact, abs=0.01)
    ap = float(metrics_from_sums(t)["ap"][0])
    assert 0.3 < ap < 0.9


def test_window_helper_matches_table(wind):
    lo, hi = window(wind[0])
    assert lo < hi


def _naive(sc, score, thr, step_min, placebo=None):
    """The same rules written as plain loops over devices and rows, as a cross-check of the vectorised evaluation."""
    step = pd.Timedelta(minutes=step_min)
    n_steps = int(np.ceil((sc.t_hi - sc.t_lo) / step))
    out = {"ev_n": 0, "ev_det": 0, "delay": 0.0, "chance": 0.0, "al_tp": 0, "al_fp": 0, "al_ign": 0, "neg": 0, "fa": 0}
    tab = sc.table.assign(score=score, label=sc.labels)
    for dev, g in tab.groupby("device_id"):
        s = np.full(n_steps, np.nan)
        lab = np.full(n_steps, GREY)
        idx = ((g["timestamp"] - sc.t_lo) // step).to_numpy()
        s[idx], lab[idx] = g["score"].to_numpy(), g["label"].to_numpy()
        flag = np.array([x > thr if np.isfinite(x) else False for x in s])
        gap, minlen = int(round(6 * 60 / step_min)), int(round(0.5 * 60 / step_min))
        alarm = np.zeros(n_steps, dtype=bool)
        i = 0
        while i < n_steps:
            if not flag[i]:
                i += 1
                continue
            j = i
            last = i
            while j + 1 < n_steps and (j + 1 - last) <= gap:
                j += 1
                if flag[j]:
                    last = j
            if flag[i:last + 1].sum() >= minlen:
                alarm[i:last + 1] = True
            i = last + 1
        gi = sorted(tab["device_id"].unique()).index(dev)
        for r in sc.truth[sc.truth["device_id"] == dev].itertuples():
            t0 = int((r.start - sc.t_lo) // step)
            t1 = int(np.ceil((r.end - sc.t_lo) / step))
            rows = [k for k in range(t0, t1) if lab[k] == POS]
            hits = [k for k in rows if alarm[k]]
            out["ev_n"] += 1
            if hits:
                out["ev_det"] += 1
                out["delay"] += (hits[0] - t0) * step_min / 60
            if placebo is not None:
                out["chance"] += np.mean([any(p[gi, k] for k in rows) for p in placebo])
        k = 0
        while k < n_steps:
            if not alarm[k]:
                k += 1
                continue
            m = k
            while m < n_steps and alarm[m]:
                m += 1
            if (lab[k:m] == POS).any():
                out["al_tp"] += 1
            elif (lab[k:m] == GREY).all():
                out["al_ign"] += 1
            else:
                out["al_fp"] += 1
            k = m
        out["neg"] += int((lab == NEG).sum())
        out["fa"] += int(((lab == NEG) & alarm).sum())
    return out


def test_the_vectorised_evaluation_equals_a_plain_loop_implementation(wind_scenario):
    sc, split = wind_scenario
    null = make_scenario(split, None, 0)
    rng = np.random.default_rng(5)
    for thr in (0.7, 1.3):
        score = rng.gamma(2.0, 0.5, len(sc.table)) + 2.0 * (sc.labels == POS) * (rng.random(len(sc.table)) < 0.4)
        s_null = rng.gamma(2.0, 0.5, len(null.table))
        pl = [alarms_of(null, s_null, thr, 10.0)]
        sums, ev = evaluate(sc, score, thr, 10.0, placebo=pl)
        want = _naive(sc, score, thr, 10.0, pl)
        got = sums[["ev_n", "ev_det", "ev_delay_h", "pl_det", "al_tp", "al_fp", "al_ign", "neg_rows", "fa_rows"]].sum()
        assert got["ev_n"] == want["ev_n"] and got["ev_det"] == want["ev_det"]
        assert got["ev_delay_h"] == pytest.approx(want["delay"], abs=1e-6) and got["pl_det"] == pytest.approx(want["chance"])
        assert (got["al_tp"], got["al_fp"], got["al_ign"]) == (want["al_tp"], want["al_fp"], want["al_ign"])
        assert got["neg_rows"] == want["neg"] and got["fa_rows"] == want["fa"]


def test_a_method_without_information_has_no_excess_recall_and_an_auc_of_one_half(wind_scenario):
    """Random scores calibrated like a real method: recall equals the placebo recall up to sampling, AUROC is one half."""
    sc, split = wind_scenario
    null = make_scenario(split, None, 0)
    rec, pla, auc = [], [], []
    for k in range(60):
        rng = np.random.default_rng(k)
        cal = rng.random(20000)
        thr = float(np.quantile(cal, 0.99))
        s_null = rng.random(len(null.table))
        s = rng.random(len(sc.table))
        sums, _ = evaluate(sc, s, thr, 10.0, cal_scores=cal, placebo=[alarms_of(null, s_null, thr, 10.0)])
        m = {k2: float(v[0]) for k2, v in metrics_from_sums(sums[COLUMNS].sum().to_numpy()).items()}
        rec.append(m["recall"])
        pla.append(m["recall_placebo"])
        auc.append(m["auroc"])
    assert abs(np.mean(rec) - np.mean(pla)) < 0.15
    assert abs(np.mean(auc) - 0.5) < 0.03


def test_the_placebo_of_a_perfect_method_is_what_the_untouched_alarms_hit(wind_scenario):
    sc, split = wind_scenario
    null = make_scenario(split, None, 0)
    always = np.ones(len(null.table))
    pl = [alarms_of(null, always, 0.5, 10.0)]  # alarms everywhere on the untouched block
    sums, ev = evaluate(sc, np.zeros(len(sc.table)), 0.5, 10.0, placebo=pl)
    assert sums["pl_det"].sum() == len(sc.truth) and sums["ev_det"].sum() == 0 and (ev["chance"] == 1).all()
    sums, ev = evaluate(sc, np.zeros(len(sc.table)), 0.5, 10.0, placebo=[np.zeros_like(pl[0])])
    assert sums["pl_det"].sum() == 0


def test_the_calibrated_threshold_keeps_alarms_inside_the_budget_and_rules_keep_theirs():
    from scada_fault_bench.base import ALARM_BUDGET, _segments, alarm_share, calibrate_budget

    rng = np.random.default_rng(1)
    n = 6000
    t = pd.date_range("2019-01-01", periods=n, freq="10min", tz="UTC")
    dev = np.array(["A"] * n)
    s = np.abs(pd.Series(rng.normal(size=n)).rolling(12, min_periods=1).mean().to_numpy()) * 3
    grey = np.zeros(n, dtype=bool)
    thr = calibrate_budget(s, dev, t, grey, 10.0)
    segs = _segments(dev, pd.DatetimeIndex(t).as_unit("ns").asi8, 600 * 10**9)
    assert alarm_share(s, thr, segs, ~grey, 10.0) <= ALARM_BUDGET + 1e-12
    assert alarm_share(s, thr * 0.9, segs, ~grey, 10.0) > alarm_share(s, thr, segs, ~grey, 10.0)
    assert calibrate_budget(s, dev, t, grey, 10.0, fixed=True) == 0.5
    assert calibrate_budget(np.full(n, np.nan), dev, t, grey, 10.0) == float("inf")
    # a gap in time splits the series: no alarm is merged across it
    t2 = t.delete(slice(3000, 3500))
    assert len(_segments(np.array(["A"] * len(t2)), pd.DatetimeIndex(t2).as_unit("ns").asi8, 600 * 10**9)) == 2


def test_the_rank_transform_is_monotone_and_saturates():
    from scada_fault_bench.evaluate import T_MAX, rank_transform

    cal = np.sort(np.random.default_rng(0).random(1000))
    t = rank_transform(np.array([-1.0, 0.1, 0.5, 0.9, 0.999, 5.0, np.nan]), cal)
    assert (np.diff(t[:6]) >= 0).all() and t[5] == T_MAX and t[6] == 0 and t[0] == 0


def test_a_short_zero_power_hour_in_sun_is_not_grey_but_a_whole_down_day_is(pv):
    from scada_fault_bench.scenario import grey_mask

    df, meta = pv
    d = df.copy()
    inv = d["device_id"] == "INV1"
    hour = inv & (d["timestamp"] >= "2017-04-10 18:00") & (d["timestamp"] < "2017-04-10 19:00")  # 11:00 local sun in the synthetic plant
    day = inv & (d["timestamp"] >= "2017-04-20") & (d["timestamp"] < "2017-04-21")
    d.loc[hour | day, "p_ac"] = 0.0
    g = grey_mask(d, meta)
    sun = d["timestamp"].map(d[d["device_id"] == "MET"].set_index("timestamp")["poa"]) > 300
    assert not (g & hour & sun).any()
    assert (g & day).sum() >= (day & sun).sum() > 0


def test_positive_rows_are_never_grey_and_the_truth_starts_at_the_first_positive_row(pv):
    """A temperature drift is injected around the clock, but a PV plant can only be asked to show it in daylight."""
    from scada_fault_bench.scenario import grey_mask

    df, meta = pv
    split = make_split(df, meta, 1, name="syn", n_folds=3, spec=spec("pv", 120))
    sc = make_scenario(split, "thermal_drift", 0, repeats=1)
    grey = grey_mask(split.test, meta)
    assert not (grey & (sc.labels == POS)).any()
    for r in sc.truth.itertuples():
        rows = (sc.table["device_id"] == r.device_id) & (sc.labels == POS) & (sc.table["timestamp"] >= r.window_start) & (sc.table["timestamp"] < r.window_end)
        t = sc.table.loc[rows, "timestamp"]
        assert t.min() == r.start and t.max() + pd.Timedelta(meta.interval) == r.end and len(t) == r.n_affected
    assert (sc.labels == POS).sum() == sc.truth["n_affected"].sum()


def test_care_events_keep_their_downtime_rows_as_affected_and_grey_only_outside_events():
    from scada_fault_bench.real import _scenario
    from scada_fault_bench.synthetic import synthetic_wind

    df, meta = synthetic_wind(n_turbines=1, days=60, seed=2)
    df = df.copy()
    df["x_train_test"] = np.where(df["timestamp"] < df["timestamp"].min() + pd.Timedelta(days=30), "train", "prediction")
    df["iec_category"] = "full_performance"
    start, end = pd.Timestamp("2019-02-10", tz="UTC"), pd.Timestamp("2019-02-17", tz="UTC")
    inside = (df["timestamp"] >= start) & (df["timestamp"] < end)
    df.loc[inside, "iec_category"] = "forced_outage"  # the status of a CARE event is typically downtime
    df.loc[(df["timestamp"] >= "2019-02-25") & (df["timestamp"] < "2019-02-26"), "iec_category"] = "forced_outage"  # a real stop elsewhere
    ev = pd.DataFrame({"device_id": ["T01"], "start": [start], "end": [end], "status": ["anomaly"]})
    sc = _scenario(df, meta, ev, 7)
    ts = sc.table["timestamp"]
    assert (sc.labels[((ts >= start) & (ts < end)).to_numpy()] == POS).all()
    assert (sc.labels[((ts >= "2019-02-25") & (ts < "2019-02-26")).to_numpy()] == GREY).all()
    assert (sc.labels[((ts >= end) & (ts < end + pd.Timedelta(days=1))).to_numpy()] == GREY).all()
    assert sc.truth["n_affected"].iloc[0] == 7 * 144 and sc.ftype == "real_fault"
