import numpy as np
import pandas as pd
import pytest
from conftest import window

from scada_fault_bench.base import MERGE_GAP_HOURS, MIN_ALARM_HOURS, alarm_mask, calibrate
from scada_fault_bench.data import DatasetSpec, make_split
from scada_fault_bench.evaluate import COLUMNS, HIST_BINS, evaluate, hist_edges
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
    assert ev["delay_h"].max() <= 12  # an alarm may start only at the first affected row of the burst (merge gap fills the rest)
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


def _naive(sc, score, thr, step_min, tol_days=1.0):
    """The same rules written as plain loops over devices and rows, as a cross-check of the vectorised evaluation."""
    step = pd.Timedelta(minutes=step_min)
    n_steps = int(np.ceil((sc.t_hi - sc.t_lo) / step))
    out = {"ev_n": 0, "ev_det": 0, "delay": 0.0, "al_tp": 0, "al_fp": 0, "al_ign": 0, "neg": 0, "fa": 0}
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
        wins = []
        for r in sc.truth[sc.truth["device_id"] == dev].itertuples():
            t0 = int((r.start - sc.t_lo) // step)
            t1 = int(np.ceil((r.end - sc.t_lo) / step))
            lo, hi = max(t0, 0), min(t1 + int(tol_days * 24 * 60 / step_min), n_steps)
            wins.append((lo, hi))
            out["ev_n"] += 1
            hit = np.flatnonzero(alarm[lo:hi])
            if len(hit):
                out["ev_det"] += 1
                out["delay"] += 0.0 if lo + hit[0] <= t0 else (lo + hit[0] - t0) * step_min / 60
        k = 0
        while k < n_steps:
            if not alarm[k]:
                k += 1
                continue
            m = k
            while m < n_steps and alarm[m]:
                m += 1
            if any(k < hi and m > lo for lo, hi in wins):
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
    sc, _ = wind_scenario
    rng = np.random.default_rng(5)
    for thr in (0.7, 1.3):
        score = rng.gamma(2.0, 0.5, len(sc.table)) + 2.0 * (sc.labels == POS) * (rng.random(len(sc.table)) < 0.4)
        sums, ev = evaluate(sc, score, thr, 10.0)
        want = _naive(sc, score, thr, 10.0)
        got = sums[["ev_n", "ev_det", "ev_delay_h", "al_tp", "al_fp", "al_ign", "neg_rows", "fa_rows"]].sum()
        assert got["ev_n"] == want["ev_n"] and got["ev_det"] == want["ev_det"]
        assert got["ev_delay_h"] == pytest.approx(want["delay"], abs=1e-6)
        assert (got["al_tp"], got["al_fp"], got["al_ign"]) == (want["al_tp"], want["al_fp"], want["al_ign"])
        assert got["neg_rows"] == want["neg"] and got["fa_rows"] == want["fa"]


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
