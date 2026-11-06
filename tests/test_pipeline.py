import json

import numpy as np
import pandas as pd
import pytest
from scada_schema import write_parquet

from scada_fault_bench import cli, data, report
from scada_fault_bench.base import METHODS
from scada_fault_bench.data import DatasetSpec
from scada_fault_bench.evaluate import COLUMNS
from scada_fault_bench.runner import RunConfig, applicable_methods, run_dataset
from scada_fault_bench.summary import summarise, write_summary
from scada_fault_bench.synthetic import synthetic_pv, synthetic_wind


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A data folder with one synthetic wind farm and one synthetic PV plant, registered as datasets."""
    root = tmp_path_factory.mktemp("bench")
    wind, wm = synthetic_wind(n_turbines=5, days=120, seed=3)
    pv, pm = synthetic_pv(days=120, seed=3, inverters=2)
    write_parquet(wind, wm, root / "syn_wind.parquet")
    write_parquet(pv, pm, root / "syn_pv.parquet")
    specs = {"syn_wind": DatasetSpec("syn_wind", "wind", "2019-01-01", "2019-05-01", "Synthetic wind", 30),
             "syn_pv": DatasetSpec("syn_pv", "pv", "2017-03-01", "2017-06-29", "Synthetic PV", 30)}
    old = dict(data.DATASETS)
    data.DATASETS.update(specs)
    yield root
    data.DATASETS.clear()
    data.DATASETS.update(old)


def run_tiny(root, out, **kw):
    cfg = RunConfig(out=out, data=root, folds=(0, 1), methods=("zscore", "rules", "cusum"), types=("flatline", "string_loss", "inverter_outage"), **kw)
    for name in ("syn_wind", "syn_pv"):
        run_dataset(name, cfg)
    return out


def test_a_run_writes_shards_that_summarise_into_every_selection(tiny, tmp_path):
    out = run_tiny(tiny, tmp_path / "r1")
    shards = sorted(p.name for p in (out / "shards").iterdir())
    assert "syn_wind.blocks.parquet" in shards and "syn_pv.truth.parquet" in shards and "syn_wind.runs.parquet" in shards
    blocks = pd.read_parquet(out / "shards" / "syn_wind.blocks.parquet")
    assert set(blocks["method"]) == {"zscore", "rules", "cusum"}
    assert set(blocks["ftype"]) == {"null", "flatline"}  # string_loss and inverter_outage do not apply to wind
    assert set(blocks["fold"]) == {0, 1} and set(COLUMNS) <= set(blocks.columns)
    S = summarise(out, n_boot=50)
    assert {e["profile"] for e in S["overall"]} == {"wind", "pv"}
    assert {e["type"] for e in S["types"] if e["profile"] == "pv"} == {"string_loss", "flatline", "inverter_outage"}
    assert len(S["cells"]) == 3 * 1 + 3 * 3 and S["config"]["folds"] == [0, 1]
    for e in S["cells"] + S["types"] + S["overall"]:
        for m in e.values():
            if isinstance(m, dict) and m.get("value") is not None and m["lo"] is not None:
                assert m["lo"] - 0.15 <= m["value"] <= m["hi"] + 0.15  # the point value lies in its interval up to resampling noise
    rules = next(e for e in S["overall"] if e["profile"] == "wind" and e["method"] == "rules")
    assert rules["fa_h_per_year"]["value"] is not None


def test_two_runs_give_identical_sums(tiny, tmp_path):
    a = run_tiny(tiny, tmp_path / "a")
    b = run_tiny(tiny, tmp_path / "b")
    for name in ("syn_wind", "syn_pv"):
        x = pd.read_parquet(a / "shards" / f"{name}.blocks.parquet")
        y = pd.read_parquet(b / "shards" / f"{name}.blocks.parquet")
        pd.testing.assert_frame_equal(x, y)
        pd.testing.assert_frame_equal(pd.read_parquet(a / "shards" / f"{name}.truth.parquet"), pd.read_parquet(b / "shards" / f"{name}.truth.parquet"))


def test_the_event_counts_follow_levels_and_repeats(tiny, tmp_path):
    out = run_tiny(tiny, tmp_path / "c")
    tw = pd.read_parquet(out / "shards" / "syn_wind.truth.parquet")
    tp = pd.read_parquet(out / "shards" / "syn_pv.truth.parquet")
    assert len(tw[(tw["type"] == "flatline") & (tw["fold"] == 0)]) <= 10 and set(tw["level"]) <= {1, 2, 3, 4, 5}
    assert len(tp[(tp["type"] == "string_loss") & (tp["fold"] == 0)]) <= 5  # one event per level on a PV plant
    ev = pd.read_parquet(out / "shards" / "syn_wind.events.parquet")
    assert {"detected", "delay_h", "level"} <= set(ev.columns) and len(ev) > 0


def test_summary_json_report_tables_and_numbers(tiny, tmp_path):
    out = run_tiny(tiny, tmp_path / "d")
    p = write_summary(out, n_boot=50)
    S = report.load_summary(p)
    nums = report.numbers(S)
    assert nums["n_methods"] == "3" and nums["n_datasets"] == "2" and int(nums["n_events"].replace(",", "")) == sum(e["n"] for e in S["events"])
    t = report.table_overall(S, "wind")
    assert t.count("\n") == 1 + 3 and "`zscore`" in t and "|" in t
    assert report.table_types(S, "pv").splitlines()[0].count("|") == 2 + 3
    assert "flatline" in report.table_faults(S) and "`rules`" in report.table_methods(S) and "Synthetic wind" in report.table_datasets(S)
    assert report.cell({"x": {"value": None, "lo": None, "hi": None}}, "x") == "n/a"
    assert report.cell({"x": {"value": 0.5, "lo": 0.25, "hi": 0.75}}, "x") == "0.50 (0.25-0.75)"
    json.loads(p.read_text(encoding="utf-8"))


def test_a_failing_method_is_recorded_and_does_not_stop_the_run(tiny, tmp_path, monkeypatch):
    from scada_fault_bench.baselines import ZScore

    def boom(self, train, meta):
        raise RuntimeError("broken on purpose")

    monkeypatch.setattr(ZScore, "fit", boom)
    out = run_tiny(tiny, tmp_path / "e")
    runs = pd.read_parquet(out / "shards" / "syn_wind.runs.parquet")
    assert runs.loc[runs["method"] == "zscore", "error"].str.contains("broken on purpose").all()
    assert (runs.loc[runs["method"] == "rules", "error"] == "").all()
    S = summarise(out, n_boot=20)
    assert any("broken on purpose" in r["errors"] for r in S["runs"])


def test_methods_that_do_not_apply_to_a_profile_are_not_run():
    ok, why = applicable_methods("wind", ("pv_autoencoder", "zscore"))
    assert ok == ["zscore"] and why == {}
    assert "pv_autoencoder" not in applicable_methods("wind")[0] and "nbm_rc" not in applicable_methods("pv")[0]


def test_cli_lists_injects_and_runs_a_plugin(tiny, tmp_path, capsys):
    assert cli.main(["list"]) == 0
    text = capsys.readouterr().out
    assert "kelmarsh_2019" in text and "derating" in text and "zscore" in text and "care_A" in text
    out = tmp_path / "inj" / "x"
    assert cli.main(["inject", "--dataset", "syn_wind", "--type", "flatline", "--fold", "1", "--data", str(tiny), "--out", str(out)]) == 0
    table = pd.read_parquet(str(out) + ".table.parquet")
    truth = pd.read_csv(str(out) + ".truth.csv")
    assert set(table["label"].unique()) <= {0, 1, 2} and (table["label"] == 1).sum() == truth["n_affected"].sum()
    res = tmp_path / "plug"
    assert cli.main(["run", "--datasets", "syn_wind", "--folds", "0", "--types", "flatline", "--methods", "my_method", "--plugin", "plugin_example:Mine", "--data", str(tiny), "--out", str(res)]) == 0
    assert "my_method" in METHODS
    blocks = pd.read_parquet(res / "shards" / "syn_wind.blocks.parquet")
    assert set(blocks["method"]) == {"my_method"}
    assert cli.main(["summarise", "--out", str(res), "--n-boot", "20"]) == 0
    assert (res / "summary.json").exists()
    del METHODS["my_method"]


def test_cli_rejects_a_plugin_that_is_not_a_detector(tmp_path):
    with pytest.raises(SystemExit):
        cli._plugin("numpy:ndarray")


def test_a_missing_dataset_says_how_to_prepare_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare"):
        data.load_dataset("kelmarsh_2019", tmp_path)


def test_test_blocks_cover_the_period_without_overlap_with_training(tiny):
    df, meta = data.load_dataset("syn_wind", tiny)
    for fold in range(4):
        sp = data.make_split(df, meta, fold, name="syn_wind")
        assert sp.t_hi - sp.t_lo == pd.Timedelta(days=30)
        buf = pd.Timedelta(days=data.BUFFER_DAYS)
        assert not ((sp.train["timestamp"] > sp.t_lo - buf) & (sp.train["timestamp"] < sp.t_hi + buf)).any()
        assert sp.test["timestamp"].min() >= sp.t_lo and sp.test["timestamp"].max() < sp.t_hi
        assert sp.cal.mean() == pytest.approx(0.25, abs=0.1)
        assert len(sp.train) + len(sp.test) <= len(df)
    with pytest.raises(ValueError):
        data.make_split(df, meta, 7, name="syn_wind")


def test_the_nbm_and_autoencoder_adapters_are_listed_as_unavailable_when_missing():
    for name in ("nbm_rc", "pv_autoencoder"):
        ok, why = METHODS[name].available()
        assert isinstance(ok, bool) and (ok or why)
    assert np.isfinite(1.0)
