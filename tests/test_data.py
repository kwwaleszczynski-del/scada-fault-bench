"""Checks on the real prepared tables (marked ``data``: skipped when ``data/`` or ``$SCADA_FAULT_BENCH_DATA`` has no tables)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scada_schema import validate

from scada_fault_bench.base import METHODS
from scada_fault_bench.data import DATASETS, data_dir, dataset_path, load_dataset, make_split
from scada_fault_bench.faults import LEVEL_COUNT, types_for
from scada_fault_bench.real import REAL_NAME, care_events
from scada_fault_bench.scenario import NEG, POS, make_scenario

pytestmark = pytest.mark.data

AVAILABLE = [d for d in DATASETS if dataset_path(d).exists()]


@pytest.fixture(scope="module", params=AVAILABLE or [pytest.param(None, marks=pytest.mark.skip(reason="no prepared datasets (scada-fault-bench prepare)"))])
def real_split(request):
    name = request.param
    df, meta = load_dataset(name)
    return name, make_split(df, meta, 1, name=name)


def test_the_table_is_canonical_and_the_split_is_clean(real_split):
    name, sp = real_split
    assert not validate(sp.test, sp.meta).errors
    assert sp.test["timestamp"].min() >= sp.t_lo and sp.test["timestamp"].max() < sp.t_hi
    assert len(sp.train) > 2 * len(sp.test) * 0.9
    assert set(sp.test["device_id"]) == set(sp.train["device_id"])


def test_every_fault_type_is_injected_at_every_level_and_the_table_stays_valid(real_split):
    name, sp = real_split
    repeats = 2 if sp.meta.profile == "wind" else 1
    for t in types_for(sp.meta.profile):
        sc = make_scenario(sp, t, 0, repeats=repeats)
        assert set(sc.truth["level"]) == set(range(1, LEVEL_COUNT + 1)), (name, t)
        assert len(sc.truth) == LEVEL_COUNT * repeats, (name, t)
        assert not validate(sc.table, sp.meta).errors, (name, t)
        assert (sc.labels == POS).sum() == sc.truth["n_affected"].sum()
        assert (sc.labels == NEG).mean() > 0.2
        again = make_scenario(sp, t, 0, repeats=repeats)
        pd.testing.assert_frame_equal(sc.truth, again.truth)


def test_a_baseline_scores_the_real_table_and_sees_a_large_injected_fault(real_split):
    name, sp = real_split
    meta = sp.meta
    t = "power_loss" if meta.profile == "wind" else "string_loss"
    sc = make_scenario(sp, t, 0, repeats=1)
    m = METHODS["zscore"]().fit(sp.train, meta)
    s = m.score(sc.table, meta)
    big = sc.truth[sc.truth["level"] == LEVEL_COUNT].iloc[0]
    sel = ((sc.table["device_id"] == big.device_id) & (sc.table["timestamp"] >= big.start) & (sc.table["timestamp"] < big.end)).to_numpy() & (sc.labels == POS)
    neg = sc.labels == NEG
    assert np.nanmedian(s[sel]) > np.nanmedian(s[neg])


@pytest.mark.skipif(not care_events(), reason="CARE farm A not prepared")
def test_care_tables_have_a_training_section_and_labelled_events():
    ids = care_events()
    assert len(ids) == 22
    n_anom = 0
    for e in ids[:4]:
        df = pd.read_parquet(data_dir() / f"{REAL_NAME}_{e}.parquet", columns=["timestamp", "x_train_test"])
        ev = pd.read_parquet(data_dir() / f"{REAL_NAME}_{e}.events.parquet")
        assert set(df["x_train_test"]) == {"train", "prediction"}
        n_anom += int((ev["status"] == "anomaly").sum())
    assert n_anom >= 1
