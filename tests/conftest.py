import pandas as pd
import pytest

from scada_fault_bench.synthetic import synthetic_pv, synthetic_wind


@pytest.fixture(scope="session")
def wind():
    return synthetic_wind(n_turbines=5, days=120, seed=1)


@pytest.fixture(scope="session")
def pv():
    return synthetic_pv(days=120, seed=1, inverters=2)


def window(df):
    t = df["timestamp"]
    return t.min(), t.max() + pd.Timedelta("10min")
