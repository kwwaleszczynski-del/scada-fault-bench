import numpy as np

from scada_fault_bench.base import Detector


class Mine(Detector):
    """A user's detector as documented in the README: distance of power from its training mean."""

    name = "my_method"
    title = "My method"
    description = "absolute distance of active power from its training mean"
    profiles = ("wind",)

    def fit(self, train, meta):
        self.mean = float(train["p_ac"].mean())
        return self

    def score(self, table, meta):
        return np.abs(table["p_ac"].to_numpy(dtype="float64") - self.mean)
