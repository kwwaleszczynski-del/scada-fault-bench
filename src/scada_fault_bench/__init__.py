"""An open benchmark for fault detection in wind and PV SCADA data (see the README)."""

from scada_fault_bench import baselines, portfolio  # noqa: E402,F401 - importing the package registers the methods in base.METHODS
from scada_fault_bench._version import __version__

__all__ = ["__version__"]
