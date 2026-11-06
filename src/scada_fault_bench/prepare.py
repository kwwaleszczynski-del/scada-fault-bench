"""Build the canonical tables of the benchmark from the original open files, with the loaders of ``scada-schema``.

Each dataset of :data:`scada_fault_bench.data.DATASETS` has a matching entry in ``scada_schema.study.SPECS`` (same name, same
loader, same arguments): the benchmark reads the table that ``scada-schema`` reads, so the numbers of both repositories refer
to the same data. The IEC 61400-26 category of every row is attached from the event log where the dataset has one (it marks
known stops as grey rows, see :mod:`scada_fault_bench.scenario`). The raw archives are looked for as ``scada-schema`` does
(``$SCADA_SCHEMA_<NAME>_DIR``, ``$SCADA_SCHEMA_DATA/<name>``, ``./data/<name>``); ``fetch=True`` downloads them first.

Usage:
    prepare("kelmarsh_2019", out_dir="data", fetch=True)
"""

from __future__ import annotations

import logging
from pathlib import Path

from scada_schema import write_parquet
from scada_schema.loaders._paths import raw_dir
from scada_schema.study import MODULES, SPECS

from scada_fault_bench.data import DATASETS, dataset_path

log = logging.getLogger(__name__)

__all__ = ["prepare"]


def prepare(name: str, out_dir: str | Path | None = None, fetch: bool = False, raw: str | Path | None = None) -> Path:
    """Load one benchmark dataset from its original files and write ``<out_dir>/<name>.parquet``."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {', '.join(DATASETS)}")
    spec = next(s for s in SPECS if s.name == name)
    mod = MODULES[spec.loader]
    folder = raw_dir(spec.loader, raw)
    kw = dict(spec.kwargs)
    if fetch:
        log.info("fetching %s into %s", spec.loader, folder)
        if spec.loader == "pvdaq":
            mod.fetch(folder, system=kw["system"], years=kw.get("years") or [2016])
        else:
            mod.fetch(folder, **({"years": kw["years"]} if "years" in kw else {}))
    loaded = mod.load(folder, **kw)
    if spec.states and len(loaded.events):
        from scada_schema.states import attach_states

        attach_states(loaded)
    dest = dataset_path(name, out_dir)
    write_parquet(loaded.table, loaded.meta, dest)
    log.info("wrote %s: %d rows, %d devices", dest, len(loaded.table), loaded.table["device_id"].nunique())
    return dest
