"""Command line: ``scada-fault-bench list | prepare | inject | run | summarise``.

Usage:
    scada-fault-bench list
    scada-fault-bench prepare --datasets kelmarsh_2019 pvdaq_34 --fetch
    scada-fault-bench inject --dataset kelmarsh_2019 --type derating --fold 1 --out out/derating
    scada-fault-bench run --datasets kelmarsh_2019 --folds 0 1 --out results
    scada-fault-bench run --datasets kelmarsh_2019 --plugin my_package.my_module:MyDetector --out results
    scada-fault-bench summarise --out results
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

from scada_fault_bench import __version__

log = logging.getLogger("scada_fault_bench")


def _plugin(spec: str) -> None:
    """Register a user's detector: ``package.module:ClassName`` (a subclass of ``Detector`` with a ``name``)."""
    from scada_fault_bench.base import METHODS, Detector, register

    mod, _, attr = spec.partition(":")
    cls = getattr(importlib.import_module(mod), attr)
    if not (isinstance(cls, type) and issubclass(cls, Detector)):
        raise SystemExit(f"{spec} is not a subclass of scada_fault_bench.base.Detector")
    if cls.name not in METHODS:
        register(cls)


def _list(args) -> int:
    from scada_fault_bench import baselines, portfolio  # noqa: F401
    from scada_fault_bench.base import METHODS
    from scada_fault_bench.data import DATASETS
    from scada_fault_bench.faults import FAULT_TYPES

    print("datasets (injected faults):")
    for n, d in DATASETS.items():
        print(f"  {n:20s} {d.profile:5s} {d.start}..{d.end}")
    print("dataset with real labelled faults: care_A (CARE to Compare, farm A)")
    print("fault types:")
    for n, f in FAULT_TYPES.items():
        print(f"  {n:20s} {'+'.join(f.profiles()):8s} levels {list(f.levels)} {f.unit}")
    print("methods:")
    for n, m in METHODS.items():
        ok, why = m.available()
        print(f"  {n:20s} {m.family:9s} {'+'.join(m.profiles):8s} {'available' if ok else 'NOT available: ' + why}")
    return 0


def _prepare(args) -> int:
    from scada_fault_bench.prepare import prepare
    from scada_fault_bench.real import REAL_NAME, prepare_care

    for name in args.datasets:
        if name == REAL_NAME:
            prepare_care(args.data, fetch=args.fetch, raw=args.raw)
        else:
            prepare(name, args.data, fetch=args.fetch, raw=args.raw)
    return 0


def _inject(args) -> int:
    from scada_fault_bench.data import load_dataset, make_split
    from scada_fault_bench.scenario import make_scenario

    df, meta = load_dataset(args.dataset, args.data)
    split = make_split(df, meta, args.fold, name=args.dataset)
    sc = make_scenario(split, args.type, args.seed, repeats=args.repeats or (2 if meta.profile == "wind" else 1))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = sc.table.assign(label=sc.labels)
    table.to_parquet(out.with_suffix(".table.parquet"))
    sc.truth.to_csv(out.with_suffix(".truth.csv"), index=False)
    log.info("wrote %s.table.parquet (%d rows, label 0 healthy / 1 affected / 2 grey) and %s.truth.csv (%d events)", out, len(table), out, len(sc.truth))
    return 0


def _run(args) -> int:
    from scada_fault_bench.runner import RunConfig, run_dataset

    for p in args.plugin or []:
        _plugin(p)
    cfg = RunConfig(out=args.out, data=args.data, folds=tuple(args.folds), seeds=args.seeds, repeats=args.repeats,
                    methods=tuple(args.methods) if args.methods else None, types=tuple(args.types) if args.types else None)
    for name in args.datasets:
        run_dataset(name, cfg)
    return 0


def _summarise(args) -> int:
    from scada_fault_bench.summary import write_summary

    p = write_summary(args.out, n_boot=args.n_boot, seed=args.seed)
    log.info("wrote %s", p)
    return 0


def build_parser() -> argparse.ArgumentParser:
    from scada_fault_bench.data import DATASETS
    from scada_fault_bench.real import REAL_NAME

    ap = argparse.ArgumentParser(prog="scada-fault-bench", description="Open benchmark for fault detection in wind and PV SCADA data.")
    ap.add_argument("--version", action="version", version=f"scada-fault-bench {__version__}")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list", help="datasets, fault types and methods, and whether each method can run here")
    s.set_defaults(fn=_list)
    names = [*DATASETS, REAL_NAME]
    inj_names = list(DATASETS)

    def common(sp):
        sp.add_argument("--data", default=None, help="folder with <dataset>.parquet files (default $SCADA_FAULT_BENCH_DATA or ./data)")

    s = sub.add_parser("prepare", help="build canonical tables from the original open files with scada-schema")
    common(s)
    s.add_argument("--datasets", nargs="+", choices=names, default=names)
    s.add_argument("--fetch", action="store_true", help="download the original files first")
    s.add_argument("--raw", default=None, help="folder of the original files (default as in scada-schema)")
    s.set_defaults(fn=_prepare)
    s = sub.add_parser("inject", help="write one test block with injected faults of one type, its truth and row labels")
    common(s)
    s.add_argument("--dataset", choices=inj_names, required=True)
    s.add_argument("--type", required=True)
    s.add_argument("--fold", type=int, default=1)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--repeats", type=int, default=None)
    s.add_argument("--out", required=True, help="path prefix of the output files")
    s.set_defaults(fn=_inject)
    s = sub.add_parser("run", help="run methods on datasets, write shards to <out>/shards")
    common(s)
    s.add_argument("--datasets", nargs="+", choices=names, default=names)
    s.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3])
    s.add_argument("--seeds", type=int, default=1)
    s.add_argument("--repeats", type=int, default=None)
    s.add_argument("--methods", nargs="+", default=None)
    s.add_argument("--types", nargs="+", default=None)
    s.add_argument("--plugin", nargs="+", default=None, help="extra detectors as package.module:ClassName")
    s.add_argument("--out", default="results")
    s.set_defaults(fn=_run)
    s = sub.add_parser("summarise", help="bootstrap the shards into <out>/summary.json")
    s.add_argument("--out", default="results")
    s.add_argument("--n-boot", type=int, default=1000)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=_summarise)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
