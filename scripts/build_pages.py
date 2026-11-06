"""Build docs/data.js (the data of the project page) from results/summary.json.

``docs/index.html`` has no numbers of its own: it draws ``window.DATA``, which is this file's output; ``tests/test_docs.py``
compares the committed ``docs/data.js`` with the rebuilding.

Usage:
    python scripts/build_pages.py            # writes docs/data.js
    python scripts/build_pages.py --check    # exit 1 when docs/data.js differs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scada_fault_bench import report  # noqa: E402

SUMMARY = ROOT / "results" / "summary.json"
DATA = ROOT / "docs" / "data.js"
ND = 4


def _round(x):
    if isinstance(x, float):
        return round(x, ND)
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return x


def build_data(S: dict) -> str:
    keep = {k: S[k] for k in ("cells", "datasets", "types", "overall", "real", "null", "events", "levels", "methods", "fault_types", "datasets_info", "order", "config", "real_events")}
    keep["titles"] = {t: report.title_of(t) for t in S["order"]["types"]}
    keep["numbers"] = report.numbers(S)
    return "window.DATA = " + json.dumps(_round(keep), separators=(",", ":"), sort_keys=True) + ";\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = build_data(report.load_summary(SUMMARY))
    if args.check:
        return 0 if DATA.read_text(encoding="utf-8") == text else 1
    DATA.write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
