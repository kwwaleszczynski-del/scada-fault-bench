"""Render README.md from docs/README.template.md and results/summary.json: every number in the README comes from the run.

Usage:
    python scripts/render_readme.py            # writes README.md
    python scripts/render_readme.py --check    # exit 1 when README.md differs from the rendering
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scada_fault_bench import report  # noqa: E402

TEMPLATE = ROOT / "docs" / "README.template.md"
SUMMARY = ROOT / "results" / "summary.json"
README = ROOT / "README.md"


def top_sentence(S: dict, profile: str) -> str:
    """The methods whose F1 interval overlaps the interval of the highest F1: the honest reading of a ranking with intervals."""
    ent = [e for e in S["overall"] if e["profile"] == profile and e["f1"]["value"] is not None]
    if not ent:
        return ""
    best = max(ent, key=lambda e: e["f1"]["value"])
    lo = best["f1"]["lo"] if best["f1"]["lo"] is not None else best["f1"]["value"]
    near = [e["method"] for e in ent if (e["f1"]["hi"] if e["f1"]["hi"] is not None else e["f1"]["value"]) >= lo]
    names = ", ".join(f"`{m}`" for m in near)
    if len(near) == len(ent):
        return f"The F1 intervals of all methods overlap with the interval of the highest value (`{best['method']}`): {names}."
    return f"The highest F1 is that of `{best['method']}`; the methods whose F1 interval overlaps its interval are {names}, so the data do not separate them."


def render(template: str, S: dict) -> str:
    nums = report.numbers(S)
    nums["top_wind"] = top_sentence(S, "wind")
    nums["top_pv"] = top_sentence(S, "pv")
    tables = {
        "overall_wind": report.table_overall(S, "wind"), "overall_pv": report.table_overall(S, "pv"),
        "types_wind": report.table_types(S, "wind"), "types_pv": report.table_types(S, "pv"),
        "real": report.table_real(S), "faults": report.table_faults(S), "methods": report.table_methods(S), "datasets": report.table_datasets(S),
    }
    text = re.sub(r"\{\{table:(\w+)\}\}", lambda m: tables[m.group(1)], template)
    text = re.sub(r"\{\{(\w+)\}\}", lambda m: nums[m.group(1)], text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    S = report.load_summary(SUMMARY)
    text = render(TEMPLATE.read_text(encoding="utf-8"), S)
    if args.check:
        return 0 if README.read_text(encoding="utf-8") == text else 1
    README.write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
