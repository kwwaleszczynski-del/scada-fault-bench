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
    """What the intervals allow to say: which methods have an excess recall whose interval stays above zero, and which have the highest AUROC."""
    ent = [e for e in S["overall"] if e["profile"] == profile and e["recall_excess"]["value"] is not None]
    if not ent:
        return ""
    above = [e["method"] for e in ent if e["recall_excess"]["lo"] is not None and e["recall_excess"]["lo"] > 0]
    auc = [e for e in ent if e["auroc"]["value"] is not None]
    best = max(auc, key=lambda e: e["auroc"]["value"])
    lo = best["auroc"]["lo"] if best["auroc"]["lo"] is not None else best["auroc"]["value"]
    near = [e["method"] for e in auc if (e["auroc"]["hi"] if e["auroc"]["hi"] is not None else e["auroc"]["value"]) >= lo]
    names = ", ".join(f"`{m}`" for m in near)
    ex = ", ".join(f"`{m}`" for m in above) if above else "none"
    return (f"The excess recall has an interval above zero for: {ex}. The highest AUROC is that of `{best['method']}`; the methods whose AUROC interval overlaps its interval are {names}"
            " (the intervals describe the sampling of weeks and events in this benchmark, so a method outside that list is not shown to be worse on other data, and one inside it is not shown to be equal).")


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
