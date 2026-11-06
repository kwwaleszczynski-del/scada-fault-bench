"""Tables and numbers for the README and the project page, all taken from ``results/summary.json``.

Nothing here contains a result: every figure is read from the summary, so the README (which ``scripts/render_readme.py`` builds
from ``docs/README.template.md``) and the page (``scripts/build_pages.py``) cannot disagree with the run.

Usage:
    S = load_summary("results/summary.json")
    md = table_overall(S, "wind")
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["cell", "load_summary", "numbers", "table_datasets", "table_faults", "table_methods", "table_overall", "table_real", "table_types"]

TYPE_TITLES = {
    "anemometer_drift": "anemometer drift", "nacelle_dir_offset": "nacelle direction offset", "derating": "derating", "power_loss": "power loss",
    "thermal_drift": "temperature drift", "flatline": "flatline", "unit_jump": "unit jump", "clock_shift": "clock shift", "string_loss": "string loss",
    "shading": "shading", "inverter_outage": "inverter outage", "pyranometer_drift": "pyranometer drift",
}


def load_summary(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cell(e: dict, key: str, nd: int = 2, ci: bool = True, scale: float = 1.0) -> str:
    """``value (lo-hi)`` of one metric of a summary entry, or ``n/a`` when it is undefined (no events, no alarms)."""
    m = e[key]
    if m["value"] is None:
        return "n/a"
    v = f"{m['value'] * scale:.{nd}f}"
    if not ci or m["lo"] is None or m["hi"] is None:
        return v
    return f"{v} ({m['lo'] * scale:.{nd}f}-{m['hi'] * scale:.{nd}f})"


def _by(S: dict, kind: str, **keys) -> dict:
    return {e["method"]: e for e in S[kind] if all(e.get(k) == v for k, v in keys.items())}


def _methods(S: dict, profile: str | None = None) -> list[str]:
    m = S["order"]["methods"]
    return [x for x in m if profile is None or profile in S["methods"][x]["profiles"]]


def title_of(t: str) -> str:
    return TYPE_TITLES.get(t, t.replace("_", " "))


def numbers(S: dict) -> dict[str, str]:
    c = S["config"]
    ev = S["events"]
    inj = sum(e["n"] for e in ev)
    real = S["real_events"][0] if S["real_events"] else {"anomalies": 0, "normal_datasets": 0}
    wind_ds = [d for d in S["order"]["datasets"] if S["datasets_info"][d]["profile"] == "wind"]
    pv_ds = [d for d in S["order"]["datasets"] if S["datasets_info"][d]["profile"] == "pv"]
    wind_types = [t for t in S["order"]["types"] if "wind" in S["fault_types"][t]["profiles"]]
    pv_types = [t for t in S["order"]["types"] if "pv" in S["fault_types"][t]["profiles"]]
    n = {
        "n_events": f"{inj:,}", "n_datasets": str(len(wind_ds) + len(pv_ds)), "n_wind_datasets": str(len(wind_ds)), "n_pv_datasets": str(len(pv_ds)),
        "n_methods": str(len(S["order"]["methods"])), "n_baselines": str(sum(S["methods"][m]["family"] == "baseline" for m in S["order"]["methods"])),
        "n_portfolio": str(sum(S["methods"][m]["family"] == "portfolio" for m in S["order"]["methods"])),
        "n_types": str(len(S["order"]["types"])), "n_wind_types": str(len(wind_types)), "n_pv_types": str(len(pv_types)),
        "n_levels": str(c["levels"]), "n_folds": str(len(c["folds"])), "n_boot": str(c["n_boot"]),
        "n_real_anomalies": str(real["anomalies"]), "n_real_normal": str(real["normal_datasets"]),
        "test_days_wind": str(c["test_days"]["wind"]), "test_days_pv": str(c["test_days"]["pv"]), "buffer_days": str(c["buffer_days"]),
        "cal_pct": f"{100 * (1 - c['cal_quantile']):.1f}", "merge_gap_h": f"{c['merge_gap_hours']:g}", "min_alarm_h": f"{c['min_alarm_hours']:g}",
        "tolerance_days": f"{c['tolerance_days']:g}", "guard_days": f"{c['guard_days']:g}", "min_affected_h": f"{c['min_affected_hours']:g}",
        "version": c["version"], "scada_schema_version": c["scada_schema"],
        "seeds_pv": str(c["seeds"]),
    }
    for prof in ("wind", "pv"):
        n[f"events_{prof}"] = f"{sum(e['n'] for e in ev if S['datasets_info'][e['dataset']]['profile'] == prof):,}"
    return n


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(head: list[str], rows: list[list[str]], align: list[str] | None = None) -> str:
    align = align or ["l"] + ["r"] * (len(head) - 1)
    sep = ["---:" if a == "r" else ":---" for a in align]
    return "\n".join([_row(head), _row(sep), *[_row(r) for r in rows]])


def table_overall(S: dict, profile: str) -> str:
    ent = _by(S, "overall", profile=profile)
    rows = []
    for m in _methods(S, profile):
        e = ent.get(m)
        if e is None:
            continue
        mi = S["methods"][m]
        rows.append([f"{mi['title']} (`{m}`)", cell(e, "f1"), cell(e, "recall"), cell(e, "precision"), cell(e, "delay_h", 0), cell(e, "fa_h_per_year", 0), cell(e, "auroc"),
                     cell(e, "ap")])
    return _table(["Method", "F1", "Recall", "Precision", "Delay (h)", "False alarms (h/yr)", "AUROC", "AP"], rows)


def table_types(S: dict, profile: str, key: str = "recall") -> str:
    types = [t for t in S["order"]["types"] if profile in S["fault_types"][t]["profiles"]]
    ent = {(e["method"], e["type"]): e for e in S["types"] if e["profile"] == profile}
    rows = []
    for m in _methods(S, profile):
        rows.append([f"`{m}`", *[cell(ent[(m, t)], key, ci=False) if (m, t) in ent else "n/a" for t in types]])
    return _table(["Method", *[title_of(t) for t in types]], rows)


def table_real(S: dict) -> str:
    ent = _by(S, "real", dataset="care_A")
    rows = []
    for m in _methods(S, "wind"):
        e = ent.get(m)
        if e is None:
            continue
        rows.append([f"{S['methods'][m]['title']} (`{m}`)", cell(e, "recall"), cell(e, "precision"), cell(e, "delay_h", 0), cell(e, "fa_h_per_year", 0), cell(e, "auroc"), cell(e, "ap")])
    return _table(["Method", "Recall", "Precision", "Delay (h)", "False alarms (h/yr)", "AUROC", "AP"], rows)


def table_faults(S: dict) -> str:
    rows = []
    for t in S["order"]["types"]:
        f = S["fault_types"][t]
        lv = ", ".join(f"{x:g}" for x in f["levels"])
        grade = "duration" if f["level_kind"] == "duration" else "size"
        rows.append([f"`{t}`", "+".join(f["profiles"]), f["summary"], f"{lv} {f['unit']} ({grade})"])
    return _table(["Type", "Profile", "What is injected", "Five levels"], rows, ["l", "l", "l", "l"])


def table_methods(S: dict) -> str:
    rows = []
    for m in S["order"]["methods"]:
        mi = S["methods"][m]
        thr = "fixed" if mi["fixed_threshold"] else "calibrated"
        rows.append([f"`{m}`", "+".join(mi["profiles"]), mi["family"], thr, mi["description"] + (f" Source: {mi['source']}." if mi["source"] else "")])
    return _table(["Name", "Profile", "Family", "Threshold", "Description"], rows, ["l", "l", "l", "l", "l"])


def table_datasets(S: dict) -> str:
    cnt: dict[str, int] = {}
    for e in S["events"]:
        cnt[e["dataset"]] = cnt.get(e["dataset"], 0) + e["n"]
    rows = []
    for d in S["order"]["datasets"]:
        i = S["datasets_info"][d]
        rows.append([f"`{d}`", i["title"], i["profile"], f"{i['start']} to {i['end']}", f"{cnt.get(d, 0):,}"])
    return _table(["Name", "Dataset", "Profile", "Period (UTC)", "Injected events"], rows, ["l", "l", "l", "l", "r"])
