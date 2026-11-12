"""The committed documents are what the run produced: README.md, docs/data.js, and the release metadata agree."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

from scada_fault_bench import report
from scada_fault_bench.base import METHODS
from scada_fault_bench.faults import FAULT_TYPES

ROOT = Path(__file__).resolve().parents[1]
# names of the author's employer and work e-mail, written in pieces so that this file does not contain them
PRIVATE = re.compile("|".join(["(?<![a-z])" + "er" + "gy(?![a-z])", "@" + "er" + "gy", "konrad" + r"\." + "waleszczynski"]), re.I)


def script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def summary():
    p = ROOT / "results" / "summary.json"
    if not p.exists():
        pytest.skip("results/summary.json not built (scada-fault-bench run, then summarise)")
    return report.load_summary(p)


def test_readme_is_the_template_rendered_from_the_summary(summary):
    r = script("render_readme")
    text = r.render((ROOT / "docs" / "README.template.md").read_text(encoding="utf-8"), summary)
    assert (ROOT / "README.md").read_text(encoding="utf-8") == text
    assert "{{" not in text


def test_readme_has_no_hand_typed_result_numbers():
    """Numbers in the prose must come from placeholders: the template may contain only counts that are part of a name or a version."""
    tpl = (ROOT / "docs" / "README.template.md").read_text(encoding="utf-8")
    prose = re.sub(r"```.*?```", "", tpl, flags=re.S)
    prose = re.sub(r"\{\{[^}]+\}\}|\]\([^)]*\)|`[^`]*`|https?://\S+|doi \S+|10\.\d+/[\w.]+", "", prose)
    allowed = {"4", "0", "1", "2", "3", "5", "95", "14", "12"}
    found = set(re.findall(r"\b\d+(?:\.\d+)*\b", prose)) - allowed - {"4.0"}  # CC BY 4.0
    assert not found, f"hand-typed numbers in the README template: {sorted(found)}"


def test_the_readme_tables_cover_every_method_and_fault_type(summary):
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for m in summary["order"]["methods"]:
        assert f"`{m}`" in text
    for t in summary["order"]["types"]:
        assert f"`{t}`" in text
    assert set(summary["order"]["methods"]) <= set(METHODS) and set(summary["order"]["types"]) <= set(FAULT_TYPES)


def test_the_summary_is_internally_consistent(summary):
    c = summary["config"]
    assert c["n_boot"] >= 1000 and len(c["folds"]) >= 2
    assert sum(e["n"] for e in summary["events"]) > 0
    n_cells = len(summary["order"]["methods"]) * len(summary["order"]["datasets"]) * len(summary["order"]["types"])
    assert 0 < len(summary["cells"]) <= n_cells
    for e in summary["overall"] + summary["types"] + summary["cells"] + summary["datasets"] + summary["real"]:
        for k, m in e.items():
            if isinstance(m, dict) and m.get("value") is not None and m["lo"] is not None:
                if k in ("precision", "recall", "recall_placebo", "f1", "auroc", "ap"):
                    assert -1e-9 <= m["lo"] <= m["hi"] <= 1 + 1e-9, (e["method"], k, m)
                    assert m["lo"] - 0.05 <= m["value"] <= m["hi"] + 0.05, (e["method"], k, m)  # the point value is inside its interval up to resampling noise
    assert len({d["dataset"] for d in summary["datasets"]}) >= 3
    assert len({t["type"] for t in summary["types"]}) >= 6
    assert not [r for r in summary["runs"] if r["errors"]], "a method failed in the committed run"
    assert c["seeds_wind"] >= 1 and c["seeds_pv"] >= 1 and c["block_weeks"] >= 2 and 0 < c["alarm_budget"] < 0.1
    for e in summary["overall"]:
        assert e["n"]["events"] > 0 and e["recall_placebo"]["value"] is not None, e["method"]
        assert abs(e["recall_excess"]["value"] - (e["recall"]["value"] - e["recall_placebo"]["value"])) < 1e-9
    for m, info in summary["methods"].items():
        if info["scope"]:
            seen = {e["type"] for e in summary["types"] if e["method"] == m}
            assert seen <= set(info["scope"]), (m, seen)


def test_pages_data_is_built_from_the_summary(summary):
    b = script("build_pages")
    assert (ROOT / "docs" / "data.js").read_text(encoding="utf-8") == b.build_data(summary)


def test_project_page_uses_only_data_from_data_js_and_pins_its_script():
    html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    assert "window.DATA" in html and "plotly" in html.lower() and 'data.js' in html
    assert "integrity=" in html and 'prefers-color-scheme: dark' in html and 'data-theme' in html
    body = re.sub(r"<style.*?</style>|<script.*?</script>", "", html, flags=re.S)
    text = re.sub(r"<[^>]+>", "", body).replace("CC BY-SA 4.0", "").replace("CC BY 4.0", "")
    assert not re.findall(r"\d+\.\d+", text), "numbers typed into the page text"


# ----------------------------------------------------------------------------------------------- release metadata
def _release():
    return json.loads((ROOT / "release.json").read_text(encoding="utf-8"))


def test_version_is_the_same_everywhere():
    from scada_fault_bench import __version__

    v = _release()["version"]
    assert v == __version__
    assert f"version: {v}" in (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert f"## [{v}]" in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")


def test_dois_are_well_formed_and_consistent_once_assigned():
    rel = _release()
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    for key in ("doi_concept", "doi_version"):
        if rel[key] is not None:
            assert re.fullmatch(r"10\.5281/zenodo\.\d+", rel[key])
            assert rel[key] in cff
    if rel["doi_version"]:
        assert rel["doi_concept"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    placeholder = "A DOI is assigned by Zenodo when the first release is archived"
    assert (placeholder in readme) == (rel["doi_concept"] is None)


def test_pypi_flag_matches_the_readme():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert ("Not on PyPI yet" not in readme) == _release()["published_on_pypi"]


def test_release_workflow_uses_trusted_publishing_and_no_token():
    wf = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "id-token: write" in wf and "name: pypi" in wf and "PYPI_PUBLISH" in wf and "pypa/gh-action-pypi-publish" in wf
    assert "secrets." not in wf and "password" not in wf.lower()


def test_release_metadata_is_valid_and_free_of_private_names():
    z = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    assert z["upload_type"] == "software" and z["license"] == "MIT" and z["creators"] == [{"name": "Waleszczyński, Konrad"}]
    for f in (".zenodo.json", "CITATION.cff", "pyproject.toml", "release.json", "docs/RELEASING.md", "README.md", "docs/METHODS.md", "docs/index.html"):
        text = (ROOT / f).read_text(encoding="utf-8")
        assert not PRIVATE.search(text), f
        emails = [e for e in re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", text) if not e.endswith("users.noreply.github.com")]
        assert not emails, (f, emails)


def test_no_traces_of_tools_or_private_data_in_tracked_text():
    import subprocess

    out = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    words = ["co-" + "authored-by", "cl" + "aude", "anth" + "ropic", "super" + "powers"]  # written in pieces so that this file does not contain them
    pat = re.compile("|".join(words), re.I)
    for f in out.stdout.split():
        p = ROOT / f
        if p.suffix in (".parquet", ".png", ".ico") or not p.is_file() or f == "tests/test_docs.py":
            continue
        assert not pat.search(p.read_text(encoding="utf-8", errors="ignore")), f
        assert not f.lower().startswith(("." + words[1], words[1])), f
