# Releasing scada-fault-bench

Nothing is published automatically. `.github/workflows/release.yml` builds and checks the sdist and wheel for every published GitHub release;
uploading to PyPI (Trusted Publishing, no token anywhere in the repository) happens only when the repository variable `PYPI_PUBLISH` is `true`.
The version, the DOIs and whether the package is on PyPI live in `release.json`; the README, `CITATION.cff` and the tests read them from there.

First release, in this order: switch the repository on at https://zenodo.org/account/settings/github/ (metadata from `.zenodo.json`); add a pending
trusted publisher at https://pypi.org/manage/account/publishing/ (project `scada-fault-bench`, owner `kwwaleszczynski-del`, repository
`scada-fault-bench`, workflow `release.yml`, environment `pypi`); create the GitHub environment `pypi` and the variable `PYPI_PUBLISH` = `true`;
replace `unreleased` in `CHANGELOG.md` by the date and set `date_released` in `release.json`; publish a release with the tag `v0.1.0`; afterwards set
`published_on_pypi` and the two DOIs in `release.json`, add them to `CITATION.cff`, run `python scripts/render_readme.py` and push.

Checked locally before a release: `python -m build && twine check dist/*`, a fresh virtual environment with `pip install dist/*.whl` and
`scada-fault-bench list`, and `pytest -q`.
