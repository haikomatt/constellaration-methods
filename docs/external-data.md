# External data for `construction.nae_db_seed_screen` (E21)

`src/constellaration_methods/construction/nae_db_seed_screen.py` reproduces the E21 experiment
(generator route A': a near-axis QI database seed screen). Two external inputs it needs are **not
shipped with this repository** -- neither is this project's own artefact, both are large, and one
carries a licence of its own. This page states exactly what each is, where to get it, and the local
layout the module expects. Every function that touches them fails with a clear, actionable error
(pointing back here) when they are absent; the module's own tests never need either (see the module
docstring and `tests/construction/test_nae_db_seed_screen.py`).

## 1. The near-axis QI database

**What it is.** A public, second-order near-axis QI (quasi-isodynamic) construction database of
803,532 configurations, built by an 8-parameter analytic scan (axis curvature/torsion Fourier
coefficients, elongation function, on-axis mirror ratio, well width) with no optimisation against
any benchmark and no leaderboard lineage anywhere.

**Attribution.** E. Rodriguez and G. G. Plunk, "tailored initial conditions for optimization"
(near-axis QI database v3), arXiv:2601.08400 (2026), Max Planck Institute for Plasma Physics.
Dataset: Zenodo, doi:[10.5281/zenodo.18220133](https://doi.org/10.5281/zenodo.18220133), licensed
**CC BY 4.0**. If you use this database, cite the paper and the Zenodo record; this repository's own
use of it is documented per seed in every manifest `nae_db_seed_screen.build_batch` writes (database
file, row ID, and the eight input features), and in `docs/results/p3/E21/`.

**Size.** Roughly 16 GB: a `dataframe.pkl` summary table (803,532 rows, 87 columns) plus the
per-configuration sweep JSON files each row's `db_file`/`ID` fields point into.

**Local layout this module expects**, relative to whatever directory you pass as `--data-repo`
(default: this repository's own root):

```
data/qi_nae_db/
  dataframe.pkl          the summary table (not read directly by this module -- selection
                          happens upstream of it; see below)
  <sweep files>.json      the per-configuration files a selection's db_file paths resolve to
```

A selection JSON (the `--selection`/`--selection-extra` input to the `build` subcommand) is a list
of entries, each carrying at minimum `seed_name`, `N` (field-period count), `db_file` (path to its
sweep file, relative to `--data-repo` unless absolute), `ID` (the row's 1-based position within that
file), `iota` (the dataframe's stored near-axis rotational transform, checked against the rebuild),
and `features` (the eight input features, carried through to the manifest). Building this file from
the summary table is a separate selection step (cuts and ranking on the database's own columns);
this module starts from its output.

## 2. The authors' pyQIC fork

**What it is.** The database's configurations are signed-Frenet-frame constructions; the ordinary
PyPI `qic` (pyQIC) package cannot rebuild them at second order -- only the database authors' own
fork can, because it carries `Qic.mag_well_reshape`, the MHD magnetic-well completion this module
applies (`nae_db_seed_screen.rebuild_and_export`), and the second-order (`order="r2"`) construction
path the rebuild uses.

**Where to get it.** GitHub, user `SebereX`, branch/PR `change_structure`, commit `2a2e443d`. Clone
it and install it (e.g. `pip install -e .` from the fork's checkout) into its own virtual
environment -- do not install it into the same environment as this package's pinned
`constellaration`/`vmecpp` stack; keep the two separate, as the working campaign this was extracted
from did (a "fork venv" used only for the rebuild step, entirely outside this repository's own
Docker image).

**How this module checks it.** `nae_db_seed_screen._require_qic()` imports `qic.Qic` and additionally
checks for `Qic.mag_well_reshape`; if either the import fails or that attribute is missing (i.e. a
vanilla PyPI `qic` is installed instead of the fork), it raises a `RuntimeError` that names this file.

## 3. What runs where

Per the working campaign's convention (kept here for anyone reproducing the experiment): the `build`
subcommand -- the actual database rebuild, using the fork -- runs OUTSIDE any container, in the
fork's own pinned virtual environment, on host CPU cores reserved for that kind of work. The
`placement` subcommand and the canonical (VMEC++) evaluation of the exported boundaries are ordinary
uses of this repository's own pinned image and do not need the fork or the database at all -- they
only need the boundary files the `build` step already wrote.

## 4. The `ijson` dependency

`load_raw_config` streams each sweep JSON file incrementally (files run up to roughly 50 MB) using
the `ijson` package -- an ordinary, freely installable dependency, not tied to the fork or the
database's licence. Install it with:

```bash
pip install -e ".[generator]"
```

This does **not** install the pyQIC fork (section 2); that step is a separate, manual clone-and-pip-
install outside this repository, by design.
