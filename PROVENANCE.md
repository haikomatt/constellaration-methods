# Provenance

## P1 and P2: own lineage

`docs/results/submission/best_own_geometrical.json` (P1) and
`docs/results/submission/best_feasible_simple_to_build.json` (P2) are both **own lineage**: built
by this repository's own optimizer (`engines.alm_ngopt_optimizer`, an Augmented-Lagrangian /
Nevergrad descent with canonical-in-loop scoring) from an analytic seed
(`constellaration.initial_guess.generate_rotating_ellipse` for P1; an own-lineage near-axis/descent
climb for P2), with **no leader boundary anywhere in either run's invocation or lineage**. Full
detail, including the exact seed parameters, random seeds, algorithm settings, and a fresh
reload-and-re-evaluate verification against the benchmark's own scorer, is in
`docs/results/submission/PROVENANCE.json` (P2) and
`docs/results/submission/PROVENANCE_own_geometrical.json` (P1). Both are submittable as own work.

## P3: davidkh lineage, measurement only

Every P3 (`MHDStableQIStellarator`) result in `docs/results/p3/` traces back to `davidkh`'s public
P2 (`SimpleToBuildQIStellarator`) leaderboard entry as its starting boundary. That entry is public
(the leaderboard is a public record), but **no reuse terms for derivatives of public leaderboard
boundaries have been agreed with its author or with Proxima Fusion**. Consequently:

- No P3 result in this repository is, or is presented as, a submission. Every P3 artefact here is
  measurement and method: how far a restoration/front-mapping method gets from that public start,
  not a design anyone could submit today.
- No P3 boundary coordinates are redistributed. Every `REPORT.json`/`curve.json` under
  `docs/results/p3/` has been passed through `tools/strip_boundaries.py`, which recursively removes
  the keys that carry Fourier boundary coefficients (`x`, `boundary`, `boundary_dict`, `r_cos`,
  `z_sin`, `r_sin`, `z_cos`) from every nested structure. `tools/check_no_boundaries.py` is the
  corresponding CI-style check that this held for everything actually committed.
- Public leaderboard handles referenced in the code and docs (`davidkh`, `phanerozoic`,
  `canyon-ant`) are kept, not redacted -- they are the public record every measurement here is
  defined against, and every mention of a derived boundary carries this provenance note.
- D1 (`docs/results/p3/D1/`), the boundary-diversity measurement, includes `phanerozoic`'s public
  fronts, `davidkh`'s and `canyon-ant`'s public entries, and this repository's own P3 results, all
  compared as symmetry-quotiented physical-space distances -- again no boundary coordinates are
  redistributed, only distances, cluster counts, and the two summary plots.

See `docs/methods.md` for what each P3 experiment (E13, E15-E20b) actually shows and what it
refutes, and the README results table for the headline numbers.

## Near-axis QI database (CC BY 4.0)

This repository's own construction-search modules (`construction.p3_construction_proxies`,
`construction.p3_construction_search`, `construction.p3_construction_bank_eval`) and the near-axis
engines (`engines.near_axis_search`, `engines.nae_generator`) are a **live parameter search** over
`constellaration.mhd.near_axis_configuration.generate`'s own near-axis parameter space (per its
Goodman-et-al. construction) -- they do not read or ship a copy of any downloaded database.

`construction.nae_db_seed_screen` (E21, generator route A', added in this repository's second pass)
is different: it reads a public, external near-axis QI construction database directly.

- **Database.** E. Rodriguez and G. G. Plunk, near-axis QI database v3, arXiv:2601.08400 (2026),
  Max Planck Institute for Plasma Physics. Dataset: Zenodo,
  doi:[10.5281/zenodo.18220133](https://doi.org/10.5281/zenodo.18220133), licensed **CC BY 4.0**.
  803,532 configurations, built by an analytic construction with no optimisation against any
  benchmark and no leaderboard boundary anywhere in its own lineage.
- **Not shipped.** Neither the database (roughly 16 GB) nor the authors' pyQIC fork needed to
  rebuild its configurations is distributed with this repository. `docs/external-data.md` states
  exactly what each is, where to obtain it, and the local layout the code expects; the module fails
  with a clear, actionable error when they are absent, and its own tests need neither (fixtures and
  injected fakes only).
- **Lineage of E21's own seeds.** Every E21 seed is **own lineage by construction**: rebuilt from
  the database's stored analytic parameters, completed with the authors' own MHD magnetic-well
  transform, and exported at a chosen aspect ratio -- no leaderboard boundary anywhere. At the same
  time, every seed **derives from a public third-party database under its own licence**: this is a
  different kind of provenance from the P1/P2 own-lineage results above (which start from an
  in-house analytic seed with no external dataset at all), and is why it is called out separately
  here rather than folded into the "own lineage" language above.
  Attribution and licence terms for anyone reusing E21's seeds or this module: cite the paper and
  the Zenodo record above; CC BY 4.0 requires attribution, not share-alike or non-commercial terms,
  so this repository's own Apache-2.0 licence is unaffected, but the database's own terms travel
  with any seed derived from it.
- **Results.** `docs/results/p3/E21/` (REPORT.json, placement summary, `STATUS.md`); see the
  README's results table and `docs/methods.md`'s cold-start section for what E21 shows.

## Pinned versions

`constellaration==0.2.6` (PyPI), which transitively pins `vmecpp==0.4.11` and
`scikit-learn==1.5.2` (pinned explicitly and defensively in `pyproject.toml`/`Dockerfile` so a
future `constellaration` bump cannot silently drag `scikit-learn`'s version along). Every result in
`docs/results/` was produced with exactly this stack. See `pyproject.toml` and `Dockerfile`.
