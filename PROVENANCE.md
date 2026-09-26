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

## Near-axis QI database (CC BY 4.0) -- not applicable in this pass

The ConStellaration benchmark's own near-axis-expansion QI database is CC BY 4.0 licensed. This
repository's construction-search modules (`construction.p3_construction_proxies`,
`construction.p3_construction_search`, `construction.p3_construction_bank_eval`) and the near-axis
engines (`engines.near_axis_search`, `engines.nae_generator`) are a **live parameter search** over
`constellaration.mhd.near_axis_configuration.generate`'s own near-axis parameter space (per its
Goodman-et-al. construction) -- they do not read or ship a copy of the downloaded database. The
code that would read that database directly is E21's generator (`phase1/generator-build`), which
per the build plan is deferred to a second pass of this repository, landing only after its source
PR merges. When that code is added, this section will carry the database's CC BY 4.0 attribution
and licence text alongside it.

## Pinned versions

`constellaration==0.2.6` (PyPI), which transitively pins `vmecpp==0.4.11` and
`scikit-learn==1.5.2` (pinned explicitly and defensively in `pyproject.toml`/`Dockerfile` so a
future `constellaration` bump cannot silently drag `scikit-learn`'s version along). Every result in
`docs/results/` was produced with exactly this stack. See `pyproject.toml` and `Dockerfile`.
