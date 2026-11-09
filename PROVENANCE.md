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

## P3: davidkh lineage, measurement only (except the own-lineage results E21 to E34 -- see below)

Every P3 (`MHDStableQIStellarator`) result in `docs/results/p3/` other than E21 and the own-lineage
results listed under "Near-axis QI database" below (E22 to E34) traces back to `davidkh`'s public P2
(`SimpleToBuildQIStellarator`) leaderboard entry as its starting boundary. That entry is public (the leaderboard is a public record), but **no reuse terms
for derivatives of public leaderboard boundaries have been agreed with its author or with Proxima
Fusion**. Consequently, for every davidkh-lineage result:

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
- **E22, E22b, E22c, E23.0, E23, E23b, E23c, E24r-B-2 and E26: own lineage, submittable at Matt's
  discretion.** `docs/results/p3/E22/` (the restoration link that took E21's confirmed seed from
  worst-of-three V 0.1443 to 0.0, strictly feasible on all five rows), `docs/results/p3/E22b/` and
  `docs/results/p3/E22c/` (the same restoration chain repeated, unchanged, from two further
  database seeds -- db00588 to worst-of-three V 0.0085 feasible, and db00397 to worst-of-three
  V 0.0049 feasible; adjudicated 2026-09-27, repeatability label 3 of 3), and
  `docs/results/p3/E23.0/` (the baseline measurement of E22's archive, confirmed hypervolume 82.13)
  are all **own lineage**: each design traces to the Rodriguez and Plunk near-axis QI database
  (CC BY 4.0) via a database seed and this repository's own restoration/front-confirmation code,
  with **no leaderboard boundary anywhere in its lineage**. `docs/results/p3/E23/` (the seeded
  population front grown from E22's confirmed point: 53 confirmed members, hypervolume 89.71,
  adjudicated 2026-09-27), `docs/results/p3/E23b/` (the second front link, anchors mode from E23's
  confirmed front: 52 confirmed members, hypervolume 92.61, adjudicated 2026-09-27) and
  `docs/results/p3/E23c/` (the third front link, anchors mode from E23b's confirmed front: 65
  confirmed members, hypervolume 94.28, adjudicated 2026-09-28/30 -- the chain then stopped on its
  pre-registered 2% rule, no E23d) design the same own lineage as E22 -- every member of every link
  is a variation on E22's own-lineage boundary produced by this repository's own front-mapping
  code, with no leaderboard boundary anywhere in any member's lineage. `docs/results/p3/E26/` (the
  mirror 0.25 -> 0.23 definition-change warm re-optimisation run on E23c's confirmed front: 46
  confirmed members, hypervolume 91.65, survival 0.972 of E23c's 94.28, adjudicated 2026-10-03) is
  own lineage by the same construction -- every member descends from E23c's own-lineage front via
  this repository's own re-optimisation code, with no leaderboard boundary anywhere in its lineage.
  This is unlike every davidkh-lineage P3 result above, which is measurement and method only and
  not submittable. E22's, E22b's, E22c's, E23.0's, E23's, E23b's, E23c's and E26's boundary exports
  (`docs/results/p3/E22/own-lineage/`, `docs/results/p3/E22b/own-lineage/`,
  `docs/results/p3/E22c/own-lineage/`, `docs/results/p3/E23.0/own-lineage/`,
  `docs/results/p3/E23/own-lineage/`, `docs/results/p3/E23b/own-lineage/`,
  `docs/results/p3/E23c/own-lineage/` and `docs/results/p3/E26/own-lineage/`) are shipped
  un-stripped for exactly this reason -- they are own lineage and clean provenance, so
  `tools/strip_boundaries.py`'s davidkh-derivative rationale does not apply to them, and
  `tools/check_no_boundaries.py` exempts those exact paths by an explicit allowlist (see that
  script's `BOUNDARY_ALLOWED_PREFIXES`). Attribution and licence terms for the database travel with
  these designs the same way as for E21's seeds (see above). The E24/E24r breadth banks
  (`docs/results/p3/E24-a85/`, `docs/results/p3/E24-b14/`, `docs/results/p3/E24r-a85/`,
  `docs/results/p3/E24r-b14/`, `docs/results/p3/E24r-a85-2/`, `docs/results/p3/E24r-b14-2/`) are own
  lineage by the same construction. **E24r-B-2** (`docs/results/p3/E24r-b14-2/`, the second
  restoration link from the aspect-14 bank's best seed: worst-of-three V 0.0, aspect 11.97, verified
  in both evaluator versions, adjudicated 2026-10-14) is the fourth own-lineage feasible design and
  is treated exactly as the others: its two exports
  (`docs/results/p3/E24r-b14-2/own-lineage/`) are shipped un-stripped under an explicit allowlist
  entry, and attribution and licence terms for the database travel with it. The banks themselves
  (boundaries reported only as counts, never redistributed), the link-1 results and the E24r-A-2
  near miss (infeasible) ship no boundary, so there is nothing to exempt for them. E25 (the union
  front) failed at launch three times (a merger bug, a container-name collision, missing anchor
  boundary files, all fixed) and then ran; it carries a correction, see `docs/results/p3/E25/STATUS.md`
  and `docs/methods.md`. All of
  E22, E22b, E22c, E23.0, E23, E23b, E23c, E24r-B-2 and E26, and the refresh set below, are
  **submittable at Matt's discretion**; as of this writing, **no submission has been made**. See `docs/results/p3/E22/STATUS.md`,
  `docs/results/p3/E22b/STATUS.md`, `docs/results/p3/E22c/STATUS.md`,
  `docs/results/p3/E23.0/RESULT.md`, `docs/results/p3/E23/STATUS.md`,
  `docs/results/p3/E23b/STATUS.md`, `docs/results/p3/E23c/STATUS.md`,
  `docs/results/p3/E26/STATUS.md`, `docs/results/p3/E24r-b14-2/STATUS.md`, the README's results
  table, and `docs/methods.md`'s cold-start and cheap-front-mapping sections for what each shows.
- **The refresh set (E24r link 3 to E34, 2026-10-22 to 2026-11-06): eleven own-lineage designs.**
  The designs are E22, E22b, E22c (the aspect-10 cold start), E24r-B-2 (aspect 11.97), E24r-A-3
  (`docs/results/p3/E24r-a85-3/`, aspect 7.37), E29 (`E29r-c100-2/`, 8.62), E30a (`E30ar-3/`, 7.64),
  E30b (`E30br-2/`, 9.43), E32 (`E32r-1/`, 10.66), E33a (`E33ar-1/`, 10.45) and E33b (`E33br-1/`,
  10.68). Each descends from the Rodriguez and Plunk database (CC BY 4.0) through a database seed and
  this repository's own restoration (`p3_final_restoration.py`), with **no leaderboard boundary
  anywhere in its lineage**; the database terms travel with them as above. E27 and E31a/E31b are
  strict-nominal restorations of designs already in the list, not new designs. The fronts grown from
  them (`docs/results/p3/E25-front/` onward, 27 `REPORT.json` files) are own lineage by the same
  construction and ship without coordinates. Their design exports
  (`docs/results/p3/E24r-a85-3/own-lineage/`, `E27/`, `E29r-c100-2/`, `E30ar-3/`, `E30br-2/`,
  `E31a-nominal/`, `E31b-nominal/`, `E32r-1/`, `E33ar-1/`, `E33br-1/`, each `own-lineage/`) are shipped
  un-stripped under the same allowlist rule in `tools/check_no_boundaries.py`. The statement above that
  P3 work is measurement only, with no coordinates redistributed, is for the davidkh-derived work
  alone: it is unchanged for those results. E34 (`docs/results/p3/E34-recheck-026/`) re-scored all
  eleven designs and 546 front members under constellaration 0.2.6 and ships residuals only. The E25
  and E25b fronts were built in E22's representation; the two E24r anchors in E25 were drawn with E22's
  major radius, so E25 was not the union it was registered as (a correction, with its guards, in
  `docs/methods.md`). **No submission has been made.**

## Pinned versions

`constellaration==0.2.6` (PyPI), which transitively pins `vmecpp==0.4.11` and
`scikit-learn==1.5.2` (pinned explicitly and defensively in `pyproject.toml`/`Dockerfile` so a
future `constellaration` bump cannot silently drag `scikit-learn`'s version along). Every result in
`docs/results/` was produced with exactly this stack. See `pyproject.toml` and `Dockerfile`.
