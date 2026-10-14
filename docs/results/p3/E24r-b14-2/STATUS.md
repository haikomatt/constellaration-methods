# E24r bank B, link 2 (aspect ~12.0) status

**Label: Supported (goal). The fourth own-lineage, clean-provenance P3-feasible design, and the
first off the aspect-10 line.**

Runner `p3_final_restoration.py` (the E22 command, unchanged), started from link 1's adjudicated
best point (worst-of-three V 0.0143 at aspect 12.0; see `docs/results/p3/E24r-b14/STATUS.md`), which
itself came from bank B's best confirmed database seed `db00537_n2_s231`. Container
`p3-e24r-b14-2`, cores 2-13. Chain rule: material bar = start V - 0.010 = 0.0043; goal benchmark
feasibility. Runner label `verified_p3_feasibility`, stop reason `feasible_confirmed`.

**Adjudication (three evaluations, worst of three).** Point `P:L1:t0.5`, residuals
[-0.262, -0.039, -0.101, -0.011, -0.074], worst-of-three **V 0.0, strictly nominal on all five
rows**. Two further adjudicated points are also feasible (V 0.0033 and 0.0059, both inside the 0.01
tolerance). Raw: iota/N 0.316, log10 qi -3.638, edge mirror 0.225, flux compression 0.890, vacuum
well +0.0074, **aspect 11.97, min_len 8.72**, solo hypervolume 77.99.

**Fidelity re-check (chain rule, before the word "verified"), 2026-10-14.** Both exports
(`own-lineage/feasible_boundary_P1L1.json` and its toroidal reflection
`own-lineage/feasible_boundary_P1L1_reflected.json`) were re-scored with `score_mhd_stable.py` in
fresh processes in both images. In `phase1`: feasible for both exports, V -0.0111 (see
`recheck_phase1.json`, `recheck_phase1.log`). In `leaderboard-fidelity` (constellaration 0.2.1, the
board's version, which takes no `|iota|`): the un-reflected export fails the iota sign only, and the
reflected export is feasible with metrics identical to 1e-4 to the `phase1` run (see
`recheck_leaderboard-fidelity.json`, `recheck_leaderboard-fidelity.log`). Verified.

**Cost.** 882 optimisation evaluations, 11 verification evaluations, 1.34 h on 12 cores; no
accepted step was needed (the recovery's first line search produced the feasible point). This is
the cheapest of the four feasible designs: 1.3 h against 4.1 to 6.4 h for E22, E22b and E22c, from a
seed that had already been through one link (link 1, 3.4 h).

**What this shows.** The generator's operating aspect matters. Seeded from the aspect-14 bank
(canonical aspect 11.9 to 13.7) and run through the identical chain, the pipeline reaches
benchmark feasibility at aspect 12 with a gradient scale length (min_len 8.72) above every member
of the aspect-10 own-lineage front (maximum 8.24). One seed, one bank, two links; it is the first
design off the aspect-10 line, not a front, and its solo hypervolume (77.99) is below the
aspect-10 designs' 81.6 to 83.1. The conclusion that the aspect-14 bank is better for P3 rests on
one design.

**Provenance and publication.** Own lineage by construction: the seed traces to the Rodriguez and
Plunk near-axis QI database v3 (CC BY 4.0, Zenodo doi:10.5281/zenodo.18220133) with no leaderboard
boundary anywhere in its lineage; see `PROVENANCE.md`. Submittable at Matt's discretion; no
submission has been made.

**Files.** `REPORT.json` (the runner's full adjudication record, boundary and candidate
coefficient arrays stripped by `tools/strip_boundaries.py`); `events.jsonl` (the run's event
trace); `best_recheck.json` (the runner's own re-check of the best point); `recheck_phase1.*` and
`recheck_leaderboard-fidelity.*` (the fidelity re-check evidence); `own-lineage/` (the two
boundary exports, exempted from `tools/check_no_boundaries.py` by an explicit allowlist).
