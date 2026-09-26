# E22 status

**Label: Supported (goal). First own-lineage, clean-provenance P3-feasible boundary.**

Runner `p3_final_restoration.py` (line-searched quadratic minimax restoration, chain rule,
worst-of-three), started from the confirmed E21 seed `db00538_n2_s231` (worst-of-three V 0.1443).
Stop reason: `recovery_failed` (the recovery's line search produced the feasible point and then
could not qualify a further step; the ceilings did not bind).

**Adjudication (three evaluations, worst of three).** Point `R0:L5:t0.5`, residuals
`[-0.2566, -0.0071, -0.0266, -0.0082, -0.0844]`, **V = 0.0, feasible under the official predicate**
on the recorded evaluation and two fresh confirmations (identical to 1e-9; deterministic evaluator).
Raw: iota/N 0.314, log10 qi -3.525, edge mirror 0.243, flux compression 0.893, vacuum well +0.0084,
aspect 10.21, min_len 7.34; solo hypervolume 81.6. Strictly nominal on all five rows (every residual
negative, not merely within the 0.01 tolerance). A second feasible point `R0:L5:t1.0` (V 0.0006,
also feasible) sits on the same line.

**Fidelity re-check (chain rule, before the word "verified").** The exported boundary
(`own-lineage/feasible_boundary_R0L5.json`) re-scored in a fresh process in `phase1`: feasible,
identical metrics (see `recheck.log`, `recheck_phase1.json`). In `leaderboard-fidelity`
(constellaration 0.2.1, the board's version, which takes no `|iota|`): the original export fails
the iota row only (negative rotational transform), the toroidally reflected export
(`own-lineage/feasible_boundary_R0L5_reflected.json`, the E13 procedure) is **feasible with metrics
identical to 1e-4 in both images** (see `recheck_reflected.log`,
`recheck_reflected_leaderboard-fidelity.json`). Verified.

**Cost.** 1,780 optimisation evaluations (two full Jacobians of 880, 20 trials), 11 verification
evaluations, 4.10 h on 12 cores; 3 accepted steps (0.1443 -> 0.0735 -> 0.0672 -> 0.0655) then the
recovery line search from 0.0640 to 0.0 in one line model; 0 non-convergent evaluations; new-mode
(m or n 9-10) energy fraction of the accepted displacements 0.05-0.07%: the finish was bought inside
the 8x8 block.

**Pipeline cost, definition to feasible own-lineage design (E21 + E22).** Host build 1.4 h (960
seeds), canonical screen 1.4 h (960 + 24 evaluations), restoration 4.1 h (1,791 evaluations): about
7 h wall and ~2,800 canonical evaluations from the database filter to a verified feasible design.
Bars: material (0.1343) and substantial (0.030) cleared on the way; **goal** reached; chain rule:
goal -> stop chaining.

**What this shows, and what it does not yet show.** A generator with no leaderboard lineage plus
the existing restoration chain reaches P3 feasibility from a cold start in one box day.
Repeatability across seeds (E22b/E22c) and a front from this point (E23) are separate, not-yet
folded-in registrations; see `docs/methods.md` and the README's results table for their status.

**Provenance and publication.** Own lineage by construction: the seed traces to the Rodriguez and
Plunk near-axis QI database v3 (CC BY 4.0, Zenodo doi:10.5281/zenodo.18220133) with no leaderboard
boundary anywhere in its lineage -- see `PROVENANCE.md`. Unlike every other P3 result in this
repository, this design is submittable at Matt's discretion; no submission has been made. Nothing
here is public beyond this repository.

**Files.** `REPORT.json` (the runner's full adjudication record, boundary/candidate coefficient
arrays stripped by `tools/strip_boundaries.py`); `events.jsonl` (the run's event trace, no boundary
coordinates ever appear in it); `recheck.log`, `recheck_reflected.log`, `recheck_phase1.json`,
`recheck_reflected_phase1.json`, `recheck_leaderboard-fidelity.json`,
`recheck_reflected_leaderboard-fidelity.json` (the fidelity re-check evidence: two exports, each
re-scored fresh in both the `phase1` and `leaderboard-fidelity` images); `own-lineage/` (the two
boundary exports themselves, own lineage and clean provenance, exempted from
`tools/check_no_boundaries.py` by an explicit allowlist -- see that script).
