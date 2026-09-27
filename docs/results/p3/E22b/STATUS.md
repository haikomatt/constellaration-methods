# E22b status

**Label: Supported (goal). Repeatability link 1 of 2 (with E22c): a second own-lineage,
clean-provenance P3-feasible boundary from a different database seed.**

Runner `p3_final_restoration.py` (line-searched quadratic minimax restoration, chain rule,
worst-of-three), started from database seed `db00588_n2_s231` (start worst-of-three V 0.1743), run
unchanged from the E22 lock (E22 command, same pre-registration). Stop reason: `recovery_failed`
(the recovery's line search produced the feasible point and then could not qualify a further step,
matching E22's pattern).

**Adjudication (three evaluations, worst of three).** Point `R0:L15:t0.5`, worst-of-three
**V 0.0085, feasible under the official predicate.** The mirror residual (0.0085) and flux residual
(0.0009) sit inside the 0.01 tolerance; iota, QI and vacuum well are strictly satisfied. Raw
(fresh re-check): iota/N 0.273, log10 qi -3.646, edge mirror 0.2521, flux compression 0.9008,
vacuum well +0.0104, aspect 10.19, min_len 7.44; solo hypervolume 82.8.

**Fidelity re-check (chain rule, before the word "verified").** The exported boundary
(`own-lineage/feasible_boundary_R0L15.json`) re-scored in a fresh process in `phase1`: feasible,
identical metrics (see `recheck.log`, `recheck_phase1.json`). In `leaderboard-fidelity`
(constellaration 0.2.1, the board's version, which takes no `|iota|`): the original export fails
the iota row only (negative rotational transform), and the toroidally reflected export
(`own-lineage/feasible_boundary_R0L15_reflected.json`, the E13 procedure) is **feasible with
metrics identical to 1e-4 in both images** (see `recheck_reflected.log`,
`recheck_reflected_leaderboard-fidelity.json`). Verified.

**Cost.** 1,833 optimisation evaluations, 11 verification evaluations, 6.44 h on 12 cores; 5
accepted steps (0.1743 -> 0.0415) then the recovery line search reached feasibility; two full
Jacobians; 0 non-convergent evaluations.

**What this shows.** Run from a different database seed than E22, under the identical lock and
command, the pipeline again reaches benchmark feasibility from a cold start. Adjudicated alongside
E22c on 2026-09-27 at 04:20 as repeatability 3 of 3 (E22, E22b, E22c) -- see `docs/methods.md`'s
cold-start section and the README's headline table for the folded-in repeatability label.

**Provenance and publication.** Own lineage by construction: the seed traces to the Rodriguez and
Plunk near-axis QI database v3 (CC BY 4.0, Zenodo doi:10.5281/zenodo.18220133) with no leaderboard
boundary anywhere in its lineage -- see `PROVENANCE.md`. Submittable at Matt's discretion; no
submission has been made.

**Files.** `REPORT.json` (the runner's full adjudication record, boundary/candidate coefficient
arrays stripped by `tools/strip_boundaries.py`); `events.jsonl` (the run's event trace, no boundary
coordinates ever appear in it); `recheck.log`, `recheck_reflected.log`, `recheck_phase1.json`,
`recheck_reflected_phase1.json`, `recheck_leaderboard-fidelity.json`,
`recheck_reflected_leaderboard-fidelity.json` (the fidelity re-check evidence: two exports, each
re-scored fresh in both the `phase1` and `leaderboard-fidelity` images); `own-lineage/` (the two
boundary exports themselves, own lineage and clean provenance, exempted from
`tools/check_no_boundaries.py` by an explicit allowlist -- see that script).
