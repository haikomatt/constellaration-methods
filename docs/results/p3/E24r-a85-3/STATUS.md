# E24r bank A, link 3 (aspect 7.37) status

**Label: Supported (goal). The fifth own-lineage, clean-provenance P3-feasible design, and the first at low aspect.**

The third restoration link from the aspect-8.5 bank's link-2 point (worst-of-three V 0.02116; see
`docs/results/p3/E24r-a85-2/STATUS.md`), the link-2 command unchanged, ceilings 2,400 optimisation
evaluations, 12 verification, 12 h. Container `p3-e24r-a85-3`, cores 2-13. Registered and locked
2026-10-22 (`docs/preregistration/20261022T102328Z-E24r3-E25b-lock.md`). Bars: material V <= 0.0112,
goal verified P3 feasibility with a feasible fresh re-check, null gain < 0.0025.

**Result.** Runner label `verified_p3_feasibility`, stop `feasible_confirmed`, 1,801 optimisation
evaluations, 3.4 h. Point `R10:T15:d0.00002:t0.10`, residuals [-0.276, 0.009972, 0.009995, 0.007284,
-0.167], identical on both confirmations; worst-of-three V 0.01. It is tolerance-feasible, **not
strictly nominal**: the mirror row sits 5e-6 inside the tolerance. Raw: iota/N -0.319, log10 qi
-3.465, mirror 0.2525, flux 0.907, well +0.017, **aspect 7.37, min_len 5.06**, solo hypervolume about
76.6.

**Fidelity re-check (2026-10-22), both images.** `phase1`: feasible for both exports (V 0.009995).
`leaderboard-fidelity` (constellaration 0.2.1): the reflected export is feasible with identical
residuals to six decimals; the un-reflected export fails the iota sign only. Under the frozen
benchmark version 0.2.6 both exports are feasible (E34, `docs/results/p3/E34-recheck-026/`).

Exports: `own-lineage/best_boundary.json` and `own-lineage/best_boundary_reflected.json`, shipped
un-stripped under the own-lineage allowlist (database seed, CC BY 4.0, no leaderboard boundary in the
lineage). `best_candidate.json` is the candidate record with its coordinates removed.
