# E23 status

**Label: Material. A 53-member own-lineage front, confirmed hypervolume 89.71.**

Runner `b3_nsga2_mhd.py` (seeded population, canonical evaluation in the loop, E17's protocol),
seeded from E22's confirmed feasible point (`R0:L5:t0.5`, worst-of-three V 0.0, solo hypervolume
81.60 -- see `docs/results/p3/E22/STATUS.md`). Population 48, coefficient-space box, init scale
0.02, 12 workers. Stop reason: `max_evals`. Run: 2,976 evaluations, 62 generations, 7.2 h;
provisional hypervolume rose 81.60 -> 89.71 in one link. `search-REPORT.json` is the run's own
record (`results/p3-e23-nsga2/REPORT.json` in the source campaign, no boundary coordinates ever
appear in it).

**Adjudication (E15/E17 procedure, cap amended before any confirmation ran).** The nondominated
tolerance-feasible set had 53 members, needing 106 fresh evaluations against the registered cap of
50 confirmations (100 evaluations); the cap was raised to 110 so every nondominated member could be
confirmed -- a budget guard copied from E17 (25 members), and confirming more members can only add
retained members, so the change could not move the outcome toward the bars. Recorded before the
confirmation ran. Confirmation used 106 canonical evaluations (cap 110): **2,458 tolerance-feasible
archive rows, 53 nondominated, 53 retained** (two fresh confirmations each, complete, <= 0.01,
within 1e-4 of the archived value), **confirmed hypervolume 89.71343477929015** (reference
(1.0, 20.0), library primitive). 46 of 53 retained members are strictly nominal on every row (the
remaining 7 satisfy the official tolerance but not the strict bound). Aspect ratio spans
9.888869-10.340135 (span 0.451; the pre-registered breadth label -- widened aspect ratio -- is not
met); min_len spans 7.512630-7.880825. Best singleton member `G60:24` (solo hypervolume 86.532152,
aspect 9.909995, min_len 7.576026). The E22 seed itself is dominated by its own offspring and does
not appear in the retained front. `REPORT.json` is the confirmation's full record
(`results/p3-e23-front/REPORT.json` in the source campaign, boundary/candidate coefficient arrays
stripped by `tools/strip_boundaries.py`); `verification.jsonl` is the 106 fresh confirmation
evaluations (boundary coordinates stripped from every line -- the source file carries them, this
copy does not).

**Label: MATERIAL** (>= 5 confirmed members met; hypervolume 89.71 sits below the substantial bar
111.30 and the goal 124.23, both unmet). Placement: N = 2 throughout (the leaders' family measured
at N = 3; D1 comparison undefined for this front).

**Comparison, for the record.** The first davidkh-lineage front (E15) had 9 members at hypervolume
111.30 from a single strictly feasible point with better per-member objectives (min_len roughly
9-10 at aspect roughly 10.6). This own-lineage front has six times the members at 80% of that
hypervolume; the difference is per-member objective quality (min_len 7.51-7.88 here, well short of
E15's members), not feasibility or breadth -- neither front widened aspect ratio.

**Chain rule.** Material -> a second front link (E23b) is running from this confirmed front,
identical settings, seeded via the anchors mode (E17's protocol); see
`docs/preregistration/20260927T102152Z-E23b-lock.md` and its pop-size amendment
(`docs/preregistration/20260927T102255Z-E23b-lock-amended.md` -- the builder refuses a population
smaller than the anchor count, so pop-size moved from 48 to 56, the smallest multiple of four above
53; the first container exited at argument parsing before any evaluation ran). E23b's own result is
a separate, not-yet-folded-in registration.

**Provenance and publication.** Own lineage by construction, unchanged from E22: every member
traces to the Rodriguez and Plunk near-axis QI database v3 (CC BY 4.0, Zenodo
doi:10.5281/zenodo.18220133) via E22's restoration chain, with no leaderboard boundary anywhere in
its lineage -- see `PROVENANCE.md`. Submittable at Matt's discretion; no submission has been made.

**Files.** `search-REPORT.json` (the seeded-population run's full record, no boundary coordinates);
`REPORT.json` (the confirmation's full record, boundary/candidate coefficient arrays stripped);
`verification.jsonl` (the 106 fresh confirmation evaluations, boundary coordinates stripped);
`own-lineage/` (all 53 retained members' boundary exports plus their toroidally reflected
counterparts -- 106 files, own lineage and clean provenance, exempted from
`tools/check_no_boundaries.py` by an explicit allowlist -- see that script).
