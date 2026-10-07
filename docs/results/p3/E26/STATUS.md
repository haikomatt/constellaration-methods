# E26 status

**Label: Substantial (E18b's bars). Own-lineage definition change, warm re-optimisation.**

The E18b analogue run on the own lineage instead of the davidkh lineage: from the best confirmed
own-lineage front available (E23c's confirmed 65-member front, hypervolume 94.28), the same
builder (`b3_nsga2_mhd.py`) with `--constraint-override mirror_upper=0.23` (edge mirror ratio bound
0.25 -> 0.23, the same change as E18b), 3,000 evaluations budget, confirmation under the same
override. Container `p3-e26`, cores 2-13.

**Run.** 2,992 evaluations, 44 generations, 1 non-convergent evaluation. **First tolerance-feasible
point under the new definition at evaluation 24** -- one of the carried-over anchors' own
re-evaluation, already inside the band under the tighter bound. 2,169 feasible rows in the archive.
`search-REPORT.json` is the run's own record, no boundary coordinates.

**Adjudication.** Confirmed front under the new definition: **46 members, confirmed hypervolume
91.65208552153834**, 13 of 46 strictly nominal on every row; aspect spans 10.11-10.53, min_len
spans 7.99-8.27. **Survival 0.972** of the old front's 94.28 (i.e. 91.65/94.28) -- compare E18b on
the davidkh lineage, survival 0.995. `REPORT.json` is the confirmation's full record, boundary and
candidate coefficient arrays stripped by `tools/strip_boundaries.py`; `verification.jsonl` is the
92 fresh confirmation evaluations (boundary coordinates stripped from every line).

**Label: SUBSTANTIAL** (E18b's bars: material = first confirmed feasible point under the new
definition within 1,500 evaluations -- met at evaluation 24; substantial = confirmed front >= 5
members under the new definition -- met with 46).

**Reading.** Warm re-optimisation under a changed definition costs almost nothing on the own
lineage too -- a definition change that invalidated the whole front recovers to 97.2% of the old
hypervolume within three dozen evaluations of the first feasible point. The own-lineage front is
slightly less robust to the mirror tightening than the derived one (more members end up sitting
near the mirror bound: 13 of 46 strictly nominal here versus the tighter margins E18b reported).

**Provenance and publication.** Own lineage by construction: every member traces to the Rodriguez
and Plunk near-axis QI database v3 (CC BY 4.0, Zenodo doi:10.5281/zenodo.18220133) via E22's
restoration chain and E23/E23b/E23c's front-mapping, with no leaderboard boundary anywhere in its
lineage -- see `PROVENANCE.md`. Submittable at Matt's discretion; no submission has been made.

**Files.** `search-REPORT.json` (the re-optimisation run's full record, no boundary coordinates);
`REPORT.json` (the confirmation's full record, boundary/candidate coefficient arrays stripped);
`verification.jsonl` (the fresh confirmation evaluations, boundary coordinates stripped);
`own-lineage/` (all 46 retained members' boundary exports plus their toroidally reflected
counterparts -- 92 files, own lineage and clean provenance, exempted from
`tools/check_no_boundaries.py` by an explicit allowlist -- see that script).
