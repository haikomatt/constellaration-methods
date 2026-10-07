# E23b status

**Label: Material. Second front link, chain continues (+3.2%).**

Runner `b3_nsga2_mhd.py` (anchors mode, E17's protocol), seeded from E23's confirmed 53-member
front (`--anchors results/p3-e23-front/REPORT.json --representation
results/p3-e22-538/start_serialised.json`). The builder refuses a population smaller than the
anchor count (53 anchors > 48), so population moved from 48 to 56, the smallest multiple of four
above 53 (amendment recorded before any evaluation ran; the first container exited at argument
parsing). Population 56, coefficient-space box, init scale 0.02, 12 workers. Run (container
`p3-e23b-r2`, 2026-09-27 11:23-18:35): 2,968 evaluations, 53 generations, stop reason `max_evals`;
provisional hypervolume rose 89.71 -> 92.61. `search-REPORT.json` is the run's own record
(`results/p3-e23b-nsga2/REPORT.json` in the source campaign, no boundary coordinates ever appear in
it).

**Adjudication (E17's protocol, cap 200 so every nondominated member could be confirmed; the
anchors are re-evaluated as generation 0, so the union with E23's members is inside the archive).**
Confirmation (18:35-18:58) used 104 canonical evaluations (cap 200): 2,667 tolerance-feasible
archive rows, **52 nondominated, 52 retained** (two fresh confirmations each, complete, <= 0.01,
within 1e-4 of the archived value), **confirmed hypervolume 92.60524959613949** (+2.89, **+3.2%**
on E23's 89.71; above the 2% chain threshold 91.51). 31 of 52 retained members are strictly nominal
on every row (E23: 46 of 53 -- the front is moving into the tolerance band, the leaders' regime).
Aspect ratio spans 9.80-10.38 (span 0.576); min_len spans 7.57-8.09. Best singleton member
`G52:40` (solo hypervolume 88.36, aspect 9.99, min_len 7.83). `REPORT.json` is the confirmation's
full record (`results/p3-e23b-front/REPORT.json` in the source campaign, boundary/candidate
coefficient arrays stripped by `tools/strip_boundaries.py`); `verification.jsonl` is the 104 fresh
confirmation evaluations (boundary coordinates stripped from every line).

**Label: MATERIAL** (>= 5 confirmed members met; hypervolume 92.61 sits below the substantial bar
111.30 and the goal 124.23, both unmet). Per-link gain, falling as expected for a fixed seed and
representation: +9.9% (E23), +3.2% (E23b).

**Chain rule.** Material, and the gain cleared the 2% continuation threshold -> a third front link
(E23c) launched from this confirmed front, identical settings; see
`docs/results/p3/E23c/STATUS.md`.

**Provenance and publication.** Own lineage by construction, unchanged from E22/E23: every member
traces to the Rodriguez and Plunk near-axis QI database v3 (CC BY 4.0, Zenodo
doi:10.5281/zenodo.18220133) via E22's restoration chain and E23's front-mapping, with no
leaderboard boundary anywhere in its lineage -- see `PROVENANCE.md`. Submittable at Matt's
discretion; no submission has been made.

**Files.** `search-REPORT.json` (the anchors-mode run's full record, no boundary coordinates);
`REPORT.json` (the confirmation's full record, boundary/candidate coefficient arrays stripped);
`verification.jsonl` (the 104 fresh confirmation evaluations, boundary coordinates stripped);
`own-lineage/` (all 52 retained members' boundary exports plus their toroidally reflected
counterparts -- 104 files, own lineage and clean provenance, exempted from
`tools/check_no_boundaries.py` by an explicit allowlist -- see that script).
