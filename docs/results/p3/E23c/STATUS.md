# E23c status

**Label: Material. Third front link, chain stops (+1.81%, below the 2% rule).**

Runner `b3_nsga2_mhd.py` (anchors mode, identical to E23b), seeded from E23b's confirmed 52-member
front (`--anchors results/p3-e23b-front/REPORT.json --representation
results/p3-e22-538/start_serialised.json`), population 56, coefficient-space box, init scale 0.02,
12 workers. Run (container `p3-e23c`, 2026-09-27 19:00/21:20 to 2026-09-28 01:37): 2,968
evaluations, 53 generations, stop reason `max_evals`; provisional hypervolume rose 92.61 and higher
(confirmed below). `search-REPORT.json` is the run's own record (`results/p3-e23c-nsga2/REPORT.json`
in the source campaign, no boundary coordinates ever appear in it).

**Adjudication (same protocol, cap 200; run by `scripts/p3_front_chain_runner.sh`, the registered
host runner, 01:37-01:54).** 104 canonical evaluations used (cap 200): **65 nondominated, 65
retained** (two fresh confirmations each, complete, <= 0.01, within 1e-4 of the archived value),
**confirmed hypervolume 94.28361300367145** (+1.71, **+1.81%** on E23b's 92.61 -- below the 2%
chain threshold 94.46). 44 of 65 retained members are strictly nominal on every row. Aspect ratio
spans 9.77-10.49 (span 0.72); min_len spans 7.64-8.24. `REPORT.json` is the confirmation's full
record (`results/p3-e23c-front/REPORT.json` in the source campaign, boundary/candidate coefficient
arrays stripped by `tools/strip_boundaries.py`); `verification.jsonl` is the fresh confirmation
evaluations (boundary coordinates stripped from every line).

**Label: MATERIAL** (>= 5 confirmed members met; hypervolume 94.28 sits below the substantial bar
111.30 and the goal 124.23, both unmet).

**Chain stop.** `scripts/p3_front_chain_runner.sh` logged `chain stop: gain < 2%` at 01:54:41 and
exited; no E23d launched -- the chain ended on the registered rule, not the clock (the hard stop,
2026-09-30 04:00, was never reached). Per-link gain, falling as expected for a fixed seed and
representation: +9.9% (E23), +3.2% (E23b), +1.81% (E23c).

**Chain summary (E23 -> E23b -> E23c, one seed, one representation, ~17 h total wall across three
links).** Own-lineage hypervolume **81.60 (E22 baseline) -> 89.71 -> 92.61 -> 94.28**, i.e. +15.5%
total from the single feasible seed; member count **2 -> 53 -> 52 -> 65**. Against the
davidkh-lineage fronts: E15 111.30 (9 members), E17 124.23 (25 members) -- substantial not reached;
the gap is per-member objective quality (min_len 7.6-8.2 here versus 9-10 at the same aspect on the
davidkh lineage), which the population builder narrows slowly (+1.7 per link at the end).

**What this chain has not tested.** Whether 94.28 (or the ~+15-16% pattern) is a property of this
one E22 seed, or general across the generator's other confirmed cold-start seeds (E22b
`db00588_n2_s231`, E22c `db00397_n2_s231`) -- no front-repeatability run is registered for those
two. Also untested: whether the plateau is a pop-size/representation ceiling (unchanged pop 56 and
coefficient box across all three links) or a property of the design itself.

**Provenance and publication.** Own lineage by construction, unchanged from E22/E23/E23b: every
member traces to the Rodriguez and Plunk near-axis QI database v3 (CC BY 4.0, Zenodo
doi:10.5281/zenodo.18220133), with no leaderboard boundary anywhere in its lineage -- see
`PROVENANCE.md`. Submittable at Matt's discretion; no submission has been made.

**Files.** `search-REPORT.json` (the anchors-mode run's full record, no boundary coordinates);
`REPORT.json` (the confirmation's full record, boundary/candidate coefficient arrays stripped);
`verification.jsonl` (the fresh confirmation evaluations, boundary coordinates stripped);
`own-lineage/` (all 65 retained members' boundary exports plus their toroidally reflected
counterparts -- 130 files, own lineage and clean provenance, exempted from
`tools/check_no_boundaries.py` by an explicit allowlist -- see that script).
