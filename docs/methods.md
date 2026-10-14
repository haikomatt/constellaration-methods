# Methods

Every claim below carries its evidence and a grade in the campaign's own discipline:
**Supported** (pre-registered, locked, canonically confirmed, bar met), **Refuted** (same
discipline, bar missed or labelled "null"), **Measured** (locked measurement probe, no bar),
**Open** (not tested). Provenance note first: every feasible P3 design this repo holds descends
from `davidkh`'s public P2 design; nothing is submittable without reuse terms; all P3 results are
measurement and method.

## P3 (`MHDStableQIStellarator`): what this method can say, and what it refutes

### "Methods that let us re-optimise quickly under a changed definition"

**Supported (E18b, 2026-09-24).** Definition change: edge mirror ratio bound 0.25 -> 0.23, which
made all 25 members of a confirmed front infeasible by 0.027-0.096.

Warm start from that front with a seeded population (48, canonical evaluation in the loop): first
confirmed feasible point under the new definition at evaluation 165; confirmed 12-member front
under the new definition at hypervolume 123.64, i.e. 99.5% of the old front's 124.23, within 2,976
evaluations (~8 h on 12 cores).

Bars (material: first feasible <= 1,500; substantial: >= 5-member confirmed front) both met. One
change, one front, one size; the number is a first data point, not a law.

**Supported, own lineage (E26, 2026-10-03): the same definition change, repeated on the own
lineage.** The E18b analogue, run on E23c's confirmed 65-member own-lineage front (hypervolume
94.28) instead of the davidkh-lineage front: the identical mirror 0.25 -> 0.23 change, warm-started
with the same seeded-population builder under the new definition, 2,992 evaluations, 44
generations, 1 non-convergent. First tolerance-feasible point under the new definition at
**evaluation 24** (an anchor's own re-evaluation already inside the band); confirmed front **46
members, hypervolume 91.65**, 13 of 46 strictly nominal, aspect 10.11-10.53, min_len 7.99-8.27.
**Survival 0.972** of the old front's 94.28 -- compare E18b's 0.995 on the davidkh lineage. Label:
**substantial** (E18b's bars: material -- first feasible <= 1,500 -- met at evaluation 24;
substantial -- confirmed front >= 5 members -- met with 46). One change, one front, repeated once
on a second lineage; the same caveat as E18b applies: one change, one front, one size, not a law.
Reading: warm re-optimisation under a changed definition costs almost nothing on the own lineage
either, though the own-lineage front is slightly less robust to the mirror tightening than the
derived one (more members end up near the mirror bound). See docs/results/p3/E26/STATUS.md.

**Refuted for sampler seeds and first-order constructions (E18a, E19, E20/E20b).** From our own
near-axis seed at confirmed V 0.0536: a seeded population could not beat the seed in 2,976
evaluations; a second-order local model with measured off-diagonal curvature (12 directions, 156
probes) and its diagonal control both predicted no descending step, measurements agreeing to 1e-4,
confirmed bests 0.0519 and 0.0516. Registered comparison label: off-diagonal curvature does not
matter. The construction survey behind those seeds is itself refuted at every aspect tried (E20,
E20b -- see "What is refuted" below): a first-order near-axis generator cannot supply a P3-range
twist that is also an equilibrium.

**Supported, repeated (3 of 3), from a second-order near-axis database seed (E21 + E22/E22b/E22c):
first own-lineage, clean-provenance P3-feasible boundary, then repeated on two further seeds.**
E21 tries a different generator instead: a public, CC BY
4.0, second-order near-axis QI database (Rodriguez and Plunk, arXiv:2601.08400; dataset
doi:10.5281/zenodo.18220133 -- see PROVENANCE.md and docs/external-data.md), filtered for P3-shaped
constructions and completed with the authors' own MHD magnetic-well transform. The route was
justified by a Phase 0 screen: 24 database seeds at aspect 8-9, evaluated canonically, converged in
VMEC at P3-range twist 24 times out of 24 (our own first-order sampler never does), with iota
satisfied 24/24 and mirror satisfied 19/24 -- only the well, flux and QI rows failed, and the well
is a construction parameter of the database with a lever the same Phase 0 screen demonstrated
canonically. A registered screen of 960 selected seeds (`construction.nae_db_seed_screen`,
docs/results/p3/E21/) confirmed a best seed at worst-of-three V = 0.1443, with iota, mirror and
vacuum well satisfied at confirmation and canonical aspect ratio 10.41 -- inside the pre-registered
material bar (confirmed V <= 0.30 with those three rows satisfied at aspect <= 12). QI and flux
compression remained unsatisfied at the seed.

E22 then ran the campaign's proven restoration chain (`p3_final_restoration.py`, the E7/E19b
protocol, line-searched quadratic minimax, chain rule, worst-of-three) from that confirmed seed:
worst-of-three V fell from 0.1443 to **0.0, strictly feasible on all five rows** (residuals
[-0.2566, -0.0071, -0.0266, -0.0082, -0.0844]; raw iota/N 0.314, log10 qi -3.525, edge mirror 0.243,
flux compression 0.893, vacuum well +0.0084, aspect 10.21, min_len 7.34, solo hypervolume 81.6),
confirmed on the recorded evaluation and two fresh confirmations (identical to 1e-9). Fidelity
re-check: the exported boundary re-scores identically in a fresh `phase1` process; in
`leaderboard-fidelity` (constellaration 0.2.1, the board's version) the original export fails the
iota row only (negative rotational transform), and the toroidally reflected export (the E13
procedure) is feasible with metrics identical to 1e-4 in both images. Verified in both evaluator
versions. Cost: 1,780 optimisation evaluations, 11 verification evaluations, 4.10 h on 12 cores; 3
accepted steps (0.1443 -> 0.0735 -> 0.0672 -> 0.0655) then a recovery line search from 0.0640 to
0.0 in one line model. Bars met: material (0.1343) and substantial (0.030) cleared on the way;
**goal** (benchmark-feasible on all three evaluations) reached; the chain rule stops chaining at
goal. This is the first own-lineage, clean-provenance P3-feasible design the campaign holds --
unlike every davidkh-lineage P3 result, it descends from a CC BY 4.0 database with no leaderboard
boundary anywhere in its lineage, and is submittable at Matt's discretion (see PROVENANCE.md); no
submission has been made.

**Repeatability (E22b, E22c; adjudicated 2026-09-27): 3 of 3.** Two further database seeds ran the
identical lock and command. E22b (seed `db00588`, start worst-of-three V 0.174): best point
`R0:L15:t0.5`, worst-of-three **V 0.0085, feasible** (mirror residual 0.0085 and flux 0.0009 inside
the 0.01 tolerance, iota/QI/well strict); raw aspect 10.19, min_len 7.44, solo hypervolume 82.8;
1,833 optimisation + 11 verification evaluations, 6.44 h. E22c (seed `db00397`, start
worst-of-three V 0.191, a QI- and flux-binding start): best point `R0:L16:t0.5`, worst-of-three
**V 0.0049, feasible** (QI residual 0.0049 inside tolerance, iota/mirror/flux/well strict); raw
aspect 10.30, min_len 7.57, solo hypervolume 83.1; 1,852 optimisation + 11 verification evaluations,
4.08 h. Both fidelity-rechecked the same way as E22: feasible fresh in `phase1`; in
`leaderboard-fidelity` the un-reflected export fails the iota sign only and the reflected export is
feasible with metrics identical to 1e-4 in both images. The pre-registered repeatability label --
"the pipeline reaches feasibility from a cold start" -- is met 3 of 3 across three distinct
database seeds, each QI- or flux-binding at the seed, each in one link (4.1 / 6.4 / 4.1 h; 1,780 /
1,833 / 1,852 optimisation evaluations), with the finish coming from the recovery's first line
search in all three. All three designs are two-period, aspect 10.2-10.3, gradient scale length
7.3-7.6, and sit within 0.03 of every bound except iota and well: the same constraint corner the
leaderboard designs occupy, reached from a different family. A front from this point (E23) is
material, confirmed hypervolume 89.71 with 53 members -- see the cheap-front-mapping section below
and docs/results/p3/E22b/STATUS.md, docs/results/p3/E22c/STATUS.md and docs/results/p3/E23/STATUS.md.

**Pipeline cost: what a "new definition to good designs" costs here.** Proxima's question was how
quickly this method reaches a good design under a changed or fresh definition. The honest answer
for a cold start, definition to a feasible own-lineage design (E21 + E22/E22b/E22c): host build
1.4 h (960 seeds), canonical screen 1.4 h (960 + 24 evaluations) -- a shared 2.8 h generator step --
then restoration 4.1 / 6.4 / 4.1 h (1,791 / 1,844 / 1,863 evaluations, optimisation plus
verification) for E22/E22b/E22c respectively: about 6.9-9.2 h wall and ~2,775-2,850 canonical
evaluations from the database filter to a verified feasible design, repeated 3 of 3 across
independent database seeds. Caveat, stated plainly: this is the fixed P3 definition
(not a changed one -- E18b above is the changed-definition number), and the design reached each
time is a low-scoring but feasible one: aspect 10.2-10.3 and min_len 7.3-7.6 sit near the bottom of
the board's range, not a competitive front member.

**Where the generator's sweet spot sits (E24, E24r; 2026-10-02/04).** The own-lineage front sits
entirely at aspect ~9.8-10.5 because every cold-start seed used so far (E21/E22/E22b/E22c) came
from the same aspect-10.4 bank. Two further banks were built from the same selection and screened
canonically: aspect 8.5 (901 seeds, canonical achieved 7.2-8.4) and aspect 14.0 (960 seeds,
canonical achieved 11.9-13.7). Both are material (seeds exist with confirmed V <= 0.30 and
iota/mirror/well satisfied): bank A best confirmed 0.2761 (7 of 901 seeds clear the bar; QI tail
empty at this aspect, flux tail 16% <= 0.9); **bank B best confirmed 0.1136, with 214 of 960 seeds
clearing the bar** (E21 at aspect 10.4 had 151 of 960) -- the generator's best operating aspect for
P3 reads as **~12, not ~10**. One restoration link ran from each bank's best seed, the E22 command
unchanged: bank A reached worst-of-three **V 0.0751 at aspect 7.4**, infeasible (material by the
chain rule, not substantial); bank B reached worst-of-three **V 0.0143 at aspect 12.0, min_len
8.80**, infeasible but **substantial** (<= 0.030) -- and the objective value reached there (min_len
8.80) is above every member of the own-lineage front (max 8.24) and in the range of the
davidkh-lineage E15 front at the same aspect. Neither link reached feasibility in one pass. A
second restoration link from each bank's adjudicated best point (identical command, material bar =
the link's V minus 0.010) was adjudicated 2026-10-14, and it is the breadth result:

- **Bank B, link 2: feasible (E24r-B-2).** Point `P:L1:t0.5`, worst-of-three **V 0.0, strictly
  nominal on all five rows**: iota/N 0.316, log10 qi -3.638, edge mirror 0.225, flux compression
  0.890, vacuum well +0.0074, **aspect 11.97, min_len 8.72**, solo hypervolume 77.99. 882
  optimisation evaluations, 1.34 h (link 1 took 3.4 h). Fidelity re-check: feasible in
  `phase1` for both exports (V -0.0111); in `leaderboard-fidelity` the reflected export is feasible
  with identical metrics and the un-reflected export fails the iota sign only. The fourth
  own-lineage feasible design and the first off the aspect-10 line, with a scale length above every
  member of the aspect-10 front. Its solo hypervolume (77.99) is below the aspect-10 designs'
  81.6 to 83.1, so this is breadth in aspect, not a better score.
- **Bank A, link 2: near miss (E24r-A-2).** Best confirmed worst-of-three **V 0.0212 at aspect
  ~7.4**, infeasible, with QI, mirror and flux tied at 0.021; 1,843 evaluations, 4.08 h. Substantial
  but not feasible; the chain rule would fund a link 3 (bar 0.0112), which was not launched.

Reading, with its limits: the generator's best operating aspect for P3 reads as about 12 to 14;
feasibility was reached at 12 and has not yet been reached at 7.4. That rests on one design from
bank B and one near miss from bank A, each a single seed; it does not show that low-aspect designs
are out of reach, only that two links were not enough from the best aspect-8.5 seed. The union
front over all four own-lineage anchors (E25) has not run. See
docs/results/p3/E24-a85/STATUS.md, docs/results/p3/E24-b14/STATUS.md,
docs/results/p3/E24r-a85/STATUS.md and docs/results/p3/E24r-b14/STATUS.md.

### "Methods that can map Pareto fronts cheaply"

**Supported with a stated limit (E15-E17, 2026-09-18/19).** From one strictly feasible point, a
bounded front builder confirmed 35 members at hypervolume 120.05 in 11 h, and a seeded population
raised that to 124.23 with 25 members in 9 h, every member confirmed twice in fresh processes. The
limit: hypervolume on this board is bought by aspect-ratio spread (leaders span aspect 5-24; ours
spans 0.56), and one member accounts for 96% of our hypervolume. Neither method widened aspect; the
registered breadth label was missed. Cheap front mapping near a known design: yes. Cheap front
mapping across the design space: not shown.

**Material, own lineage, the same limit (E23, 2026-09-27).** The identical seeded-population
protocol from E22's own-lineage feasible point (2,976 evaluations, 62 generations, 7.2 h) produced
2,458 tolerance-feasible archive rows, 53 nondominated, all 53 retained after two fresh
confirmations each: confirmed hypervolume 89.71, up from the 81.60 seed (+9.9%). 46 of 53 are
strictly nominal on every row; aspect spans 9.89-10.34 (span 0.451, breadth label not met); min_len
spans 7.51-7.88; best singleton `G60:24` solo hypervolume 86.53. Label: **material** (>= 5
confirmed members met; the substantial bar 111.30 and goal 124.23 unmet). For the record: the
first davidkh-lineage front (E15) had 9 members at hypervolume 111.30 from a single strictly
feasible point with better objectives (min_len roughly 9-10 at aspect roughly 10.6); this
own-lineage front has six times the members at 80% of the hypervolume, the difference being
per-member objective quality, not feasibility or breadth. Same limit as E15-E17: cheap front
mapping near a known design, not across the design space -- neither davidkh-lineage nor
own-lineage front-mapping has widened aspect. Chain rule: material -> a second front link (E23b) is
running from E23's confirmed front, identical settings. See docs/results/p3/E23/STATUS.md.

**Chain close (E23b, E23c; adjudicated 2026-09-27/30).** Two further links ran under the
registered chain rule (each link seeded via anchors mode from the previous link's confirmed front,
identical pop 56 and coefficient box, stop if the gain falls below 2%). E23b: 2,968 evaluations, 53
generations; 2,667 tolerance-feasible archive rows, 52 nondominated, 52 retained, confirmed
hypervolume **92.61** (+2.89, +3.2% on E23's 89.71, above the 91.51 continuation threshold); 31 of
52 strictly nominal; aspect 9.80-10.38, min_len 7.57-8.09. E23c: identical settings from E23b's
confirmed front; 2,968 evaluations, 53 generations; 65 nondominated, 65 retained, confirmed
hypervolume **94.28** (+1.71, +1.81% on E23b's 92.61, below the 94.46 continuation threshold); 44
of 65 strictly nominal; aspect 9.77-10.49, min_len 7.64-8.24. **The chain stopped on this
pre-registered 2% rule** (`scripts/p3_front_chain_runner.sh` logged `chain stop: gain < 2%` and
exited; no E23d launched) -- not on the clock (the hard stop, 2026-09-30 04:00, was never reached).
Both links are labelled **material** (the substantial bar 111.30 and goal 124.23 remain unmet).
Chain summary, one seed and one representation across three links, ~17 h total wall: own-lineage
confirmed hypervolume **81.60 (E22 baseline) -> 89.71 -> 92.61 -> 94.28**, i.e. +15.5% total; member
count 2 -> 53 -> 52 -> 65. Per-link gain fell as expected for a fixed seed and representation:
+9.9%, +3.2%, +1.81%. Not tested: whether this pattern is a property of the E22 seed specifically,
or general across the generator's other confirmed cold-start seeds (E22b, E22c) -- no
front-repeatability run is registered for those two; and whether the plateau is a pop-size/
representation ceiling (unchanged across all three links) or a property of the design itself. See
docs/results/p3/E23b/STATUS.md and docs/results/p3/E23c/STATUS.md.

### "How diverse can near-optimal solutions be?"

**Measured (D1, locked 2026-09-23).** 30,444 feasible boundaries (phanerozoic's 45 public fronts,
davidkh, canyon-ant, ours), symmetry-quotiented physical-space RMS distance in minor radii,
near-equivalent = both feasible and within 0.05 on both objectives. Within one entrant's history:
1,003,439 pairs, 0% as far apart as random feasible pairs (d_rand 0.388): rigid. Across entrants:
38,613 pairs, 97.9% as far apart as random, entirely between two families: the leaderboard family
(phanerozoic, davidkh's P3 entry, canyon-ant; pairwise 0.015-0.048, consistent with shared ancestry
or convergence) and ours (0.44-0.46 from all of them). Answer: yes, at least two geometrically
distinct families reach the same metrics at aspect ~10.2-10.8; within a family the map from metrics
to shape is tight. A designer choosing among equals has a real choice between families, and none
within one. Cost proxies (surface curvature, high-mode spectral energy) are reported as proxies; no
coil evaluator.

### "If you are willing to write up or open-source your pipeline"

**Available -- this repository.** Pre-registration snapshots committed at every lock; canonical
worst-of-three adjudication in the runnable scripts, not in the agent; problem definition as
configuration (`--constraint-override`); seeded population front-mapping with fresh-process
confirmation; the front-from-archive confirmation tool; galleries and hypervolume-vs-evaluation
curves; the diversity probe.

### Their reference point (~178 internal HVI, beaten by the board)

**Measured.** Board top three at 2026-09-18: 207.53 (668-member front), 205.82 (43 members),
187.91. Our confirmed best 124.23. The gap is breadth, not per-member quality: each leader's best
single member is worth ~120-126, like ours.

## Supporting evidence

- E13 (2026-09-18): first strictly feasible P3 boundary (all five residuals negative), reproduced
  in two library versions; the seed for everything above. Cause of the breakthrough not isolated
  (fresh model at a new point vs the corrections alone; matched test never funded).
- Surrogate for the flux row (E10/E13 held-out): fails its own bar (sign agreement 7/9 < 0.80);
  the reviewer's rms convention could not be reproduced by any of 96 conventions; verdict
  invariant.
- Evaluation cost: 7x7 median 100 s, 8x8 median 81 s per canonical evaluation on one core; 12
  workers.

## What is refuted, and where the open problem is

- Own-lineage cold start: three method families null at seed 176 (population; diagonal
  second-order; second-order with measured cross-curvature), best confirmed 0.0516, never
  feasible. Seeds 49 and 243 stalled earlier (0.142; 0.49 with the mirror traded away). All three
  seeds came from a near-axis construction survey (405 sampled, 61 with positive well, 0
  feasible).
- Where the seeds sit (measured 2026-09-25 with the D1 metric): the near-axis seed 176 is 0.62
  minor radii from the leaders' family and 0.82 from ours; after the whole restoration chain moved
  it 1.17 minor radii it sits 0.40 from the leaders and 0.76 from ours, i.e. as far from every
  known feasible family as an unrelated design. The chain converged to a worst-residual minimum in
  a region no feasible design is known to occupy. Local methods cannot fix a start in the wrong
  region.
- Off-diagonal curvature: measured, 1-5x the diagonal in the chain directions, and irrelevant to
  the stall.
- Front breadth: two methods, no widening in aspect (E16, E17).
- Own-lineage near-axis construction (E20, E20b, 2026-09-25): the proxy search works
  (constructions with iota/nfp 0.73, mirror 0.14, QI residual 1e-4 at aspect 8.6 exist in the
  near-axis space), but at minor radius 0.1 every construction with P3-range twist fails to
  converge as an equilibrium and every convergent one misses mirror by 2-4x and QI by two orders.
  **Refuted at this aspect ratio.** At aspect ~14 and ~20 (E20b), constructions with P3-range
  twist never form equilibria at any aspect tried; the low-twist ones converge and their boundary
  QI improves with aspect but stays 1.5-2 decades short of the bound at aspect 20, and flux
  compression, which the construction cannot see, is 1.1-1.9 against 0.9 on every one. **Refuted
  at every aspect tried:** own-lineage P3 feasibility needs a different construction order or
  generator (second-order near-axis with the well and a flux-aware target, or a generator trained
  on the feasible families).

## P1 (`GeometricalProblem`): a representational trap, and the discipline that caught it

Score **0.97522**, own lineage (rotating-ellipse analytic seed, no leader boundary -- see
[PROVENANCE.md](../PROVENANCE.md)). The method is a direct ALM/Nevergrad descent
(`engines.alm_ngopt_optimizer`) with canonical-in-loop scoring, no LLM in this repository's
version.

Building the P1 search surfaced a specific failure mode worth recording on its own. A search
pinned to a coarse Fourier truncation (4x4) can drive its cheap-fidelity score down into
non-physical corners of an impoverished representation, reporting confident progress the whole
way, while the real low-elongation solutions (which need a richer 8x8 truncation) are simply
unrepresentable in that space. Solver effort cannot fix a representation gap: different solver
versions, more iterations, relaxed tolerances, and two continuation ladders all agreed the wall was
real, because none of them touch the representation. Freeing the mode budget to 8x8 flipped 100 of
101 previously-"non-physical" designs to confirmed-feasible with the same code and the same
discipline -- only how the shape was described had changed.

**Honesty guard, verbatim:** "I did not beat the leaderboard. Warm-starting my search from a top
design let it *refine* that design (the best it found sat under 1% away from the seed) -- it did
not discover its own shape. [...] this is a finding about *representation and proxy-trust*, not a
defect in the benchmark. The benchmark is well-built; the dataset is valid high-fidelity data; the
coarse boundary parameterization is a reasonable default."

## P2 (`SimpleToBuildQIStellarator`): the QI frontier audit

Score **0.5818**, own lineage (climb-lifted 4x4 -> 8x8 design, c8 climb bank call_index 30548 --
see [PROVENANCE.md](../PROVENANCE.md)). "Reaching QI feasibility" (canonical log10 qi <= -4) is a
feasibility step, not the score; the leaderboard ranks the coil-simplicity objective
(`minimum_normalized_magnetic_gradient_scale_length`) among feasible designs. This campaign's
own-lineage pipeline (near-axis database -> target generation -> DESC descent -> canonical
confirmation) moved the confirmed frontier from -2.14 to -2.541 canonical log10 qi, and a
deliberate due-diligence pass over the dataset, paper, and upstream code -- run before committing
to a larger compute campaign -- caught that some of that -2.54 wall was self-inflicted (a skipped
stock-recipe verbatim run descends far past it) and some of it is a genuine, honestly-measured
frontier (the corrected recipe still didn't reach -4 in 3.5 hours on a 16-core box against a paper
budget of 96 vCPUs).

**Honesty guard, verbatim:** "We did not plant a flag at -4. We produced something more durable: a
fully own-lineage pipeline [...] a precise map of what does and does not set the QI wall, a
demystification of how the leaderboard frontier is actually reached, and the intellectual honesty
to catch our own divergence with a deliberate audit rather than defend a wall."
