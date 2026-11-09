# Methods

Every claim below carries its evidence and a grade in the campaign's own discipline:
**Supported** (pre-registered, locked, canonically confirmed, bar met), **Refuted** (same
discipline, bar missed or labelled "null"), **Measured** (locked measurement probe, no bar),
**Open** (not tested). Provenance note first: the P3 designs and fronts that descend from `davidkh`'s public P2
design are measurement and method only, and nothing derived from it is submittable without reuse
terms. The eleven own-lineage P3-feasible designs (E22 onward) descend from a public CC BY 4.0
database and our own restoration, with no leaderboard boundary in their lineage (PROVENANCE.md).

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
front over the own-lineage anchors (E25) did run, and carries a correction (below). See
docs/results/p3/E24-a85/STATUS.md, docs/results/p3/E24-b14/STATUS.md,
docs/results/p3/E24r-a85/STATUS.md and docs/results/p3/E24r-b14/STATUS.md.

### "Methods that reach feasibility from a database seed at a chosen aspect"

**Supported, six of six (E24, E29, E30a, E30b, E32, E33; 2026-10-02 to 2026-11-05): the
bank-then-restore pipeline.** The cold-start result above was one bank at one aspect. The same chain
now runs as a single pipeline, with the requested aspect as its only argument
(`scripts/p3_bank_chain_runner.sh TAG ASPECT MAX_ASPECT`):

1. **Bank.** Rebuild the E21 selection with the authors' MHD completion at W +0.02 and a geometry gate
   (rmin/r00 >= 0.5), exported at the requested aspect by the boundary's own estimate; screen every
   seed canonically and confirm the best 12 at worst of three. Bars: null (no seed), material
   (confirmed V <= 0.30 with iota, mirror and well satisfied, canonical aspect no higher than the
   stated maximum), substantial (<= 0.10), goal (feasible).
2. **Restoration chain.** The E24r command unchanged from the bank's best confirmed seed, then each
   next link from the last link's adjudicated best, at most four links, continuing only while a link
   is material (V falls by at least 0.010) and stopping on verified feasibility.
3. **Verification.** A feasible point is re-checked in both evaluator images through the reflected
   export before it is called verified (and, from E34, under the frozen benchmark version).
4. **Front.** If a link reaches feasibility, a front grown from that design alone, in its own
   representation (next section).

Six banks were run through it. Every bank was material or better, every chain reached a verified
feasible design, and each took one to three restoration links of a few hours:

| Bank (requested aspect) | Best confirmed seed V | Links | Design aspect | min_len | Solo HV | Final link |
|---|---|---|---|---|---|---|
| E24 A (8.5) | 0.2761 | 3 | 7.37 | 5.06 | about 76.6 | 3.4 h, tolerance-feasible |
| E24 B (14) | 0.1136 | 2 | 11.97 | 8.72 | 77.99 | 1.3 h, strictly nominal |
| E29 (10.0) | 0.2006 | 2 | 8.62 | 6.17 | 81.59 | 1.8 h, tolerance-feasible |
| E30a (9.0) | 0.2566 | 3 | 7.64 | 5.31 | 77.98 | 3.1 h, tolerance-feasible |
| E30b (11.0) | 0.1801 | 2 | 9.43 | 6.70 | 81.43 | 1.3 h, strictly nominal |
| E32 (12.5) | 0.1241 | 1 | 10.66 | 7.60 | 80.30 | 2.7 h, tolerance-feasible |

Two further seeds from E32's bank (E33a `db00588_n2_s231`, E33b `db00538_n2_s364`) also reached
feasibility in one link each: aspect 10.45 (min_len 7.72, solo 83.36, the highest of any design) and
10.68 (min_len 7.56, solo 79.80). With the three cold-start designs of E22, E22b and E22c, that is
**eleven own-lineage P3-feasible designs, aspect 7.37 to 11.97, every one feasible under
constellaration 0.2.6** (E34 below). Banks come out below the aspect requested (8.5 gave 7.4, 14 gave
12.0, 10 gave 8.6, 9 gave 7.6, 11 gave 9.4, 12.5 gave 10.7), so the request is a lever, not a target.

Limits, stated plainly. Six of six is six banks, each with one chain from its best seed; it supports
"the pipeline reaches feasibility when aimed at an aspect in 8.5 to 14", not a rate. The designs are
not leaderboard-scale: solo hypervolumes are 76 to 83 and the best single front we have grown from
them is 97.3, against a board top of 285.25 (context only, nothing derived from it). Most are
tolerance-feasible rather than strictly nominal, so their margins are thin. See
`docs/results/p3/E24r-a85-3/`, `E29r-c100-2/`, `E30ar-3/`, `E30br-2/`, `E32r-1/`, `E33ar-1/` and
`E33br-1/` (each with a STATUS.md).

**The strict-nominal restoration option (E27, E31; partial).** `p3_final_restoration.py` takes
`--success-tol` (default: the benchmark tolerance, unchanged). At `0.0` the run stops only on a point
confirmed strictly nominal on every evaluation, and each adjudicated point now carries a
`strictly_nominal` flag whatever the option. It was added for the E24r link-3 design, which sat 5e-6
inside the mirror tolerance. E27 (from that design): 1,760 evaluations, **0 accepted steps**, worst V
0.0100 to 0.0060 from probe points, not strictly nominal (bars: goal strictly nominal, material <=
0.0050, null > 0.0075; **partial**). E31a (the E29 design) did reach strictly nominal (worst row
-0.00074) and E31b (the E30a design) went 0.0051 to 0.00218; both again with zero accepted
trust-region steps. Reading: the probes find the gain, the step model does not, so strict nominal
at low aspect is not reliably reachable by this method. See `docs/results/p3/E27/STATUS.md`.

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

**Own-representation fronts, and where they grow (E28 to E33; suggested).** A front is grown from one
design in that design's own representation: `b3_nsga2_mhd.py --representation <the restoration's
start_serialised.json>`, canonical evaluation in the loop, population 56, init scale 0.005 (a quarter
of E25's, because the margins are small), 24 h or 3,000 evaluations, confirmation cap 300, the anchor
render check below, and the 2% chain rule. Each design's fronts, in the order they were grown:

| Design (aspect) | Solo HV | Front links (HV) | Best / solo | Members, best front |
|---|---|---|---|---|
| E28, link-3 and E27 designs (7.37) | 76.59 | 77.02, 77.55 | 1.013 | 22 |
| E30a (7.64) | 77.98 | 78.96, 79.79 | 1.023 | 23 |
| E31b, E30a restored (7.64) | 77.95 | 78.65, 79.30 | 1.017 | 11 |
| E29 (8.62) | 81.59 | 84.30, 85.67 | 1.050 | 33 |
| E31a, E29 restored (8.61) | 81.67 | 84.06, 85.66 | 1.049 | 41 |
| E30b (9.43) | 81.43 | 86.05, 88.74, 90.15 | 1.107 | 89 |
| E33a (10.45) | 83.36 | 87.22, 89.11, 90.92, 92.10 | 1.105 | 61 |
| E33b (10.68) | 79.80 | 86.25, 88.21, 90.89, 92.30 | 1.157 | 128 |
| E32 (10.66) | 80.30 | 85.36, 89.97, 94.01, 95.71 | 1.192 | 66 |

The fronts at aspect 9.1 to 11.0 reach **10 to 19% above their anchor's own hypervolume**; those at
7.3 to 8.6 reach **1 to 5%**. E30b and E32 were the observation. E33 was the pre-registered test: two
more seeds near aspect 10.8 from E32's bank, with the prediction "best front >= 1.10 x solo supported,
< 1.05 x refuted". Both seeds supported it (1.105, which clears the bar narrowly, and 1.157). The best
own-representation fronts are E32fd 95.7, E33bfd 92.3, E33afd 92.1 and E30bfc 90.2; the best single
front overall remains E25b at 97.3 (72 members, built in E22's representation, see the correction
below). The descriptive hypervolume of every confirmed own-lineage design and front together is
117.41; it is not a bar.

Grade: **suggested**, not supported. Two seeds on one bank, over a region defined after the fact by
E30b and E32; the prediction was written for the region and not for any mechanism, and aspect is
confounded with other things that change along it (min_len, scale length, how the bank's seeds
differ). What the evidence does say is that region, and not margin (next section) and not the
particular seed, is the better explanation of which fronts grew. Reading for breadth: a front grown
from one design stays close to it, and the way to a wider front is more designs at more aspects, not
more generations from one. Not tested: whether the pattern holds on banks aimed at other aspects, or
under a different init scale.

**Refuted for two designs: margin at the anchor does not make a front grow (E31; 2026-10-29/31).**
E30b's anchor was strictly nominal and its fronts grew (+10.7%); the tolerance-edge anchors' did not
(+1.3%, +2.3%, +5.0%). E31 tested whether margin was the cause by restoring the two tolerance-edge
designs from E29 and E30a toward strict nominal and growing fronts from the restored versions, so
each design was compared with itself. Prediction: the best E31 front exceeds the best front from the
tolerance-edge version of the same design by at least 2.0 (supported), not at all (refuted), partial
between. E31a: 85.657 against 85.67, **refuted**. E31b: 79.30 against 79.79, **refuted**. Two cases,
so graded no stronger than suggested, but the direction is clear: a small margin won by restoration
does not let a front grow. See `docs/results/p3/E31a-nominal/STATUS.md` and
`docs/results/p3/E31b-nominal/STATUS.md`.

**A correction: the E25 union front did not draw its anchors as stated (E25, E25b; 2026-10-26).** The
engine never moves `r_cos(0,0)`, the major radius, so the run's template supplies it for every
coordinate vector. E22's template has 0.8143, the link-3 lineage 0.8016 and the aspect-12 lineage
0.8153. E25 seeded the link-3 design's exact coordinates and evaluated them at aspect 7.49 with worst
residual 0.026, not 7.37 and 0.0100; the aspect-12 design came out at 11.959 instead of 11.973. The
union script had checked mode counts, field periods and scaling, and nothing else. We found it on
2026-10-26 by comparing the anchors' stored residuals with their generation-0 evaluations.

What stands: every E25 and E25b front member is a real boundary, evaluated and confirmed canonically
in E22's representation, so the fronts, their hypervolumes (95.81, 97.34) and their labels (material)
stand. What is withdrawn: E25 was not a union of the five verified designs, and our earlier reading
that the low-aspect design dropped out because its perturbations were infeasible was wrong, because
the design was never evaluated as itself. An audit of E25's archive afterwards showed that only E22's
own anchor had reproduced exactly; E22b and E22c were re-rendered by 6e-4 and 1e-3 in residual, the
aspect-12 design by 7e-4 and the link-3 design by 0.016.

Two guards now enforce the invariant at the points where it can break, and a third checks the result:

- `p3_union_anchors.union_anchors` refuses anchors whose `r_cos(0,0)` differ (`fixed_major_radius`,
  tolerance 1e-12).
- `p3_front_builder.load_anchor_x` refuses an anchor whose major radius differs from the run's
  representation (`_refuse_major_radius_mismatch`).
- `p3_anchor_render_check.py`: once generation 0 holds the anchors, each anchor's stored confirmed
  residuals must be reproduced by its verbatim evaluation in the archive within 1e-4, or the runner
  stops the container and records a technical failure. Its first version crashed on a confirmed
  front's members, which carry a tag but no coordinates (E28b, 2026-10-26); it failed safe, and was
  fixed under the regression ratchet (the coordinates are now resolved from the source archive by
  tag). On the stopped archive the fixed check showed all 11 anchors reproduced exactly. From E28
  onward every front passed it.

A low-aspect front therefore has to be built in its own representation, which is what E28 onward do.
Two earlier launch defects of the same family (the missing `<tag>.boundary.json` files of the
re-launch, 2026-10-22, and the container-name collision before it) are recorded in
`docs/results/p3/E25/STATUS.md`.

**Measured (E34, 2026-11-06): every design and every front member is feasible under the frozen
benchmark version.** Proxima froze the leaderboard on constellaration 0.2.6 on 2026-10-08
(constellaration-bench discussion 7; the Space's `requirements.txt` agrees). Every
"leaderboard-fidelity" check until then used 0.2.1. A new image pinned to 0.2.6
(`docker/leaderboard-0.2.6/Dockerfile`, vmecpp 0.4.11) scored 574 boundaries on 12 cores with no
errors: **all eleven verified designs, with both the plain and the reflected export**, and **all 546
confirmed members of the ten best fronts as plain exports** (E25b 72/72, E28b 22/22, E29fb 33/33,
E30afb 23/23, E30bfc 89/89, E31afb 41/41, E31bfb 11/11, E32fd 66/66, E33afd 61/61, E33bfd 128/128).
Residuals and objectives match the phase1 and 0.2.1 checks to the reported digits. So the toroidal
reflection used for 0.2.1 is no longer needed (the Space has since fixed its iota sign, per
discussion 3), and every own-lineage design and front is board-verified on the board's own version.
No submission has been made. Results: `docs/results/p3/E34-recheck-026/`.

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
