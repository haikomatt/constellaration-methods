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

**Refuted for cold start (E18a 2026-09-23, E19a/E19b 2026-09-24/25).** From our own near-axis seed
at confirmed V 0.0536: a seeded population could not beat the seed in 2,976 evaluations; a
second-order local model with measured off-diagonal curvature (12 directions, 156 probes) and its
diagonal control both predicted no descending step, measurements agreeing to 1e-4, confirmed bests
0.0519 and 0.0516. Registered comparison label: off-diagonal curvature does not matter. Cold start
to feasibility from our own seeds is the open problem, and we can say precisely where it is
(see "What is refuted" below).

**Material, provisional (E21, 2026-09-26): a better cold start from a public database.** The
own-lineage cold-start seeds above all came from this repository's first-order near-axis
construction search, which "What is refuted" below shows cannot supply a P3-range twist that is
also an equilibrium. E21 tries a different generator instead: a public, CC BY 4.0, second-order
near-axis QI database (Rodriguez and Plunk, arXiv:2601.08400; dataset doi:10.5281/zenodo.18220133 --
see PROVENANCE.md and docs/external-data.md), filtered for P3-shaped constructions and completed
with the authors' own MHD magnetic-well transform. The route was justified by a Phase 0 screen: 24
database seeds at aspect 8-9, evaluated canonically, converged in VMEC at P3-range twist 24 times
out of 24 (our own first-order sampler never does), with iota satisfied 24/24 and mirror satisfied
19/24 -- only the well, flux and QI rows failed, and the well is a construction parameter of the
database with a lever the same Phase 0 screen demonstrated canonically. A registered screen of 960
selected seeds (`construction.nae_db_seed_screen`, docs/results/p3/E21/) confirmed a best seed at
worst-of-three V = 0.1443, with iota, mirror and vacuum well satisfied at confirmation and canonical
aspect ratio 10.41 -- inside the pre-registered material bar (confirmed V <= 0.30 with those three
rows satisfied at aspect <= 12). QI and flux compression remain unsatisfied at the seed. "Material,
provisional" here means exactly this: the seed clears the bar on its own confirmed score, before any
restoration descent has been run from it -- see docs/results/p3/E21/STATUS.md for the registered
follow-on this label is provisional against.

### "Methods that can map Pareto fronts cheaply"

**Supported with a stated limit (E15-E17, 2026-09-18/19).** From one strictly feasible point, a
bounded front builder confirmed 35 members at hypervolume 120.05 in 11 h, and a seeded population
raised that to 124.23 with 25 members in 9 h, every member confirmed twice in fresh processes. The
limit: hypervolume on this board is bought by aspect-ratio spread (leaders span aspect 5-24; ours
spans 0.56), and one member accounts for 96% of our hypervolume. Neither method widened aspect; the
registered breadth label was missed. Cheap front mapping near a known design: yes. Cheap front
mapping across the design space: not shown.

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
