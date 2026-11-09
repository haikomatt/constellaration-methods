# constellaration-methods: full overview

A self-contained, tested Python package and thin CLIs that reproduce an independent research
campaign's methods on the three [ConStellaration](https://github.com/proximafusion/constellaration)
stellarator-optimisation problems (P1 `GeometricalProblem`, P2 `SimpleToBuildQIStellarator`, P3
`MHDStableQIStellarator`), with the campaign's evidence discipline built in: pre-registration
snapshots committed before every locked experiment, and canonical worst-of-three re-evaluation
before anything is called feasible. This is not a leaderboard kit -- it is the method: a
feasibility-restoration engine, seeded population front-mapping, front-from-archive confirmation,
a boundary-diversity metric, and the packaging/verification scripts that prove every number here
against the benchmark's own evaluator.

The working campaign this was extracted from stays private; this repository is a curated,
import-clean, leak-checked copy of the parts of it that are reusable methods rather than a
day-to-day research log.

## Results

| Problem | Score | Lineage | Provenance | Status |
|---|---|---|---|---|
| P1 -- `GeometricalProblem` | **0.97522** | own | rotating-ellipse analytic seed, random_seed 20260907/20260908, NO leader boundary, 4h+8h+10h ALM/NGOpt descent, resumed via checkpoint ([docs/results/submission/PROVENANCE_own_geometrical.json](results/submission/PROVENANCE_own_geometrical.json)) | submittable (own lineage) |
| P2 -- `SimpleToBuildQIStellarator` | **0.5818** | own | climb-lifted design (4x4 -> 8x8 higher-mode content), c8 climb bank call_index 30548 ([docs/results/submission/PROVENANCE.json](results/submission/PROVENANCE.json)) | submittable (own lineage) |
| P3 -- `MHDStableQIStellarator` | no submission | own (eleven designs) and davidkh (public P2 leaderboard entry) | eleven own-lineage P3-feasible designs, aspect 7.37 to 11.97, every one feasible under constellaration 0.2.6, from database seeds through our restoration; best single front hypervolume 97.3, best own-representation front 95.7. The P3 work that descends from davidkh's design is measurement and method only; nothing is submitted | own lineage: submittable at Matt's discretion; davidkh-derived: measurement only |
| P3 / E13 (2026-09-18) | V=0.0 (strictly feasible) | davidkh | first strictly feasible P3 boundary, all five residuals negative, reproduced in two library versions ([docs/results/p3/E13](results/p3/E13)) | Supported |
| P3 / E15+E16 (2026-09-18) | HV 120.05, 35 members | davidkh | bounded front builder from the E13 point, 11h wall-clock ([docs/results/p3/E16](results/p3/E16)) | Supported (stated limit -- see docs/methods.md) |
| P3 / E17 (2026-09-19) | HV 124.23, 25 members | davidkh | seeded population front-mapping, 9h wall-clock, every member confirmed twice in fresh processes ([docs/results/p3/E17](results/p3/E17)) | Supported (stated limit) |
| P3 / E18a (2026-09-23) | V=0.0536 -> no improvement | own (176 lineage) | seeded NSGA-II cold start from the own-lineage restoration chain end, 2,976 canonical evaluations, no offspring beat the seed ([docs/results/p3/E18a](results/p3/E18a)) | Refuted |
| P3 / E18b (2026-09-24) | HV 123.64 (99.5% of E17), 12 members | davidkh | warm-start re-optimisation under a changed definition (edge mirror bound 0.25 -> 0.23), first feasible point at evaluation 165 ([docs/results/p3/E18b](results/p3/E18b)) | Supported |
| P3 / E21 (2026-09-26) | seed V 0.144 (material, confirmed seed, before descent) | own (public near-axis DB, CC BY 4.0) | a near-axis QI database seed screen (960 seeds, generator route A'), best confirmed seed inside the material region (iota/mirror/well satisfied at aspect 10.41), QI and flux still open ([docs/results/p3/E21](results/p3/E21)) | Material |
| P3 / E22 (2026-09-26) | V 0.0 (strictly feasible), solo hypervolume 81.6 | own (public near-axis DB, CC BY 4.0) + this pipeline | first own-lineage, clean-provenance P3-feasible boundary: strictly feasible on all five rows (V 0.0 worst of three), verified in both evaluator versions via the reflected export; aspect 10.21, min L_grad_B 7.34, solo hypervolume 81.6; from database filter to verified feasible design in about 7 h wall and ~2,800 canonical evaluations ([docs/results/p3/E22](results/p3/E22)) | repeatability 3 of 3, then eight more designs from aspect 7.4 to 12 (E24r to E33); front chain E23-E23c to 94.28 (65 members), stopped by rule |
| P3 / E22b (2026-09-27) | V 0.0085 (feasible), solo hypervolume 82.8 | own (public near-axis DB, CC BY 4.0) + this pipeline | repeatability link 1/2: database seed db00588, start worst-of-three V 0.174, best point R0:L15:t0.5, worst-of-three V 0.0085 feasible under the official predicate (mirror 0.0085 and flux 0.0009 inside tolerance, iota/QI/well strict); aspect 10.19, min_len 7.44, solo hypervolume 82.8; 1,833 optimisation + 11 verification evaluations, 6.44 h ([docs/results/p3/E22b](results/p3/E22b)) | Supported (goal) |
| P3 / E22c (2026-09-27) | V 0.0049 (feasible), solo hypervolume 83.1 | own (public near-axis DB, CC BY 4.0) + this pipeline | repeatability link 2/2: database seed db00397, start worst-of-three V 0.191, best point R0:L16:t0.5, worst-of-three V 0.0049 feasible under the official predicate (QI 0.0049 inside tolerance, iota/mirror/flux/well strict); aspect 10.30, min_len 7.57, solo hypervolume 83.1; 1,852 optimisation + 11 verification evaluations, 4.08 h ([docs/results/p3/E22c](results/p3/E22c)) | Supported (goal) |
| P3 / E23.0 (2026-09-26) | confirmed hypervolume 82.13 (two retained points) | own (public near-axis DB, CC BY 4.0) + this pipeline | baseline measurement of the E22 archive (E15 procedure): 2 tolerance-feasible rows, 2 nondominated, 2 retained after two fresh confirmations each; the own-lineage starting score for the E23 front link ([docs/results/p3/E23.0](results/p3/E23.0)) | Measured |
| P3 / E23 (2026-09-27) | confirmed hypervolume 89.71, 53 members | own (public near-axis DB, CC BY 4.0) + this pipeline | seeded population front from the E22 point: 2,976 evaluations, 62 generations, 7.2 h; 2,458 tolerance-feasible archive rows, 53 nondominated, 53 retained after two fresh confirmations each; confirmed hypervolume 89.71 from the 81.60 seed (+9.9%); 46 of 53 strictly nominal; aspect 9.89-10.34, min_len 7.51-7.88; best singleton solo 86.53 ([docs/results/p3/E23](results/p3/E23)) | Material (against bars 5 members / 111.30 / 124.23; breadth label not met); chained to E23b |
| P3 / E23b (2026-09-27) | confirmed hypervolume 92.61, 52 members (+3.2% on E23) | own (public near-axis DB, CC BY 4.0) + this pipeline | second front link, anchors mode from E23's confirmed front (pop moved 48 -> 56, the anchor-count amendment), 2,968 evaluations, 53 generations; 2,667 tolerance-feasible archive rows, 52 nondominated, 52 retained; 31 of 52 strictly nominal; aspect 9.80-10.38, min_len 7.57-8.09 ([docs/results/p3/E23b](results/p3/E23b)) | Material; above the 2% continuation threshold, chained to E23c |
| P3 / E23c (2026-09-28/30) | confirmed hypervolume 94.28, 65 members (+1.81% on E23b) | own (public near-axis DB, CC BY 4.0) + this pipeline | third front link, anchors mode from E23b's confirmed front, pop 56, 2,968 evaluations; 65 nondominated, 65 retained; 44 of 65 strictly nominal; aspect 9.77-10.49, min_len 7.64-8.24 ([docs/results/p3/E23c](results/p3/E23c)) | Material; chain stops (gain 1.81% below the 2% rule) |
| P3 / E24 bank A, aspect 8.5 (2026-10-02/04) | 7 of 901 confirmed seeds at V <= 0.30, canonical aspect 7.2-8.4, best 0.2761 | own (public near-axis DB, CC BY 4.0) | breadth screen rebuilding the E21 selection at aspect 8.5; QI tail empty at this aspect (0% at log10 qi <= -2.5), flux tail 16% <= 0.9 ([docs/results/p3/E24-a85](results/p3/E24-a85)) | Material |
| P3 / E24 bank B, aspect 14 (2026-10-02/04) | 214 of 960 confirmed seeds at V <= 0.30, canonical aspect 11.9-13.7, best 0.1136 | own (public near-axis DB, CC BY 4.0) | breadth screen at aspect 14 (E21 at aspect 10.4: 151 seeds); the generator's best operating aspect for P3 reads as ~12, not ~10 ([docs/results/p3/E24-b14](results/p3/E24-b14)) | Material |
| P3 / E24r bank A (2026-10-03) | worst-of-three V 0.0751 at aspect 7.4, infeasible | own (public near-axis DB, CC BY 4.0) + this pipeline | one restoration link from bank A's best confirmed seed, 1,853 evaluations, 2.9 h ([docs/results/p3/E24r-a85](results/p3/E24r-a85)) | Material by the chain rule (not substantial); link 2 below |
| P3 / E24r bank B (2026-10-03) | worst-of-three V 0.0143 at aspect 12.0, min_len 8.80, infeasible | own (public near-axis DB, CC BY 4.0) + this pipeline | one restoration link from bank B's best confirmed seed, 1,831 evaluations, 3.4 h; objective value above every own-lineage front member ([docs/results/p3/E24r-b14](results/p3/E24r-b14)) | Substantial (<= 0.030); not feasible; link 2 below |
| P3 / E24r bank B, link 2 (2026-10-04) | V 0.0, strictly nominal on all five rows (worst of three); solo hypervolume 77.99 | own (public near-axis DB, CC BY 4.0) + this pipeline | second restoration link from the aspect-14 bank's link-1 point (seed `db00537_n2_s231`, start V 0.0143), point `P:L1:t0.5`; iota/N 0.316, log10 qi -3.638, mirror 0.225, flux 0.890, well +0.0074; aspect 11.97, min_len 8.72; 882 evaluations, 1.34 h; fidelity re-check feasible in both evaluator versions via the reflected export (the un-reflected export fails the iota sign only) ([docs/results/p3/E24r-b14-2](results/p3/E24r-b14-2)) | Supported (goal): fourth own-lineage feasible design, first off the aspect-10 line |
| P3 / E24r bank A, link 2 (2026-10-04) | best confirmed worst-of-three V 0.0212 at aspect ~7.4, infeasible | own (public near-axis DB, CC BY 4.0) + this pipeline | second restoration link from bank A's link-1 point (start V 0.0751); QI, mirror and flux tied at 0.021; 1,843 evaluations, 4.08 h ([docs/results/p3/E24r-a85-2](results/p3/E24r-a85-2)) | Substantial, not feasible; link 3 above reached feasibility |
| P3 / E26 (2026-10-03) | confirmed hypervolume 91.65, 46 members; survival 0.972 of E23c's 94.28 | own (public near-axis DB, CC BY 4.0) + this pipeline | mirror 0.25 -> 0.23 definition change, warm re-optimisation from E23c's confirmed front; first tolerance-feasible point at evaluation 24; 13 of 46 strictly nominal; aspect 10.11-10.53, min_len 7.99-8.27 ([docs/results/p3/E26](results/p3/E26)) | Substantial (E18b's bars) |
| P3 / E24r bank A, link 3 (2026-10-22) | V 0.009995 (feasible, tolerance-feasible), aspect 7.37, min_len 5.06; solo hypervolume about 76.6 | own (public near-axis DB, CC BY 4.0) + this pipeline | third restoration link from bank A's link-2 point, the link command unchanged, 1,801 evaluations, 3.4 h; point `R10:T15:d0.00002:t0.10`, residuals [-0.276, 0.009972, 0.009995, 0.007284, -0.167], mirror 5e-6 inside the tolerance; verified in `phase1` and, through the reflected export, in `leaderboard-fidelity` ([docs/results/p3/E24r-a85-3](results/p3/E24r-a85-3)) | Supported (goal): fifth own-lineage feasible design, first at low aspect |
| P3 / E25 (2026-10-23) | confirmed hypervolume 95.81, 45 members | own (public near-axis DB, CC BY 4.0) + this pipeline | union front over five anchors, 2,968 evaluations, 53 generations, aspect 10.06-12.03, min_len 7.61-8.79. Re-launched after a third launch defect (anchor boundary files not written). **Corrected 2026-10-26:** the two E24r anchors were drawn with E22's major radius, as different shapes; E25 was not a union of the verified designs, though the fronts are real confirmed designs ([docs/results/p3/E25-front](results/p3/E25-front), [STATUS](results/p3/E25/STATUS.md)) | Material (bar 5 members); substantial (111.30) not met; label stands, union claim withdrawn |
| P3 / E25b (2026-10-23) | confirmed hypervolume 97.34, 72 members (55 strictly nominal) | own (public near-axis DB, CC BY 4.0) + this pipeline | from E25's confirmed front, 2,968 evaluations, aspect 10.01-11.96; +1.6% on E25, so the chain stopped ([docs/results/p3/E25b-front](results/p3/E25b-front)) | Material; best single own-lineage front |
| P3 / E27 (2026-10-23) | worst V 0.0100 -> 0.0060, not strictly nominal; aspect 7.372, min_len 5.046 | own (public near-axis DB, CC BY 4.0) + this pipeline | strict-nominal restoration of the link-3 design with the new `--success-tol 0.0`: 1,760 evaluations, 3.3 h, 0 accepted steps, gain from probe points; verified in both images ([docs/results/p3/E27](results/p3/E27)) | Partial (goal not met, null not met) |
| P3 / E28 (2026-10-26) | confirmed hypervolume 77.02, 11 members (anchors 76.59) | own (public near-axis DB, CC BY 4.0) + this pipeline | low-aspect front in its own representation from the E27 and link-3 designs, init scale 0.005, anchor render check passed, 2,968 evaluations, aspect 7.348-7.390 ([docs/results/p3/E28-front](results/p3/E28-front)) | Material (substantial 80.42 not met) |
| P3 / E28b (2026-10-27) | confirmed hypervolume 77.55, 22 members | own (public near-axis DB, CC BY 4.0) + this pipeline | from E28's confirmed front; +0.68%, so the chain stopped. A first attempt stopped on a defect in the render check (it crashed on front members that carry no coordinates), fixed under the regression ratchet ([docs/results/p3/E28b-front](results/p3/E28b-front)) | Material |
| P3 / E29 bank, aspect 10.0 request (2026-10-26/27) | best confirmed seed V 0.2006 (960 of 960 built) | own (public near-axis DB, CC BY 4.0) + this pipeline | the E24 pipeline with the aspect changed ([docs/results/p3/E29r-c100-2](results/p3/E29r-c100-2) for the chain) | Material |
| P3 / E29r, links 1 and 2 (2026-10-27) | link 1 V 0.2006 -> 0.0245; link 2 V 0.0083 (feasible), aspect 8.617, min_len 6.167, solo 81.59 | own (public near-axis DB, CC BY 4.0) + this pipeline | link 2: 883 evaluations, point `P:T0:d0.00250:t0.25`, residuals [-0.250, 0.00248, 0.00157, 0.00827, -0.142]; verified in both images ([docs/results/p3/E29r-c100-2](results/p3/E29r-c100-2)) | Supported (goal): sixth own-lineage feasible design, in the 7.4-10 gap |
| P3 / E29f, E29fb (2026-10-27) | hypervolume 84.30 (30 members), 85.67 (33 members) | own (public near-axis DB, CC BY 4.0) + this pipeline | fronts in the design's own representation; +1.63% on the second link, so the chain stopped ([docs/results/p3/E29fb-front](results/p3/E29fb-front)) | Material; substantial by 0.007 (bar 85.666) |
| P3 / E30a (2026-10-28) | bank best 0.2566; link 3 V 0.0051 (feasible), aspect 7.637, min_len 5.308, solo 77.98 | own (public near-axis DB, CC BY 4.0) + this pipeline | request aspect 9.0 came out at 7.64; links 0.2566 -> 0.0408 -> 0.0154 -> 0.0051, link 3 1,793 evaluations; verified in both images. Fronts E30af 17 members HV 78.96, E30afb 23 members HV 79.79 (+1.04%, stopped) ([docs/results/p3/E30ar-3](results/p3/E30ar-3), [E30afb-front](results/p3/E30afb-front)) | Supported (goal): seventh own-lineage feasible design; fronts Material |
| P3 / E30b (2026-10-29) | bank best 0.1801; link 2 V 0.0, strictly nominal, aspect 9.430, min_len 6.704, solo 81.43 | own (public near-axis DB, CC BY 4.0) + this pipeline | request aspect 11.0; links 0.1801 -> 0.0298 -> 0.0, link 2 882 evaluations, every row satisfied with at least 0.0086 to spare; verified in both images. Fronts E30bf 24 members HV 86.05, E30bfb 56 members HV 88.74, E30bfc 89 members (74 strictly nominal) HV 90.15 (+1.59%, stopped) ([docs/results/p3/E30br-2](results/p3/E30br-2), [E30bfc-front](results/p3/E30bfc-front)) | Supported (goal): eighth feasible design, second strictly nominal one, in the 8.6-10.2 gap; fronts Substantial, Substantial, Goal |
| P3 / E31a (2026-10-31) | E29 design restored to strictly nominal (worst row -0.00074); fronts HV 84.06, 85.657 | own (public near-axis DB, CC BY 4.0) + this pipeline | restoration 1,760 evaluations, 0 accepted steps, verified in both images, solo 81.67; margin prediction: 85.657 against the tolerance-edge best 85.67 ([docs/results/p3/E31a-nominal](results/p3/E31a-nominal)) | Restoration goal; margin prediction **Refuted** |
| P3 / E31b (2026-10-31) | E30a design V 0.0051 -> 0.00218 (not strictly nominal); fronts HV 78.65, 79.30 | own (public near-axis DB, CC BY 4.0) + this pipeline | restoration 0 accepted steps, verified in both images, solo 77.95; margin prediction: 79.30 against the tolerance-edge best 79.79 ([docs/results/p3/E31b-nominal](results/p3/E31b-nominal)) | Restoration material; margin prediction **Refuted** |
| P3 / E32 (2026-11-01) | bank best 0.1241; link 1 V 0.0099 (feasible), aspect 10.659, min_len 7.596, solo 80.30 | own (public near-axis DB, CC BY 4.0) + this pipeline | request aspect 12.5; link 1 1,777 evaluations, QI at the edge; verified in both images. Fronts E32f 28 members HV 85.36, E32fb 51 HV 89.97, E32fc 72 HV 94.01, E32fd 66 members (56 strictly nominal) HV 95.71 (+1.81%, stopped; also the last registered link) ([docs/results/p3/E32r-1](results/p3/E32r-1), [E32fd-front](results/p3/E32fd-front)) | Supported (goal): ninth feasible design, in the 10.3-11.9 gap; fronts Substantial, Goal, Goal, Goal |
| P3 / E33a (2026-11-05) | link 1 V 0.0087 (feasible), aspect 10.445, min_len 7.724, solo 83.36 | own (public near-axis DB, CC BY 4.0) + this pipeline | second seed from E32's bank (`db00588_n2_s231`, 0.1577 at aspect 10.86); verified in both images. Fronts HV 87.22, 89.11, 90.92, 92.09 (61 members, +1.29%, stopped); best / solo 1.105 ([docs/results/p3/E33ar-1](results/p3/E33ar-1), [E33afd-front](results/p3/E33afd-front)) | Supported (goal): tenth feasible design; region prediction **Supported** (1.105 >= 1.10) |
| P3 / E33b (2026-11-05) | link 1 V 0.0099 (feasible), aspect 10.678, min_len 7.560, solo 79.80 | own (public near-axis DB, CC BY 4.0) + this pipeline | third seed from E32's bank (`db00538_n2_s364`, 0.1474 at aspect 10.81); verified in both images. Fronts HV 86.25, 88.21, 90.89, 92.30 (128 members, 114 strictly nominal, +1.55%, stopped); best / solo 1.157 ([docs/results/p3/E33br-1](results/p3/E33br-1), [E33bfd-front](results/p3/E33bfd-front)) | Supported (goal): eleventh feasible design; region prediction **Supported** (1.157) |
| P3 / E34 (2026-11-06) | 574 boundaries scored, all feasible under constellaration 0.2.6 | own (public near-axis DB, CC BY 4.0) + this pipeline | fidelity re-check under the leaderboard's frozen benchmark version: all eleven designs feasible with both the plain and the reflected export, and all 546 confirmed members of the ten best fronts feasible as plain exports; portfolio hypervolume 117.41 stands on the board's own version ([docs/results/p3/E34-recheck-026](results/p3/E34-recheck-026), [Dockerfile](../docker/leaderboard-0.2.6/Dockerfile)) | Measured: every own-lineage design and front is board-verified |

The pre-registered repeatability label -- "the pipeline reaches feasibility from a cold start" --
is now met 3 of 3: three distinct database seeds (E22, E22b, E22c), each QI- or flux-binding at the
seed, each reaching a benchmark-feasible design in one link (4.1 / 6.4 / 4.1 h; 1,780 / 1,833 /
1,852 optimisation evaluations), with the finish coming from the recovery's first line search in
all three.

Six banks aimed at six aspects (8.5, 14, 10, 9, 11, 12.5) then each produced a verified feasible
design in one to three restoration links of a few hours, and two more seeds from the last bank did
the same: eleven own-lineage designs from aspect 7.37 to 11.97, every one feasible under
constellaration 0.2.6, the benchmark's frozen version. The pipeline is described in
[docs/methods.md](methods.md) (bank-then-restore). Fronts grown from a single design in its own
representation reach 10 to 19% above the design's own hypervolume at aspect 9 to 11 and 1 to 5% at
7.3 to 8.6 (suggested, two pre-registered seeds); the margin explanation was refuted (E31). The best
single front is 97.3, against a board top of 285.25 (context only): the gap is breadth.

For the record: the first davidkh-lineage front (E15) had 9 members at hypervolume 111.30 from a
single strictly feasible point with better objectives (min_len roughly 9-10 at aspect roughly
10.6); this own-lineage front (E23) has six times the members at 80% of the hypervolume, the
difference being per-member objective quality, not feasibility or breadth.

Chain summary (E23 -> E23b -> E23c, one seed, one representation, ~17 h total wall across three
links): own-lineage confirmed hypervolume rose 81.60 (the E22 baseline) -> 89.71 -> 92.61 -> 94.28,
i.e. +15.5% total from the single feasible seed, member count 2 -> 53 -> 52 -> 65. The chain
stopped on its pre-registered 2% rule (E23c's gain over E23b, 1.81%, fell below the 2% continuation
threshold), not on the clock; no E23d launched.

Full grades (Supported / Refuted / Measured / Open), every number, and the P1/P2 method sections
are in [docs/methods.md](methods.md). Provenance detail (what "own lineage" and "davidkh
lineage" mean, and the near-axis QI database's CC BY 4.0 attribution where it applies) is in
[PROVENANCE.md](../PROVENANCE.md).

## Honesty guards

This repository follows the same discipline as the write-up it was extracted from -- every number
above ships with the run that produced it, and every P3 result is labelled by what it does and
does not show:

> Provenance note first: the P3 work that descends from davidkh's public P2 design is
> measurement and method only. The eleven own-lineage designs descend from a public CC BY 4.0
> database and our restoration; none has been submitted.

> [E25 correction] Our first attempt to merge designs from different seeds into one front drew the
> same coordinates as different shapes, because the major radius is fixed by each run's template and
> not by the design. The fronts were real designs, but the merge was not what we said it was. We
> withdrew that claim; the pipeline now refuses such a merge and checks that every front's first
> generation reproduces its anchors.

> [E17/E16] The limit: hypervolume on this board is bought by aspect-ratio spread (leaders span
> aspect 5-24; ours spans 0.56), and one member accounts for 96% of our hypervolume. Neither
> method widened aspect; the registered breadth label was missed. Cheap front mapping near a known
> design: yes. Cheap front mapping across the design space: not shown.

> [E18a] Population search without a landscape model cannot finish a cold start the local model
> left at 0.054; it only refines inside feasibility.

> [P1] I did not beat the leaderboard. Warm-starting my search from a top design let it refine
> that design -- it did not discover its own shape.

No literal single "we do not claim" paragraph exists in the source write-ups this repo was built
from; the guards above are verbatim sentences pulled from them, chosen because each states a limit
plainly. See `docs/methods.md` for the full grade-by-grade breakdown these are drawn from.

## Quickstart

```bash
podman build -t localhost/constellaration-methods:test .
# or: docker build -t localhost/constellaration-methods:test .

# Run the full test suite inside the pinned image (constellaration==0.2.6,
# vmecpp==0.4.11, scikit-learn==1.5.2 -- see pyproject.toml):
podman run --rm -v "$(pwd)":/workspace:Z --entrypoint pytest \
  localhost/constellaration-methods:test tests/ -q

# Host-runnable subset (pure logic, no physics stack required):
pip install -e ".[dev]"
pytest tests/ -q
```

## Layout

```
src/constellaration_methods/
  engines/       direct optimizers and boundary/near-axis machinery
                 (alm_ngopt_optimizer, near_axis_search, nae_generator,
                 boundary_grid, boundary_generator, thread_limits,
                 trajectory_logger)
  restoration/   the P3 feasibility-restoration engine and its final runner
                 (p3_feasibility_restoration, p3_final_restoration,
                 p3_promote_candidate)
  fronts/        Pareto-front mapping and confirmation
                 (p3_front_builder, b3_nsga2_mhd, p3_front_from_archive,
                 p3_curve_from_archive), the union-anchors merge with its
                 major-radius refusal (p3_union_anchors) and the anchor render
                 check (p3_anchor_render_check)
  evaluation/    scoring, canonical confirmation, and submission packaging
                 (score_mhd_stable, confirm_candidates, rescore_trajectory,
                 qi_audit_instrument, package_submission, verify_p1)
  construction/  near-axis construction search and its cheap proxies
                 (p3_construction_proxies, p3_construction_search,
                 p3_construction_bank_eval); the near-axis QI database seed
                 screen (nae_db_seed_screen, E21 -- external data, see
                 docs/external-data.md)
  analysis/      diversity metrics, galleries, and rendering
                 (boundary_diversity, boundary_symmetry,
                 p3_design_diversity_probe, p3_front_gallery,
                 render_boundary_3d, visualize_boundary, dashboard_style)
  problems/      problem-definition overrides (problem_with_overrides)
scripts/         one thin CLI per runner, plus prereg_snapshot.sh and the bank-then-restore
                 chain runners (p3_chain_lib.sh, p3_bank_chain_runner.sh,
                 p3_nominal_front_runner.sh)
docker/          leaderboard-0.2.6/Dockerfile, the image pinned to the benchmark's frozen
                 version (E34); the root Dockerfile is the pinned test environment
tools/           strip_boundaries.py / check_no_boundaries.py / check_leaks.sh
tests/           every test of the included modules, moved and import-rewritten
docs/
  methods.md            the method write-up, grades kept, every number
  external-data.md       what nae_db_seed_screen (E21) needs that this repository does not
                         ship -- the near-axis QI database (CC BY 4.0) and the authors'
                         pyQIC fork -- and exactly how to obtain and lay them out
  preregistration/      35 pre-registration snapshots (E12..E33, D1, atlas-probe)
  results/               P1/P2 submission packages (own lineage, boundaries included);
                         P3 REPORT.json/curve.json (davidkh lineage, boundary
                         coordinates stripped); own-lineage design exports under
                         p3/*/own-lineage/ (allowlisted), own-lineage front REPORTs
                         without coordinates; D1 diversity report; E21 (near-axis
                         database seed screen, provisional, see docs/results/p3/E21/STATUS.md);
                         a MANIFEST with sha256 + source path for every file
scale/README.md  the GKE/KubeRay fleet substrate this pipeline could scale onto --
                 what is demonstrated there today, what is not, and the port as
                 the open item
```

## Scale

A separate, already-built substrate (GKE + KubeRay, scale-to-zero, chaos-tested preemption
resilience, an observability gate, a keyless CD path) exists for running workloads like this one
at scale. It has not been used to run any of the physics in this repository. See
[scale/README.md](../scale/README.md) for the honest demonstrated/not-demonstrated split.

## Licence and provenance

Apache-2.0, copyright 2026 Matt Dawson -- see [LICENSE](../LICENSE) and [NOTICE](../NOTICE). Public
leaderboard handles referenced throughout (`davidkh`, `phanerozoic`, `canyon-ant`) are the public
record the P3 measurements are defined against; every mention of a derived boundary carries its
provenance line -- see [PROVENANCE.md](../PROVENANCE.md). No P3 boundary coordinates are
redistributed by this repository (`tools/check_no_boundaries.py` enforces this in CI).
