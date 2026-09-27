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
| P3 -- `MHDStableQIStellarator` | no submission | davidkh (public P2 leaderboard entry) | every feasible P3 design we hold descends from davidkh's public P2 design; nothing is submittable without reuse terms; all P3 results are measurement and method | measurement only |
| P3 / E13 (2026-09-18) | V=0.0 (strictly feasible) | davidkh | first strictly feasible P3 boundary, all five residuals negative, reproduced in two library versions ([docs/results/p3/E13](results/p3/E13)) | Supported |
| P3 / E15+E16 (2026-09-18) | HV 120.05, 35 members | davidkh | bounded front builder from the E13 point, 11h wall-clock ([docs/results/p3/E16](results/p3/E16)) | Supported (stated limit -- see docs/methods.md) |
| P3 / E17 (2026-09-19) | HV 124.23, 25 members | davidkh | seeded population front-mapping, 9h wall-clock, every member confirmed twice in fresh processes ([docs/results/p3/E17](results/p3/E17)) | Supported (stated limit) |
| P3 / E18a (2026-09-23) | V=0.0536 -> no improvement | own (176 lineage) | seeded NSGA-II cold start from the own-lineage restoration chain end, 2,976 canonical evaluations, no offspring beat the seed ([docs/results/p3/E18a](results/p3/E18a)) | Refuted |
| P3 / E18b (2026-09-24) | HV 123.64 (99.5% of E17), 12 members | davidkh | warm-start re-optimisation under a changed definition (edge mirror bound 0.25 -> 0.23), first feasible point at evaluation 165 ([docs/results/p3/E18b](results/p3/E18b)) | Supported |
| P3 / E21 (2026-09-26) | seed V 0.144 (material, confirmed seed, before descent) | own (public near-axis DB, CC BY 4.0) | a near-axis QI database seed screen (960 seeds, generator route A'), best confirmed seed inside the material region (iota/mirror/well satisfied at aspect 10.41), QI and flux still open ([docs/results/p3/E21](results/p3/E21)) | Material |
| P3 / E22 (2026-09-26) | V 0.0 (strictly feasible), solo hypervolume 81.6 | own (public near-axis DB, CC BY 4.0) + this pipeline | first own-lineage, clean-provenance P3-feasible boundary: strictly feasible on all five rows (V 0.0 worst of three), verified in both evaluator versions via the reflected export; aspect 10.21, min L_grad_B 7.34, solo hypervolume 81.6; from database filter to verified feasible design in about 7 h wall and ~2,800 canonical evaluations ([docs/results/p3/E22](results/p3/E22)) | repeatability 3 of 3 (E22b, E22c); front E23 material (53 members, 89.71); E23b running |
| P3 / E22b (2026-09-27) | V 0.0085 (feasible), solo hypervolume 82.8 | own (public near-axis DB, CC BY 4.0) + this pipeline | repeatability link 1/2: database seed db00588, start worst-of-three V 0.174, best point R0:L15:t0.5, worst-of-three V 0.0085 feasible under the official predicate (mirror 0.0085 and flux 0.0009 inside tolerance, iota/QI/well strict); aspect 10.19, min_len 7.44, solo hypervolume 82.8; 1,833 optimisation + 11 verification evaluations, 6.44 h ([docs/results/p3/E22b](results/p3/E22b)) | Supported (goal) |
| P3 / E22c (2026-09-27) | V 0.0049 (feasible), solo hypervolume 83.1 | own (public near-axis DB, CC BY 4.0) + this pipeline | repeatability link 2/2: database seed db00397, start worst-of-three V 0.191, best point R0:L16:t0.5, worst-of-three V 0.0049 feasible under the official predicate (QI 0.0049 inside tolerance, iota/mirror/flux/well strict); aspect 10.30, min_len 7.57, solo hypervolume 83.1; 1,852 optimisation + 11 verification evaluations, 4.08 h ([docs/results/p3/E22c](results/p3/E22c)) | Supported (goal) |
| P3 / E23.0 (2026-09-26) | confirmed hypervolume 82.13 (two retained points) | own (public near-axis DB, CC BY 4.0) + this pipeline | baseline measurement of the E22 archive (E15 procedure): 2 tolerance-feasible rows, 2 nondominated, 2 retained after two fresh confirmations each; the own-lineage starting score for the E23 front link ([docs/results/p3/E23.0](results/p3/E23.0)) | Measured |
| P3 / E23 (2026-09-27) | confirmed hypervolume 89.71, 53 members | own (public near-axis DB, CC BY 4.0) + this pipeline | seeded population front from the E22 point: 2,976 evaluations, 62 generations, 7.2 h; 2,458 tolerance-feasible archive rows, 53 nondominated, 53 retained after two fresh confirmations each; confirmed hypervolume 89.71 from the 81.60 seed (+9.9%); 46 of 53 strictly nominal; aspect 9.89-10.34, min_len 7.51-7.88; best singleton solo 86.53 ([docs/results/p3/E23](results/p3/E23)) | Material (against bars 5 members / 111.30 / 124.23; breadth label not met); E23b chained and running |

The pre-registered repeatability label -- "the pipeline reaches feasibility from a cold start" --
is now met 3 of 3: three distinct database seeds (E22, E22b, E22c), each QI- or flux-binding at the
seed, each reaching a benchmark-feasible design in one link (4.1 / 6.4 / 4.1 h; 1,780 / 1,833 /
1,852 optimisation evaluations), with the finish coming from the recovery's first line search in
all three.

For the record: the first davidkh-lineage front (E15) had 9 members at hypervolume 111.30 from a
single strictly feasible point with better objectives (min_len roughly 9-10 at aspect roughly
10.6); this own-lineage front (E23) has six times the members at 80% of the hypervolume, the
difference being per-member objective quality, not feasibility or breadth.

Full grades (Supported / Refuted / Measured / Open), every number, and the P1/P2 method sections
are in [docs/methods.md](methods.md). Provenance detail (what "own lineage" and "davidkh
lineage" mean, and the near-axis QI database's CC BY 4.0 attribution where it applies) is in
[PROVENANCE.md](../PROVENANCE.md).

## Honesty guards

This repository follows the same discipline as the write-up it was extracted from -- every number
above ships with the run that produced it, and every P3 result is labelled by what it does and
does not show:

> Provenance note first: every feasible P3 design we hold descends from davidkh's public P2
> design; nothing is submittable without reuse terms; all P3 results are measurement and method.

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
                 p3_curve_from_archive)
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
scripts/         one thin CLI per runner, plus prereg_snapshot.sh
tools/           strip_boundaries.py / check_no_boundaries.py / check_leaks.sh
tests/           every test of the included modules, moved and import-rewritten
docs/
  methods.md            the method write-up, grades kept, every number
  external-data.md       what nae_db_seed_screen (E21) needs that this repository does not
                         ship -- the near-axis QI database (CC BY 4.0) and the authors'
                         pyQIC fork -- and exactly how to obtain and lay them out
  preregistration/      16 pre-registration snapshots (E12..E21, D1, atlas-probe)
  results/               P1/P2 submission packages (own lineage, boundaries included);
                         P3 REPORT.json/curve.json (davidkh lineage, boundary
                         coordinates stripped); D1 diversity report; E21 (near-axis
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
