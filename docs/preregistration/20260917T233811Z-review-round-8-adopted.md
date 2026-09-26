---
title: "constellaration — P3 reopen: three pre-registered experiments before close (2026-09-16)"
status: active
tags: [constellaration, plan, mhd_stable, preregistered]
type: plan
created: 2026-09-16
parent_plan: "[[mhd-stable-entry-plan]]"
depends_on: "[[p3-final-feasibility-attempt-2026-09-16]]"
related: "[[h5-well-feasible-region-is-constructible]]"
---

> [!note] Status: draft — Matt (2026-09-16): "2 & 3 & 4 then 1, not ready to close".
> Three experiments, run in cost order E2 -> E4 -> E3, each locked at its own
> launch, each adjudicated by the final protocol's rules and labels
> ([[p3-final-feasibility-attempt-2026-09-16]] §3-4) with NO external review.
> Canonical only; every evaluation archived; worst-V-of-three confirmations.

## Shared rules (all three)
Acceptance: V = max(0, five canonical residuals) falls by eps (1e-4 unless
repeatability says otherwise); finite residuals; raw well >= 0; anything
within tolerance stays within it; no regression allowances. Trust region
0.00125 start / 0.0025 cap / 1e-5 floor. Stop on: 8 consecutive
non-material trials, radius floor (after the one permitted recovery),
> 30% non-convergence in the last 20 trials, runner error, ceilings.
Adjudication: three lowest-V distinct complete candidates, two fresh
confirmations each, worst V. Labels: verified P3 feasibility (all residuals
<= 0.01 on all three evaluations) | substantial (<= 0.100) | material
beyond existing evidence (E2/E4: <= 0.115, i.e. clearly below the retained
0.1238 / 0.1653 respectively; E3: <= 0.130) | gate not met | technical
limitation. Feasibility is the official predicate, not optimiser
acceptance; a feasible-looking point gets priority confirmation.

## E2 — second-order (line-searched) restoration from the retained 0.1238 boundary
> LAUNCHED 2026-09-16, unchanged; ceilings 600 opt / 12 verif / 2 Jacobians / 8 h (opt cutoff 7 h); material bar 0.115. Output `results/p3-e2-176/`.
> **RESULT: progress gate not met by 0.0003** — confirmed V 0.115331 (from 0.123814; 0.0085 in 459 evals, all converged, 8 accepted steps). The line search fixed step size (first 3 steps = fastest verified progress of the campaign) but not the QI/mirror/flux three-way tie, now equal to 4 decimals with well at zero. Stalled at the radius floor after the recovery. `results/p3-e2-176/RESULT.md`.
**Why:** first-order steps died at the eps/radius floors with the model
accurate (rho 0.5-1.0 at radii <= 3e-4) — the signature of curvature at
the QI/mirror/flux tie, not of a wall. Never tried.
**Design:** start `results/p3-final-176/best_candidate_V0.12381.json`
(repeatability gate as before); fresh 160-evaluation Jacobian (Jacobian
#1); direction d from the same LP (well capped at 0); then a **3-point
quadratic line model per residual along d**: evaluate x + d/2 and x + d,
fit r_i(t) = a_i t^2 + b_i t + c_i through t = 0, 1/2, 1, choose t* on a
grid in [0.1, 1.5] minimising predicted V(t) subject to predicted well
<= 0; evaluate x + t* d canonically and apply the acceptance rule to THAT
point (the two line points are archived and may be accepted if they
satisfy the rule and are better than the incumbent). Broyden updates from
every converged evaluation (dx <= 2 x cap); drift checks/partial refresh
as before; one recovery from the lowest-V eligible archive point with
Jacobian #2. Ceilings: 600 optimisation evaluations, 12 verification, 2
Jacobians, 8 h. Cost ~3 evaluations per trial instead of 1.
**Outcomes:** as shared rules; "material beyond existing evidence" =
confirmed V <= 0.115. A stall at the same ~0.124 tie with the line model
accurate = "second-order steps do not escape the tie either" — recorded,
not "locally optimal".

## E4 — final protocol on seed 49's retained candidate
> First launch stopped at the repeatability gate through a runner defect (control evaluated under the wrong nfp; 0 optimisation evals spent; fixed, tested). Second attempt found a second runner defect: a NaN (frozen) Jacobian column poisoned the Broyden update and the LP returned zero steps after one accepted step (0.1653 -> 0.1519); fixed, tested, replayed. RELAUNCHED (third) 2026-09-16, unchanged rules; same ceilings; material bar 0.115. Output `results/p3-e4-49/`; aborted runs kept as `-aborted-*`.
> **RESULT: progress gate not met** — confirmed V 0.141766 (from 0.165294; 12 accepted steps). Same tie signature as 176 (here QI + mirror two-way, flux/well slack). Recovery's fresh Jacobian outperformed the Broyden model (rho 4.3 first step). `results/p3-e4-49/RESULT.md`.
**Start:** `results/m1-post-ladder-d49/candidate_k29_0.165.json` (V 0.1653;
iota 0.097, qi 0.165, mirror 0.161, flux satisfied, well +0.018 raw).
Repeatability gate; fresh Jacobian; E2's line-searched protocol (so E2 and
E4 share one method); ceilings 600 / 12 / 2 / 8 h. Known risk: this region
had VMEC non-convergence on 16/24 checkpoint rows; the 30%-in-last-20
stop and the "technical limitation" label cover it. "Material" = <= 0.115.

## E3 — the three-of-five seed (243): coarse descent then restoration
**Why:** all 61 well-feasible seeds already satisfy flux at the seed;
flux is lost DURING descent. Seed 243 (nfp 3, aspect 18.3) satisfies
mirror, flux AND well at the seed (canonical: iota 0.866, qi 0.550,
mirror -0.365, flux -0.414, well -0.088); only iota and QI are violated.
The preservation rule then holds three constraints while two descend —
a lineage variant the campaign has not run. Seed 249 also violates mirror
(0.84) and is excluded.
**Stage A (coarse):** direct ALM descent (`m1_p3_migration_descent.py
--arm direct --seeds 243 --aspect-bound 22 --maxit 60 --per-seed-wall-min
240`, iota fix in), per-k history + canonical checkpoint evaluation.
Gate to stage B: best canonically evaluated point with V <= 0.30 AND
mirror/flux/well still within tolerance at that point. If the ALM trades
them away (as it did for the 20 M1 seeds), that is E3's finding: the
coarse descent, not the seed, destroys the profile; record and stop.
**Stage B:** E2's protocol from the stage-A candidate; ceilings 600 / 12 /
2 / 8 h; "material" = <= 0.130 (this lineage has no prior evidence).
Aspect is a P3 objective, not a constraint: 18 is scorable but poor;
noted, not optimised.

> **LOCKED 2026-09-16 10:35: stage A launched** unchanged (direct arm, 4x4, aspect bound 22, maxit 60, per-seed wall 240 min, budgets 300/+150/1500, 5 h). Out `results/p3-e3-243-stageA/`. Stage B gate as written: a canonically evaluated checkpoint with V <= 0.30 AND mirror/flux/well each <= 0.01; otherwise "the ALM trades them away" is E3's result. Representation stays 4x4 (E6b did not break the tie).

> **RESULT (2026-09-16 14:42): stage-B gate NOT met; the ALM traded mirror away at once.** Stage A ran to its 240-min wall (7,297 search evaluations, many non-convergent; search best violation 0.452). Canonical checkpoints (17 evaluated, 1 non-convergent): best canonical V **0.490** at k7-k9 (iota/nfp 0.137 vs >= 0.25 is the binding residual 0.45; mirror 0.373 vs <= 0.25, i.e. traded away from the seed's satisfied -0.365; QI log10 -1.94; flux 0.627 and well +0.052 stayed satisfied). No checkpoint has V <= 0.30, and mirror is violated at every one. Per the pre-registration that IS E3's result: the direct ALM arm does not preserve mirror while iota and QI descend from this seed; no stage B. `results/p3-e3-243-stageA/canonical/`.

## Order, cost, close
E2 (~3-5 h) -> E4 (~3-5 h) -> E3 (4 h + ~5 h). Then close (option 1):
final claim per the reviewer's §6 wording, updated with these three
results; dossier triage (stale pre-registrations -> superseded/dormant).

## Extended queue (standing authorisation 2026-09-16; each to be written up fully and locked before its launch)
Order after E3, cheapest-first, every one canonical-only under the shared rules:
- **E5 — multi-start.** The same line-searched protocol from the other
  retained candidates (E2's endpoint, 49's, 48's k13, C-176 k25, the paired
  incumbents): does the same tie appear at the same V everywhere (a
  representation limit) or does one lineage break through? ~8 h each.
- **E6 — more coordinates at the tie (PROMOTED to next; full pre-registration below).**
- **E7 — full second-order model.** A quasi-Newton (SR1/BFGS) model of each
  residual, not just along one line; SQP step with a trust region. Build
  cost is mine, compute similar to E2.
- **E7 as amended after E10-pilot-2 (pre-registered and LOCKED 2026-09-16 15:17, launched):
  diagonal second-order restoration from step-scale probes.** Why: at the
  coefficient box's step scale every residual is a convex bowl in nearly
  every new-mode direction (QI V-shaped 22/24, mirror 20/24, quadratic term
  ~3x the linear term), so the linear model overshoots and the radius
  collapses. Model: r_i + J_i d + 1/2 sum_j max(C_ij, 0) d_j^2 with J and C
  from the SAME central-difference probes taken at h_j = 0.01 x box
  multiplier (new modes 0.07-1.5 scaled units = ~1e-4 coefficient units, the
  scale of the accepted steps; old modes 0.01 as before). Step: minimise the
  slack over the box (SLSQP), preservation rows, well cap and tolerance
  exactly as the LP; a failed solve is a rejected trial. Not SR1 (the queue's
  original wording): SR1 needs gradients we never measure. Everything else
  is the final-attempt protocol with the line search: no allowances, raw
  well >= 0, eps 1e-4, ratio trust policy, one recovery, block rule,
  worst-of-three adjudication. Declared changes: trust radius 0.0025 start /
  0.01 cap (4x E6b's cap; on the 32 pilot coordinates the quadratic model's
  own optimum sits at ~0.012 trust units, so E6b's 0.0025 cap would bind
  before the bowl bottom); Broyden updates on J continue (guard 2 x cap in
  trust units); C is refreshed only by full or partial probe refreshes.
  Start: E6b's adjudicated best (0.110645, 8x8, coefficient box). Fresh J0 +
  C0 at step scale = 576 evaluations (Jacobian #1 of 2). Ceilings: 1,000
  optimisation evaluations, 12 verification, 12 h (optimisation cutoff 10
  h). Material bar **0.105** (clearly below E6b's 0.1106); substantial
  0.100; feasible 0.01 on all three evaluations. Outcomes: "second order
  breaks the tie" = confirmed worst-of-three V <= 0.105; "does not" =
  confirmed >= 0.108 with the floor/eps signature recurring, which would be
  strong evidence that the tie is a property of the local bowl at 8x8 and
  not of the model order; between = material progress, recorded as such.
  Also report: rho of the quadratic predictions (the mechanistic check that
  the bowl model is right), and whether the QP's steps end interior to the
  box (bowl bottom reached) or on it. Dry check on the 32 pilot-2
  coordinates only: the model predicts 0.1106 -> 0.1003 with the three-way
  tie persisting at the bowl bottom. Command: `python3
  scripts/p3_final_restoration.py --checkpoint
  results/m1-post-ladder-d176/seed_0176/checkpoint.json --start-candidate
  results/p3-e6b-176-8x8/best_candidate_V0.11065_8x8.json --out-dir
  results/p3-e7-176-8x8 --workers 12 --line-search --coeff-space-box
  --second-order --delta0 0.0025 --delta-cap 0.01 --material-bar 0.105
  --max-opt-evals 1000 --max-verif-evals 12 --total-hours 12
  --opt-cutoff-hours 10` (podman, `--network none`, cores 2-13). Code
  15521d6 (818 tests incl. an end-to-end runner smoke on a toy model).
  > **RESULT (2026-09-16 17:54): "second order breaks the tie" — confirmed worst-of-three V
  > **0.086607** (bar 0.105; substantial 0.100 also met; label: substantial but
  > infeasible progress).** 0.110645 -> 0.086607 in 791 evals (576 model, 215
  > trials), 15 accepted, 0 non-convergent, 2.6 h; median rho 0.99 on accepted
  > steps (the bowl model is right); new modes 17-27% of every step. The
  > three-way tie opened (QI sole worst after step 5) and re-formed at 0.087
  > (QI 0.0866 / mirror 0.0862 / flux 0.0855) at the model's bowl bottom;
  > radius floor at 2.5 h; the recovery could not be funded under the 1,000
  > ceiling (stop reason, not a defect). Best point:
  > `results/p3-e7-176-8x8/best_candidate_V0.08661_8x8.json`; RESULT.md there.
  > Largest single-experiment gain of the campaign (more than E2+E4+E6+E6b
  > combined). Follow-up: E7b (fresh step-scale model at this point).
  > **E7b — E7 continued with a fresh model (pre-registered and LOCKED 2026-09-16 17:55,
  > launched).** Why: E7 stopped at the bowl bottom of a model built 0.024
  > earlier and refreshed only partially; the recovery could not be funded.
  > Protocol: identical to E7 (second order, step-scale probes, coefficient
  > box, line search, cap 0.01, no allowances, raw well >= 0, one recovery,
  > worst-of-three). Start: E7's adjudicated best (0.086607). Fresh J0 + C0
  > = 576 evaluations. Ceilings: 1,400 optimisation evaluations (two
  > 576-probe models plus ~250 trials fit), 12 verification, 14 h
  > (optimisation cutoff 12 h). Bars: material beyond E7 = confirmed
  > worst-of-three V <= **0.080**; substantial (0.100) already met, not
  > re-claimed; feasible = 0.01 on all three evaluations. "Bowl bottom
  > again" = confirmed >= 0.084 with the floor/eps signature, read as the
  > diagonal second-order limit at 8x8 from this lineage (the next lever
  > would be off-diagonal curvature or a new lineage, not more of the same).
  > Between 0.080 and 0.084 = material progress, recorded as such. Also
  > report rho and whether the tie re-forms. Command: as E7 with
  > `--start-candidate results/p3-e7-176-8x8/best_candidate_V0.08661_8x8.json
  > --out-dir results/p3-e7b-176-8x8 --material-bar 0.080 --max-opt-evals
  > 1400 --total-hours 14 --opt-cutoff-hours 12`.
  > **RESULT (2026-09-16 21:36): material beyond E7 — confirmed worst-of-three V **0.065547**
  > (bar 0.080; label substantial but infeasible).** 0.086607 -> 0.065547 in
  > 1,273 evals (two models, 121 in trials), 8 accepted, 0 non-convergent,
  > median rho 1.08, 3.7 h; each fresh model bought 0.01-0.02 then bottomed
  > out (tie re-formed: QI 0.0656 / mirror 0.0651 / flux 0.0651); stall after
  > the qualified recovery. Best `results/p3-e7b-176-8x8/best_candidate_V0.06555_8x8.json`;
  > RESULT.md there. Cumulative E7 + E7b: 0.1106 -> 0.0655.
  > **E7c — the chain continues (pre-registered and LOCKED 2026-09-16 21:36, launched):**
  > identical protocol and ceilings from E7b's best. Bars: material beyond E7b
  > = confirmed <= **0.055**; feasible = 0.01 on all three evaluations;
  > "rebuild buys nothing" = confirmed >= 0.063 (less than 0.0025 gain from a
  > fresh model) with the floor signature = the chain's limit, stop chaining.
  > Between = material, chain again. Out `results/p3-e7c-176-8x8/`.
  > **Chain rule, fixed 2026-09-16 23:00 (applies from E7d onward, so no bar is
  > chosen per run):** each link starts from the previous link's adjudicated
  > best with the identical command and ceilings; material bar = previous
  > confirmed best minus 0.010; feasible = 0.01 on all three evaluations;
  > chain limit = confirmed best within 0.0025 of the start (a rebuild buys
  > nothing) with the floor signature -> stop and close. Before any
  > "verified" claim the best boundaries are re-evaluated in the
  > leaderboard-pinned image (`private-image:leaderboard-fidelity`,
  > constellaration 0.2.1 / vmecpp 0.4.2): the P1 validation reproduced the
  > board exactly but did not exercise QI, flux or well. Adjudication note
  > for the write-up: confirmations are identical to 1e-6 because the
  > evaluator is deterministic; worst-of-three guards against process state
  > leakage, not measurement noise.
  > **E7c RESULT (2026-09-17 01:16): material, chain again — confirmed worst-of-three V
  > **0.056868** (bar 0.055 missed by 0.0019; chain limit 0.063 cleared by
  > 0.0061).** 0.065547 -> 0.056868 in 1284 evals, 10 accepted, 0
  > non-convergent, median rho 1.03, 3.654 h. First model bought only
  > 0.0033, the recovery model 0.0053. Per-link gain falling: 0.024 / 0.021
  > / 0.0087. Tie persists (QI 0.0569 / mirror 0.0566 / flux 0.0563). Best
  > `results/p3-e7c-176-8x8/best_candidate_V0.05687_8x8.json`. Next per the
  > chain rule: E7d, bar 0.047. Cumulative E7-E7c: 0.1106 -> 0.0569.
- **E8 — the population family (NSGA-II, dossier B3).** Pop 200 on P3
  seeded from all retained candidates + H5 seeds; real evaluate() on the
  final population; SUPPORTED > 129.8 (published ALM-NGOpt reference).
  The one different method family still queued. ~24 h.
- **E9 — construction with the preservation rule from the seed** (E3
  generalised): every H5 seed with >= 3 constraints satisfied, stage A + B.
- **E10 — smoothed-gate flux surrogate for the search model (added 2026-09-16
  after E6b's mechanism).** The benchmark's flux compression gates a grid
  mean with a hard heaviside; a logistic gate of relative width w (units of
  the drift rms) computed from the SAME equilibrium gives a differentiable
  model of flux for the Jacobian only; canonical acceptance unchanged.
  **E10-pilot (Phase 0 evidence, ~95 evaluations, ~12 min, DRAFT, lock at
  launch after E3 stage A frees the cores):** at E6b's best point
  (`best_candidate_V0.11065_8x8.json`, 8x8): centre + (+h, -h) probes for 24
  new-mode and 8 old-mode coordinates (h 0.01, the run's h) + re-evaluation
  of the 30 largest archived E6b line points, with widths w in {0.02, 0.05,
  0.1, 0.2}. Success for a width: (a) forward/backward FD sign agreement
  >= 90% in the new-mode columns (real flux: 31%); (b) the surrogate's
  first-order prediction of the flux change at the 30 line points (using
  the surrogate's own FD gradient restricted to the probed coordinates is
  NOT possible, so: compare the surrogate CHANGE centre -> point with the
  real flux change) has sign agreement >= 80% and Spearman >= 0.6. If a
  width passes both, pre-register E10 (final runner, flux row of J from the
  surrogate at that width, acceptance canonical, start E6b's best, coefficient
  box, 700 evals). If none passes, E10 is dropped and E7 (SR1 secant
  curvature) stands as the next 8x8 test. Every pilot evaluation is
  canonical and archived; no label is claimed from the pilot.
  > **E10-pilot LOCKED 2026-09-16 14:42, launched as drafted** (24 new + 8 old
  > coordinates sampled with seed 0, h 0.01, 30 largest E6b line points,
  > widths 0.02/0.05/0.1/0.2, 95 canonical evaluations, out
  > `results/p3-e10-pilot/`). Pass rule unchanged.
  > **RESULT (2026-09-16 14:57): no width passes both criteria -> E10 as specified is
  > DROPPED.** 95/95 evaluations converged. (a) FD sign agreement in the
  > new-mode columns at h = 0.01: real flux 0.375; surrogate 0.625 / 0.667 /
  > 0.625 / 0.542 at w = 0.02 / 0.05 / 0.1 / 0.2 (bar 0.90): FAIL at every
  > width. (b) step-scale prediction at the 30 largest E6b line points:
  > sign agreement 0.87 / 0.93 / 0.93 / 0.80, Spearman 0.61 / 0.62 / 0.66 /
  > 0.58 (bars 0.80 / 0.60): PASS at w = 0.02, 0.05, 0.1. Reading: smoothing
  > the gate makes flux predictive at the scale of the accepted steps but
  > not at the h = 0.01 probe scale (probe displacement ~1e-5 coefficient
  > units in new modes), so the residual roughness at that scale is not only
  > the gate. Old-mode columns also improve (0.375 -> 0.75-0.875).
  > `results/p3-e10-pilot/REPORT.json`. (Analysis crashed once on a JSON key
  > type; fixed with a regression test, re-run from the archive, no
  > evaluation repeated.)
  > **E10-pilot-2 — probes at step scale (pre-registered and LOCKED 2026-09-16 14:58,
  > launched):** same point, same 24 new + 8 old coordinates (seed 0), but the
  > probe step is h x box multiplier per coordinate (new modes 0.07-1.5 scaled
  > units = ~1e-4 coefficient units, the scale of E6b's accepted steps; old
  > modes unchanged at 0.01), no line points, 65 canonical evaluations, out
  > `results/p3-e10-pilot2/`. Question: is the finite difference smooth at the
  > step scale? Pass = real flux OR some surrogate width reaches FD sign
  > agreement >= 0.90 in the new-mode columns. Decision: pass with real flux
  > -> E10 becomes "per-coordinate probe scale" only (the final runner's
  > Jacobian probes at h_j = h x mult_j, real metrics, acceptance unchanged);
  > pass only with a surrogate width -> E10 = probe scale + surrogate flux row
  > at that width; fail -> E10 dropped for good, E7 next. No label from the
  > pilot.
  > **RESULT (2026-09-16 15:10): FAIL -> E10 dropped for good.** 65/65 converged. FD sign
  > agreement in the new-mode columns at step scale: real flux 0.50; surrogate
  > 0.75 / 0.79 / 0.75 / 0.71 (w 0.02 / 0.05 / 0.1 / 0.2); bar 0.90.
  > `results/p3-e10-pilot2/REPORT.json`.
  > **Post hoc, decisive for E7's design:** the disagreements are NOT noise.
  > Classifying each probed new-mode direction by the signs of (r+ - r0) and
  > (r- - r0): QI is V-shaped (both raise it) in 22/24, mirror in 20/24, flux
  > in 12/24, iota 7/24, well 6/24; Λ-shaped: 0/24 for QI, mirror and flux.
  > Median |quadratic|/|linear| term at the probe (= box) scale: QI 3.2,
  > mirror 2.9, flux 1.0. The point sits in a convex bowl of QI and mirror
  > with respect to every new mode; a linear model overshoots at the box edge,
  > the ratio policy shrinks the radius, and progress dies at the floor. The
  > smoothed-gate surrogate turns flux monotone in 19/24, so the gate does
  > contribute flux's V-shape, but QI/mirror curvature is the larger effect
  > and is not a gate artefact. A diagonal second-order model from the same
  > probes (curvature = (r+ - 2 r0 + r-)/h_j^2 at step scale) captures all of
  > this at zero extra evaluation cost: E7's design below is amended
  > accordingly (diagonal curvature from step-scale probes, not SR1, which
  > would need gradients we never have).
- **E11 probe — external seeds on the P3 constraints (measurement, no label;
  recorded 2026-09-17 01:17 before running).** Matt's question: is it the method
  or the lineage? Step 1: canonical P3 residuals, raw metrics and would-be
  solo score of davidkh's public boundaries we hold (leaderboard-validation
  `davidkh_top` / `davidkh_2nd`, P1 winners; `family-study/targets/
  01_davidkh_0.6361.json`, the 7x7 QI design). One high-fidelity forward
  model each, `scripts/score_mhd_stable.py --boundary`. Step 2 (only if a
  design has V <= 0.30 with at most two constraints violated): pre-register
  a second-order restoration from it (E5-style multi-start with an external
  seed), same protocol, bar = V/2 or better. Out `results/p3-e11-probe/`.
  > **Step 1 RESULT (2026-09-17 01:20):** davidkh's 7x7 QI design (nfp 3):
  > canonical residuals iota -0.044, QI -0.144, mirror -0.207 (all satisfied,
  > large margins; raw log10_qi -4.005, mirror 0.198, iota/nfp 0.261), flux
  > +0.111 (raw 1.0001), well +0.305 (raw -0.0305); V = 0.3047. Objectives:
  > aspect 10.01, min gradient scale length 12.72 -> would-be solo score
  > 137.2 (above the 129.8 ALM-NGOpt baseline by itself). The two P1
  > geometrical winners cannot be QI-evaluated ("Minimum of B is at the
  > boundary"): not P3 material. `results/p3-e11-probe/report.json`.
  > **Step 2 goes ahead with one declared deviation:** V 0.3047 is 1.6% over
  > the 0.30 threshold; the condition that threshold stood for (at most two
  > constraints violated, QI family intact) is met with margin. E11 runs
  > BEFORE E7d (more informative; Matt: "more interested in the method").
  > **E11 — second-order restoration from davidkh's QI design (pre-registered
  > and LOCKED 2026-09-17 01:21, launched).** Start: `results/p3-e11-probe/
  > start_candidate_davidkh_V0.3047.json` (native 7x7, 224 coordinates,
  > scaling 1.0, no coefficient box: every coordinate is an "old" mode).
  > Protocol: the E7 second-order protocol unchanged (step-scale probes at
  > h 0.01 = the cap-scale for native modes, diagonal curvature, SLSQP
  > minimax step, line search, no allowances, raw well >= 0 once reached,
  > tolerance preserved for the satisfied rows iota/QI/mirror, eps 1e-4,
  > ratio trust policy 0.0025 / 0.01, one recovery, worst-of-three). Fresh
  > model = 448 evaluations. Ceilings: 1,200 optimisation evaluations, 12
  > verification, 12 h (cutoff 10 h). Bars: material = confirmed
  > worst-of-three V <= **0.15** (V/2); substantial <= 0.10; **feasible =
  > 0.01 on all three evaluations** (this would be the campaign's first
  > verified P3-feasible boundary AND a would-be solo score near 137 if the
  > objectives survive; both to be checked in the leaderboard-pinned image
  > before any claim). Method-vs-lineage reading: feasible or V <= 0.05 here
  > = the method transfers and the 176 lineage was the limit; stall above
  > 0.15 with the floor signature = the method itself is limited on the
  > well/flux pair too. Also report aspect and scale length at every
  > accepted step (objectives are not protected by the protocol; their
  > drift decides whether a feasible point is worth anything). Command:
  > `python3 scripts/p3_final_restoration.py --checkpoint
  > results/m1-post-ladder-d176/seed_0176/checkpoint.json --start-candidate
  > results/p3-e11-probe/start_candidate_davidkh_V0.3047.json --out-dir
  > results/p3-e11-davidkh-7x7 --workers 12 --line-search --second-order
  > --delta0 0.0025 --delta-cap 0.01 --material-bar 0.15 --max-opt-evals
  > 1200 --max-verif-evals 12 --total-hours 12 --opt-cutoff-hours 10`
  > (podman, `--network none`, cores 2-13). E7d follows E11.
  > **Run 1 ABORTED by a declared defect (2026-09-17 02:22):** after the 448-probe
  > model, all 8 trials failed inside the solver ("inequality constraints
  > incompatible"): the runner applied the raw-well cap (well residual <= 0)
  > unconditionally, and the acceptance rule required raw well >= 0 on every
  > accepted step. Every earlier start already had well >= 0, so this never
  > bit; this start has well -0.030. The pre-registration says "raw well >= 0
  > once reached", so the fix (8a58830, regression tests first) applies the
  > cap and the acceptance requirement only once the incumbent's well is
  > non-negative, and never gives it back after that. Aborted dir kept
  > (`results/p3-e11-davidkh-7x7/`; its REPORT adjudicated a probe point at
  > 0.270). **Run 2 launched** with the identical command except
  > `--out-dir results/p3-e11-davidkh-7x7-run2 --reuse-jacobian-archive
  > results/p3-e11-davidkh-7x7` (the 448 paid probes rebuild J and C; charged
  > as model #1; repeatability gate re-run). Bars unchanged.
  > **Run 2 ABORTED by a second declared defect (2026-09-17 02:34):** 8 trials, 16
  > line-point evaluations, 0 steps evaluated: the line-search selector
  > discarded every grid point whose PREDICTED well was still positive
  > (same unconditional rule, a different site), and the recovery selector
  > had it too. Fixed in 5a84972 (regression tests first); all five well
  > checks in the runner are now "once reached". The 16 line points showed
  > the well responding strongly (0.305 -> 0.249 at a full step of the
  > first direction, with QI spent to +0.011 there and intact at the half
  > step), so the landscape is workable. Aborted dir kept
  > (`results/p3-e11-davidkh-7x7-run2/`). **Run 3 launched**, identical
  > command with `--out-dir results/p3-e11-davidkh-7x7-run3` and the same
  > `--reuse-jacobian-archive results/p3-e11-davidkh-7x7`. Bars unchanged.
  > Lesson for the write-up: every prior start had well >= 0, so three
  > sites encoding "well must be >= 0" as unconditional passed 30+ runs and
  > the toy end-to-end test (whose toy also starts with well satisfied);
  > the toy test now needs a well-violated variant.
  > **Run 3 RESULT (2026-09-17 04:42): progress gate not met (bar 0.15), but NOT a stall.**
  > Adjudicated best 0.17746 (a rejected trial with QI +0.154); last accepted
  > (preservation-respecting) V **0.19814** from 0.3047 in 537 evals, 13 of
  > 27 trials accepted, 0 non-convergent, median rho 1.001, 2.1 h. Stopped
  > by the protocol's fixed recovery-qualification threshold (V <= 0.130),
  > written for 0.13-class starts, while still descending. The descent
  > bought well (raw -0.0305 -> -0.0198) with QI margin (log10_qi -4.005 ->
  > -3.471, now at the tolerance edge) and let flux rise (1.0001 -> 1.040);
  > objectives largely kept (aspect 10.25, scale length 12.22, would-be
  > solo 129). `results/p3-e11-davidkh-7x7-run3/RESULT.md`.
  > **E11b (pre-registered and LOCKED 2026-09-17 04:42, launched):** chain link from the
  > run-3 incumbent (`incumbent_candidate_V0.19814_7x7.json`), identical
  > protocol and ceilings, fresh 448-probe model, ONE declared parameter:
  > `--recovery-qualify-v 0.30` (the recovery qualifies on >= 1 material
  > step alone; the 0.130 threshold is a 0.13-class artefact; parameter
  > added in 97e9ac2, default unchanged). Bars unchanged: material 0.15,
  > substantial 0.10, feasible 0.01; chain limit = gain < 0.0025. The
  > question it answers: can well and flux both fall with QI pinned at its
  > tolerance? Out `results/p3-e11b-davidkh-7x7/`.
  > **E11b RESULT (2026-09-17 07:40): progress gate not met (adjudicated 0.15575 on a
  > QI-broken probe; bar 0.15); accepted incumbent V **0.16314** from
  > 0.19814 (gain 0.035), 11/19 accepted, median rho 1.00, 985 evals, 2.9 h,
  > stalled after recovery.** Answer to E11b's question: partly yes: the
  > first joint flux+well steps on this seed happened (P:T14-T16), then
  > flux and well tied at 0.163 with QI pinned at its edge. Non-convergence
  > climbing (12.6% of evaluations; model columns 193 -> 155). Objectives
  > aspect 10.39 / scale length 12.21. `results/p3-e11b-davidkh-7x7/RESULT.md`.
  > **E11c (LOCKED 2026-09-17 07:40, launched):** chain link 3, identical, from the E11b
  > incumbent; bar 0.146 by the chain rule (0.15575 - 0.010); feasible 0.01;
  > chain limit gain < 0.0025; additionally, if non-convergent probes exceed
  > half the columns in either model, record "technical limitation of this
  > seed under VMEC++ at high fidelity" and stop the chain.
  > **E11c RESULT (2026-09-17 10:53): bar 0.146 MET — adjudicated best 0.11363** (a
  > rejected trial with QI +0.114; label material_but_infeasible_progress); accepted incumbent V
  > **0.14064** from 0.16314 (gain 0.0225), 9/27 accepted, rho 1.00,
  > 981 evals, 3.055 h; models 157 and 214 of 224 columns (non-convergence
  > eased at this point: 83/981). Flux and well descend together with QI
  > pinned; the adjudicated points show the pair goes lower still if QI is
  > released past tolerance. `results/p3-e11c-davidkh-7x7/RESULT.md`. (The
  > box was rebooted at 10:47, after REPORT.json was written: nothing lost.)
  > **E11d (LOCKED 2026-09-17 10:53, launched):** link 4, identical, from the E11c
  > incumbent; bar 0.104 (chain rule); feasible 0.01; chain limit gain <
  > 0.0025; half-columns stop rule as E11c.
  > **E11d RESULT (2026-09-17 13:49): bar 0.104 missed by 0.0050 — adjudicated best
  > 0.10899; accepted incumbent V **0.11625** from 0.14064 (gain 0.0244),
  > 10/25 accepted, rho 1.00, 983 evals, 2.929 h; models 216/218 of
  > 224 columns; non-convergent 14. Flux fell below the well for the first
  > time; both descend with QI pinned. Raw: well -0.0116, flux
  > 1.0028, aspect 10.48, scale length 11.92.
  > `results/p3-e11d-davidkh-7x7/RESULT.md`. Chain rule: material, chain
  > again. **E11e (LOCKED 2026-09-17 13:49, launched):** link 5, identical, from the
  > E11d incumbent; bar 0.099; feasible 0.01; chain limit gain < 0.0025;
  > half-columns stop rule.
  > **E11e RESULT (2026-09-17 17:02): bar 0.099 MET — adjudicated best **0.05016**, the
  > lowest confirmed V of the campaign; accepted incumbent V **0.05190** from
  > 0.11625 (gain 0.064, largest link of either chain), 10/25 accepted,
  > rho 1.00, 979 evals, 3.208 h; models 219/217 of 224; non-convergent 12.
  > Mirror margin drawn down for the first time (-0.230 -> -0.170); QI at
  > its edge. Raw: well -0.0052, flux 0.9465, mirror 0.2074, aspect
  > 10.63, scale length 11.27 (would-be solo 115).
  > `results/p3-e11e-davidkh-7x7/RESULT.md`. **E11f (LOCKED 2026-09-17 17:02, launched):**
  > link 6, identical, from the E11e incumbent; bar 0.04; feasible 0.01;
  > chain limit gain < 0.0025; half-columns stop rule.
  > **E11f RESULT (2026-09-17 19:51): CHAIN LIMIT — accepted V 0.05081 from 0.05190 (gain
  > 0.0011 < 0.0025); adjudicated best 0.04978; 2/11 accepted, 19 solves
  > with no descent, 929 evals, 2.8 h.** The davidkh chain ends at
  > confirmed **0.04978** (campaign best): raw well -0.0051, flux 0.9456,
  > log10_qi -3.473 (edge), mirror 0.2075, aspect 10.63, scale length
  > 11.31. Stop chaining this seed. `results/p3-e11f-davidkh-7x7/RESULT.md`,
  > best `best_candidate_V0.05081_7x7.json`. Next: E7d (176 lineage, bar 0.047) so both chains
  > have a measured limit, then close (fidelity check in the pinned image
  > for both bests, entitled claim rewrite, capstone section).
  > **E7d (LOCKED 2026-09-17 19:51, launched):** 176-lineage chain link 4 from
  > E7c's best (0.056868), identical command to E7c, bar 0.047 (chain rule),
  > feasible 0.01, chain limit gain < 0.0025. Out `results/p3-e7d-176-8x8/`.
  > **E7d RESULT (2026-09-17 23:27): 176 chain STOPPED — adjudicated best 0.05357 (bar
  > 0.047 not met); accepted V 0.05443 from 0.056868 (accepted gain 0.0024 <
  > 0.0025; confirmed gain 0.0033, marginally above it). Declared judgement:
  > the rule's conditions leave this gap uncovered; stopped on the collapsed
  > trend (0.024 / 0.021 / 0.0087 / 0.0033). Three-way tie persists (QI
  > 0.0544 / mirror 0.0533 / flux 0.0524). `results/p3-e7d-176-8x8/RESULT.md`.
  > **Both chains are at their limits. Close-out begins** (fidelity check,
  > entitled claim, capstone, triage); no further runs without Matt.

## Close-out 1 — fidelity check in the leaderboard-pinned image (2026-09-17 23:39)
Image rebuilt from `docker/leaderboard-fidelity/Dockerfile` (constellaration
0.2.1 / vmecpp 0.4.2, verified by pip). Both chain bests exported as
leaderboard-format boundaries (`results/p3-close-out/*.boundary.json`,
`scripts/p3_promote_candidate.py --to-boundary`; the export reproduces each
run's residuals in the phase1 image to 1e-5).
- **176 lineage** (`lineage176_chain_incumbent_V0.0544`): pinned = phase1
  on all five residuals to 1e-6 (V 0.054433). Fidelity holds.
- **davidkh chain** (`davidkh_chain_incumbent_V0.0508`): QI, mirror, flux,
  well and both objectives identical to 1e-6, but the IOTA residual is
  +2.049 in the pinned image vs -0.049 in ours: `constellaration` 0.2.1
  computes `lb - iota` with no absolute value (problems.py line 174),
  0.2.6 uses `lb - |iota|`, and the davidkh seed has iota/nfp = -0.262. A
  negative-iota stellarator is the mirror image of a positive-iota one, so
  the equivalent boundary is a reflection. Tested both handedness flips:
  z -> -z (negate z_sin) flips iota but is NOT a symmetry of the
  benchmark's flux metric (0.946 -> 1.20 raw; the heaviside gate's sign
  convention assumes one handedness), while phi -> -phi (n -> -n for m > 0,
  z_sin[0, :] negated) reproduces EVERY residual to 1e-5 in BOTH images
  with iota/nfp = +0.2623 (`davidkh_chain_incumbent_V0.0508.phireflected.
  boundary.json`). **Verdict: fidelity holds for both bests; the davidkh
  representative to quote or submit is the phi-reflected one.** Consequence
  for the record: davidkh's own P2 design has negative iota and would fail
  P3's iota constraint under 0.2.1 as written; under 0.2.6 it passes. Which
  version the live board runs is not knowable from here; the reflected
  boundary is safe under both.
Then close: claim wording updated; dossier triage; capstone P3 section.

## E6 — 8x8 restoration from E2's tied 0.1153 boundary (pre-registered 2026-09-16, Matt: E6 before E3)
**Why:** two lineages (176 at 0.115, 49 at ~0.142) show the same signature:
the QI-family constraints tie, every direction that lowers one raises
another, steps collapse to eps with the model accurate. In 80 coordinates
there may be no direction left; the P2 winners' QI lives in high-order
dressing above 4x4. E6 tests whether more coordinates move the tie.
**Start:** E2's confirmed best (V 0.115331) re-expressed at 8x8 with the
new modes at zero: `results/p3-e2-176/best_candidate_V0.11533_8x8.json`
(288 coordinates; round-trip through the engine's own machinery exact).
The repeatability gate must reproduce the 4x4 residuals to 1e-4 (same
shape) — a failure there is a representation defect, stop.
**Protocol:** the final-attempt protocol with the line search (as E2),
no regression allowances, well >= 0, tolerance preserved; fresh Jacobian
at 8x8 = 576 evaluations (Jacobian #1 of 2); pilot gate V <= 0.130 at 100
evaluations after the Jacobian (already met at the start; the gate's
job is the block rule); block rule every 200; one recovery.
Ceilings: 1,400 optimisation evaluations, 12 verification, 2 Jacobians,
16 h (optimisation cutoff 14 h). Trust region and eps unchanged.
**Outcomes:** material beyond existing evidence = confirmed V <= 0.110
(clearly below 0.1153); substantial <= 0.100; feasible = 0.01. "8x8 does
not move the tie" = confirmed best >= 0.113 with the eps/floor signature
recurring = strong evidence that the tie is not a coordinate-count
limit at 8x8 either (recorded, not "locally optimal"). Also report: the
first accepted step's size vs E2's first step; whether the new modes
carry the step (fraction of |d| in |m|>4 or |n|>4 coordinates).
Cost ~4-8 h. Then E3 on whichever representation E6 justifies.
> LAUNCHED 2026-09-16 after E4, unchanged; phase 0 on the 8x8 start reproduced the 4x4 residuals exactly. Output `results/p3-e6-176-8x8/`.
> **RESULT: progress gate not met** — confirmed V 0.111728 (from 0.115331; 1,288 evals, 8 steps, same tie signature). **But the new modes were never in play:** 0.1% of the step energy, because the engine's spectrum scaling gives them a 55x smaller coefficient reach inside the same box, while per coefficient they are ~5x MORE sensitive. E6 does not answer the 8x8 question -> E6b. `results/p3-e6-176-8x8/RESULT.md`.

## E6b — 8x8 with an equal box in coefficient space (pre-registered 2026-09-16)
**Why:** E6's trust box in scaled coordinates suppressed the 208 new modes
(scale 0.0009 vs 0.05 per unit). Give every coordinate the same reach in
COEFFICIENT units: per-coordinate radius delta_j = delta x s_ref / scale_j
with s_ref = the median old-mode scale (old modes keep exactly E6's box;
new modes get a 55x larger scaled box, i.e. the same coefficient box).
**Start:** E6's start (0.115331, 8x8), reusing E6's linearisation-0
Jacobian (576 paid evaluations, exact at that point); no new Jacobian
unless the recovery needs one (Jacobian #2 allowed). Repeatability gate as
before (skipped if the archive's start row is reused; recorded).
**Protocol:** otherwise identical to E6 (line search, no allowances, well
>= 0, tolerance preserved, eps 1e-4, ratio trust policy on delta with the
per-coordinate multiplier fixed, one recovery, stop rules). Report the
displacement-energy fraction in new modes for every accepted step (it must
be substantial for E6b to count as a test at all; if it stays < 5% the
experiment is "inconclusive: new modes still unused", not a negative).
Ceilings: 700 optimisation evaluations (a second Jacobian fits), 12
verification, 10 h. Material bar 0.110 (E6's), substantial 0.100.
**Outcomes:** breaks the tie = confirmed V <= 0.110 with >= 20% of step
energy in new modes; does not = tie recurs with new modes used (>= 20%)
-> the strongest available evidence that coordinate count is not the
limit at 8x8; inconclusive = new modes still < 5% of step energy.

> **LOCKED 2026-09-16 08:04** (launch follows this edit). Implementation declared before
> launch: (1) **control condition kept literally**: old (4x4) modes have
> multiplier exactly 1 (E6's box); each of the 208 new modes gets
> median(old scale)/scale_j, range 7.4-148 (median ~55). (2) The
> repeatability gate is still run (3 verification evals), stricter than the
> pre-registration's "may skip". (3) E6's J0 (576 probes, 288/288 columns
> OK, start dx = 0.0) is loaded from `results/p3-e6-176-8x8/archive.jsonl`
> and **charged as full Jacobian #1 of 2**, so one fresh recovery Jacobian
> may still be taken. (4) The Broyden-update guard (step <= 2 delta_cap) is
> measured in trust units |d_j|/m_j; in raw scaled coordinates every
> coefficient-box step would fail it and secant updates would be silently
> disabled. (5) Trust radius delta itself is unchanged (0.00125 start,
> 0.0025 cap, ratio policy); only the per-coordinate multiplier differs.
> **Dry run on E6's J0 (linear model, NOT a result):** isotropic box at
> delta0 predicts V 0.115 -> 0.098 with 0.2% of step energy in new modes;
> coefficient box predicts 0.115 -> 0.021 with 28% in new modes (at the
> cap it predicts feasibility, 27%). The linear model will overpredict; the
> ratio policy decides. Per coefficient unit the new modes' Jacobian columns
> are ~3x larger than the old ones' (median 48 vs 16).
> Command: `python3 scripts/p3_final_restoration.py --checkpoint
> results/m1-post-ladder-d176/seed_0176/checkpoint.json --start-candidate
> results/p3-e2-176/best_candidate_V0.11533_8x8.json --out-dir
> results/p3-e6b-176-8x8 --workers 12 --line-search --coeff-space-box
> --reuse-jacobian-archive results/p3-e6-176-8x8 --material-bar 0.110
> --max-opt-evals 700 --max-verif-evals 12 --total-hours 10
> --opt-cutoff-hours 8` (podman, `--network none`, cores 2-13).

> **In-flight observation 08:31 (post hoc, NOT pre-registered; does not
> change the labels):** 3 accepted steps in 11 trials (0.11533 -> 0.11266,
> new-mode energy 27-28% each), then the radius collapsed to the floor:
> every rejected trial had flux RISING while the model predicted it falling,
> even at delta 1e-5; the drift check flagged the flux row; recovery started
> at 0.43 h from the 0.11266 point (fresh Jacobian, 576 evals). Diagnostic
> from E6's own probes (no new evaluations): for flux in the 208 new-mode
> directions the forward/backward one-sided differences disagree in sign 69%
> of the time and the median second difference |r+ - 2r0 + r-| (2.2e-3) is
> 2.3x the first difference (9.5e-4). Flux is not smooth at the probe scale
> in high-order modes; iota and well are smooth everywhere (second
> difference ~1e-6); QI is rough in the OLD modes too (sign flips 50%).
> Interpretation to test at adjudication: a first-order model cannot steer
> flux with the new modes, which is a method limit, not evidence about the
> geometry.
> **Mechanism (08:32, read from the library, `constellaration/mhd/
> turbulent_transport.py`):** flux compression = grid mean over field lines
> of grad r . grad r gated by `np.heaviside(curvature_drift, 0)`. The gate is
> a hard step: a tiny boundary change flips grid points across the
> curvature zero crossing and switches their whole contribution on or off,
> so the metric is piecewise-constant at fine scales and has no useful
> derivative there. Ripple-like high-order modes move the zero crossings
> most, which is why the new-mode flux columns are rough while iota and
> well (smooth equilibrium integrals) are not. Consequence: the linearised
> restoration is mismatched to the flux constraint at the scales the
> coefficient box reaches; the tie at ~0.11 includes a constraint this
> method cannot steer finely. Candidate follow-up (to pre-register only
> after E6b adjudicates): a smoothed-gate surrogate (sigmoid in place of
> heaviside) for the Jacobian only, canonical acceptance unchanged; or a
> derivative-free search over the new modes inside the coefficient box.
> Grid (library defaults, `forward_model.py`): one surface at s = 0.49,
> 101 field lines x 64 toroidal points = 6,464 gated points; a single gate
> flip moves the mean by about one part in 6,464 of the local integrand.
> **Even response (post hoc, from the same probes):** in 69% of the 208
> new-mode directions the +h and -h probes shift raw flux in the SAME
> direction (old modes: 43%); median |dflux| per new-mode probe 8.7e-4 (~5
> gate flips), never exactly zero. The new modes start at exactly zero
> amplitude, and a surface-averaged metric has no first-order response to
> an absent mode, only a second-order one; for ripple that is an increase
> either way. The central-difference Jacobian therefore reads the new-mode
> flux columns as ~0, the LP spends those coordinates on QI and mirror, and
> flux rises on the step regardless of sign. Two consequences for the
> queue: (a) any first-order method starting from zero-padded modes will
> repeat this; (b) E7 (second-order model) is now the well-motivated next
> step at 8x8, because a curvature term captures "new modes cost flux
> quadratically" and lets a QP trade it. Both to be pre-registered only
> after E6b adjudicates.
> **Curvature check (post hoc, E6b's recovery-Jacobian probes, no new
> evaluations):** the finite-difference diagonal curvature of flux in the
> new modes is positive in 94% of directions (median 16 per scaled unit^2)
> but it is NOT a usable quadratic model: it predicts a flux rise of ~13
> for a delta0 coefficient-box step where 0.03 was measured. The second
> difference at h = 0.01 is dominated by the stepwise gate noise (~1e-3),
> not by smooth curvature. Design consequence for E7: curvature must come
> from secant pairs at the step scale actually taken (SR1 from evaluated
> points), never from the h-probes; and the cleaner fix for the Jacobian is
> a smoothed-gate flux surrogate computed from the same equilibrium (E10
> candidate), with canonical acceptance unchanged. E6b's best is within
> 1.2e-4 trust units of the JR centre, so JR remains valid there.

> **RESULT (2026-09-16 10:31): progress gate not met; pre-registered reading "does not break the tie".** Confirmed worst-of-three V **0.110645** (bar 0.110, missed by 0.0006) from 0.115331; 696 evals (576 = recovery Jacobian; 120 in 32 trials), 7 accepted steps, 0 non-convergent, 2.4 h. New modes carried **26-28%** of the step energy on every accepted step (valid test; >= 20% reading applies). Tie recurred: QI 0.1082 / mirror 0.1081 / flux 0.1107, well spent to raw 8e-5, radius floor after the qualified recovery. Best 8x8 point so far: `results/p3-e6b-176-8x8/best_candidate_V0.11065_8x8.json`. Mechanism above (flux heaviside gate + even response of zero-amplitude modes) qualifies the reading: coordinates added to THIS first-order method do not break the tie; E7 (second-order) is the motivated next test at 8x8. `results/p3-e6b-176-8x8/RESULT.md`.

## Review round 8 (2026-09-18) — adopted
Verdict: numerical progress supported (45 adjudicated records match archives
and confirmations); mechanism and stopping claims too strong; two
implementation restrictions to fix before any new lever. Adopted in full:
1. Claim replaced by the reviewer's wording (brief §4); accepted endpoints
   (0.05443 / 0.05081, cross-version checked) kept distinct from lowest
   archived confirmed points (0.05357 / 0.04978, the latter reopening QI).
2. E7d: stopped by declared discretion, confirmed gain 0.0033 above the
   0.0025 criterion; E11f met it (start-to-confirmed 0.0021).
3. Mechanisms narrowed (convexity partly imposed by clipping; even response
   = bowl-like component only; flux metric piecewise smooth with possible
   jumps; rho validates the line predictor, not the multivariate model; E7
   changed probe scale and cap with model order).
4. Iota: 0.2.1 check is historical compatibility; live Space reportedly
   updated and unpinned; the midplane-reflection discrepancy is an
   unresolved orientation sensitivity, not attributed to the gate.
5. Provenance: public-seed chain is a refinement of davidkh's entry; reuse
   terms not established; no submission without them.
6. Next: (i) align the step model's satisfied-row bound with acceptance
   (tol, well 0 once reached) and make the line search select within ALL
   preservation rows and keep the best admissible sampled point; (ii) log
   the multivariate model's own prediction error separately from the line
   fit; (iii) only then a matched test of limited off-diagonal curvature;
   defer NSGA-II and smoothing. Confirmed in code/logs before adopting: the
   bound is `min(tol, 0)` at p3_feasibility_restoration.py:235/294; the line
   search discarded 34 admissible sampled points across the chains.
7. Audit qualifications: evaluation count to be reconciled from the logs
   (reviewer: 13,873 reopen incl. close-out; 14,361 incl. the final
   attempt); pre-registration immutability to be made checkable by dated
   snapshots committed to the repo at every lock (from now on).
