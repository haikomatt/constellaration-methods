"""Final canonical feasibility attempt from seed 176's lineage — the reviewer's
closing protocol, transcribed as ceilings and rules (vault
p3-final-feasibility-attempt-2026-09-16). Reuses the tested pure parts of
p3_feasibility_restoration (LP, Broyden, trust region, stall, evaluator).

Phases:
  0. Repeatability: reconstruct + serialise the start boundary (Arm F trial 3),
     evaluate it twice in fresh canonical processes plus the 0.152095 control;
     require <= 1e-4 per component vs the archive; eps = max(1e-4, 5 e_rep).
  1. Fresh 160-evaluation Jacobian at the start (full Jacobian #1 of 2).
  2. Pilot: 100 optimisation evaluations; funding gate V <= 0.130.
  3. Continuation in 200-evaluation blocks; progress >= min(0.002, 0.1(V-0.01)).
  4. ONE recovery on the first failed gate / 8 non-material trials / floor /
     refresh request: lowest-V complete archive point meeting preservation,
     fresh Jacobian (#2), radius reset, 20 trials; qualifies only with >= 1
     material step and V <= 0.130.
  5. Stop conditions: recovery failure, runner error, > 30% non-convergence
     in the last 20 trials, hour-20 optimisation cutoff, evaluation ceilings.
  6. Adjudication: archive frozen; three lowest-V distinct complete
     candidates re-evaluated twice each in fresh processes; worst V of three;
     official feasibility predicate; labels. Any benchmark-feasible-looking
     point gets priority confirmation immediately.
Acceptance: V_new <= V_old - eps, all residuals finite, raw well >= 0,
tolerance-satisfied constraints preserved. NO regression allowances.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from constellaration_methods.restoration.p3_feasibility_restoration import (
    TOL,
    Budget,
    CanonicalEvaluator,
    RatioTrustRegion,
    RestorationLPInfeasible,
    StallCounter,
    broyden_update,
    central_difference_curvature,
    central_difference_jacobian,
    curvature_from_archive,
    directions_from_archives,
    drift_rows,
    fill_with_svd,
    jacobian_from_archive,
    probe_points,
    quadratic_model_step_terms,
    restoration_step,
    restoration_step_quadratic,
    subspace_curvature_from_probes,
    subspace_probe_points,
    worst_columns,
)


# --------------------------------------------------------------------------- #
# pure rules (tested without VMEC)
# --------------------------------------------------------------------------- #
def eps_from_repeatability(archived, repeats, max_component: float = 1e-4):
    """(ok, e_rep, eps): ok iff every repeat agrees with the archive within
    max_component per residual; e_rep = largest discrepancy; eps =
    max(1e-4, 5 e_rep). An empirical safeguard, not a statistical bound."""
    archived = np.asarray(archived, float)
    e_rep = 0.0
    for rr in repeats:
        rr = np.asarray(rr, float)
        if rr.shape != archived.shape or not np.all(np.isfinite(rr)):
            return False, float("inf"), float("inf")
        e_rep = max(e_rep, float(np.abs(rr - archived).max()))
    ok = e_rep <= max_component
    return ok, e_rep, max(1e-4, 5.0 * e_rep)


def representation_of(cand: dict) -> dict:
    """The (boundary template, mode budget, nfp, scaling) a candidate must be
    rebuilt with. Every candidate carries its own; never borrow the run's
    (E4 2026-09-16: the control was evaluated under the wrong nfp)."""
    rep = dict(
        boundary_dict=cand["boundary_dict"],
        mpol=int(cand["mpol"]),
        ntor=int(cand["ntor"]),
        nfp=int(cand["nfp"]),
        scaling=float(cand.get("scaling", 1.0)),
    )
    if cand.get("fake_quadratic"):
        rep["fake_quadratic"] = True  # test seam: toy model, no VMEC
        rep["fake_well_violated"] = bool(cand.get("fake_well_violated", False))
        rep["fake_near_feasible"] = bool(cand.get("fake_near_feasible", False))
        if cand.get("fake_coupling") is not None:  # E19 item 3: toy ratchet test
            rep["fake_coupling"] = cand["fake_coupling"]
    return rep


def V_of(r) -> float:
    return max(0.0, float(np.asarray(r, float).max()))


def accept_final(r_old, r_new, tol: float = TOL, eps: float = 1e-4) -> bool:
    """Final-attempt acceptance: evaluation succeeded with finite residuals;
    V_new <= V_old - eps; raw well >= 0; tolerance-satisfied constraints stay
    satisfied. No start-anchored allowances of any kind."""
    if r_new is None:
        return False
    r_old, r_new = np.asarray(r_old, float), np.asarray(r_new, float)
    if not np.all(np.isfinite(r_new)):
        return False
    if not (V_of(r_new) <= V_of(r_old) - eps):
        return False
    if r_old[4] <= 0.0 and r_new[4] > 0.0:
        return False  # raw well >= 0, once reached, is never given back
    return not any((r_old[i] <= tol) and (r_new[i] > tol) for i in range(len(r_old)))


def well_cap_for(r) -> dict:
    """The raw-well cap for the restoration step: residual <= 0 (raw well
    >= 0) once the incumbent's well is non-negative; no cap while the well
    is still violated (E11: applying it unconditionally made every step
    infeasible from a start with well < 0)."""
    return {4: 0.0} if float(np.asarray(r, float)[4]) <= 0.0 else {}


def recovery_qualified(material_steps: int, v: float, threshold: float) -> bool:
    """The single recovery qualifies with >= 1 material step AND V at or below
    the threshold (the final protocol's 0.130; E11 makes it a declared
    parameter because a 0.30 start cannot reach 0.130 in 20 trials)."""
    return material_steps >= 1 and v <= threshold


def block_progress_required(v_block_start: float) -> float:
    return max(0.0, min(0.002, 0.1 * (v_block_start - 0.01)))


@dataclass
class Ceilings:
    t_start: float
    total_hours: float = 24.0
    opt_cutoff_hours: float = 20.0
    max_opt_evals: int = 1200
    max_verif_evals: int = 20
    max_full_jacobians: int = 2
    opt_used: int = 0
    verif_used: int = 0
    full_jacobians: int = 0

    def can_optimise(self, now: float, n: int) -> bool:
        return (
            now - self.t_start
        ) < self.opt_cutoff_hours * 3600 and self.opt_used + n <= self.max_opt_evals

    def charge_opt(self, n: int):
        if self.opt_used + n > self.max_opt_evals:
            raise RuntimeError("optimisation evaluation ceiling")
        self.opt_used += n

    def can_full_jacobian(self) -> bool:
        return self.full_jacobians < self.max_full_jacobians

    def charge_full_jacobian(self):
        if not self.can_full_jacobian():
            raise RuntimeError("full-Jacobian ceiling")
        self.full_jacobians += 1

    def charge_verif(self, n: int):
        if self.verif_used + n > self.max_verif_evals:
            raise RuntimeError("verification evaluation ceiling")
        self.verif_used += n

    def past_total(self, now: float) -> bool:
        return (now - self.t_start) >= self.total_hours * 3600


def select_recovery_point(archive, incumbent_r, tol: float = TOL):
    """Lowest-V complete canonical archive point that keeps every constraint
    the incumbent already has within tolerance, with raw well >= 0 once the
    incumbent has it. Probes are eligible (a RESTART, not an accepted step).
    Ties -> earliest."""
    inc = np.asarray(incumbent_r, float)
    well_reached = inc[4] <= 0.0
    best = None
    for rec in archive:
        if not rec.get("complete") or rec.get("r") is None:
            continue
        rr = np.asarray(rec["r"], float)
        if not np.all(np.isfinite(rr)) or (well_reached and rr[4] > 0.0):
            continue
        if any((inc[i] <= tol) and (rr[i] > tol) for i in range(len(inc))):
            continue
        v = V_of(rr)
        if best is None or v < best[0]:
            best = (v, rec)
    return None if best is None else best[1]


def confirmations_complete(confirmations, n_rows: int = 5) -> bool:
    """True iff there are EXACTLY 2 confirmation vectors, each finite with
    n_rows components. Fix 3 (review): adjudicate_worst_v used to be handed
    an already-filtered list (failed confirmations dropped by the caller)
    and could declare feasibility from as little as one complete vector.
    Feasibility must never be declared without two complete confirmations."""
    confs = list(confirmations)
    if len(confs) != 2:
        return False
    for c in confs:
        arr = np.asarray(c, float)
        if arr.shape != (n_rows,) or not np.all(np.isfinite(arr)):
            return False
    return True


def adjudicate_worst_v(recorded, confirmations, tol: float = TOL):
    """Worst V across the recorded evaluation and its confirmations; official
    feasibility iff EVERY residual <= tol on all of them AND the
    confirmations are complete (confirmations_complete: exactly two,
    finite, five-component vectors)."""
    allr = [np.asarray(recorded, float)] + [np.asarray(c, float) for c in confirmations]
    worst = max(V_of(r) for r in allr)
    feasible = confirmations_complete(confirmations) and all(np.all(r <= tol) for r in allr)
    return worst, feasible


def label_outcome(
    worst_v: float, feasible: bool, technical: bool, material_bar: float = 0.130
) -> str:
    if technical:
        return "technical_limitation"
    if feasible:
        return "verified_p3_feasibility"
    if worst_v <= 0.100:
        return "substantial_but_infeasible_progress"
    if worst_v <= material_bar:
        return "material_but_infeasible_progress"
    return "progress_gate_not_met"


def quadratic_line_model(r0, r_half, r_full):
    """Per-residual quadratic r_i(t) = A_i t^2 + B_i t + C_i through t = 0, 1/2, 1
    (E2: the second-order line model along the LP direction)."""
    r0, rh, r1 = (np.asarray(v, float) for v in (r0, r_half, r_full))
    C = r0
    A = 2.0 * (r1 - 2.0 * rh + r0)
    B = r1 - r0 - A
    return A, B, C


def best_t_on_line(A, B, C, t_grid, require_well_nonpositive: bool = True, r0=None, tol=None):
    """t* on the grid minimising predicted V(t) = max_i r_i(t), subject to the
    predicted well residual (row 4) <= 0 once the well has been reached
    (E11: while the well is still violated the requirement is dropped) and,
    when r0/tol are given, to every row satisfied at r0 (<= tol) staying
    within tol (review round 8, 2026-09-18: the well-only check let the
    selector pick points that broke QI). (None, None) if no grid point
    qualifies."""
    A, B, C = (np.asarray(v, float) for v in (A, B, C))
    preserved = (
        [i for i in range(len(C)) if np.asarray(r0, float)[i] <= tol] if r0 is not None else []
    )
    best = (None, None)
    for t in t_grid:
        r = A * t * t + B * t + C
        if require_well_nonpositive and r[4] > 0.0:
            continue
        if any(r[i] > tol for i in preserved):
            continue
        v = float(max(0.0, r.max()))
        if best[0] is None or v < best[1]:
            best = (float(t), v)
    return best


SURROGATE_WIDTHS = [0.02, 0.05, 0.1, 0.2]


def representation_with_surrogate(rep: dict, log_flux_surrogate: bool) -> dict:
    """The evaluation payload template, optionally asking the evaluator to
    store the smoothed-gate flux surrogate at every canonical evaluation
    (E10 widths; no extra equilibrium solve). Returns a new dict."""
    if not log_flux_surrogate:
        return dict(rep)
    return dict(rep, surrogate_widths=list(SURROGATE_WIDTHS))


def model_prediction_error(r0, J, C, d, r_meas, lowrank=None) -> dict:
    """The multivariate diagonal-quadratic model's OWN prediction at a step d
    against the measured residuals: predicted r, largest absolute miss and
    its row, and rho_model = measured V gain / predicted V gain (review
    round 8: the trial rho validated the quadratic line predictor fitted to
    fresh points, not this model). C may be None (first-order model).

    E19: `lowrank` = (U, S), if given, adds the measured off-diagonal
    subspace cross-term contribution (quadratic_model_step_terms) on top of
    C's diagonal term -- so the logged 'max_abs_err'/'rho_model' reflect the
    model the runner actually stepped with, cross terms included, when
    --subspace-curvature is on. None when C is None (no lowrank without a
    second-order model in the first place) or lowrank is None (byte-identical
    to the pre-E19 model)."""
    r0 = np.asarray(r0, float)
    d = np.asarray(d, float)
    Jc = np.where(np.isnan(J), 0.0, np.asarray(J, float))
    pred = r0 + Jc @ d
    if C is not None:
        pred = pred + quadratic_model_step_terms(C, d, lowrank)
    r_meas = np.asarray(r_meas, float)
    err = pred - r_meas
    worst = int(np.argmax(np.abs(err)))
    pred_gain = float(r0.max() - pred.max())
    meas_gain = float(r0.max() - r_meas.max())
    rho = meas_gain / pred_gain if abs(pred_gain) > 1e-12 else None
    return {
        "predicted_r": pred.tolist(),
        "max_abs_err": float(abs(err[worst])),
        "worst_row": worst,
        "rho_model": rho,
    }


def direction_seed_from_file(path, n: int) -> np.ndarray:
    """Raw vectors from a --direction-file JSON ({'directions': [[...], ...]}),
    Gram-Schmidt orthonormalised in file order, dropping near-duplicates
    (cosine > 0.99) and anything already in the span -- the same rule
    directions_from_archives applies to accepted-step displacements. A
    deterministic, hand-specified alternative to --direction-archives (E19
    item 3's toy ratchet test needs an exact, reproducible subspace rather
    than one replayed from a real lineage archive)."""
    raw = json.loads(Path(path).read_text())["directions"]
    kept: list = []
    for vector in raw:
        vector = np.asarray(vector, float)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-12:
            continue
        u = vector / norm
        if kept and max(abs(float(u @ q)) for q in kept) > 0.99:
            continue
        for q in kept:
            u = u - float(u @ q) * q
        norm2 = float(np.linalg.norm(u))
        if norm2 < 1e-6:
            continue
        kept.append(u / norm2)
    return np.asarray(kept, float).reshape(-1, n) if kept else np.zeros((0, n))


def build_subspace_directions(direction_archives, direction_file, J, n: int, k: int):
    """(U, n_from_archives, n_from_svd): the E19 subspace direction set.
    Seeded from --direction-file if given (deterministic, for tests), else
    from --direction-archives' accepted-step displacements, else no seed at
    all; filled to k with the current Jacobian J's top right-singular
    vectors (fill_with_svd) either way. n_from_archives + n_from_svd == the
    returned U's row count (<= k)."""
    if direction_file:
        seed = direction_seed_from_file(direction_file, n)
    elif direction_archives:
        seed = directions_from_archives(direction_archives, n, k)
    else:
        seed = np.zeros((0, n))
    n_from_archives = seed.shape[0]
    U = fill_with_svd(seed, J, k)
    n_from_svd = U.shape[0] - n_from_archives
    return U, n_from_archives, n_from_svd


def per_row_offdiag_norm_ratio(S) -> list:
    """Frobenius norm of the off-diagonal part of each residual row's
    measured K x K curvature block, over the Frobenius norm of its diagonal
    -- how much curvature the pre-E19 diagonal-only model was missing for
    that row. None for a row whose diagonal norm is (numerically) zero,
    rather than a division by zero or an unbounded ratio."""
    S = np.asarray(S, float)
    out = []
    for row in S:
        diag = np.diag(np.diag(row))
        off = row - diag
        diag_norm = float(np.linalg.norm(diag))
        out.append(None if diag_norm < 1e-12 else float(np.linalg.norm(off) / diag_norm))
    return out


def best_admissible_sampled(incumbent_r, samples, tol: float, eps: float):
    """(t, r) of the lowest-V sampled line point that passes acceptance against
    the incumbent, or (None, None). Review round 8: an already-evaluated
    admissible point must never be discarded because the unconstrained line
    minimum predicted something better (34 such losses across the chains)."""
    best = (None, None)
    for t, r in samples:
        if r is None or not accept_final(incumbent_r, r, tol, eps):
            continue
        if best[1] is None or V_of(r) < V_of(best[1]):
            best = (float(t), np.asarray(r, float))
    return best


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def _fresh_evaluate(rep: dict, xs, tags, workers: int, log_path: Path, budget: Budget):
    """Evaluate in a FRESH process pool (fresh canonical processes)."""
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        ev = CanonicalEvaluator(rep, pool, log_path, budget)
        return ev.evaluate(xs, tags)


def probe_scale(h: float, radius_mult, second_order: bool):
    """Per-coordinate probe step for the Jacobian: h everywhere for the
    first-order runs; h * box multiplier (step scale) for E7's second-order
    model, whose curvature is only meaningful at the scale the steps use."""
    if not second_order or radius_mult is None:
        return h
    return h * np.asarray(radius_mult, float)


def step_in_trust_units(d, mult) -> float:
    """max_j |d_j| / m_j: a step's size measured against the per-coordinate
    box (m = 1 everywhere for the isotropic box). The Broyden-update guard
    (|step| <= 2 delta_cap) is applied in these units so E6b's wide new-mode
    box does not disable secant updates."""
    d = np.asarray(d, float)
    if mult is None:
        return float(np.abs(d).max())
    return float((np.abs(d) / np.asarray(mult, float)).max())


def run(args) -> dict:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ceil = Ceilings(
        t_start=t0,
        total_hours=args.total_hours,
        opt_cutoff_hours=args.opt_cutoff_hours,
        max_opt_evals=args.max_opt_evals,
        max_verif_evals=args.max_verif_evals,
    )
    events_path = out / "events.jsonl"
    archive_path = out / "archive.jsonl"  # every paid evaluation
    verif_path = out / "verification.jsonl"

    def log(**kw):
        kw["t_h"] = round((time.time() - t0) / 3600, 3)
        events_path.open("a").write(json.dumps(kw) + "\n")
        print(json.dumps(kw), flush=True)

    # ---- start point: reconstruct from the paired archive
    if args.start_candidate:
        cand = json.loads(Path(args.start_candidate).read_text())
        x0 = np.asarray(cand["x"] if "x" in cand else cand["x_scaled"], float)
        r_arch = np.asarray(cand["r"] if "r" in cand else cand["residuals"], float)
        rep = representation_of(cand)
        start_row = {"tag": f"candidate:{Path(args.start_candidate).name}"}
        args.start_log = args.start_candidate
    else:
        src = [json.loads(ln) for ln in Path(args.start_log).read_text().splitlines() if ln.strip()]
        start_row = [r for r in src if r["tag"].startswith(args.start_tag)][0]
        x0 = np.asarray(start_row["x"], float)
        r_arch = np.asarray(start_row["r"], float)
        ck = json.loads(Path(args.checkpoint).read_text())
        rep = dict(
            boundary_dict=ck["boundary_dict"],
            mpol=int(ck["max_poloidal_mode"]),
            ntor=int(ck["max_toroidal_mode"]),
            nfp=int(ck["n_field_periods"]),
            scaling=float(ck.get("infinity_norm_spectrum_scaling", 1.0)),
        )
    rep = representation_with_surrogate(rep, args.log_flux_surrogate)
    n = len(x0)
    (out / "start_serialised.json").write_text(
        json.dumps(
            {
                "x": x0.tolist(),
                "archived_r": r_arch.tolist(),
                **rep,
                "source": {"log": args.start_log, "tag": start_row["tag"]},
            },
            indent=1,
        )
    )
    control = json.loads(Path(args.control_candidate).read_text())
    xc, rc = np.asarray(control["x"], float), np.asarray(control["r"], float)

    # ---- E6b: equal box in coefficient space (per-coordinate radius multipliers)
    radius_mult, old_mask, scale_vec = None, None, None
    if args.coeff_space_box:
        from constellaration_methods.restoration.p3_promote_candidate import (
            coefficient_space_multiplier,
            old_mode_mask,
        )

        if not args.start_candidate:
            raise SystemExit("--coeff-space-box needs a promoted --start-candidate")
        src_budget = cand.get("source", {})
        old_mask, scale_vec = old_mode_mask(cand, int(src_budget["mpol"]), int(src_budget["ntor"]))
        radius_mult = coefficient_space_multiplier(scale_vec, old_mask)
        (out / "coeff_space_box.json").write_text(
            json.dumps(
                {
                    "source_budget": src_budget,
                    "n_old": int(old_mask.sum()),
                    "n_new": int((~old_mask).sum()),
                    "mult_old_range": [
                        float(radius_mult[old_mask].min()),
                        float(radius_mult[old_mask].max()),
                    ],
                    "mult_new_range": [
                        float(radius_mult[~old_mask].min()),
                        float(radius_mult[~old_mask].max()),
                    ],
                    "radius_mult": radius_mult.tolist(),
                    "old_mask": old_mask.tolist(),
                },
                indent=1,
            )
        )

    def energy_share(d):
        if old_mask is None:
            return None
        from constellaration_methods.restoration.p3_promote_candidate import (
            new_mode_energy_fraction,
        )

        return new_mode_energy_fraction(d, scale_vec, old_mask)

    # ---- phase 0: repeatability (verification budget)
    vbudget = Budget(cap=args.max_verif_evals)
    rs, raws = _fresh_evaluate(
        rep, [x0, x0], ["REP:start:1", "REP:start:2"], args.workers, verif_path, vbudget
    )
    rs_c, _ = _fresh_evaluate(
        representation_of(control),
        [xc],
        ["REP:control:1"],
        args.workers,
        verif_path,
        vbudget,
    )
    rs = list(rs) + list(rs_c)
    ceil.charge_verif(3)
    ok, e_rep, eps = eps_from_repeatability(
        r_arch,
        [rr for rr in rs[:2] if rr is not None] if all(rr is not None for rr in rs[:2]) else [],
        args.repeat_tol,
    )
    okc, e_c, _ = eps_from_repeatability(rc, [rs[2]] if rs[2] is not None else [], args.repeat_tol)
    log(
        event="repeatability",
        ok=bool(ok and okc),
        e_rep=e_rep,
        e_control=e_c,
        eps=eps,
        start_repeats=[None if rr is None else rr.tolist() for rr in rs[:2]],
        control=None if rs[2] is None else rs[2].tolist(),
    )
    if not (ok and okc):
        rep_out = {
            "label": "technical_limitation",
            "reason": "repeatability failed",
            "e_rep": e_rep,
            "e_control": e_c,
        }
        (out / "REPORT.json").write_text(json.dumps(rep_out, indent=2))
        return rep_out
    r0 = r_arch.copy()
    if args.phase0_only:
        rep_out = {
            "label": "phase0_only",
            "repeatability_ok": True,
            "e_rep": e_rep,
            "e_control": e_c,
            "eps": eps,
            "verif_evals": ceil.verif_used,
            "wall_h": round((time.time() - t0) / 3600, 3),
        }
        (out / "REPORT.json").write_text(json.dumps(rep_out, indent=2))
        return rep_out

    obudget = Budget(cap=args.max_opt_evals)
    ctx = mp.get_context("spawn")
    archive = []  # in-memory mirror of archive.jsonl rows that have residuals
    incumbent_x, incumbent_r = x0.copy(), r0.copy()
    accepted = []
    stop_reason = None
    technical = False
    success_point = None  # holder: set the instant priority_confirm confirms
    # feasibility (fix 2, E13 defect: the return value used to be ignored, so
    # the loop ran on -- 452 extra evaluations after the feasible point)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        ev = CanonicalEvaluator(rep, pool, archive_path, obudget)

        def evaluate(xs, tags):
            ceil.charge_opt(len(xs))
            rs_, raws_ = ev.evaluate(xs, tags)
            for x_, t_, r_ in zip(xs, tags, rs_):
                archive.append(
                    {
                        "tag": t_,
                        "x": np.asarray(x_).tolist(),
                        "r": None if r_ is None else r_.tolist(),
                        "complete": r_ is not None and bool(np.all(np.isfinite(r_))),
                    }
                )
            return rs_, raws_

        h_probe = probe_scale(args.h, radius_mult, args.second_order)

        def full_jacobian(xc_, label, rc_=None):
            """(J, ok); with --second-order also refreshes the diagonal
            curvature C (nonlocal) from the same probes at step scale."""
            nonlocal C
            ceil.charge_full_jacobian()
            pts, meta = probe_points(xc_, args.h, h_per_coord=h_probe)
            rs_, _ = evaluate(pts, [f"{label}:{j}:{'+' if s_ > 0 else '-'}" for j, s_ in meta])
            J_, ok_ = central_difference_jacobian(rs_, meta, n=n, h=h_probe)
            if args.second_order:
                C, okc = central_difference_curvature(rs_, meta, n=n, h=h_probe, r0=rc_)
                log(
                    event="curvature",
                    label=label,
                    ok_cols=int(sum(okc)),
                    frac_positive=float(np.nanmean(C > 0)),
                    median_new_mode_qi_mirror_flux=[
                        float(np.nanmedian(C[i][~old_mask])) for i in (1, 2, 3)
                    ]
                    if old_mask is not None
                    else None,
                )
            log(
                event="jacobian",
                label=label,
                ok_cols=int(sum(ok_)),
                opt_used=ceil.opt_used,
                probe_scale="step" if args.second_order else "h",
            )
            return J_, ok_

        C = None

        # ---- phase 1: Jacobian at the start (fresh, or reused from a prior
        # run at the identical start point and representation: E6b reuses
        # E6's 576 paid J0 probes; it still counts as full Jacobian #1 of 2)
        if args.reuse_jacobian_archive:
            prev = Path(args.reuse_jacobian_archive)
            prev_start = json.loads((prev / "start_serialised.json").read_text())
            dx = float(np.abs(np.asarray(prev_start["x"], float) - incumbent_x).max())
            same_rep = all(prev_start[k] == rep[k] for k in ("mpol", "ntor", "nfp", "scaling"))
            if dx > 1e-12 or not same_rep:
                raise SystemExit(
                    f"--reuse-jacobian-archive: start mismatch (dx={dx:.2e}, same_rep={same_rep})"
                )
            ceil.charge_full_jacobian()
            J, okc_ = jacobian_from_archive(prev / "archive.jsonl", "J0", n, h_probe)
            if args.second_order:
                C, _ = curvature_from_archive(prev / "archive.jsonl", "J0", n, h_probe, incumbent_r)
            log(
                event="jacobian",
                label="J0",
                ok_cols=int(sum(okc_)),
                opt_used=ceil.opt_used,
                reused_from=str(prev),
                start_dx=dx,
            )
        else:
            J, okc_ = full_jacobian(incumbent_x, "J0", incumbent_r)

        # ---- E19: off-diagonal (subspace) curvature. getattr defaults keep
        # this a no-op for callers (existing tests included) whose args don't
        # carry these attributes at all -- --subspace-curvature 0 (the
        # default) is byte-identical to the pre-E19 runner.
        k_sub = getattr(args, "subspace_curvature", 0) or 0
        direction_archives = getattr(args, "direction_archives", None)
        direction_file = getattr(args, "direction_file", None)
        if k_sub > 0 and not args.second_order:
            raise SystemExit("--subspace-curvature requires --second-order")
        subspace = None  # {"U", "S", "K", "n_from_archives", "n_from_svd", "h",
        #                    "per_row_offdiag_norm_ratio"} once measured
        subspace_evals_total = 0
        steps_by_model: dict = {}  # E19 item 2: tally of restoration_step_quadratic's
        # fallback-ladder rung ("subspace"/"subspace_half"/"diagonal") across every trial

        def subspace_lowrank():
            return None if subspace is None else (subspace["U"], subspace["S"])

        def measure_subspace(J_now, x_now, r_now, label):
            """Build U (archives/file, filled with SVD) and measure S at the
            given point through the SAME worker pool and evaluation-budget
            accounting as every other probe (`evaluate`); updates the
            nonlocal `subspace` and cumulative `subspace_evals_total`. A
            no-op when --subspace-curvature is 0 or no directions are
            available at all."""
            nonlocal subspace, subspace_evals_total
            if k_sub <= 0:
                return
            U, n_from_archives, n_from_svd = build_subspace_directions(
                direction_archives, direction_file, J_now, n, k_sub
            )
            if U.shape[0] == 0:
                log(event="subspace_unavailable", label=label)
                subspace = None
                return
            h_sub = args.delta0
            pts, meta = subspace_probe_points(x_now, U, h_sub)
            tags = [
                f"{label}:SU{tag[1]}:{'+' if tag[2] > 0 else '-'}"
                if tag[0] == "diag"
                else f"{label}:SU{tag[1]}x{tag[2]}:{'+' if tag[3] > 0 else '-'}"
                for tag in meta
            ]
            rs_, _ = evaluate(pts, tags)
            S = subspace_curvature_from_probes(rs_, meta, r_now, U, h_sub)
            ratios = per_row_offdiag_norm_ratio(S)
            subspace_evals_total += len(pts)
            subspace = {
                "U": U,
                "S": S,
                "K": int(U.shape[0]),
                "n_from_archives": int(n_from_archives),
                "n_from_svd": int(n_from_svd),
                "h": h_sub,
                "per_row_offdiag_norm_ratio": ratios,
            }
            log(
                event="subspace_measured",
                label=label,
                K=subspace["K"],
                n_from_archives=subspace["n_from_archives"],
                n_from_svd=subspace["n_from_svd"],
                evals=len(pts),
                per_row_offdiag_norm_ratio=ratios,
                h=h_sub,
            )

        # --reuse-jacobian-archive may reuse J/C but always re-measures S: it
        # never reuses a prior run's subspace probes (E19 item 2).
        measure_subspace(J, incumbent_x, incumbent_r, "J0")

        tr = RatioTrustRegion(delta=args.delta0, cap=args.delta_cap, floor=1e-5)
        stall = StallCounter(max_in_row=8, min_gain=eps)
        recovery_used = False
        phase = "pilot"
        pilot_start_used = ceil.opt_used
        block_start_used, block_start_v = ceil.opt_used, V_of(incumbent_r)
        recent = []  # last 20 trial convergence flags
        n_since_drift = 0
        trial = 0

        def priority_confirm(x_, r_, tag_, raw_=None):
            nonlocal success_point
            # any benchmark-feasible-looking point: confirm immediately (verification budget)
            if r_ is not None and np.all(np.asarray(r_) <= TOL) and vbudget.remaining >= 2:
                rs2, _ = _fresh_evaluate(
                    rep,
                    [x_, x_],
                    [f"CONF:{tag_}:1", f"CONF:{tag_}:2"],
                    args.workers,
                    verif_path,
                    vbudget,
                )
                ceil.charge_verif(2)
                wv, feas = adjudicate_worst_v(r_, [rr for rr in rs2 if rr is not None], TOL)
                log(
                    event="priority_confirmation",
                    tag=tag_,
                    worst_v=wv,
                    feasible=bool(feas),
                )
                if feas:
                    success_point = {
                        "x": np.asarray(x_, float).tolist(),
                        "r": np.asarray(r_, float).tolist(),
                        "raw": raw_,
                        "tag": tag_,
                    }
                    log(event="success", tag=tag_, worst_v=wv)
                return feas
            return False

        def do_recovery():
            nonlocal \
                J, \
                incumbent_x, \
                incumbent_r, \
                recovery_used, \
                tr, \
                stall, \
                block_start_used, \
                block_start_v
            recovery_used = True
            pick = select_recovery_point(archive, incumbent_r, TOL)
            if pick is None or not ceil.can_full_jacobian():
                log(
                    event="recovery_unavailable",
                    reason="no eligible point" if pick is None else "jacobian ceiling",
                )
                return False
            log(
                event="recovery_restart",
                tag=pick["tag"],
                v=V_of(pick["r"]),
                note="RESTART, not an accepted step",
            )
            incumbent_x, incumbent_r = (
                np.asarray(pick["x"], float),
                np.asarray(pick["r"], float),
            )
            if not ceil.can_optimise(time.time(), 2 * n):
                return False
            J, _ = full_jacobian(incumbent_x, "JR", incumbent_r)
            measure_subspace(J, incumbent_x, incumbent_r, "JR")  # E19: re-measured
            tr = RatioTrustRegion(delta=args.delta0, cap=args.delta_cap, floor=1e-5)
            stall = StallCounter(max_in_row=8, min_gain=eps)
            v_at_restart = V_of(incumbent_r)
            material = 0
            for k in range(20):
                if not ceil.can_optimise(time.time(), 1):
                    break
                acc = one_trial(f"R{k}")
                if acc:
                    material += 1
                if success_point is not None:
                    break
            qualified = recovery_qualified(material, V_of(incumbent_r), args.recovery_qualify_v)
            log(
                event="recovery_result",
                material_steps=material,
                v=V_of(incumbent_r),
                v_at_restart=v_at_restart,
                qualified=bool(qualified),
            )
            block_start_used, block_start_v = ceil.opt_used, V_of(incumbent_r)
            return qualified

        def one_trial(tag_prefix):
            nonlocal J, incumbent_x, incumbent_r, n_since_drift, trial
            model_used = None
            try:
                if not args.second_order:
                    d, t_pred = restoration_step(
                        incumbent_r,
                        J,
                        tr.delta,
                        TOL,
                        caps=well_cap_for(incumbent_r),
                        radius_mult=radius_mult,
                    )
                else:
                    d, t_pred, model_used = restoration_step_quadratic(
                        incumbent_r,
                        J,
                        C,
                        tr.delta,
                        TOL,
                        caps=well_cap_for(incumbent_r),
                        radius_mult=radius_mult,
                        lowrank=subspace_lowrank(),
                    )
                    steps_by_model[model_used] = steps_by_model.get(model_used, 0) + 1
            except RestorationLPInfeasible as e:
                log(event="lp_infeasible", trial=trial, delta=tr.delta, msg=str(e)[:100])
                tr.reject()
                stall.record(accepted=False, gain=None)
                trial += 1
                return False
            pred_r = incumbent_r + np.where(np.isnan(J), 0.0, J) @ d
            if args.second_order:
                pred_r = pred_r + quadratic_model_step_terms(C, d, subspace_lowrank())
            t_star = 1.0
            if args.line_search:
                # E2: 3-point quadratic line model per residual along d
                (r_half, r_full), (raw_half, raw_full) = evaluate(
                    [incumbent_x + 0.5 * d, incumbent_x + d],
                    [f"{tag_prefix}:L{trial}:t0.5", f"{tag_prefix}:L{trial}:t1.0"],
                )
                raw_of_t = {0.5: raw_half, 1.0: raw_full}
                model_err = {
                    f"t{tt}": model_prediction_error(
                        incumbent_r,
                        J,
                        C if args.second_order else None,
                        tt * d,
                        rr,
                        lowrank=subspace_lowrank() if args.second_order else None,
                    )
                    for tt, rr in ((0.5, r_half), (1.0, r_full))
                    if rr is not None
                }
                log(
                    event="model_error",
                    tag=f"{tag_prefix}:L{trial}",
                    **{
                        f"{k}_{f}": v[f]
                        for k, v in model_err.items()
                        for f in ("max_abs_err", "worst_row", "rho_model")
                    },
                )
                recent.extend([r_half is not None, r_full is not None])
                for tt, rr in ((0.5, r_half), (1.0, r_full)):
                    if (
                        rr is not None
                        and step_in_trust_units(tt * d, radius_mult) <= 2 * args.delta_cap
                    ):
                        J, _ = broyden_update(J, tt * d, rr - incumbent_r)
                    if rr is not None:
                        priority_confirm(
                            incumbent_x + tt * d,
                            rr,
                            f"{tag_prefix}:L{trial}:t{tt}",
                            raw_of_t.get(tt),
                        )
                        if success_point is not None:
                            # feasibility confirmed at a line-search point:
                            # stop right here, no more evaluations this trial
                            trial += 1
                            return False
                if r_half is None or r_full is None:
                    t_star, v_line = None, None
                else:
                    A_, B_, C_ = quadratic_line_model(incumbent_r, r_half, r_full)
                    t_star, v_line = best_t_on_line(
                        A_,
                        B_,
                        C_,
                        np.linspace(0.1, 1.5, 29),
                        require_well_nonpositive=bool(incumbent_r[4] <= 0.0),
                        r0=incumbent_r,
                        tol=TOL,
                    )
                    if t_star is not None:
                        pred_r = A_ * t_star**2 + B_ * t_star + C_
                log(
                    event="line_model",
                    tag=f"{tag_prefix}:L{trial}",
                    t_star=t_star,
                    v_pred=v_line,
                    r_half=None if r_half is None else r_half.tolist(),
                    r_full=None if r_full is None else r_full.tolist(),
                )
                t_samp, r_samp = best_admissible_sampled(
                    incumbent_r, ((0.5, r_half), (1.0, r_full)), TOL, eps
                )
                if t_star is None:
                    if r_samp is None:
                        tr.reject()
                        stall.record(accepted=False, gain=None)
                        trial += 1
                        return False
                    t_star, pred_r = (
                        t_samp,
                        r_samp,
                    )  # no admissible model point: take the measured one
                elif r_samp is not None and V_of(r_samp) <= v_line:
                    t_star, pred_r = t_samp, r_samp
                d = t_star * d
                t_pred = float(pred_r.max())
            (r_new,), (raw_new,) = evaluate(
                [incumbent_x + d],
                [f"{tag_prefix}:T{trial}:d{tr.delta:.5f}:t{t_star:.2f}"],
            )
            recent.append(r_new is not None)
            if r_new is not None and step_in_trust_units(d, radius_mult) <= 2 * args.delta_cap:
                J, _ = broyden_update(J, d, r_new - incumbent_r)
            ok_ = accept_final(incumbent_r, r_new, TOL, eps)
            if not ok_ and args.line_search and r_samp is not None:
                # the model's point failed but an already-evaluated line point
                # passes every gate: keep the measured improvement (review round 8)
                log(
                    event="sampled_point_kept",
                    tag=f"{tag_prefix}:T{trial}",
                    t=t_samp,
                    v=V_of(r_samp),
                    v_rejected=None if r_new is None else V_of(r_new),
                )
                d, r_new, raw_new = t_samp * d / t_star, r_samp, raw_of_t[t_samp]
                ok_ = True
            v_old = V_of(incumbent_r)
            gain = None if r_new is None else v_old - V_of(r_new)
            pred_impr = v_old - t_pred
            rho = None if (r_new is None or pred_impr <= 1e-12) else gain / pred_impr
            log(
                event="trial",
                tag=f"{tag_prefix}:T{trial}",
                delta=tr.delta,
                predicted_r=pred_r.tolist(),
                measured_r=None if r_new is None else r_new.tolist(),
                v_old=v_old,
                v_new=None if r_new is None else V_of(r_new),
                rho=rho,
                accepted=bool(ok_),
                nonconv=r_new is None,
                model_used=model_used,
                opt_used=ceil.opt_used,
                new_mode_energy=energy_share(d),
            )
            if r_new is not None:
                priority_confirm(
                    incumbent_x + d,
                    r_new,
                    f"{tag_prefix}:T{trial}",
                    raw_new.get("raw"),
                )
            if ok_:
                accepted.append(
                    {
                        "tag": f"{tag_prefix}:T{trial}",
                        "v": V_of(r_new),
                        "r": r_new.tolist(),
                        "raw": raw_new.get("raw"),
                        "rho": rho,
                        "opt_used": ceil.opt_used,
                        "new_mode_energy": energy_share(d),
                    }
                )
                incumbent_x, incumbent_r = incumbent_x + d, r_new
                (out / "incumbent.json").write_text(
                    json.dumps(
                        {
                            "x": incumbent_x.tolist(),
                            "r": incumbent_r.tolist(),
                            "raw": raw_new.get("raw"),
                            **rep,
                        },
                        indent=1,
                    )
                )
                tr.update(rho if rho is not None else 0.0)
                stall.record(accepted=True, gain=gain)
                n_since_drift += 1
            else:
                tr.reject()
                stall.record(accepted=False, gain=None)
            trial += 1
            return bool(ok_)

        def drift_maintenance():
            nonlocal J, n_since_drift
            if n_since_drift < args.drift_every or not ceil.can_optimise(time.time(), 4):
                return
            n_since_drift = 0
            rng = np.random.default_rng(args.seed + trial)
            dirs = rng.standard_normal((2, n))
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            pts = np.concatenate([incumbent_x + args.h * dirs, incumbent_x - args.h * dirs])
            rs_, _ = evaluate(
                pts,
                [f"D{trial}:{i}:+" for i in range(2)] + [f"D{trial}:{i}:-" for i in range(2)],
            )
            meas = [
                (rs_[i] - rs_[2 + i]) / (2 * args.h)
                for i in range(2)
                if rs_[i] is not None and rs_[2 + i] is not None
            ]
            if not meas:
                return
            disc = drift_rows(J, dirs[: len(meas)], np.asarray(meas))
            bad = [i for i in range(5) if disc[i] > 0.25]
            log(event="drift_check", per_row=[float(v) for v in disc], bad_rows=bad)
            if bad and sum(1 for i in range(5) if disc[i] > 0.5) < 2:
                cols = sorted({c for i in bad for c in worst_columns(J, i, 8)})
                if ceil.can_optimise(time.time(), 2 * len(cols)):
                    pts, meta = probe_points(incumbent_x, args.h, coords=cols, h_per_coord=h_probe)
                    rs2, _ = evaluate(
                        pts,
                        [f"P{trial}:{j}:{'+' if s_ > 0 else '-'}" for j, s_ in meta],
                    )
                    Jp, okp = central_difference_jacobian(rs2, meta, n=n, h=h_probe)
                    for c_, o_ in zip(cols, okp):
                        if o_:
                            J[:, c_] = Jp[:, c_]
                    if args.second_order:
                        Cp, okcp = central_difference_curvature(
                            rs2, meta, n=n, h=h_probe, r0=incumbent_r
                        )
                        for c_, o_ in zip(cols, okcp):
                            if o_:
                                C[:, c_] = Cp[:, c_]
                    log(event="partial_refresh", rows=bad, columns=len(cols))
            elif sum(1 for i in range(5) if disc[i] > 0.5) >= 2:
                log(event="full_refresh_requested")
                raise _RefreshRequested()

        class _RefreshRequested(Exception):
            pass

        # ---- phases 2-5
        try:
            while True:
                if success_point is not None:
                    stop_reason = "feasible_confirmed"
                    break
                if not ceil.can_optimise(time.time(), 1):
                    stop_reason = "budget_or_deadline"
                    break
                if len(recent) >= 20 and sum(1 for c in recent[-20:] if not c) > 6:
                    stop_reason = "nonconvergence>30%_in_last_20"
                    technical = True
                    break
                try:
                    one_trial("P" if phase == "pilot" else "C")
                    if success_point is not None:
                        stop_reason = "feasible_confirmed"
                        break
                    drift_maintenance()
                except _RefreshRequested:
                    if recovery_used:
                        stop_reason = "refresh_requested_after_recovery"
                        break
                    if not do_recovery():
                        stop_reason = "recovery_failed"
                        break
                    phase = "continuation"
                    continue
                if stall.stalled or tr.delta <= tr.floor:
                    if recovery_used:
                        stop_reason = "stalled_after_recovery"
                        break
                    if not do_recovery():
                        stop_reason = "recovery_failed"
                        break
                    phase = "continuation"
                    continue
                if phase == "pilot" and ceil.opt_used - pilot_start_used >= 100:
                    if V_of(incumbent_r) <= 0.130:
                        log(event="pilot_gate", met=True, v=V_of(incumbent_r))
                        phase = "continuation"
                        block_start_used, block_start_v = (
                            ceil.opt_used,
                            V_of(incumbent_r),
                        )
                    else:
                        log(event="pilot_gate", met=False, v=V_of(incumbent_r))
                        if recovery_used or not do_recovery():
                            stop_reason = "pilot_gate_failed"
                            break
                        phase = "continuation"
                elif phase == "continuation" and ceil.opt_used - block_start_used >= 200:
                    need = block_progress_required(block_start_v)
                    got = block_start_v - V_of(incumbent_r)
                    log(
                        event="block_gate",
                        required=need,
                        achieved=got,
                        met=bool(got >= need),
                    )
                    if got >= need:
                        block_start_used, block_start_v = (
                            ceil.opt_used,
                            V_of(incumbent_r),
                        )
                    else:
                        if recovery_used or not do_recovery():
                            stop_reason = "block_gate_failed"
                            break
        except Exception as e:  # noqa: BLE001
            stop_reason = f"runner_error: {type(e).__name__}: {str(e)[:200]}"
            technical = True
            log(event="runner_error", msg=stop_reason)
    log(
        event="optimisation_stopped",
        reason=stop_reason,
        opt_used=ceil.opt_used,
        v_incumbent=V_of(incumbent_r),
        n_accepted=len(accepted),
    )

    # ---- phase 6: adjudication (archive frozen; fresh processes)
    complete = [a for a in archive if a["complete"]]
    complete.sort(key=lambda a: V_of(a["r"]))
    cands, seen = [], set()
    for a in complete:
        key = tuple(np.round(np.asarray(a["x"]), 10))
        if key in seen:
            continue
        seen.add(key)
        cands.append(a)
        if len(cands) == 3:
            break
    results = []
    for a in cands:
        if vbudget.remaining < 2:
            break
        rs2, _ = _fresh_evaluate(
            rep,
            [np.asarray(a["x"]), np.asarray(a["x"])],
            [f"ADJ:{a['tag']}:1", f"ADJ:{a['tag']}:2"],
            args.workers,
            verif_path,
            vbudget,
        )
        ceil.charge_verif(2)
        confs = [rr for rr in rs2 if rr is not None]
        wv, feas = adjudicate_worst_v(a["r"], confs, TOL)
        tech = not confirmations_complete(confs)
        results.append(
            {
                "tag": a["tag"],
                "recorded_v": V_of(a["r"]),
                "confirmations": [c.tolist() for c in confs],
                "worst_v": wv,
                "feasible": bool(feas),
                "technical": tech,
                "r": a["r"],
                "x": a["x"],
            }
        )
        log(
            event="adjudicated",
            tag=a["tag"],
            worst_v=wv,
            feasible=bool(feas),
            technical=tech,
        )
    best = min(results, key=lambda r_: r_["worst_v"]) if results else None
    label = (
        label_outcome(
            best["worst_v"],
            best["feasible"],
            best["technical"] or technical,
            args.material_bar,
        )
        if best
        else "technical_limitation"
    )
    report = {
        "label": label,
        "material_bar": args.material_bar,
        "line_search": bool(args.line_search),
        "stop_reason": stop_reason,
        "eps": eps,
        "e_rep": e_rep,
        "start_v": V_of(r0),
        "incumbent_v": V_of(incumbent_r),
        "success_point": success_point,
        "n_accepted": len(accepted),
        "accepted": accepted,
        "opt_evals": ceil.opt_used,
        "verif_evals": ceil.verif_used,
        "full_jacobians": ceil.full_jacobians,
        "recovery_used": recovery_used,
        "coeff_space_box": bool(args.coeff_space_box),
        "second_order": bool(args.second_order),
        "reused_jacobian": args.reuse_jacobian_archive,
        "new_mode_energy_accepted": [a.get("new_mode_energy") for a in accepted],
        "adjudication": results,
        "best": best,
        "wall_h": round((time.time() - t0) / 3600, 3),
        "note": (
            "0.130 and 0.100 are NOT P3 success; feasibility = official predicate "
            "on all three evaluations"
        ),
    }
    # E19: only present when --subspace-curvature > 0, so a K=0 run's REPORT
    # is byte-identical to the pre-E19 runner (no new key at all, not even
    # null). K/n_from_archives/n_from_svd/per_row_offdiag_norm_ratio/h reflect
    # the LATEST measurement (J0, or JR if a recovery remeasured it);
    # evals_used is cumulative across every measurement in the run.
    if k_sub > 0:
        report["subspace"] = {
            "K": subspace["K"] if subspace else k_sub,
            "n_from_archives": subspace["n_from_archives"] if subspace else 0,
            "n_from_svd": subspace["n_from_svd"] if subspace else 0,
            "evals_used": subspace_evals_total,
            "per_row_offdiag_norm_ratio": (
                subspace["per_row_offdiag_norm_ratio"] if subspace else None
            ),
            "h": subspace["h"] if subspace else args.delta0,
            "steps_by_model": dict(steps_by_model),
        }
    (out / "REPORT.json").write_text(json.dumps(report, indent=2))
    print(
        "\n=== FINAL ATTEMPT REPORT ===\n"
        + json.dumps(
            {k: v for k, v in report.items() if k not in ("accepted", "adjudication", "best")},
            indent=2,
        ),
        flush=True,
    )
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="representation source (D-176 checkpoint)")
    ap.add_argument(
        "--start-log",
        default="results/p3-restoration-176-paired/full/evaluations.jsonl",
    )
    ap.add_argument("--start-tag", default="T3:")
    ap.add_argument(
        "--control-candidate",
        default="results/p3-restoration-176-run2/candidate2_0.152095.json",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--h", type=float, default=0.01)
    ap.add_argument("--delta0", type=float, default=0.00125)
    ap.add_argument("--delta-cap", type=float, default=0.0025)
    ap.add_argument("--drift-every", type=int, default=3)
    ap.add_argument("--repeat-tol", type=float, default=1e-4)
    ap.add_argument("--total-hours", type=float, default=24.0)
    ap.add_argument("--opt-cutoff-hours", type=float, default=20.0)
    ap.add_argument("--max-opt-evals", type=int, default=1200)
    ap.add_argument("--max-verif-evals", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--phase0-only",
        action="store_true",
        help="run the repeatability gate and stop (preparation check)",
    )
    ap.add_argument(
        "--start-candidate",
        default=None,
        help="candidate JSON (x/r/representation); a fresh Jacobian is built",
    )
    ap.add_argument(
        "--line-search",
        action="store_true",
        help="E2: quadratic line model along the LP direction (2 extra evals per trial)",
    )
    ap.add_argument(
        "--material-bar",
        type=float,
        default=0.130,
        help="label bar for material progress beyond existing evidence",
    )
    ap.add_argument(
        "--coeff-space-box",
        action="store_true",
        help=(
            "E6b: per-coordinate trust radius giving every mode the median "
            "old-mode reach in coefficient units (needs a promoted "
            "--start-candidate carrying its source budget)"
        ),
    )
    ap.add_argument(
        "--recovery-qualify-v",
        type=float,
        default=0.130,
        help=(
            "V the single recovery must reach (with >= 1 material step) to "
            "qualify; 0.130 = the final protocol"
        ),
    )
    ap.add_argument(
        "--log-flux-surrogate",
        action="store_true",
        help=(
            "store the smoothed-gate flux surrogate (E10 widths) with every "
            "canonical evaluation; no extra VMEC cost"
        ),
    )
    ap.add_argument(
        "--second-order",
        action="store_true",
        help=(
            "E7: probes at step scale (h x box multiplier), diagonal curvature "
            "from the same probes, quadratic restoration step (SLSQP); "
            "acceptance unchanged"
        ),
    )
    ap.add_argument(
        "--reuse-jacobian-archive",
        default=None,
        help=(
            "a prior run's out-dir: reuse its J0 probes (archive.jsonl) at the "
            "identical start; charged as full Jacobian #1"
        ),
    )
    ap.add_argument(
        "--subspace-curvature",
        type=int,
        default=0,
        help="E19: K, the off-diagonal subspace curvature rank measured over K "
        "directions (0 = off; byte-identical to the pre-E19 runner). Requires "
        "--second-order.",
    )
    ap.add_argument(
        "--direction-archives",
        nargs="*",
        default=None,
        help="E19: prior p3_final_restoration out-dirs whose accepted-step "
        "displacements seed the K subspace directions (most recent LAST); "
        "filled to K with the current Jacobian's top singular vectors",
    )
    ap.add_argument(
        "--direction-file",
        default=None,
        help="E19: JSON {'directions': [[...], ...]} of raw vectors to seed the "
        "K subspace directions directly, bypassing --direction-archives "
        "(deterministic; mainly for the toy ratchet test)",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
