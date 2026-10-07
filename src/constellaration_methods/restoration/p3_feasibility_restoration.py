"""Canonical, constrained feasibility-restoration trial on a retained P3
candidate (reviewer round 5, 2026-09-15; pre-registered in the vault as
p3-feasibility-restoration-plan-2026-09-15).

Question: does a locally measurable direction improve the mirror and QI
violations together on seed 176's retained direct-resume candidate without
sacrificing iota, flux or well? The ALM ran with all five physics penalties
saturated at 1e9, so its plateau may be the weighted objective, not the
physics. This trial is penalty-free and canonical throughout:

1. r(x): the five signed canonical normalised residuals
   (`MHDStableQIStellarator._normalized_constraint_violations`: iota,
   log10_qi, mirror, flux, well; positive = violated). No objective.
2. Jacobian by central differences over all n=80 scaled coordinates (2n
   high-fidelity evaluations), plus a stability check: k coordinates
   repeated at h/2 and a few random unit directions; unstable -> INCONCLUSIVE.
3. Restoration subproblem: minimise t s.t. r_i + J_i d <= t (violated i),
   r_i + J_i d <= tol (satisfied i, preserved), |d_j| <= Delta. A linear
   program (scipy linprog).
4. Trust region with CANONICAL acceptance: accept only if the measured max
   violation falls and no satisfied constraint becomes violated; reject
   (incl. non-convergent) -> halve Delta. Re-linearise after an accepted
   step. Keep the last canonically verified incumbent throughout.
5. Hard cap on canonical evaluations (probes, trials, verification).

Every evaluation is logged to JSONL (resumable); the outcome is classified
by the pre-registered rule at the end. Usage (inside the pinned image):
    podman run --rm --network none -v "$(pwd)":/workspace:Z -w /workspace \\
        -e OMP_NUM_THREADS=1 --pids-limit 8192 --ulimit nproc=16384:16384 \\
        --entrypoint taskset localhost/constellaration-methods:test -c 2-13 \\
        python3 scripts/p3_feasibility_restoration.py \\
            --checkpoint results/m1-post-ladder-d176/seed_0176/checkpoint.json \\
            --out-dir results/p3-restoration-176 --workers 12 --budget 600
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from constellaration_methods.problems.overrides import (
    normalised_constraint_violations,
    problem_with_overrides,
)

RESIDUAL_NAMES = ("iota", "log10_qi", "mirror", "flux", "well")
TOL = 0.01  # the benchmark's relative tolerance, applied to normalised residuals


# --------------------------------------------------------------------------- #
# pure parts (tested without VMEC)
# --------------------------------------------------------------------------- #
def _step_of(h, j: int) -> float:
    """h may be one scalar or one step per coordinate (E7: step-scale probes)."""
    return float(h) if np.ndim(h) == 0 else float(np.asarray(h, float)[j])


def probe_points(x0: np.ndarray, h: float, coords=None, h_per_coord=None):
    """Central-difference probe points: for each coordinate j, x0 + h_j e_j and
    x0 - h_j e_j, with h_j = h everywhere unless h_per_coord is given.
    Returns (points (2k, n), meta [(j, +1), (j, -1), ...])."""
    x0 = np.asarray(x0, dtype=float)
    coords = list(range(len(x0))) if coords is None else list(coords)
    steps = h if h_per_coord is None else np.asarray(h_per_coord, float)
    pts, meta = [], []
    for j in coords:
        for sgn in (+1, -1):
            p = x0.copy()
            p[j] += sgn * _step_of(steps, j)
            pts.append(p)
            meta.append((j, sgn))
    return np.asarray(pts), meta


def _probe_pairs(residuals, meta):
    plus, minus = {}, {}
    for rr, (j, sgn) in zip(residuals, meta):
        (plus if sgn > 0 else minus)[j] = None if rr is None else np.asarray(rr, float)
    return plus, minus


def central_difference_jacobian(residuals, meta, n: int, h):
    """J[i, j] = (r(x + h_j e_j) - r(x - h_j e_j))_i / (2 h_j); h is a scalar or
    one step per coordinate. A None residual (non-convergent probe) leaves
    that column NaN and marks it not ok."""
    m = next(len(rr) for rr in residuals if rr is not None)
    J = np.full((m, n), np.nan)
    plus, minus = _probe_pairs(residuals, meta)
    ok = []
    for j in sorted(set(k for k, _ in meta)):
        if plus.get(j) is None or minus.get(j) is None:
            ok.append(False)
            continue
        J[:, j] = (plus[j] - minus[j]) / (2.0 * _step_of(h, j))
        ok.append(True)
    return J, ok


def central_difference_curvature(residuals, meta, n: int, h, r0):
    """Diagonal curvature C[i, j] = (r(x + h_j e_j) - 2 r(x) + r(x - h_j e_j))_i
    / h_j^2 from the same probes as the Jacobian (E7: at step scale this is
    the bowl the linear model overshoots; at h = 0.01 it is gate noise, see
    vault p3-reopen-three-experiments-2026-09-16). NaN columns as for J."""
    r0 = np.asarray(r0, float)
    C = np.full((len(r0), n), np.nan)
    plus, minus = _probe_pairs(residuals, meta)
    ok = []
    for j in sorted(set(k for k, _ in meta)):
        if plus.get(j) is None or minus.get(j) is None:
            ok.append(False)
            continue
        C[:, j] = (plus[j] - 2.0 * r0 + minus[j]) / _step_of(h, j) ** 2
        ok.append(True)
    return C, ok


def fd_stability(J_h1_cols, J_h2_cols, dir_measured, dir_predicted, max_rel=0.25):
    """Stable if the largest relative discrepancy between the two step sizes
    (on the repeated columns) and between measured and J-predicted
    directional derivatives is below max_rel. Returns (stable, discrepancy)."""
    a, b = np.asarray(J_h1_cols, float), np.asarray(J_h2_cols, float)
    scale = np.maximum(np.abs(a), 1e-8)
    disc = float(np.nanmax(np.abs(a - b) / scale)) if a.size else 0.0
    dm, dp = np.asarray(dir_measured, float), np.asarray(dir_predicted, float)
    if dm.size and np.any(np.abs(dm) > 1e-8):
        disc = max(disc, float(np.nanmax(np.abs(dm - dp) / np.maximum(np.abs(dm), 1e-8))))
    return disc < max_rel, disc


def fd_stability_rows(J_h1_cols, J_h2_cols, dir_measured, dir_predicted):
    """Per-constraint (per-row) relative discrepancy, normalised by the ROW
    norm so a near-zero element cannot dominate (the first run reported 41.0
    from one such element). Combines the half-step check on repeated columns
    with the directional-probe check; returns one discrepancy per residual."""
    a, b = np.asarray(J_h1_cols, float), np.asarray(J_h2_cols, float)
    m = a.shape[0]
    disc = np.zeros(m)
    for i in range(m):
        rn = max(float(np.linalg.norm(a[i])), 1e-8)
        disc[i] = float(np.linalg.norm(a[i] - b[i]) / rn) if a.size else 0.0
    dm, dp = np.asarray(dir_measured, float), np.asarray(dir_predicted, float)
    if dm.size:
        for i in range(m):
            rn = max(float(np.linalg.norm(dm[:, i])), 1e-8)
            disc[i] = max(disc[i], float(np.linalg.norm(dm[:, i] - dp[:, i]) / rn))
    return disc


def _rows_by_tag(log_path):
    rows = [json.loads(ln) for ln in Path(log_path).read_text().splitlines() if ln.strip()]
    return {r_["tag"]: r_ for r_ in rows}


def jacobian_from_archive(log_path, label: str, n: int, h: float):
    """(J, ok_cols) rebuilt from a prior run's probe rows tagged
    {label}:{j}:{+|-} (evaluations.jsonl or the final runner's archive.jsonl),
    so paid central-difference probes are reused. Missing or non-convergent
    probes leave NaN columns exactly as a fresh Jacobian would."""
    by = _rows_by_tag(log_path)
    residuals, meta = [], []
    for j in range(n):
        for sgn, ch in ((+1, "+"), (-1, "-")):
            row = by.get(f"{label}:{j}:{ch}")
            residuals.append(None if (row is None or "r" not in row) else row["r"])
            meta.append((j, sgn))
    return central_difference_jacobian(residuals, meta, n=n, h=h)


def curvature_from_archive(log_path, label: str, n: int, h, r0):
    """(C, ok_cols): diagonal curvature rebuilt from a prior run's probe rows
    tagged {label}:{j}:{+|-} (E7 reuse path), companion of jacobian_from_archive."""
    by = _rows_by_tag(log_path)
    residuals, meta = [], []
    for j in range(n):
        for sgn, ch in ((+1, "+"), (-1, "-")):
            row = by.get(f"{label}:{j}:{ch}")
            residuals.append(None if (row is None or "r" not in row) else row["r"])
            meta.append((j, sgn))
    return central_difference_curvature(residuals, meta, n=n, h=h, r0=r0)


def jacobian_from_log(log_path, relin: int, n: int, h: float):
    """Rebuild the Jacobian of linearisation `relin` from a prior run's
    evaluations.jsonl (tags J{relin}:{j}:{+|-}) so paid probes are reused.
    Returns (J, ok_cols, x_at, r_at): the point it was taken at is the
    'start' row for relin 0."""
    J, ok = jacobian_from_archive(log_path, f"J{relin}", n, h)
    start = _rows_by_tag(log_path)["start"]
    return J, ok, np.asarray(start["x"], float), np.asarray(start["r"], float)


def restoration_step(
    r, J, delta: float, tol: float = TOL, linearise=None, caps=None, radius_mult=None
):
    """LP: minimise t over (d, t) s.t. r_i + J_i d <= t for violated i,
    r_i + J_i d <= tol for satisfied i, -delta m_j <= d_j <= delta m_j with
    m = radius_mult (default all ones: the isotropic box; E6b passes the
    coefficient-space multipliers so new high-order modes get the same reach
    as the old ones). Returns (d, predicted max violation t). Columns with
    NaN (non-convergent probes) are frozen at d_j = 0."""
    from scipy.optimize import linprog

    r = np.asarray(r, float)
    J = np.asarray(J, float)
    m, n = J.shape
    rows_lin = list(range(m)) if linearise is None else list(linearise)
    bad = np.any(np.isnan(J[rows_lin]), axis=0)
    Jc = np.where(np.isnan(J), 0.0, J)
    violated = r > tol
    A_ub, b_ub = [], []
    # cumulative caps (paired experiment): r_i + J_i d <= cap_i, so the LP
    # never proposes a step the anchored non-regression rule would reject
    for i, cap in (caps or {}).items():
        A_ub.append(np.concatenate([Jc[i], [0.0]]))
        b_ub.append(float(cap) - r[i])
    for i in rows_lin:
        row = np.concatenate([Jc[i], [-1.0 if violated[i] else 0.0]])
        A_ub.append(row)
        # violated rows: bounded by the slack t. Satisfied rows: preserved
        # within the benchmark tolerance (<= tol), exactly what acceptance
        # requires (review round 8, 2026-09-18: the old min(tol, 0) = 0 bound
        # demanded improvement acceptance never asked for). The raw-well
        # requirement (residual <= 0 once reached) is the caller's cap.
        b_ub.append(-r[i] if violated[i] else tol - r[i])
    c = np.concatenate([np.zeros(n), [1.0]])
    mult = np.ones(n) if radius_mult is None else np.asarray(radius_mult, float)
    bounds = [(0.0, 0.0) if bad[j] else (-delta * mult[j], delta * mult[j]) for j in range(n)] + [
        (None, None)
    ]
    res = linprog(c, A_ub=np.asarray(A_ub), b_ub=np.asarray(b_ub), bounds=bounds, method="highs")
    if not res.success:
        # never drop the caps/preservation rows to make the LP solvable
        # (reviewer round 7); the caller treats this as a rejected trial
        raise RestorationLPInfeasible(f"restoration LP infeasible: {res.message}")
    d = res.x[:n]
    # the slack t bounds only the violated rows; report the predicted max
    # over ALL rows so the log compares like with like
    pred = (Jc @ d + r)[rows_lin]
    viol_lin = violated[rows_lin]
    t = float(max(res.x[n], pred[~viol_lin].max() if np.any(~viol_lin) else -np.inf))
    return d, t


def _convexified_lowrank_blocks(Cc, U, S_offdiag) -> np.ndarray:
    """E19 fix: the per-row eigen-clipped K x K curvature block.

    Forms, for each row i, B_i = U diag(Cc_i) U^T + S_offdiag_i (the row's
    ALREADY-CLIPPED ambient diagonal curvature projected into the measured
    subspace, plus the measured off-diagonal-only cross-term block -- its
    own diagonal must already be zeroed by the caller), eigendecomposes B_i
    and clips its negative eigenvalues to 0. Vectorised over all m rows via
    numpy's batched `eigh`. See `_quad_terms_from_convexified_block` for why
    this -- rather than clipping the summed diagonal+cross total -- is what
    makes the model convex."""
    Cc = np.asarray(Cc, float)
    U = np.atleast_2d(np.asarray(U, float))
    M = np.einsum("kn,mn,ln->mkl", U, Cc, U)
    B = M + np.asarray(S_offdiag, float)
    eigvals, eigvecs = np.linalg.eigh(B)
    eigvals_plus = np.maximum(eigvals, 0.0)
    return np.einsum("mki,mi,mli->mkl", eigvecs, eigvals_plus, eigvecs)


def _quad_terms_from_convexified_block(Cc, U, B_plus, d) -> np.ndarray:
    """The (m,) vector of UNHALVED per-row quadratic terms, given the
    already eigen-clipped block B_plus (`_convexified_lowrank_blocks`):
    diag(Cc_row) contracted with d_perp, the component of d orthogonal to
    U's row space (so the model is EXACTLY the diagonal-only model outside
    the measured subspace), plus z^T B_plus_row z, z = U d (so the model is
    exactly that convex block inside the subspace). Both pieces are
    individually >= 0 -- Cc >= 0 entrywise (the existing per-entry clip),
    B_plus PSD by construction -- so the per-row total is provably >= 0 for
    ANY d, with no need to clip the combined total afterwards.

    This is deliberately NOT diag(Cc_row) + U^T(B_plus_row - U diag(Cc_row)
    U^T)U (the literal "lift the correction back through U" reading): that
    algebraically simpler form is only PSD for d purely inside or purely
    outside the subspace. A d that mixes both can drive it negative when
    U's directions are not axis-aligned and Cc_row is non-uniform, because
    diag(Cc_row) and the projector onto U's row space do not commute in
    general (verified by direct counterexample: U from a random 2-of-5-dim
    QR basis, Cc = [10, 1] projected through it, already goes indefinite).
    The d_perp form used here is provably PSD for every d, matching the
    invariant this exists to guarantee."""
    Cc = np.asarray(Cc, float)
    U = np.atleast_2d(np.asarray(U, float))
    d = np.asarray(d, float)
    z = U @ d
    d_perp = d - U.T @ z
    return Cc @ (d_perp * d_perp) + np.einsum("mij,i,j->m", B_plus, z, z)


def restoration_step_quadratic(
    r,
    J,
    C,
    delta: float,
    tol: float = TOL,
    linearise=None,
    caps=None,
    radius_mult=None,
    lowrank=None,
):
    """E7: the restoration step under a diagonal second-order model
    q_i(d) = r_i + J_i d + 1/2 sum_j max(C_ij, 0) d_j^2. Minimise the slack t
    subject to q_i(d) <= t (violated rows), q_i(d) <= tol (satisfied rows,
    as acceptance requires), q_i(d) <= cap_i (caps), |d_j| <= delta m_j; negative curvature is
    clipped to 0 (the model never rewards leaving the box faster than the
    linear term). Solved by SLSQP.

    E19: `lowrank`, if given, is (U, S) -- U (K x n) orthonormal subspace
    directions and S (m x K x K) the measured curvature block per residual
    row (`subspace_curvature_from_probes`). Each row's quadratic term gains
    the measured off-diagonal cross-terms via `_quad_terms_from_convexified_block`
    (see that function's docstring for the exact model and why it is
    convexified per row rather than clipped after summing -- the earlier
    clip-after-sum model was non-convex and non-smooth wherever cross terms
    dominated, which is exactly what made SLSQP fail on the real smoke: all
    8 attempts errored with "Iteration limit reached" or "Positive
    directional derivative for linesearch"). With lowrank=None this is a
    no-op and results are byte-identical to the pre-E19 model. NaN entries
    in S (a non-convergent subspace probe) contribute zero cross-curvature
    for the affected pair, the same fallback central_difference_curvature's
    NaN columns get via Cc.

    E19 item 2: a solver failure is never immediately a rejected trial when
    a subspace is given -- it falls through a fallback ladder, in order:
    "subspace" (the full measured cross-term block), "subspace_half" (the
    cross terms scaled by 0.5 -- the same convexification, less of the
    measured curvature to be wrong about), "diagonal" (lowrank=None,
    dropping the subspace entirely). Only if EVERY rung fails does this
    raise RestorationLPInfeasible (a rejected trial, never a relaxation).
    Returns (d, predicted max over all linearised rows, model_used), where
    model_used names the rung that produced the returned step -- always
    "diagonal" when lowrank was None to begin with, since there is only one
    rung to try."""
    from scipy.optimize import minimize

    r = np.asarray(r, float)
    J = np.asarray(J, float)
    C = np.asarray(C, float)
    m, n = J.shape
    rows_lin = list(range(m)) if linearise is None else list(linearise)
    bad = np.any(np.isnan(J[rows_lin]), axis=0) | np.any(np.isnan(C[rows_lin]), axis=0)
    Jc = np.where(np.isnan(J), 0.0, J)
    Cc = np.maximum(np.where(np.isnan(C), 0.0, C), 0.0)
    violated = r > tol
    mult = np.ones(n) if radius_mult is None else np.asarray(radius_mult, float)

    def attempt(lowrank_arg):
        Uc, B_plus = None, None
        if lowrank_arg is not None:
            U_raw, S_raw = lowrank_arg
            Uc = np.atleast_2d(np.asarray(U_raw, float))
            S_offdiag = np.nan_to_num(np.asarray(S_raw, float), nan=0.0).copy()
            k_ = Uc.shape[0]
            S_offdiag[:, np.arange(k_), np.arange(k_)] = 0.0  # drop the diagonal
            B_plus = _convexified_lowrank_blocks(Cc, Uc, S_offdiag)  # precomputed
            # once: independent of d, so the SLSQP loop below never re-runs it

        def model(d):
            if lowrank_arg is None:
                quad = Cc @ (d * d)
            else:
                quad = _quad_terms_from_convexified_block(Cc, Uc, B_plus, d)
            return r + Jc @ d + 0.5 * quad

        cons = []
        for i, cap in (caps or {}).items():
            cons.append({"type": "ineq", "fun": lambda z, i=i, cap=cap: cap - model(z[:n])[i]})
        for i in rows_lin:
            if violated[i]:
                cons.append({"type": "ineq", "fun": lambda z, i=i: z[n] - model(z[:n])[i]})
            else:
                cons.append({"type": "ineq", "fun": lambda z, i=i: tol - model(z[:n])[i]})
        bounds = [
            (0.0, 0.0) if bad[j] else (-delta * mult[j], delta * mult[j]) for j in range(n)
        ] + [(None, None)]
        z0 = np.zeros(n + 1)
        z0[n] = float(r[rows_lin].max())
        res = minimize(
            lambda z: z[n],
            z0,
            jac=lambda z: np.concatenate([np.zeros(n), [1.0]]),
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        d = res.x[:n]
        pred = model(d)[rows_lin]
        feasible_model = all(
            c["fun"](res.x) >= -1e-8 for c in cons
        )  # SLSQP may report success on an infeasible model: check every row
        if not res.success or not feasible_model:
            raise RestorationLPInfeasible(f"quadratic restoration step failed: {res.message}")
        return d, float(pred.max())

    if lowrank is None:
        d, pred = attempt(None)
        return d, pred, "diagonal"

    U_raw, S_raw = lowrank
    half_lowrank = (U_raw, np.asarray(S_raw, float) * 0.5)
    ladder = (
        ("subspace", lowrank),
        ("subspace_half", half_lowrank),
        ("diagonal", None),
    )
    last_error = None
    for model_used, lowrank_arg in ladder:
        try:
            d, pred = attempt(lowrank_arg)
            return d, pred, model_used
        except RestorationLPInfeasible as e:
            last_error = e
    raise RestorationLPInfeasible(
        f"quadratic restoration step failed on every fallback ladder rung "
        f"(subspace, subspace_half, diagonal): {last_error}"
    )


# --------------------------------------------------------------------------- #
# E19: off-diagonal (subspace) curvature -- direction selection from prior-
# lineage archives, SVD fill, subspace probes, and block extraction. The
# quadratic step's low-rank extension lives on restoration_step_quadratic
# above; these are its inputs.
# --------------------------------------------------------------------------- #
def _archive_rows_by_tag(archive_dir) -> dict:
    path = Path(archive_dir) / "archive.jsonl"
    rows: dict = {}
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        rows[row["tag"]] = row
    return rows


def _x_for_accepted_tag(rows_by_tag: dict, tag: str, r) -> np.ndarray:
    """The evaluated point's x for one REPORT.json accepted-step entry.
    Recording rule (verified against results/p3-e7[,b,c,d]-176-8x8): REPORT's
    'accepted' list stores each accepted step under a SHORT tag
    ('{phase}:T{trial}', e.g. 'P:T1', 'R2:T14') with its measured residual r,
    but not its x. The evaluated x lives in archive.jsonl under a LONGER tag
    carrying the trust-region delta and line-search t* ('P:T1:d0.00125:t0.10');
    for every accepted step across all four archives there is exactly one
    archive.jsonl row whose tag has the short tag as a ':'-prefix and whose r
    matches the accepted entry, so x is recovered by that prefix+r match."""
    target = np.asarray(r, float)
    for full_tag, row in rows_by_tag.items():
        if not full_tag.startswith(tag + ":") or row.get("r") is None:
            continue
        if np.allclose(np.asarray(row["r"], float), target, atol=1e-9):
            return np.asarray(row["x"], float)
    raise ValueError(f"no archive.jsonl row found for accepted tag {tag!r}")


def _recovery_point_x(archive_dir, rows_by_tag: dict):
    """The x of the single mid-run recovery restart, if one occurred: read
    from events.jsonl's 'recovery_restart' event, whose 'tag' is already the
    FULL archive.jsonl tag of the picked point (it is `pick["tag"]`, an
    archive dict entry -- unlike the accepted list's short tags, no prefix
    search is needed here). None if there was no recovery in this archive."""
    events_path = Path(archive_dir) / "events.jsonl"
    if not events_path.exists():
        return None
    for ln in events_path.read_text().splitlines():
        if not ln.strip():
            continue
        event = json.loads(ln)
        if event.get("event") == "recovery_restart":
            row = rows_by_tag.get(event["tag"])
            return None if row is None else np.asarray(row["x"], float)
    return None


def accepted_step_displacements(archive_dir) -> list:
    """Displacement vectors d = x_after - x_before for every accepted step of
    one p3_final_restoration run, in chronological order (REPORT.json's
    'accepted' list is already chronological: `one_trial`'s shared trial
    counter only increases). The incumbent only moves on an accepted step or
    the single mid-run recovery restart -- `p3_final_restoration.run` never
    calls `do_recovery` twice (`recovery_used` gates a second attempt) -- so
    the point BEFORE step i is the point AFTER step i-1, except for the first
    accepted step whose short-tag phase prefix is 'R<digits>' (the recovery
    block's own trials), whose 'before' point is the recovery restart's point
    instead of the previous accepted step's."""
    archive_dir = Path(archive_dir)
    report = json.loads((archive_dir / "REPORT.json").read_text())
    start = json.loads((archive_dir / "start_serialised.json").read_text())
    rows_by_tag = _archive_rows_by_tag(archive_dir)
    recovery_x = _recovery_point_x(archive_dir, rows_by_tag)
    x_before = np.asarray(start["x"], float)
    seen_recovery_step = False
    displacements = []
    for step in report.get("accepted", []):
        phase = step["tag"].split(":", 1)[0]
        is_recovery_step = phase.startswith("R") and phase[1:].isdigit()
        if is_recovery_step and not seen_recovery_step:
            seen_recovery_step = True
            if recovery_x is not None:
                x_before = recovery_x
        x_after = _x_for_accepted_tag(rows_by_tag, step["tag"], step["r"])
        displacements.append(x_after - x_before)
        x_before = x_after
    return displacements


def directions_from_archives(paths, n: int, k: int) -> np.ndarray:
    """Unit vectors of accepted-step displacements from the given archives,
    most recent first: the LAST path in `paths` is treated as the most recent
    lineage link and read first, and within one archive its own most recent
    (last) accepted step is read before its earlier ones. Each candidate
    direction is Gram-Schmidt orthonormalised against the directions already
    kept; a candidate whose cosine similarity with any already-kept direction
    exceeds 0.99 is dropped as a near-duplicate before orthonormalising (a
    residual left by projecting out a near-parallel direction is numerically
    unreliable to renormalise). Returns an (m, n) array of orthonormal rows,
    m <= k."""
    kept: list = []
    for path in reversed(list(paths)):
        for d in reversed(accepted_step_displacements(path)):
            if len(kept) >= k:
                return np.asarray(kept, float)
            norm = float(np.linalg.norm(d))
            if norm < 1e-12:
                continue
            u = np.asarray(d, float) / norm
            if kept and max(abs(float(u @ q)) for q in kept) > 0.99:
                continue
            for q in kept:
                u = u - float(u @ q) * q
            norm2 = float(np.linalg.norm(u))
            if norm2 < 1e-6:
                continue
            kept.append(u / norm2)
    return np.asarray(kept, float).reshape(-1, n) if kept else np.zeros((0, n))


def fill_with_svd(U: np.ndarray, J: np.ndarray, k: int) -> np.ndarray:
    """Pad U (m x n, m <= k) with the top right-singular vectors of J (the
    Jacobian; NaN columns from non-convergent probes treated as 0) until
    there are k rows, each Gram-Schmidt orthonormalised against the
    directions already kept. Singular vectors are already mutually
    orthonormal, so only orthogonalising against the archive-sourced rows in
    U is needed; a vector that is (numerically) already in their span is
    skipped rather than kept as a near-zero, unreliable direction."""
    U = np.atleast_2d(np.asarray(U, float))
    n = np.asarray(J, float).shape[1]
    kept = [row for row in U] if U.size else []
    Jc = np.where(np.isnan(J), 0.0, np.asarray(J, float))
    _, _, Vt = np.linalg.svd(Jc, full_matrices=False)
    for v in Vt:  # rows already ordered by decreasing singular value
        if len(kept) >= k:
            break
        u = v.copy()
        for q in kept:
            u = u - float(u @ q) * q
        norm = float(np.linalg.norm(u))
        if norm < 1e-8:
            continue
        kept.append(u / norm)
    return np.asarray(kept, float).reshape(-1, n) if kept else np.zeros((0, n))


def subspace_probe_points(x0: np.ndarray, U: np.ndarray, h: float):
    """Probe points for the measured K x K subspace curvature block: for each
    direction u_i, x0 +- h u_i (2K points, single-direction); for each pair
    i < j, x0 +- h (u_i + u_j) (2 x K(K-1)/2 = K(K-1) points, pair). Returns
    (points (2K + K(K-1), n), meta) where meta[l] is ('diag', i, sign) for a
    single-direction probe or ('pair', i, j, sign) for a pair probe, sign in
    {+1, -1}, so subspace_curvature_from_probes can match residuals back to
    the entry they measure regardless of evaluation order."""
    x0 = np.asarray(x0, float)
    U = np.atleast_2d(np.asarray(U, float))
    K = U.shape[0]
    pts, meta = [], []
    for i in range(K):
        for sign in (+1, -1):
            pts.append(x0 + sign * h * U[i])
            meta.append(("diag", i, sign))
    for i in range(K):
        for j in range(i + 1, K):
            for sign in (+1, -1):
                pts.append(x0 + sign * h * (U[i] + U[j]))
                meta.append(("pair", i, j, sign))
    return np.asarray(pts), meta


def subspace_curvature_from_probes(residuals, meta, r0, U, h) -> np.ndarray:
    """The measured K x K low-rank curvature block per residual row, from the
    probes subspace_probe_points produced.

    Diagonal S_row[i, i]: the centred second difference along u_i, the same
    form as central_difference_curvature:
        S_ii = (f(+h u_i) - 2 f0 + f(-h u_i)) / h^2
    Cross terms from the pair probes at x0 +- h(u_i + u_j):
        S_ij = [f(+h(u_i+u_j)) + f(-h(u_i+u_j)) - 2 f0] / (2 h^2) - (S_ii + S_jj) / 2
    which follows from expanding a quadratic model q(d) = f0 + J d + 1/2 d^T H d
    at d = +-h(u_i + u_j), in the orthonormal subspace coordinates where
    u_p^T H u_q = H_pq = S_pq: q(+d) + q(-d) - 2 f0 = h^2 (S_ii + 2 S_ij + S_jj),
    so the LHS above equals (S_ii + S_jj)/2 + S_ij. Verified against a known
    quadratic to 1e-8 (test_subspace_curvature_from_probes_matches_a_known_quadratic).

    A None residual (a non-convergent probe, as central_difference_curvature
    also allows) leaves every S entry that depends on it NaN, never silently
    zero or dropped."""
    r0 = np.asarray(r0, float)
    m = len(r0)
    K = np.atleast_2d(U).shape[0]
    S = np.full((m, K, K), np.nan)
    diag_plus: dict = {}
    diag_minus: dict = {}
    pair_plus: dict = {}
    pair_minus: dict = {}
    for residual, tag in zip(residuals, meta):
        value = None if residual is None else np.asarray(residual, float)
        if tag[0] == "diag":
            _, i, sign = tag
            (diag_plus if sign > 0 else diag_minus)[i] = value
        else:
            _, i, j, sign = tag
            (pair_plus if sign > 0 else pair_minus)[(i, j)] = value
    diag_curv: dict = {}
    for i in range(K):
        f_plus, f_minus = diag_plus.get(i), diag_minus.get(i)
        if f_plus is None or f_minus is None:
            continue
        diag_curv[i] = (f_plus - 2.0 * r0 + f_minus) / h**2
        S[:, i, i] = diag_curv[i]
    for i in range(K):
        for j in range(i + 1, K):
            f_plus, f_minus = pair_plus.get((i, j)), pair_minus.get((i, j))
            if f_plus is None or f_minus is None:
                continue
            if i not in diag_curv or j not in diag_curv:
                continue
            combined = (f_plus + f_minus - 2.0 * r0) / (2.0 * h**2)
            cross = combined - (diag_curv[i] + diag_curv[j]) / 2.0
            S[:, i, j] = cross
            S[:, j, i] = cross
    return S


def quadratic_model_step_terms(C, d, lowrank=None) -> np.ndarray:
    """The (m,) vector of 0.5 * (per-row quadratic term) that both
    restoration_step_quadratic's model and model_prediction_error need:
    diag(max(C, 0)) plus, if `lowrank` = (U, S) is given, the measured
    off-diagonal cross-terms, convexified per row via
    `_convexified_lowrank_blocks` / `_quad_terms_from_convexified_block`
    (see the latter's docstring for the exact model and why it eigen-clips
    the row's measured block rather than clipping the diagonal-plus-cross
    SUM: the sum-clip is non-convex and non-smooth wherever cross terms
    dominate, which is what broke SLSQP in restoration_step_quadratic on the
    real smoke). With lowrank=None this is unchanged: diag(max(C, 0)) alone,
    byte-identical to the pre-E19 model. A caller passes this straight into
    `r + J @ d + quadratic_model_step_terms(...)`."""
    C = np.asarray(C, float)
    d = np.asarray(d, float)
    Cc = np.maximum(np.nan_to_num(C, nan=0.0), 0.0)
    if lowrank is None:
        return 0.5 * Cc @ (d * d)
    U_raw, S_raw = lowrank
    Uc = np.atleast_2d(np.asarray(U_raw, float))
    S_offdiag = np.nan_to_num(np.asarray(S_raw, float), nan=0.0).copy()
    k_ = Uc.shape[0]
    S_offdiag[:, np.arange(k_), np.arange(k_)] = 0.0
    B_plus = _convexified_lowrank_blocks(Cc, Uc, S_offdiag)
    return 0.5 * _quad_terms_from_convexified_block(Cc, Uc, B_plus, d)


def accept_step(r_old, r_new, tol: float = TOL, watch=None) -> bool:
    """Canonical acceptance: measured max violation falls AND no constraint
    that was satisfied (<= tol) becomes violated (> tol). Non-convergent
    trials (None) are rejected."""
    if r_new is None:
        return False
    r_old, r_new = np.asarray(r_old, float), np.asarray(r_new, float)
    if not (r_new.max() < r_old.max()):
        return False
    newly_violated = (r_old <= tol) & (r_new > tol)
    if bool(np.any(newly_violated)):
        return False
    # watched (non-linearised, acceptance-only) rows must not worsen by > tol
    for i in watch or []:
        if r_new[i] > r_old[i] + tol:
            return False
    return True


@dataclass
class TrustRegion:
    delta: float
    cap: float = 0.1
    floor: float = 1e-4

    def reject(self):
        self.delta = max(self.floor, self.delta / 2.0)

    def accept(self):
        self.delta = min(self.cap, self.delta * 2.0)


@dataclass
class Budget:
    cap: int
    used: int = 0

    def charge(self, n: int):
        if self.used + n > self.cap:
            raise RuntimeError(f"budget cap {self.cap} would be exceeded ({self.used}+{n})")
        self.used += n

    @property
    def remaining(self) -> int:
        return self.cap - self.used

    @property
    def exhausted(self) -> bool:
        return self.used >= self.cap


@dataclass
class RatioTrustRegion:
    """Corrected policy (reviewer round 6): rho = actual/predicted improvement.
    rho < 0.25 -> shrink by 2 (also on rejection / non-convergence);
    0.25 <= rho < 0.75 -> retain; rho >= 0.75 -> expand by 1.5, capped.
    Expansion requires strong agreement, not mere improvement."""

    delta: float
    cap: float = 0.0025
    floor: float = 1e-5

    def update(self, rho: float):
        if rho < 0.25:
            self.delta = max(self.floor, self.delta / 2.0)
        elif rho >= 0.75:
            self.delta = min(self.cap, self.delta * 1.5)

    def reject(self):
        self.delta = max(self.floor, self.delta / 2.0)


def accept_step_cumulative(r_start, r_old, r_new, tol: float = TOL, others=(0, 3, 4)) -> bool:
    """Canonical acceptance for the paired experiment: measured max falls;
    measured raw well >= 0 (normalised well residual <= 0); and cumulative
    non-regression anchored to the STARTING candidate: iota, flux and well
    residuals may not exceed their start values + tol at any accepted point.
    Non-convergent trials (None) are rejected."""
    if r_new is None:
        return False
    r_start, r_old, r_new = (np.asarray(v, float) for v in (r_start, r_old, r_new))
    if not (r_new.max() < r_old.max()):
        return False
    if r_new[4] > 0.0:
        return False
    return not any(r_new[i] > r_start[i] + tol for i in others)


class RestorationLPInfeasible(RuntimeError):
    """The LP with its caps/preservation rows has no feasible step. The caller
    treats this as a rejection (shrink the trust region); the rows are never
    silently dropped (reviewer round 7)."""


def accept_step_monotone_v(
    r_start, r_old, r_new, tol: float = TOL, eps: float = 1e-4, protect=(0, 4)
) -> bool:
    """Reviewer round 7 acceptance: V = max(0, all five residuals) must fall
    materially (V_new <= V_old - eps) with flux INSIDE that maximum rather than
    separately capped; raw well >= 0 on the measured point; iota and well keep
    their anchored protection (<= start + tol); any constraint already within
    the benchmark tolerance stays within it. Non-convergent -> rejected."""
    if r_new is None:
        return False
    r_start, r_old, r_new = (np.asarray(v, float) for v in (r_start, r_old, r_new))
    v_old = max(0.0, float(r_old.max()))
    v_new = max(0.0, float(r_new.max()))
    if not (v_new <= v_old - eps):
        return False
    if r_new[4] > 0.0:
        return False
    if any(r_new[i] > r_start[i] + tol for i in protect):
        return False
    return not any((r_old[i] <= tol) and (r_new[i] > tol) for i in range(len(r_old)))


def broyden_update(J, dx, dr, min_step: float = 1e-8, max_row_change: float = 0.5):
    """Safeguarded rank-1 Broyden update J <- J + ((dr - J dx) dx^T)/(dx^T dx)
    from one converged local evaluation. Frozen (NaN) columns stay NaN and are
    excluded from the product; non-finite inputs or results are refused
    (E4 2026-09-16: a NaN column poisoned every row). Skipped if the step is
    tiny or any row's norm would change by more than max_row_change."""
    J = np.asarray(J, float)
    dx = np.asarray(dx, float)
    dr = np.asarray(dr, float)
    if not (np.all(np.isfinite(dx)) and np.all(np.isfinite(dr))):
        return J, False
    frozen = np.any(np.isnan(J), axis=0)
    Jc = np.where(np.isnan(J), 0.0, J)
    dxc = np.where(frozen, 0.0, dx)
    nn = float(dxc @ dxc)
    if nn < min_step**2:
        return J, False
    upd = np.outer(dr - Jc @ dxc, dxc) / nn
    J1 = Jc + upd
    old_norm = np.linalg.norm(Jc, axis=1)
    new_norm = np.linalg.norm(J1, axis=1)
    rel = np.abs(new_norm - old_norm) / np.maximum(old_norm, 1e-8)
    if not np.all(np.isfinite(J1)) or np.any(rel > max_row_change):
        return J, False
    J1[:, frozen] = np.nan
    return J1, True


def drift_rows(J, dirs, measured):
    """Per-row relative disagreement between measured directional derivatives
    (central differences along unit directions) and J @ u."""
    J = np.asarray(J, float)
    dirs = np.asarray(dirs, float)
    measured = np.asarray(measured, float)
    pred = np.array([J @ u for u in dirs])
    out = np.zeros(J.shape[0])
    for i in range(J.shape[0]):
        rn = max(float(np.linalg.norm(measured[:, i])), 1e-8)
        out[i] = float(np.linalg.norm(measured[:, i] - pred[:, i]) / rn)
    return out


def worst_columns(J, row: int, k: int):
    """The k columns that dominate |J[row]| (largest weight first)."""
    Jr = np.abs(np.where(np.isnan(J), 0.0, np.asarray(J, float))[row])
    return [int(j) for j in np.argsort(-Jr)[:k]]


def load_start_candidate(path, n: int):
    """A saved candidate JSON (x, r, raw, representation, jacobian pointer
    {log, relin, h}) -> (x, r, J, ok_cols, rep). The Jacobian is rebuilt from
    the referenced log and must be centred on x."""
    c = json.loads(Path(path).read_text())
    x = np.asarray(c["x"], float)
    rr = np.asarray(c["r"], float)
    jp = c["jacobian"]
    rows = [json.loads(ln) for ln in Path(jp["log"]).read_text().splitlines() if ln.strip()]
    by = {r_["tag"]: r_ for r_ in rows}
    residuals, meta = [], []
    for j in range(n):
        for sgn, ch in ((+1, "+"), (-1, "-")):
            row = by.get(f"J{jp['relin']}:{j}:{ch}")
            residuals.append(None if (row is None or "r" not in row) else row["r"])
            meta.append((j, sgn))
            if row is not None:
                assert np.allclose(
                    np.asarray(row["x"]) - x, sgn * jp["h"] * np.eye(n)[j], atol=1e-9
                ), "Jacobian not centred on the candidate"
    J, ok = central_difference_jacobian(residuals, meta, n=n, h=jp["h"])
    rep_ = {k: c[k] for k in ("boundary_dict", "mpol", "ntor", "nfp", "scaling")}
    return x, rr, J, ok, rep_


def others_worse(r_old, r_new, tol: float = TOL, others=(0, 3, 4)) -> bool:
    """Pre-registered 'do not worsen the other constraints beyond tol':
    iota, flux, well may not rise by more than tol per accepted step, whether
    or not they were already violated (run 2 counted a flux +0.028 step as
    clean because flux was already violated; that was laxer than the text)."""
    r_old, r_new = np.asarray(r_old, float), np.asarray(r_new, float)
    return bool(any(r_new[i] > r_old[i] + tol for i in others))


def classify_outcome(
    stable,
    nonconv_frac,
    final_max,
    final_well,
    accepted_steps,
    success_max=0.10,
    min_total_gain=0.02,
    max_nonconv=0.30,
) -> str:
    """Pre-registered, mutually exclusive outcomes (checked in this order)."""
    if (not stable) or nonconv_frac > max_nonconv:
        return "inconclusive"
    if final_max <= success_max and final_well >= 0.0:
        return "practical_success"
    joint = [
        s for s in accepted_steps if s["d_mirror"] < 0 and s["d_qi"] < 0 and not s["others_worse"]
    ]
    # the gain that counts is over the CLEAN joint steps only (pre-registration:
    # steps that reduce both mirror and QI AND do not worsen the others)
    total_gain = -sum(s["d_max"] for s in joint)
    if joint and total_gain >= min_total_gain:
        return "mechanistic_positive"
    return "local_negative"


# --------------------------------------------------------------------------- #
# canonical evaluation (VMEC; parallel)
# --------------------------------------------------------------------------- #
def surrogate_widths_of(payload: dict) -> tuple:
    """Relative widths of the smoothed-gate flux surrogate requested for an
    evaluation (E10); empty = the benchmark metrics only."""
    return tuple(float(w) for w in payload.get("surrogate_widths", ()))


def _fake_quadratic_residuals(
    x,
    well_violated: bool = False,
    near_feasible: bool = False,
    coupling=None,
) -> list:
    """Test seam (payload 'fake_quadratic'): a convex toy with the P3 shape
    (5 residuals, well last) and a feasible region near the origin, so the
    runner's phases can be exercised end to end without VMEC. With
    well_violated the start has well > 0 and a coordinate that lowers it
    (E11: the real davidkh start). With near_feasible the start is already
    just above TOL on every row, so a real runner reaches a confirmed
    feasible point within a small evaluation ceiling (fix 2's success-path
    test: it needs the loop to actually STOP there, not just log it).

    `coupling` (E19 item 3, payload 'fake_coupling'): an optional K x K matrix
    adding a genuine cross term 0.5 * x[:K] @ coupling @ x[:K] to residual row
    1 only (log10_qi), on top of its existing isotropic diagonal bowl. Row 1
    is the natural choice: it is violated at the default start and every
    other row's bowl stays purely diagonal, so a diagonal-only curvature
    model mismeasures exactly this one row while a subspace model spanning
    the first K coordinates measures it exactly (the toy ratchet test)."""
    x = np.asarray(x, float)
    n = len(x)
    A = np.zeros((5, n))
    A[0, 0] = -1.0
    A[1, 1] = -1.0
    A[2, 2] = -1.0
    A[3, 0] = 0.5
    A[3, 1] = 0.5
    B = np.array([0.20, 0.15, 0.12, 0.05, -0.05])
    Q = np.array([4.0, 3.0, 3.0, 1.0, 0.0])
    if well_violated:
        A[4, 3] = -1.0
        B[4] = 0.30
    if near_feasible:
        B = np.array([0.014, 0.012, 0.010, 0.006, -0.05])
    r = A @ x + B + 0.5 * Q * float(x @ x) / n
    if coupling is not None:
        Cpl = np.asarray(coupling, float)
        k = Cpl.shape[0]
        xs = x[:k]
        r[1] += 0.5 * float(xs @ Cpl @ xs)
    return [float(v) for v in r]


def _fake_front_residuals_and_raw(x, constraint_overrides=None) -> tuple:
    """Test seam (payload 'fake_front', E16): a smooth toy for the front
    builder's toy end-to-end test. Three residuals (iota, QI, mirror) are
    constants deep inside tolerance; well is likewise deeply satisfied and
    never moves (so the raw-well cap never binds); flux rises 1:1 with
    coordinate 0, so a large enough step pushes it toward the 0.01 bound
    (the constraint that binds) while a SLSQP-scale trust-region step keeps
    it comfortably inside. Objectives: aspect_ratio falls with x0 and x1
    (good, aspect is minimised) and min_len rises with x0 and x2 (good, it is
    maximised), so the +x0 direction improves both objectives at once while
    also loading the flux residual -- exactly the "objectives improve along a
    direction that also increases one residual toward the bound" shape the
    E16 pre-registration asks the toy to exercise. Feasible, comfortably
    interior start at x = 0.

    The residuals are not written down directly: the toy carries fake RAW
    measurements (the same keys `_eval_worker` records for a real
    evaluation) and normalises them with `normalised_constraint_violations`,
    so `constraint_overrides` (E18 item 1) changes the toy's residuals by
    exactly the benchmark's own formula -- a `mirror_upper=0.23` run is a
    different evaluation, not just a different REPORT.json. The raw values
    are chosen so the STOCK thresholds reproduce the original constants
    (-0.05 everywhere, flux -0.05 + x0) to floating-point precision:
    iota 0.25 * 1.05, log10 QI -3.5 * 1.05, mirror 0.25 * 0.95, flux
    0.9 * (0.95 + x0), well 0.005 (normalised by the class's 0.1 floor)."""
    x = np.asarray(x, float)
    x0 = x[0] if len(x) > 0 else 0.0
    x1 = x[1] if len(x) > 1 else 0.0
    x2 = x[2] if len(x) > 2 else 0.0
    raw = {
        "iota_over_nfp": 0.25 * 1.05,
        "log10_qi": -3.5 * 1.05,
        "edge_mirror": 0.25 * 0.95,
        "flux_compression": 0.9 * (0.95 + x0),
        "vacuum_well": 0.005,
        "aspect_ratio": 10.0 - 1.0 * x0 - 0.5 * x1,
        "min_len": 10.0 + 0.5 * x0 + 1.0 * x2,
    }
    r = normalised_constraint_violations(raw, constraint_overrides)
    return r, {k: float(v) for k, v in raw.items()}


def _eval_worker(payload: dict) -> dict:
    out = {"tag": payload["tag"]}
    t0 = time.time()
    if payload.get("fake_quadratic"):
        out["r"] = _fake_quadratic_residuals(
            payload["x"],
            bool(payload.get("fake_well_violated", False)),
            bool(payload.get("fake_near_feasible", False)),
            coupling=payload.get("fake_coupling"),
        )
        out["raw"] = {"vacuum_well": -out["r"][4] / 10.0, "flux_compression": 0.9}
        out["feasible"] = bool(max(out["r"]) <= TOL)
        out["elapsed_s"] = 0.0
        return out
    if payload.get("fake_front"):
        r, raw = _fake_front_residuals_and_raw(payload["x"], payload.get("constraint_overrides"))
        out["r"] = r
        out["raw"] = raw
        out["feasible"] = bool(max(r) <= TOL)
        out["elapsed_s"] = 0.0
        return out
    try:
        import jax.numpy as jnp
        from constellaration import forward_model

        from constellaration_methods.engines.alm_ngopt_optimizer import (
            masked_coefficients_and_scale,
        )

        _, _, _, unravel, scale = masked_coefficients_and_scale(
            payload["boundary_dict"],
            max_poloidal_mode=payload["mpol"],
            max_toroidal_mode=payload["ntor"],
            n_field_periods=payload["nfp"],
            infinity_norm_spectrum_scaling=payload["scaling"],
        )
        surface = unravel(jnp.asarray(payload["x"]) * jnp.asarray(scale))
        metrics, equilibrium = forward_model.forward_model(
            boundary=surface,
            settings=forward_model.ConstellarationSettings.default_high_fidelity(),
        )
        if metrics is None:
            out["non_convergent"] = True
            return out
        p = problem_with_overrides(payload.get("constraint_overrides"))
        widths = surrogate_widths_of(payload)
        if widths:
            # p3_flux_surrogate is out of scope for this repo (excluded module);
            # this optional surrogate-widths path will ImportError if ever
            # exercised -- acceptable since it's optional/rare and this is
            # pre-existing behaviour that just moved with the file.
            from constellaration_methods.evaluation.p3_flux_surrogate import (
                smoothed_flux_compressions,
            )

            # a surrogate failure must not lose the paid canonical evaluation
            try:
                out["flux_smooth"] = smoothed_flux_compressions(equilibrium, widths)
            except Exception as e:  # noqa: BLE001
                out["flux_smooth_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["r"] = [float(v) for v in p._normalized_constraint_violations(metrics)]
        out["raw"] = {
            "iota_over_nfp": float(metrics.edge_rotational_transform_over_n_field_periods),
            "log10_qi": float(np.log10(metrics.qi)),
            "edge_mirror": float(metrics.edge_magnetic_mirror_ratio),
            "flux_compression": float(metrics.flux_compression_in_regions_of_bad_curvature),
            "vacuum_well": float(metrics.vacuum_well),
            "aspect_ratio": float(metrics.aspect_ratio),
            "min_len": float(metrics.minimum_normalized_magnetic_gradient_scale_length),
        }
        out["feasible"] = bool(p.is_feasible(metrics))
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if (
            "did not converge" in msg
            or "JACOBIAN" in msg
            or "FATAL ERROR" in msg
            or "solver failed" in msg
        ):
            out["non_convergent"] = True
        else:
            out["error"] = f"{type(e).__name__}: {msg[:300]}"
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


class CanonicalEvaluator:
    """Batch-evaluates points canonically, logs every evaluation, charges the budget."""

    def __init__(self, rep: dict, pool, log_path: Path, budget: Budget):
        self.rep, self.pool, self.log_path, self.budget = rep, pool, log_path, budget
        self.n_nonconv = 0
        self.n_total = 0

    def evaluate(self, points, tags):
        self.budget.charge(len(points))
        payloads = [dict(self.rep, x=[float(v) for v in p], tag=t) for p, t in zip(points, tags)]
        results = list(self.pool.map(_eval_worker, payloads))
        with open(self.log_path, "a") as fh:
            for p, res in zip(points, results):
                res["x"] = [float(v) for v in p]
                fh.write(json.dumps(res) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.n_total += len(results)
        self.n_nonconv += sum(1 for r_ in results if r_.get("non_convergent"))
        return [None if ("r" not in r_) else np.asarray(r_["r"], float) for r_ in results], results


def run(args) -> dict:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ck = json.loads(Path(args.checkpoint).read_text())
    x0 = np.asarray(ck["best_x"], float)
    rep = dict(
        boundary_dict=ck["boundary_dict"],
        mpol=int(ck["max_poloidal_mode"]),
        ntor=int(ck["max_toroidal_mode"]),
        nfp=int(ck["n_field_periods"]),
        scaling=float(ck.get("infinity_norm_spectrum_scaling", 1.0)),
    )
    n = len(x0)
    budget = Budget(cap=args.budget)
    log_path = out / "evaluations.jsonl"
    rng = np.random.default_rng(args.seed)
    ctx = mp.get_context("spawn")
    events = []

    def log_event(**kw):
        kw["t"] = round(time.time() - t_start, 1)
        events.append(kw)
        (out / "events.jsonl").open("a").write(json.dumps(kw) + "\n")
        print(json.dumps(kw), flush=True)

    lin_names = [t.strip() for t in args.linearise.split(",") if t.strip()]
    lin_idx = [RESIDUAL_NAMES.index(nm) for nm in lin_names]
    watch_idx = [i for i in range(len(RESIDUAL_NAMES)) if i not in lin_idx]
    reused = None
    if args.reuse_jacobian:
        reused = jacobian_from_log(args.reuse_jacobian, relin=0, n=n, h=args.h)
    stability_rows = (
        json.loads(Path(args.stability_json).read_text()) if args.stability_json else None
    )
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        ev = CanonicalEvaluator(rep, pool, log_path, budget)
        # 0. the start point: canonically verified now, or taken from the reused log
        if reused is not None:
            J_re, ok_re, x_re, r_re = reused
            if not np.allclose(x_re, x0):
                raise SystemExit(
                    "--reuse-jacobian: its start point is not this checkpoint's best_x"
                )
            r0, raw0 = r_re, {}
            log_event(
                event="start_reused",
                r=r0.tolist(),
                max=float(r0.max()),
                linearise=lin_names,
                watch=[RESIDUAL_NAMES[i] for i in watch_idx],
            )
        else:
            (r0,), (raw0,) = ev.evaluate([x0], ["start"])
            if r0 is None:
                log_event(event="start_nonconvergent")
                return {
                    "outcome": "inconclusive",
                    "reason": "start point non-convergent",
                }
            log_event(
                event="start",
                r=r0.tolist(),
                max=float(r0.max()),
                raw=raw0.get("raw"),
                linearise=lin_names,
                watch=[RESIDUAL_NAMES[i] for i in watch_idx],
            )
        incumbent_x, incumbent_r = x0.copy(), r0.copy()
        accepted_steps = []
        stable = None
        tr = TrustRegion(delta=args.delta0, cap=args.delta_cap)
        h = args.h
        n_relin = 0
        while not budget.exhausted and n_relin < args.max_relinearisations:
            # 1. Jacobian at the incumbent (linearisation 0 reused if given)
            if n_relin == 0 and reused is not None:
                J, ok = J_re, ok_re
                stable = True  # established for the linearised rows in the prior run
                log_event(
                    event="jacobian_reused",
                    relin=0,
                    ok_cols=int(sum(ok)),
                    n=n,
                    source=args.reuse_jacobian,
                    stability_per_constraint=stability_rows,
                )
            else:
                need = 2 * n
                if budget.remaining < need + 8:
                    log_event(event="budget_short_for_jacobian", remaining=budget.remaining)
                    break
                pts, meta = probe_points(incumbent_x, h)
                rs, _ = ev.evaluate(
                    pts, [f"J{n_relin}:{j}:{'+' if s > 0 else '-'}" for j, s in meta]
                )
                J, ok = central_difference_jacobian(rs, meta, n=n, h=h)
                log_event(event="jacobian", relin=n_relin, ok_cols=int(sum(ok)), n=n)
            # 2. stability check (first linearisation only), per constraint
            if stable is None:
                k = min(args.stability_coords, n)
                coords = sorted(rng.choice(n, size=k, replace=False).tolist())
                pts2, meta2 = probe_points(incumbent_x, h / 2.0, coords=coords)
                dirs = rng.standard_normal((args.stability_dirs, n))
                dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                dpts = np.concatenate([incumbent_x + h * dirs, incumbent_x - h * dirs])
                rs2, _ = ev.evaluate(
                    np.concatenate([pts2, dpts]),
                    [f"S:{j}:{'+' if s > 0 else '-'}" for j, s in meta2]
                    + [f"D{i}:+" for i in range(len(dirs))]
                    + [f"D{i}:-" for i in range(len(dirs))],
                )
                J2, ok2 = central_difference_jacobian(rs2[: 2 * k], meta2, n=n, h=h / 2.0)
                cols = [j for j, o in zip(coords, ok2) if o and ok[j]]
                dm, dp = [], []
                for i in range(len(dirs)):
                    rp, rm = rs2[2 * k + i], rs2[2 * k + len(dirs) + i]
                    if rp is not None and rm is not None:
                        dm.append((rp - rm) / (2 * h))
                        dp.append(np.where(np.isnan(J), 0.0, J) @ dirs[i])
                disc_rows = fd_stability_rows(
                    J[:, cols],
                    J2[:, cols],
                    np.asarray(dm) if dm else np.zeros((0, 5)),
                    np.asarray(dp) if dp else np.zeros((0, 5)),
                )
                stability_rows = {
                    RESIDUAL_NAMES[i]: round(float(disc_rows[i]), 4) for i in range(5)
                }
                stable = bool(all(disc_rows[i] < 0.25 for i in lin_idx))
                log_event(
                    event="stability",
                    stable=stable,
                    per_constraint=stability_rows,
                    linearised=lin_names,
                    cols_compared=len(cols),
                    dirs_compared=len(dm),
                )
                if not stable:
                    break
            # 3-4. trust-region steps until one is accepted or the region collapses
            accepted = False
            for _trial in range(args.max_trials_per_linearisation):
                if budget.exhausted:
                    break
                d, t_pred = restoration_step(incumbent_r, J, tr.delta, TOL, linearise=lin_idx)
                (r_new,), (raw_new,) = ev.evaluate(
                    [incumbent_x + d], [f"T{n_relin}:{_trial}:d{tr.delta:.4f}"]
                )
                ok_step = accept_step(incumbent_r, r_new, TOL, watch=watch_idx)
                log_event(
                    event="trial",
                    relin=n_relin,
                    trial=_trial,
                    delta=round(tr.delta, 5),
                    predicted_max=round(t_pred, 4),
                    measured_max=(None if r_new is None else round(float(r_new.max()), 4)),
                    nonconv=r_new is None,
                    accepted=bool(ok_step),
                    r_new=(None if r_new is None else [round(float(v), 4) for v in r_new]),
                )
                if ok_step:
                    step = {
                        "d_mirror": float(r_new[2] - incumbent_r[2]),
                        "d_qi": float(r_new[1] - incumbent_r[1]),
                        "d_max": float(r_new.max() - incumbent_r.max()),
                        "others_worse": others_worse(incumbent_r, r_new, TOL)
                        or any(r_new[i] > incumbent_r[i] + TOL for i in watch_idx),
                        "raw": raw_new.get("raw"),
                    }
                    accepted_steps.append(step)
                    incumbent_x, incumbent_r = incumbent_x + d, r_new
                    (out / "incumbent.json").write_text(
                        json.dumps(
                            {
                                "x": incumbent_x.tolist(),
                                "r": incumbent_r.tolist(),
                                "raw": raw_new.get("raw"),
                                "boundary_dict": rep["boundary_dict"],
                                "mpol": rep["mpol"],
                                "ntor": rep["ntor"],
                                "nfp": rep["nfp"],
                                "scaling": rep["scaling"],
                                "n_accepted": len(accepted_steps),
                                "budget_used": budget.used,
                            },
                            indent=1,
                        )
                    )
                    tr.accept()
                    accepted = True
                    break
                tr.reject()
                if tr.delta <= tr.floor:
                    break
            n_relin += 1
            if not accepted:
                log_event(
                    event="no_accepted_step_at_linearisation",
                    relin=n_relin - 1,
                    delta=tr.delta,
                )
                if tr.delta <= tr.floor:
                    break
    nonconv_frac = ev.n_nonconv / max(1, ev.n_total)
    outcome = classify_outcome(
        stable=bool(stable) if stable is not None else False,
        nonconv_frac=nonconv_frac,
        final_max=float(incumbent_r.max()),
        final_well=float(-incumbent_r[4]),  # normalised well residual is -10*well
        accepted_steps=accepted_steps,
    )
    report = {
        "outcome": outcome,
        "start_max": float(r0.max()),
        "final_max": float(incumbent_r.max()),
        "start_r": r0.tolist(),
        "final_r": incumbent_r.tolist(),
        "n_accepted_steps": len(accepted_steps),
        "accepted_steps": accepted_steps,
        "stable": stable,
        "stability_per_constraint": stability_rows,
        "linearised": lin_names,
        "watched": [RESIDUAL_NAMES[i] for i in watch_idx],
        "nonconv_frac": round(nonconv_frac, 3),
        "evaluations_used": budget.used,
        "budget": budget.cap,
        "relinearisations": n_relin,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    (out / "REPORT.json").write_text(json.dumps(report, indent=2))
    print("\n=== RESTORATION REPORT ===\n" + json.dumps(report, indent=2), flush=True)
    return report


@dataclass
class StallCounter:
    """Stall = max_in_row consecutive trials that were rejected OR accepted
    with a gain below min_gain (fix after Arm B, 2026-09-15: null steps at the
    radius floor never triggered the rejection-only stall rule)."""

    max_in_row: int = 8
    min_gain: float = 1e-4
    n_in_row: int = 0

    def record(self, accepted: bool, gain):
        if accepted and gain is not None and gain >= self.min_gain:
            self.n_in_row = 0
        else:
            self.n_in_row += 1

    @property
    def stalled(self) -> bool:
        return self.n_in_row >= self.max_in_row

    @staticmethod
    def at_floor(delta: float, floor: float) -> bool:
        return delta <= floor


def relinearisation_fits(remaining: int, n: int) -> bool:
    return remaining >= 2 * n + 2


def run_paired(args) -> dict:
    """One arm of the paired experiment (vault p3-broyden-restoration-plan-
    2026-09-15): start from a saved candidate + its Jacobian; corrected
    ratio trust region; cumulative non-regression acceptance; arm 'full'
    re-linearises after every accepted step, arm 'broyden' reuses the
    Jacobian with safeguarded updates, drift checks and partial refreshes."""
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = args.n_coords
    x0, r0, J, ok, rep = load_start_candidate(args.start_candidate, n=n)
    budget = Budget(cap=args.allowance)
    log_path = out / "evaluations.jsonl"
    rng = np.random.default_rng(args.seed)
    ctx = mp.get_context("spawn")
    t_start = time.time()

    def log_event(**kw):
        kw["t"] = round(time.time() - t_start, 1)
        (out / "events.jsonl").open("a").write(json.dumps(kw) + "\n")
        print(json.dumps(kw), flush=True)

    tr = RatioTrustRegion(delta=args.delta0, cap=args.delta_cap)
    inc_x, inc_r = x0.copy(), r0.copy()
    accepted, refreshes, n_since_check, n_broyden = [], [], 0, 0
    log_event(
        event="start",
        arm=args.arm,
        r=r0.tolist(),
        max=float(r0.max()),
        ok_cols=int(sum(ok)),
        delta=tr.delta,
    )
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        ev = CanonicalEvaluator(rep, pool, log_path, budget)
        trial = 0
        stall = StallCounter(max_in_row=args.max_rejects_in_row, min_gain=1e-4)
        while not budget.exhausted:
            if args.policy == "anchored":
                caps = {i: float(r0[i] + TOL) for i in (0, 3, 4)}
            else:  # monotone_v: flux inside the maximum; iota/well protected
                caps = {i: float(r0[i] + TOL) for i in (0, 4)}
            caps[4] = min(caps[4], 0.0)  # raw well >= 0
            try:
                d, t_pred = restoration_step(inc_r, J, tr.delta, TOL, caps=caps)
            except RestorationLPInfeasible as e:
                log_event(
                    event="lp_infeasible",
                    arm=args.arm,
                    trial=trial,
                    delta=round(tr.delta, 6),
                    msg=str(e)[:120],
                )
                tr.reject()
                stall.record(accepted=False, gain=None)
                if stall.stalled or StallCounter.at_floor(tr.delta, tr.floor):
                    break
                trial += 1
                continue
            pred_impr = float(inc_r.max() - t_pred)
            pred_r = inc_r + np.where(np.isnan(J), 0.0, J) @ d
            (r_new,), (raw_new,) = ev.evaluate([inc_x + d], [f"T{trial}:d{tr.delta:.5f}"])
            if args.policy == "anchored":
                ok_step = accept_step_cumulative(r0, inc_r, r_new, TOL)
            else:
                ok_step = accept_step_monotone_v(r0, inc_r, r_new, TOL, eps=args.eps)
            act_impr = None if r_new is None else float(inc_r.max() - r_new.max())
            rho = None if (r_new is None or pred_impr <= 1e-12) else act_impr / pred_impr
            log_event(
                event="trial",
                arm=args.arm,
                trial=trial,
                delta=round(tr.delta, 6),
                predicted_max=round(t_pred, 4),
                predicted_r=[round(float(v), 4) for v in pred_r],
                measured_max=(None if r_new is None else round(float(r_new.max()), 4)),
                rho=(None if rho is None else round(rho, 3)),
                accepted=bool(ok_step),
                nonconv=r_new is None,
                r_new=(None if r_new is None else [round(float(v), 4) for v in r_new]),
            )
            # Broyden: learn from every converged local evaluation, accepted or not
            if (
                args.arm == "broyden"
                and r_new is not None
                and np.abs(d).max() <= 2 * args.delta_cap
            ):
                J, applied = broyden_update(J, d, r_new - inc_r)
                n_broyden += int(applied)
            if ok_step:
                clean = (r_new[2] < inc_r[2]) and (r_new[1] < inc_r[1])
                accepted.append(
                    {
                        "trial": trial,
                        "delta": tr.delta,
                        "rho": rho,
                        "d_max": float(r_new.max() - inc_r.max()),
                        "d_mirror": float(r_new[2] - inc_r[2]),
                        "d_qi": float(r_new[1] - inc_r[1]),
                        "clean_joint": bool(clean),
                        "r": r_new.tolist(),
                        "raw": raw_new.get("raw"),
                        "evaluations_at": budget.used,
                    }
                )
                inc_x, inc_r = inc_x + d, r_new
                (out / "incumbent.json").write_text(
                    json.dumps(
                        {
                            "x": inc_x.tolist(),
                            "r": inc_r.tolist(),
                            "raw": raw_new.get("raw"),
                            **rep,
                            "n_accepted": len(accepted),
                            "evaluations_used": budget.used,
                        },
                        indent=1,
                    )
                )
                tr.update(rho)
                stall.record(accepted=True, gain=-accepted[-1]["d_max"])
                if stall.stalled or StallCounter.at_floor(tr.delta, tr.floor):
                    log_event(
                        event="stalled",
                        arm=args.arm,
                        delta=tr.delta,
                        in_row=stall.n_in_row,
                    )
                    break
                n_since_check += 1
                if args.arm == "full":
                    if not relinearisation_fits(budget.remaining, n):
                        # keep trialling on the existing model until the
                        # allowance is spent (fix after Arm F, 2026-09-15:
                        # stopping here under-used the allowance)
                        log_event(
                            event="relinearisation_skipped_budget",
                            remaining=budget.remaining,
                        )
                        trial += 1
                        continue
                    pts, meta = probe_points(inc_x, args.h)
                    rs, _ = ev.evaluate(
                        pts,
                        [f"J{len(accepted)}:{j}:{'+' if sg > 0 else '-'}" for j, sg in meta],
                    )
                    J, ok = central_difference_jacobian(rs, meta, n=n, h=args.h)
                    refreshes.append(
                        {
                            "kind": "full",
                            "after_accepted": len(accepted),
                            "evaluations": 2 * n,
                        }
                    )
                    log_event(event="relinearised", arm=args.arm, ok_cols=int(sum(ok)))
                elif n_since_check >= args.drift_every:
                    n_since_check = 0
                    if budget.remaining < 4 + 2:
                        break
                    dirs = rng.standard_normal((2, n))
                    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                    pts = np.concatenate([inc_x + args.h * dirs, inc_x - args.h * dirs])
                    rs, _ = ev.evaluate(
                        pts,
                        [f"D{len(accepted)}:{i}:+" for i in range(2)]
                        + [f"D{len(accepted)}:{i}:-" for i in range(2)],
                    )
                    meas = []
                    for i in range(2):
                        if rs[i] is not None and rs[2 + i] is not None:
                            meas.append((rs[i] - rs[2 + i]) / (2 * args.h))
                    if meas:
                        disc = drift_rows(J, dirs[: len(meas)], np.asarray(meas))
                        log_event(
                            event="drift_check",
                            arm=args.arm,
                            per_row=[round(float(v), 3) for v in disc],
                        )
                        bad = [i for i in range(5) if disc[i] > 0.25]
                        if sum(1 for i in range(5) if disc[i] > 0.5) >= 2:
                            if budget.remaining < 2 * n + 2:
                                break
                            pts, meta = probe_points(inc_x, args.h)
                            rs2, _ = ev.evaluate(
                                pts,
                                [
                                    f"J{len(accepted)}:{j}:{'+' if sg > 0 else '-'}"
                                    for j, sg in meta
                                ],
                            )
                            J, ok = central_difference_jacobian(rs2, meta, n=n, h=args.h)
                            refreshes.append(
                                {
                                    "kind": "full",
                                    "after_accepted": len(accepted),
                                    "evaluations": 2 * n,
                                }
                            )
                            log_event(
                                event="full_refresh",
                                arm=args.arm,
                                reason="drift>0.5 on >=2 rows",
                            )
                        elif bad:
                            cols = sorted({c for i in bad for c in worst_columns(J, i, 8)})
                            if budget.remaining < 2 * len(cols) + 2:
                                break
                            pts, meta = probe_points(inc_x, args.h, coords=cols)
                            rs2, _ = ev.evaluate(
                                pts,
                                [
                                    f"P{len(accepted)}:{j}:{'+' if sg > 0 else '-'}"
                                    for j, sg in meta
                                ],
                            )
                            Jp, okp = central_difference_jacobian(rs2, meta, n=n, h=args.h)
                            for c_, o_ in zip(cols, okp):
                                if o_:
                                    J[:, c_] = Jp[:, c_]
                            refreshes.append(
                                {
                                    "kind": "partial",
                                    "after_accepted": len(accepted),
                                    "columns": cols,
                                    "evaluations": 2 * len(cols),
                                }
                            )
                            log_event(
                                event="partial_refresh",
                                arm=args.arm,
                                rows=bad,
                                columns=len(cols),
                            )
            else:
                tr.reject()
                stall.record(accepted=False, gain=None)
                if stall.stalled or StallCounter.at_floor(tr.delta, tr.floor):
                    log_event(
                        event="stalled",
                        arm=args.arm,
                        delta=tr.delta,
                        in_row=stall.n_in_row,
                    )
                    break
            trial += 1
    clean_gain = -sum(a["d_max"] for a in accepted if a["clean_joint"])
    report = {
        "arm": args.arm,
        "start_max": float(r0.max()),
        "final_max": float(inc_r.max()),
        "start_r": r0.tolist(),
        "final_r": inc_r.tolist(),
        "cumulative_regression_vs_start": {
            RESIDUAL_NAMES[i]: round(float(inc_r[i] - r0[i]), 4) for i in (0, 3, 4)
        },
        "n_accepted": len(accepted),
        "n_clean_joint": sum(1 for a in accepted if a["clean_joint"]),
        "clean_joint_gain": round(clean_gain, 5),
        "total_gain": round(float(r0.max() - inc_r.max()), 5),
        "evaluations_used": budget.used,
        "allowance": budget.cap,
        "clean_joint_gain_per_evaluation": round(clean_gain / max(1, budget.used), 7),
        "refreshes": refreshes,
        "n_refresh": len(refreshes),
        "n_broyden_updates": n_broyden,
        "accepted_steps": accepted,
        "nonconv_frac": round(ev.n_nonconv / max(1, ev.n_total), 3),
        "practical_success": bool(inc_r.max() <= 0.10 and inc_r[4] <= 0.0),
        "policy": args.policy,
        "eps": args.eps,
        "material_progress_0.130": bool(inc_r.max() <= 0.130 and inc_r[4] <= 0.0),
        "flux_is_worst_at_stop": bool(
            int(np.argmax(inc_r)) == 3 or inc_r[3] >= inc_r.max() - 0.005
        ),
        "wall_s": round(time.time() - t_start, 1),
    }
    (out / "REPORT.json").write_text(json.dumps(report, indent=2))
    print(
        "\n=== PAIRED ARM REPORT ===\n"
        + json.dumps(
            {k: v for k, v in report.items() if k not in ("accepted_steps", "refreshes")},
            indent=2,
        ),
        flush=True,
    )
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--h", type=float, default=0.01, help="FD step in scaled coords")
    ap.add_argument("--delta0", type=float, default=0.05)
    ap.add_argument("--delta-cap", type=float, default=0.1)
    ap.add_argument("--stability-coords", type=int, default=8)
    ap.add_argument("--stability-dirs", type=int, default=4)
    ap.add_argument("--max-trials-per-linearisation", type=int, default=6)
    ap.add_argument("--max-relinearisations", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--linearise",
        default="iota,log10_qi,mirror,flux,well",
        help="comma list of residuals to linearise in the LP; the others become "
        "acceptance-only checks (must not worsen by > tol). Amendment 2026-09-15: "
        "flux's canonical derivative is not smooth at h=0.01.",
    )
    ap.add_argument(
        "--reuse-jacobian",
        default=None,
        help="a prior run's evaluations.jsonl: reuse its linearisation-0 probes and "
        "start row instead of paying again; the stability check is taken as done "
        "in that run (pass its per-constraint result via --stability-json)",
    )
    ap.add_argument("--stability-json", default=None)
    # paired experiment (reviewer round 6)
    ap.add_argument(
        "--arm",
        choices=["full", "broyden"],
        default=None,
        help="run one arm of the paired experiment from --start-candidate",
    )
    ap.add_argument(
        "--start-candidate",
        default=None,
        help="saved candidate JSON with x/r/raw + jacobian pointer",
    )
    ap.add_argument(
        "--allowance",
        type=int,
        default=300,
        help="canonical evaluations for this arm beyond the shared Jacobian",
    )
    ap.add_argument("--n-coords", type=int, default=80)
    ap.add_argument(
        "--drift-every",
        type=int,
        default=3,
        help="broyden: directional drift check every N accepted steps",
    )
    ap.add_argument("--max-rejects-in-row", type=int, default=8)
    ap.add_argument(
        "--policy",
        choices=["anchored", "monotone_v"],
        default="anchored",
        help="anchored = cumulative iota/flux/well caps (paired experiment); "
        "monotone_v = V=max(all five) must fall by eps, flux inside V, "
        "iota/well protected (reviewer round 7)",
    )
    ap.add_argument(
        "--eps",
        type=float,
        default=1e-4,
        help="monotone_v: material improvement per step",
    )
    a = ap.parse_args()
    if a.arm:
        if not a.start_candidate:
            raise SystemExit("--arm needs --start-candidate")
        if a.delta0 == 0.05:  # the old default; the paired experiment uses the corrected policy
            a.delta0 = 0.00125
        if a.delta_cap == 0.1:
            a.delta_cap = 0.0025
        run_paired(a)
    else:
        run(a)


if __name__ == "__main__":
    main()
