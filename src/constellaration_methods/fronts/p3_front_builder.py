"""E16 -- bounded front-building from the nine verified anchors (vault
p3-reopen-three-experiments-2026-09-16, section "E16 -- bounded
front-building from the nine verified anchors"). Reuses the tested pure
parts of p3_feasibility_restoration (probes, central differences, Broyden,
ratio trust region, stall counter, canonical evaluator) and
p3_final_restoration (ceilings, fresh-process confirmation, the quadratic
line-model pattern, confirmations_complete) plus p3_front_from_archive's
nondominated/hypervolume primitives and p3_promote_candidate's boundary
export.

Method: at an anchor, probe every coordinate (h = 0.01, one probe pass, 2n
evaluations) to build a first-order + diagonal second-order model of SEVEN
rows -- the five canonical residuals plus the two objectives aspect_ratio
(minimise) and min_len (maximise) -- all from the same probes. Three
scalarisations per anchor (w = (1,0), (0,1), (1,1) on objectives normalised
by the anchor's own values) spread trial steps along the front. Each trial:
a quadratic line search on the scalarised objective (two line evaluations,
t on [0.1, 1.5]), then the step is accepted only if the canonical evaluation
is benchmark-feasible (max residual <= 0.01) AND it raises the running
front's hypervolume by >= 0.01; every accepted point is confirmed twice in
fresh processes (two complete vectors, worst-of-three <= 0.01 on every
residual) before it enters the front. Ratio trust policy on the objective
gain (0.0025 start / 0.01 cap / 1e-5 floor); Broyden updates on the
Jacobian rows after every evaluated point; stall = 6 consecutive rejections
at a scalarisation -> next scalarisation; all three stalled -> next anchor;
a full pass over the anchors gaining < 0.05 total hypervolume triggers one
rebuild-and-repeat pass from the current front, then stops.

Usage (inside the pinned image, network none, cores 2-13):
    podman run --rm --network none -v "$(pwd)":/workspace:Z -w /workspace \\
        -e OMP_NUM_THREADS=1 --pids-limit 8192 --ulimit nproc=16384:16384 \\
        --entrypoint taskset localhost/private-image:phase1 -c 2-13 \\
        python3 scripts/p3_front_builder.py \\
            --anchors results/p3-e15-front/REPORT.json \\
            --representation results/p3-e13-davidkh-7x7/start_serialised.json \\
            --out-dir results/p3-e16-front --workers 12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from constellaration_methods.fronts.p3_front_from_archive import (
    _safe_name,
    hypervolume,
    nondominated,
)
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
    probe_points,
)
from constellaration_methods.restoration.p3_final_restoration import (
    Ceilings,
    _fresh_evaluate,
    confirmations_complete,
    quadratic_line_model,
)


# --------------------------------------------------------------------------- #
# pure parts (tested without VMEC, except where noted)
# --------------------------------------------------------------------------- #
def normalise_objectives(aspect, min_len, aspect_anchor, min_len_anchor):
    """(-aspect/aspect_anchor, min_len/min_len_anchor): aspect_ratio is
    minimised so its normalised component grows as aspect falls; min_len is
    maximised so its normalised component grows as min_len rises. Both are
    scaled by the ANCHOR's own values (never the current incumbent's) so the
    three scalarisations of one anchor are directly comparable."""
    return -float(aspect) / float(aspect_anchor), float(min_len) / float(min_len_anchor)


def scalarise(aspect, min_len, aspect_anchor, min_len_anchor, weights) -> float:
    """s = w_a * (-aspect/aspect_anchor) + w_l * (min_len/min_len_anchor)."""
    w_a, w_l = weights
    obj_a, obj_l = normalise_objectives(aspect, min_len, aspect_anchor, min_len_anchor)
    return float(w_a) * obj_a + float(w_l) * obj_l


def is_benchmark_feasible(r, tol: float = TOL) -> bool:
    """max residual <= tol, all finite; None (non-convergent) is not
    feasible."""
    if r is None:
        return False
    r = np.asarray(r, float)
    if r.size == 0 or not np.all(np.isfinite(r)):
        return False
    return bool(r.max() <= tol)


def hv_gain_meets_threshold(hv_before, hv_after, threshold: float = 0.01) -> bool:
    return (float(hv_after) - float(hv_before)) >= threshold


def front_accept(r, hv_before, hv_after, tol: float = TOL, hv_gain_threshold: float = 0.01) -> bool:
    """The trial-level acceptance gate (before confirmation): benchmark
    feasibility AND a running-front hypervolume gain at or above the
    threshold."""
    return is_benchmark_feasible(r, tol) and hv_gain_meets_threshold(
        hv_before, hv_after, hv_gain_threshold
    )


def evaluate_row7(rs, raws):
    """[r0..r4, aspect_ratio, min_len] per evaluation (or None if the
    evaluation failed to converge or its raw metrics are incomplete): the
    seven-row model the front builder linearises, built from the identical
    evaluator outputs p3_feasibility_restoration already produces."""
    out = []
    for r_, raw_ in zip(rs, raws):
        raw = raw_.get("raw") if isinstance(raw_, dict) else None
        if r_ is None or raw is None or "aspect_ratio" not in raw or "min_len" not in raw:
            out.append(None)
            continue
        out.append(
            np.concatenate(
                [
                    np.asarray(r_, float),
                    [float(raw["aspect_ratio"]), float(raw["min_len"])],
                ]
            )
        )
    return out


def anchor_rows_from_report(report: dict) -> list:
    """Retained E15 front members as plain rows (tag, r, aspect, min_len,
    solo_hypervolume), in the report's own order. Resolving each tag's x
    needs the engine (load_anchor_x, below) so is kept separate and is not a
    pure function."""
    out = []
    for m in report.get("members", []):
        if not m.get("retained"):
            continue
        out.append(
            {
                "tag": m["tag"],
                "r": [float(v) for v in m["confirmed_r"]],
                "aspect": float(m["confirmed_aspect_ratio"]),
                "min_len": float(m["confirmed_min_len"]),
                "solo_hypervolume": float(m["solo_hypervolume"]),
            }
        )
    return out


def order_anchors(rows: list) -> list:
    """Anchors visited in order of solo hypervolume, lowest first: the
    weakest front members have the most room to improve, so the bounded
    search spends its first (and, under the model-build ceiling, possibly
    only) evaluations on them."""
    return sorted(rows, key=lambda r: r["solo_hypervolume"])


def worse_of_three(rs, raws):
    """The (r, raw) pair with the largest max residual among three
    evaluations of the same point (a trial's own measurement plus its two
    fresh confirmations): the conservative choice recorded for the front,
    mirroring this campaign's worst-of-N adjudication convention
    (p3_final_restoration.adjudicate_worst_v, p3_front_from_archive's
    _worse_of_two)."""
    best_i, best_v = None, None
    for i, r in enumerate(rs):
        if r is None:
            continue
        v = float(np.max(np.asarray(r, float)))
        if best_v is None or v > best_v:
            best_v, best_i = v, i
    if best_i is None:
        return None, None
    return rs[best_i], raws[best_i]


def best_t_for_front(A, B, C, t_grid, tol: float, r0=None):
    """t* on the grid maximising the predicted scalarised objective (the
    LAST row of A/B/C) subject to every residual row (all rows but the last)
    predicted <= tol; (None, None) if no grid point qualifies. Generalises
    p3_final_restoration.best_t_on_line from "minimise V" to "maximise the
    scalarised objective" -- front_step's own criterion -- via the same
    per-row quadratic line model (quadratic_line_model)."""
    A, B, C = (np.asarray(v, float) for v in (A, B, C))
    m = len(C) - 1
    best = (None, None)
    for t in t_grid:
        v = A * t * t + B * t + C
        if np.any(v[:m] > tol):
            continue
        s_pred = float(v[m])
        if best[0] is None or s_pred > best[1]:
            best = (float(t), s_pred)
    return best


@dataclass
class FrontModel:
    """The seven-row local model at an anchor/incumbent: r0 = [residuals(5),
    aspect_ratio, min_len]; J, C = (7, n) first- and (clipped, diagonal)
    second-order coefficients from one central-difference probe pass.
    aspect_anchor/min_len_anchor are the ANCHOR's own objective values (fixed
    for the anchor's whole life, never the incumbent's) and normalise every
    scalarisation. r0 is updated by the caller as the incumbent moves within
    a scalarisation; J is refined by Broyden updates; C is fixed until the
    next full model rebuild."""

    r0: np.ndarray
    J: np.ndarray
    C: np.ndarray
    aspect_anchor: float
    min_len_anchor: float


def build_model(evaluate_fn, x0, r0_row7, h: float, tag_prefix: str):
    """One probe pass (2n evaluations via evaluate_fn) -> a FrontModel
    centred at x0, plus the per-column ok flags for J and C."""
    n = len(x0)
    pts, meta = probe_points(x0, h)
    tags = [f"{tag_prefix}:{j}:{'+' if s > 0 else '-'}" for j, s in meta]
    rs, raws = evaluate_fn(pts, tags)
    rows7 = evaluate_row7(rs, raws)
    J, ok = central_difference_jacobian(rows7, meta, n=n, h=h)
    C, okc = central_difference_curvature(rows7, meta, n=n, h=h, r0=r0_row7)
    r0 = np.asarray(r0_row7, float)
    model = FrontModel(r0=r0, J=J, C=C, aspect_anchor=float(r0[5]), min_len_anchor=float(r0[6]))
    return model, ok, okc


def front_step(
    model: FrontModel,
    weights,
    delta: float,
    tol_model: float = 0.008,
    well_cap: bool = True,
):
    """SLSQP step maximising the scalarised objective model s(d) = w_a *
    (-aspect(d)/aspect_anchor) + w_l * (min_len(d)/min_len_anchor) subject to
    every residual model (rows 0-4) <= tol_model (the benchmark's 0.01 less a
    0.002 model margin), the raw-well cap (row 4 <= 0, once the model's own
    incumbent has reached it) and the trust box |d_j| <= delta. Mirrors
    p3_feasibility_restoration.restoration_step_quadratic's pattern (clip
    negative curvature to 0, SLSQP, check every constraint on the returned
    point). A failed solve -- infeasible or scipy reporting failure -- is a
    rejected trial: RestorationLPInfeasible, never a relaxation. Returns
    (d, predicted scalarised-objective gain over the incumbent)."""
    from scipy.optimize import minimize

    r0 = np.asarray(model.r0, float)
    J = np.asarray(model.J, float)
    C = None if model.C is None else np.asarray(model.C, float)
    m, n = J.shape
    w_a, w_l = weights
    bad = np.any(np.isnan(J), axis=0)
    if C is not None:
        bad = bad | np.any(np.isnan(C), axis=0)
    Jc = np.where(np.isnan(J), 0.0, J)
    Cc = None if C is None else np.maximum(np.where(np.isnan(C), 0.0, C), 0.0)

    def pred(d):
        out = r0 + Jc @ d
        if Cc is not None:
            out = out + 0.5 * Cc @ (d * d)
        return out

    bounds5 = np.full(5, float(tol_model))
    if well_cap and r0[4] <= 0.0:
        bounds5[4] = min(bounds5[4], 0.0)

    def neg_s(d):
        p = pred(d)
        return -(w_a * (-p[5] / model.aspect_anchor) + w_l * (p[6] / model.min_len_anchor))

    cons = [{"type": "ineq", "fun": (lambda d, i=i: bounds5[i] - pred(d)[i])} for i in range(5)]
    bnds = [(0.0, 0.0) if bad[j] else (-delta, delta) for j in range(n)]
    res = minimize(
        neg_s,
        np.zeros(n),
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"maxiter": 300, "ftol": 1e-12},
    )
    feasible_model = all(c["fun"](res.x) >= -1e-8 for c in cons)
    if not res.success or not feasible_model:
        raise RestorationLPInfeasible(f"front step failed: {res.message}")
    d = res.x
    gain = float(-neg_s(d) - (-neg_s(np.zeros(n))))
    return d, gain


# --------------------------------------------------------------------------- #
# anchor loading (needs the engine; not pure)
# --------------------------------------------------------------------------- #
def load_anchor_x(anchor_row: dict, boundary_dir: Path, representation: dict) -> np.ndarray:
    """The anchor's scaled coordinates at the run's representation, rebuilt
    from its exported boundary (E15's <tag>.boundary.json, sitting beside
    REPORT.json) via the engine's own masked_coefficients_and_scale, exactly
    as p3_promote_candidate.candidate_from_boundary_dict does for an external
    seed. Asserts the boundary's own mode budget and nfp match the run's
    representation (a mismatch would silently misassign coordinates)."""
    from constellaration_methods.restoration.p3_promote_candidate import (
        candidate_from_boundary_dict,
    )

    safe = _safe_name(anchor_row["tag"])
    bd_path = boundary_dir / f"{safe}.boundary.json"
    bd = json.loads(bd_path.read_text())
    cand = candidate_from_boundary_dict(bd, r=anchor_row["r"])
    if (
        cand["mpol"] != representation["mpol"]
        or cand["ntor"] != representation["ntor"]
        or cand["nfp"] != representation["nfp"]
    ):
        raise SystemExit(
            f"anchor {anchor_row['tag']}: boundary budget "
            f"{cand['mpol']}x{cand['ntor']} nfp{cand['nfp']} does not match "
            f"the representation {representation['mpol']}x{representation['ntor']} "
            f"nfp{representation['nfp']}"
        )
    return np.asarray(cand["x"], float)


def anchor_rows_from_e16_report(report: dict) -> list:
    """The E16 front's confirmed members as plain rows (tag, r, aspect,
    min_len), in the report's own order. E16's top-level shape is
    'confirmed_front' (keys tag/r/aspect_ratio/min_len), not E15's
    'members' (keys tag/retained/confirmed_r/confirmed_aspect_ratio/
    confirmed_min_len/solo_hypervolume) -- there is no per-anchor
    hypervolume to order by, so unlike order_anchors this keeps the
    report's own order rather than resorting."""
    out = []
    for m in report.get("confirmed_front", []):
        out.append(
            {
                "tag": m["tag"],
                "r": [float(v) for v in m["r"]],
                "aspect": float(m["aspect_ratio"]),
                "min_len": float(m["min_len"]),
            }
        )
    return out


def load_anchors(report_path, representation_path):
    """(anchors, representation): anchors = the front's members (E15's
    retained members ordered by solo hypervolume, or E16's confirmed_front
    in its own order), each with x resolved from its exported boundary;
    representation = the common (boundary_dict, mpol, ntor, nfp, scaling)
    every canonical evaluation payload uses. Refuses a report whose
    top-level shape is neither layout, naming the keys seen."""
    report = json.loads(Path(report_path).read_text())
    if "members" in report:
        rows = order_anchors(anchor_rows_from_report(report))
    elif "confirmed_front" in report:
        rows = anchor_rows_from_e16_report(report)
    else:
        raise ValueError(
            "load_anchors: unrecognised front report layout -- expected a "
            "top-level 'members' list (E15) or 'confirmed_front' list "
            f"(E16); got keys {sorted(report.keys())}"
        )
    rep_src = json.loads(Path(representation_path).read_text())
    representation = dict(
        boundary_dict=rep_src["boundary_dict"],
        mpol=int(rep_src["mpol"]),
        ntor=int(rep_src["ntor"]),
        nfp=int(rep_src["nfp"]),
        scaling=float(rep_src.get("scaling", 1.0)),
    )
    boundary_dir = Path(report_path).parent
    anchors = []
    for row in rows:
        x = load_anchor_x(row, boundary_dir, representation)
        anchors.append(dict(row, x=x.tolist()))
    return anchors, representation


def _is_fake(representation: dict) -> bool:
    return bool(representation.get("fake_quadratic") or representation.get("fake_front"))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _export_front(front_members: dict, representation: dict, out_dir: Path) -> dict:
    """Export every front member's boundary + toroidal reflection with
    sha256 into out_dir (MANIFEST.json). No-op for a fake/toy representation
    (its boundary_dict carries no real Fourier structure)."""
    if _is_fake(representation):
        return {}
    from constellaration_methods.restoration.p3_promote_candidate import (
        boundary_dict_of_candidate,
        reflect_toroidally,
    )

    manifest: dict = {"files": {}}
    for m in front_members.values():
        cand = dict(
            x=m["x"],
            boundary_dict=representation["boundary_dict"],
            mpol=representation["mpol"],
            ntor=representation["ntor"],
            nfp=representation["nfp"],
            scaling=representation.get("scaling", 1.0),
        )
        bd = boundary_dict_of_candidate(cand)
        bd_ref = reflect_toroidally(bd)
        safe = _safe_name(m["tag"])
        bd_path = out_dir / f"{safe}.boundary.json"
        bd_ref_path = out_dir / f"{safe}.phireflected.boundary.json"
        bd_path.write_text(json.dumps(bd, indent=2))
        bd_ref_path.write_text(json.dumps(bd_ref, indent=2))
        manifest["files"][bd_path.name] = _sha256_file(bd_path)
        manifest["files"][bd_ref_path.name] = _sha256_file(bd_ref_path)
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
SCALARISATIONS = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]


def _scalarisation_loop(
    anchor_tag,
    model: FrontModel,
    incumbent_x,
    incumbent_row7,
    weights,
    args,
    ceil: Ceilings,
    evaluate_fn,
    confirm_fn,
    front_members: dict,
    front_points_fn,
    current_hv: float,
    hv_trajectory: list,
    log,
    model_index: int,
):
    """Trials for one (anchor, scalarisation) pair, until 6 consecutive
    rejections (StallCounter), the trust-region floor, or a ceiling. Mutates
    model.J (Broyden) and front_members (on confirmed acceptance) in place.
    Returns (current_hv, n_accepted, n_trials, stop_reason_or_None)."""
    tr = RatioTrustRegion(delta=args.delta0, cap=args.delta_cap, floor=args.delta_floor)
    stall = StallCounter(max_in_row=6, min_gain=-1.0)
    n_accepted = 0
    trial = 0
    stop_reason = None
    wtag = f"w{int(weights[0])}{int(weights[1])}"

    def _reject_and_maybe_stop(reason: str, **extra):
        nonlocal trial
        log(event="trial", tag=f"{tag_prefix}:{trial}", rejected_reason=reason, **extra)
        tr.reject()
        stall.record(False, None)
        trial += 1
        if stall.stalled or tr.delta <= tr.floor:
            log(event="stall", anchor=anchor_tag, weights=wtag, trials=trial)
            return True
        return False

    while True:
        if not ceil.can_optimise(time.time(), 3):
            stop_reason = "budget_or_deadline"
            break
        tag_prefix = f"M{model_index}:{_safe_name(anchor_tag)}:{wtag}:T{trial}"
        model.r0 = np.asarray(incumbent_row7, float)
        try:
            d, pred_gain = front_step(model, weights, tr.delta, args.tol_model, well_cap=True)
        except RestorationLPInfeasible as e:
            if _reject_and_maybe_stop("lp_infeasible", msg=str(e)[:150]):
                break
            continue

        (r_half, r_full), (raw_half, raw_full) = evaluate_fn(
            [incumbent_x + 0.5 * d, incumbent_x + d],
            [f"{tag_prefix}:L0.5", f"{tag_prefix}:L1.0"],
        )
        s0 = scalarise(
            incumbent_row7[5],
            incumbent_row7[6],
            model.aspect_anchor,
            model.min_len_anchor,
            weights,
        )
        pts_line = []
        for tt, rr, raww in ((0.5, r_half, raw_half), (1.0, r_full, raw_full)):
            raw = raww.get("raw") if isinstance(raww, dict) else None
            if rr is None or raw is None:
                continue
            row7 = np.concatenate(
                [
                    np.asarray(rr, float),
                    [float(raw["aspect_ratio"]), float(raw["min_len"])],
                ]
            )
            model.J, _ = broyden_update(model.J, tt * d, row7 - incumbent_row7)
            s_ = scalarise(
                raw["aspect_ratio"],
                raw["min_len"],
                model.aspect_anchor,
                model.min_len_anchor,
                weights,
            )
            pts_line.append((tt, row7, s_))
        if len(pts_line) < 2:
            if _reject_and_maybe_stop("line_point_nonconvergent"):
                break
            continue

        (_, row_half, s_half), (_, row_full, s_full) = pts_line
        combo0 = np.concatenate([np.asarray(incumbent_row7, float)[:5], [s0]])
        comboh = np.concatenate([row_half[:5], [s_half]])
        combo1 = np.concatenate([row_full[:5], [s_full]])
        A_, B_, C_ = quadratic_line_model(combo0, comboh, combo1)
        t_star, _ = best_t_for_front(
            A_,
            B_,
            C_,
            np.linspace(0.1, 1.5, 29),
            tol=args.tol,
            r0=np.asarray(incumbent_row7)[:5],
        )
        if t_star is None:
            if _reject_and_maybe_stop("no_admissible_line_point"):
                break
            continue

        d_final = t_star * d
        (r_new,), (raw_new,) = evaluate_fn(
            [incumbent_x + d_final], [f"{tag_prefix}:F:t{t_star:.3f}"]
        )
        raw_n = raw_new.get("raw") if isinstance(raw_new, dict) else None
        if r_new is not None and raw_n is not None:
            row_new = np.concatenate(
                [
                    np.asarray(r_new, float),
                    [float(raw_n["aspect_ratio"]), float(raw_n["min_len"])],
                ]
            )
            model.J, _ = broyden_update(model.J, d_final, row_new - incumbent_row7)

        accepted = False
        if is_benchmark_feasible(r_new, args.tol):
            candidate_point = {
                "aspect_ratio": raw_n["aspect_ratio"],
                "min_len": raw_n["min_len"],
                "tag": tag_prefix,
            }
            provisional_hv = hypervolume(nondominated(front_points_fn() + [candidate_point]))
            gain = provisional_hv - current_hv
            log(
                event="trial",
                tag=tag_prefix,
                predicted_gain=pred_gain,
                measured_r=[float(v) for v in r_new],
                measured_aspect=raw_n["aspect_ratio"],
                measured_min_len=raw_n["min_len"],
                feasible=True,
                provisional_hv=provisional_hv,
                hv_gain=gain,
                t_star=t_star,
            )
            if gain >= args.hv_gain_threshold:
                (c1, c2), (rc1, rc2) = confirm_fn(incumbent_x + d_final, tag_prefix)
                confs = [c for c in (c1, c2) if c is not None]
                complete = confirmations_complete(confs)
                all_ok = complete and all(is_benchmark_feasible(c, args.tol) for c in confs)
                log(
                    event="confirmed",
                    tag=tag_prefix,
                    complete=complete,
                    ok=bool(all_ok),
                )
                if all_ok:
                    r_worst, raw_worst = worse_of_three([r_new, c1, c2], [raw_new, rc1, rc2])
                    raww = raw_worst["raw"]
                    front_members[tag_prefix] = {
                        "tag": tag_prefix,
                        "r": [float(v) for v in r_worst],
                        "aspect": float(raww["aspect_ratio"]),
                        "min_len": float(raww["min_len"]),
                        "x": (incumbent_x + d_final).tolist(),
                    }
                    new_hv = hypervolume(nondominated(front_points_fn()))
                    log(
                        event="front_hv",
                        hv=new_hv,
                        added=tag_prefix,
                        n_members=len(front_members),
                    )
                    hv_trajectory.append({"tag": tag_prefix, "hv": new_hv})
                    current_hv = new_hv
                    accepted = True
                    n_accepted += 1
        else:
            log(
                event="trial",
                tag=tag_prefix,
                feasible=False,
                measured_r=None if r_new is None else [float(v) for v in r_new],
            )

        if accepted:
            incumbent_x = incumbent_x + d_final
            member = front_members[tag_prefix]
            incumbent_row7 = np.concatenate(
                [np.asarray(member["r"], float), [member["aspect"], member["min_len"]]]
            )
            actual_gain = (
                scalarise(
                    member["aspect"],
                    member["min_len"],
                    model.aspect_anchor,
                    model.min_len_anchor,
                    weights,
                )
                - s0
            )
            rho = None if abs(pred_gain) < 1e-12 else actual_gain / pred_gain
            tr.update(rho if rho is not None else 0.0)
            stall.record(True, 1.0)
        else:
            tr.reject()
            stall.record(False, None)
        trial += 1
        if stall.stalled or tr.delta <= tr.floor:
            log(event="stall", anchor=anchor_tag, weights=wtag, trials=trial)
            break

    return current_hv, n_accepted, trial, stop_reason


def run_from_anchors(anchors: list, representation: dict, out_dir, args) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    events_path = out_dir / "events.jsonl"
    archive_path = out_dir / "archive.jsonl"
    verif_path = out_dir / "verification.jsonl"

    def log(**kw):
        kw["t_h"] = round((time.time() - t0) / 3600, 4)
        events_path.open("a").write(json.dumps(kw) + "\n")
        print(json.dumps(kw), flush=True)

    ceil = Ceilings(
        t_start=t0,
        total_hours=args.total_hours,
        opt_cutoff_hours=args.total_hours,
        max_opt_evals=args.max_opt_evals,
        max_verif_evals=args.max_verif_evals,
        max_full_jacobians=args.max_models,
    )
    obudget = Budget(cap=args.max_opt_evals)
    vbudget = Budget(cap=args.max_verif_evals)

    front_members = {
        a["tag"]: {
            "tag": a["tag"],
            "r": [float(v) for v in a["r"]],
            "aspect": float(a["aspect"]),
            "min_len": float(a["min_len"]),
            "x": list(a["x"]),
        }
        for a in anchors
    }

    def front_points():
        return [
            {"aspect_ratio": m["aspect"], "min_len": m["min_len"], "tag": m["tag"]}
            for m in front_members.values()
        ]

    initial_hv = hypervolume(nondominated(front_points()))
    current_hv = initial_hv
    hv_trajectory = [{"tag": "initial", "hv": initial_hv}]
    log(event="front_hv", hv=initial_hv, n_members=len(front_members), note="initial")

    stop_reason = None
    counts: dict = {}
    model_index = 0
    pass_num = 0
    anchors_this_pass = anchors[:1] if args.smoke else list(anchors)

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        ev = CanonicalEvaluator(representation, pool, archive_path, obudget)

        def evaluate(xs, tags):
            ceil.charge_opt(len(xs))
            return ev.evaluate(xs, tags)

        def confirm(x, tag_prefix):
            if vbudget.remaining < 2:
                return (None, None), (None, None)
            rs2, raws2 = _fresh_evaluate(
                representation,
                [x, x],
                [f"CONF:{tag_prefix}:1", f"CONF:{tag_prefix}:2"],
                args.workers,
                verif_path,
                vbudget,
            )
            ceil.charge_verif(2)
            return (rs2[0], rs2[1]), (raws2[0], raws2[1])

        while stop_reason is None:
            pass_num += 1
            log(event="pass", pass_num=pass_num, n_anchors=len(anchors_this_pass))
            hv_before_pass = current_hv
            for anchor in anchors_this_pass:
                if stop_reason:
                    break
                if not ceil.can_full_jacobian():
                    stop_reason = "model_ceiling"
                    break
                anchor_x = np.asarray(anchor["x"], float)
                n = len(anchor_x)
                if not ceil.can_optimise(time.time(), 2 * n):
                    stop_reason = "budget_or_deadline"
                    break
                anchor_row7 = np.concatenate(
                    [
                        np.asarray(anchor["r"], float),
                        [anchor["aspect"], anchor["min_len"]],
                    ]
                )
                model_index += 1
                ceil.charge_full_jacobian()
                model, ok, okc = build_model(
                    evaluate,
                    anchor_x,
                    anchor_row7,
                    args.h,
                    f"M{model_index}:{_safe_name(anchor['tag'])}",
                )
                log(
                    event="model_built",
                    anchor=anchor["tag"],
                    model_index=model_index,
                    ok_cols=int(sum(ok)),
                    ok_curvature_cols=int(sum(okc)),
                    opt_used=ceil.opt_used,
                )
                weights_list = SCALARISATIONS[:1] if args.smoke else SCALARISATIONS
                for weights in weights_list:
                    if stop_reason:
                        break
                    key = f"{anchor['tag']}|w{int(weights[0])}{int(weights[1])}"
                    current_hv, n_acc, n_trials, sr = _scalarisation_loop(
                        anchor["tag"],
                        model,
                        anchor_x.copy(),
                        anchor_row7.copy(),
                        weights,
                        args,
                        ceil,
                        evaluate,
                        confirm,
                        front_members,
                        front_points,
                        current_hv,
                        hv_trajectory,
                        log,
                        model_index,
                    )
                    counts[key] = {"trials": n_trials, "accepted": n_acc}
                    if sr:
                        stop_reason = sr
                log(event="anchor_switch", anchor=anchor["tag"])
            hv_gain_this_pass = current_hv - hv_before_pass
            log(
                event="pass_complete",
                pass_num=pass_num,
                hv_before=hv_before_pass,
                hv_after=current_hv,
                gain=hv_gain_this_pass,
            )
            if stop_reason:
                break
            if args.smoke:
                stop_reason = "smoke_complete"
                break
            if hv_gain_this_pass < args.pass_gain_threshold:
                stop_reason = "front_is_local_optimal_under_model"
                break
            anchors_this_pass = order_anchors(
                [
                    {
                        "tag": m["tag"],
                        "r": m["r"],
                        "aspect": m["aspect"],
                        "min_len": m["min_len"],
                        "x": m["x"],
                        "solo_hypervolume": hypervolume(
                            [{"aspect_ratio": m["aspect"], "min_len": m["min_len"]}]
                        ),
                    }
                    for m in front_members.values()
                ]
            )

    log(event="stopped", reason=stop_reason)
    manifest = _export_front(front_members, representation, out_dir)
    report = {
        "smoke": bool(args.smoke),
        "stop_reason": stop_reason,
        "initial_hypervolume": initial_hv,
        "final_hypervolume": current_hv,
        "hypervolume_gain": current_hv - initial_hv,
        "hypervolume_trajectory": hv_trajectory,
        "n_front_members": len(front_members),
        "confirmed_front": [
            {
                "tag": m["tag"],
                "r": m["r"],
                "aspect_ratio": m["aspect"],
                "min_len": m["min_len"],
            }
            for m in front_members.values()
        ],
        "counts": counts,
        "opt_evals": ceil.opt_used,
        "verif_evals": ceil.verif_used,
        "model_builds": ceil.full_jacobians,
        "passes": pass_num,
        "n_exported": len(manifest.get("files", {})) // 2,
        "wall_h": round((time.time() - t0) / 3600, 4),
    }
    (out_dir / "REPORT.json").write_text(json.dumps(report, indent=2))
    print(
        "\n=== E16 FRONT-BUILDER REPORT ===\n"
        + json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("hypervolume_trajectory", "confirmed_front")
            },
            indent=2,
        ),
        flush=True,
    )
    return report


def run(args) -> dict:
    anchors, representation = load_anchors(args.anchors, args.representation)
    return run_from_anchors(anchors, representation, Path(args.out_dir), args)


def apply_smoke_ceilings(args) -> None:
    """--smoke forces the pre-registered smoke ceilings (600 optimisation
    evaluations, 2 h) regardless of any --total-hours/--max-opt-evals also
    given, per the E16 pre-registration's "Smoke before the lock" clause."""
    if getattr(args, "smoke", False):
        args.total_hours = 2.0
        args.max_opt_evals = 600


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchors", required=True, help="E15's REPORT.json")
    ap.add_argument("--representation", required=True, help="E13's start_serialised.json")
    ap.add_argument("--out-dir", default=None, help="default results/p3-e16-smoke with --smoke")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--h", type=float, default=0.01)
    ap.add_argument("--delta0", type=float, default=0.0025)
    ap.add_argument("--delta-cap", type=float, default=0.01)
    ap.add_argument("--delta-floor", type=float, default=1e-5)
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--tol-model", type=float, default=0.008)
    ap.add_argument("--hv-gain-threshold", type=float, default=0.01)
    ap.add_argument("--pass-gain-threshold", type=float, default=0.05)
    ap.add_argument("--total-hours", type=float, default=60.0)
    ap.add_argument("--max-opt-evals", type=int, default=3000)
    ap.add_argument("--max-verif-evals", type=int, default=60)
    ap.add_argument("--max-models", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="one anchor, one scalarisation, ceilings 600 evaluations / 2 h; "
        "its evaluations do not count toward E16 -- use a separate --out-dir",
    )
    return ap


def resolve_out_dir(args) -> str:
    """args.out_dir if given; else the pre-registered smoke default when
    --smoke is set; else a hard error (--out-dir is required for a real
    run)."""
    if args.out_dir is not None:
        return args.out_dir
    if args.smoke:
        return "results/p3-e16-smoke"
    raise SystemExit("--out-dir is required unless --smoke")


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    apply_smoke_ceilings(args)
    args.out_dir = resolve_out_dir(args)
    return run(args)


if __name__ == "__main__":
    main()
