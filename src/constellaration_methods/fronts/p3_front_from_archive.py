"""E15 -- confirm and export the front already in E13's archive (measurement,
pre-registered and LOCKED 2026-09-18, vault
p3-reopen-three-experiments-2026-09-16). No new optimisation.

Review round 9 (item 7a): E13's archive holds ~405 distinct tolerance-feasible
points and a provisional 9-point nondominated set. This confirms and exports
the retained set before any further search is funded.

Procedure:
  1. from results/p3-e13-davidkh-7x7/archive.jsonl, take every row with
     finite residuals and max residual <= 0.01 (`tolerance_feasible_rows`).
  2. compute the nondominated set on the exact objectives (aspect_ratio
     minimise, min_len maximise) from the archived raw metrics
     (`nondominated`).
  3. evaluate each nondominated point twice in fresh canonical processes
     (the existing evaluator, `_fresh_evaluate` / `Budget`); a point is
     RETAINED only if both confirmations are complete five-component
     vectors with max residual <= 0.01 and match the archived residuals to
     1e-4 (`retained`).
  4. recompute the front and its hypervolume (reference (1.0, 20.0), the
     library's own scorer -- `hypervolume`, which calls
     `constellaration.problems._hypervolume` directly) from the CONFIRMED
     metrics of the retained members only.

     Confirmed-field convention (worst-case, `worse_confirmation`): every
     confirmed_* field of a retained member -- confirmed_r, confirmed_V,
     confirmed_aspect_ratio, confirmed_min_len, strict_nominal and
     solo_hypervolume -- is read from the ONE confirmation whose max
     residual is worse (ties keep the first), so the hypervolume is
     computed from the same evaluation as the residual reported beside it.
     Nothing is averaged across the two confirmations; this mirrors the
     campaign's worst-of-N adjudication rule
     (`p3_final_restoration.adjudicate_worst_v`).
  5. export every retained boundary and its toroidal reflection with sha256
     (`p3_promote_candidate.boundary_dict_of_candidate` /
     `reflect_toroidally`).
  6. write REPORT.json / RESULT.md: n candidates, n nondominated, n
     retained, per-member table, confirmed hypervolume, strict-nominal flag,
     canonical evaluations used (<= 30).

Usage (inside the pinned image, network none, cores 2-13):
    podman run --rm --network none -v "$(pwd)":/workspace:Z -w /workspace \\
        -e OMP_NUM_THREADS=1 --pids-limit 8192 --ulimit nproc=16384:16384 \\
        --entrypoint taskset localhost/constellaration-methods:test -c 2-13 \\
        python3 scripts/p3_front_from_archive.py \\
            results/p3-e13-davidkh-7x7 --out-dir results/p3-e15-front --confirm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

TOL = 0.01
MAX_CANONICAL_EVALS = 30  # hard cap, locked in the E15 pre-registration


# --------------------------------------------------------------------------- #
# pure rules (no VMEC; the only exception is hypervolume(), which calls the
# library's own primitive and so needs the pinned image)
# --------------------------------------------------------------------------- #
def _is_finite_vec(v) -> bool:
    try:
        return all(math.isfinite(float(x)) for x in v)
    except (TypeError, ValueError):
        return False


def tolerance_feasible_rows(rows, tol: float = TOL):
    """Rows with a finite residual vector and max residual <= tol, in
    encounter order, de-duplicated on `x` rounded to 6 decimal places (E13's
    archive has none such, but a re-visited point during the run would
    otherwise be double-counted as two "candidates")."""
    seen = set()
    out = []
    for row in rows:
        r = row.get("r")
        if r is None or not _is_finite_vec(r):
            continue
        if max(float(v) for v in r) > tol:
            continue
        x = row.get("x")
        if x is not None:
            key = tuple(round(float(v), 6) for v in x)
            if key in seen:
                continue
            seen.add(key)
        out.append(row)
    return out


def best_v_of_rows(rows) -> float | None:
    """min(max(r)) -- the campaign's own worst-residual scalar V -- over rows
    with a complete, finite residual vector; None if none qualify (no
    convergent evaluation yet). The single place a running 'best V so far'
    is computed, reused by both the seeded runner's per-generation event
    (E18 item 3) and `p3_curve_from_archive.py`'s offline recomputation."""
    vs = [
        max(float(v) for v in row["r"])
        for row in rows
        if row.get("r") is not None and _is_finite_vec(row["r"])
    ]
    return min(vs) if vs else None


def v_curve_from_rows(rows, every: int = 100) -> list:
    """[[evaluations, best_v_so_far], ...] sampled at evaluation counts
    `every`, 2*`every`, ... (one archive row = one evaluation, in archive
    order); `best_v_so_far` is `best_v_of_rows` over every row up to that
    point, or None if nothing convergent has landed yet. The trailing
    partial block (fewer than `every` rows) is not included -- a run's
    current running total is read from its last logged 'generation' event
    instead, not from this offline recomputation."""
    return [[i, best_v_of_rows(rows[:i])] for i in range(every, len(rows) + 1, every)]


def _dominates(a: dict, b: dict) -> bool:
    """a dominates b: no worse in both objectives, strictly better in at
    least one (aspect_ratio minimise, min_len maximise)."""
    le = a["aspect_ratio"] <= b["aspect_ratio"] and a["min_len"] >= b["min_len"]
    strict = a["aspect_ratio"] < b["aspect_ratio"] or a["min_len"] > b["min_len"]
    return le and strict


def nondominated(points):
    """The subset of `points` (each a dict with at least 'aspect_ratio' and
    'min_len'; extra keys are preserved and ignored) not dominated by any
    other point in the list. Ties (identical objectives) all survive:
    neither dominates the other under the strict clause."""
    out = []
    for i, p in enumerate(points):
        if not any(_dominates(q, p) for j, q in enumerate(points) if j != i):
            out.append(p)
    return out


def hypervolume(points, ref=(1.0, 20.0)) -> float:
    """Hypervolume of `points` (each with 'aspect_ratio', 'min_len') against
    `ref`, computed with the library's own primitive -- the identical
    computation `MHDStableQIStellarator._score` performs, so this number is
    the campaign's own P3 metric, not a reimplementation of it."""
    import numpy as np
    from constellaration import problems

    X = np.array([[-float(p["min_len"]), float(p["aspect_ratio"])] for p in points])
    return float(problems._hypervolume(X=X, reference_point=np.array(ref, dtype=float)))


def retained(archived_r, confirmations, tol: float = TOL, atol: float = 1e-4) -> bool:
    """True only if `confirmations` has exactly two entries, each a complete
    (5-component), finite residual vector with max <= tol, and each within
    atol of `archived_r` component-wise. Anything incomplete, non-finite,
    over tolerance, missing (None), or not exactly two confirmations fails
    closed (review round 9, item 5: `priority_confirm`'s return value being
    ignored, and an incomplete confirmation set being accepted, were both
    live bugs -- this helper is the fix, expressed as a pure predicate)."""
    if len(confirmations) != 2:
        return False
    if not _is_finite_vec(archived_r) or len(list(archived_r)) != 5:
        return False
    arch = [float(v) for v in archived_r]
    for c in confirmations:
        if c is None:
            return False
        c = list(c)
        if len(c) != 5 or not _is_finite_vec(c):
            return False
        c = [float(v) for v in c]
        if max(c) > tol:
            return False
        if max(abs(a - b) for a, b in zip(arch, c)) > atol:
            return False
    return True


def strict_nominal(r) -> bool:
    """All five residuals strictly < 0 (every nominal P3 bound satisfied
    with margin, not merely within the benchmark's 0.01 tolerance). Fails
    closed on non-finite input."""
    if not _is_finite_vec(r):
        return False
    return all(float(v) < 0.0 for v in r)


# --------------------------------------------------------------------------- #
# driver (needs the pinned image: VMEC via _fresh_evaluate, plus
# constellaration.problems for hypervolume())
# --------------------------------------------------------------------------- #
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _safe_name(tag: str) -> str:
    return tag.replace(":", "_").replace("/", "_")


def worse_confirmation(confirmations, raws):
    """(residuals, raw metrics) of the confirmation with the larger max
    residual; ties keep the first. One evaluation supplies EVERY confirmed_*
    field of a retained member (module docstring, step 4), mirroring the
    campaign's worst-of-N adjudication rule
    (`p3_final_restoration.adjudicate_worst_v`). Callers pass only complete
    confirmations (`retained` has already rejected None/incomplete ones)."""
    worst_i = 0
    worst_v = max(float(v) for v in confirmations[0])
    for i, c in enumerate(confirmations[1:], start=1):
        v = max(float(x) for x in c)
        if v > worst_v:
            worst_i, worst_v = i, v
    return confirmations[worst_i], raws[worst_i]


def run(args) -> dict:
    from constellaration_methods.problems.overrides import parse_constraint_overrides
    from constellaration_methods.restoration.p3_feasibility_restoration import Budget
    from constellaration_methods.restoration.p3_final_restoration import _fresh_evaluate
    from constellaration_methods.restoration.p3_promote_candidate import (
        boundary_dict_of_candidate,
        reflect_toroidally,
    )

    constraint_overrides = parse_constraint_overrides(getattr(args, "constraint_override", None))
    archive_dir = Path(args.archive_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = [
        json.loads(line)
        for line in (archive_dir / "archive.jsonl").read_text().splitlines()
        if line.strip()
    ]
    start = json.loads((archive_dir / "start_serialised.json").read_text())

    feasible_rows = tolerance_feasible_rows(rows, tol=args.tol)
    points = [dict(row["raw"], tag=row["tag"], x=row["x"], r=row["r"]) for row in feasible_rows]
    front = sorted(nondominated(points), key=lambda p: p["aspect_ratio"])

    report: dict = {
        "n_candidates": len(feasible_rows),
        "n_nondominated": len(front),
        "tol": args.tol,
        "constraint_overrides": constraint_overrides,
        "members": [
            {
                "tag": p["tag"],
                "archived_aspect_ratio": p["aspect_ratio"],
                "archived_min_len": p["min_len"],
                "archived_r": p["r"],
                "archived_V": max(float(v) for v in p["r"]),
            }
            for p in front
        ],
    }

    if not args.confirm:
        (out / "REPORT.json").write_text(json.dumps(report, indent=2))
        return report

    cap = int(getattr(args, "max_confirmations", MAX_CANONICAL_EVALS))
    check_confirmation_cap(n_members=len(front), cap=cap)
    report["confirmation_cap"] = cap

    rep = dict(
        boundary_dict=start["boundary_dict"],
        mpol=int(start["mpol"]),
        ntor=int(start["ntor"]),
        nfp=int(start["nfp"]),
        scaling=float(start.get("scaling", 1.0)),
        constraint_overrides=constraint_overrides,
    )
    verif_path = out / "verification.jsonl"
    budget = Budget(cap=cap)

    xs, tags = [], []
    for p in front:
        safe = _safe_name(p["tag"])
        xs += [p["x"], p["x"]]
        tags += [f"CONF:E15:{safe}:1", f"CONF:E15:{safe}:2"]

    rs, raws = _fresh_evaluate(rep, xs, tags, args.workers, verif_path, budget)

    members = []
    manifest = {"tolerance": args.tol, "atol": 1e-4, "files": {}}
    retained_points = []
    for i, p in enumerate(front):
        c1, c2 = rs[2 * i], rs[2 * i + 1]
        raw1 = raws[2 * i].get("raw")
        raw2 = raws[2 * i + 1].get("raw")
        ok = retained(p["r"], [c1, c2], tol=args.tol, atol=1e-4)
        member = {
            "tag": p["tag"],
            "archived_aspect_ratio": p["aspect_ratio"],
            "archived_min_len": p["min_len"],
            "archived_r": [float(v) for v in p["r"]],
            "archived_V": max(float(v) for v in p["r"]),
            "confirmation_1": None if c1 is None else [float(v) for v in c1],
            "confirmation_2": None if c2 is None else [float(v) for v in c2],
            "retained": ok,
        }
        if ok:
            worse_r, worse_raw = worse_confirmation([c1, c2], [raw1, raw2])
            confirmed_r = [float(v) for v in worse_r]
            confirmed_aspect = float(worse_raw["aspect_ratio"])
            confirmed_min_len = float(worse_raw["min_len"])
            member.update(
                confirmed_r=confirmed_r,
                confirmed_V=max(confirmed_r),
                confirmed_aspect_ratio=confirmed_aspect,
                confirmed_min_len=confirmed_min_len,
                strict_nominal=strict_nominal(confirmed_r),
                solo_hypervolume=hypervolume(
                    [{"aspect_ratio": confirmed_aspect, "min_len": confirmed_min_len}],
                    ref=(1.0, 20.0),
                ),
            )
            retained_points.append(
                {
                    "tag": p["tag"],
                    "aspect_ratio": confirmed_aspect,
                    "min_len": confirmed_min_len,
                }
            )

            cand = {
                "x": p["x"],
                "boundary_dict": start["boundary_dict"],
                "mpol": start["mpol"],
                "ntor": start["ntor"],
                "nfp": start["nfp"],
                "scaling": start.get("scaling", 1.0),
            }
            bd = boundary_dict_of_candidate(cand)
            bd_ref = reflect_toroidally(bd)
            safe = _safe_name(p["tag"])
            bd_path = out / f"{safe}.boundary.json"
            bd_ref_path = out / f"{safe}.phireflected.boundary.json"
            bd_path.write_text(json.dumps(bd, indent=2))
            bd_ref_path.write_text(json.dumps(bd_ref, indent=2))
            manifest["files"][bd_path.name] = _sha256_file(bd_path)
            manifest["files"][bd_ref_path.name] = _sha256_file(bd_ref_path)
        members.append(member)

    confirmed_front = nondominated(retained_points)
    confirmed_hv = hypervolume(confirmed_front, ref=(1.0, 20.0)) if confirmed_front else 0.0

    report.update(
        n_retained=sum(1 for m in members if m["retained"]),
        canonical_evaluations_used=len(xs),
        members=members,
        confirmed_front_tags=[p["tag"] for p in confirmed_front],
        confirmed_hypervolume=confirmed_hv,
        n_strict_nominal=sum(1 for m in members if m.get("strict_nominal") is True),
    )
    (out / "REPORT.json").write_text(json.dumps(report, indent=2))
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    (out / "RESULT.md").write_text(_render_result_md(report))
    return report


def _render_result_md(report: dict) -> str:
    lines = [
        "# E15 -- confirm and export the front already in E13's archive",
        "",
        "Facts only; no interpretation.",
        "",
        f"- n candidates (tolerance-feasible, max residual <= {report['tol']}): "
        f"{report['n_candidates']}",
        f"- n nondominated (exact objectives, aspect_ratio min / min_len max): "
        f"{report['n_nondominated']}",
        f"- n retained (two fresh confirmations, complete, <= tol, within 1e-4 "
        f"of archived): {report.get('n_retained', 'n/a')}",
        f"- canonical evaluations used: {report.get('canonical_evaluations_used', 0)} "
        f"(cap {report.get('confirmation_cap', MAX_CANONICAL_EVALS)})",
        f"- n retained members satisfying every nominal bound strictly: "
        f"{report.get('n_strict_nominal', 'n/a')}",
        f"- confirmed front hypervolume (ref (1.0, 20.0), retained members only): "
        f"{report.get('confirmed_hypervolume', 'n/a')}",
        "",
        "## Per-member table",
        "",
        "| tag | archived V | confirmed residuals | confirmed aspect | "
        "confirmed min_len | solo HV | strict nominal | retained |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in report.get("members", []):
        lines.append(
            "| {tag} | {v} | {r} | {a} | {l} | {hv} | {sn} | {ret} |".format(
                tag=m["tag"],
                v=round(m["archived_V"], 6),
                r=m.get("confirmed_r", "n/a"),
                a=round(m["confirmed_aspect_ratio"], 6) if "confirmed_aspect_ratio" in m else "n/a",
                l=round(m["confirmed_min_len"], 6) if "confirmed_min_len" in m else "n/a",
                hv=round(m["solo_hypervolume"], 6) if "solo_hypervolume" in m else "n/a",
                sn=m.get("strict_nominal", "n/a"),
                ret=m["retained"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def check_confirmation_cap(n_members: int, cap: int) -> None:
    """Refuse to start confirmations that would exceed the evaluation cap.

    Two complete confirmations per member are planned; the cap defaults to
    the E15 constant and is raised only by an explicit, registered flag.
    """
    n_planned = 2 * n_members
    if n_planned > cap:
        raise RuntimeError(
            f"confirmation cap: {n_planned} confirmations for {n_members} members "
            f"exceeds the {cap}-evaluation cap"
        )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive_dir", help="e.g. results/p3-e13-davidkh-7x7")
    ap.add_argument(
        "--max-confirmations",
        type=int,
        default=MAX_CANONICAL_EVALS,
        help="cap on fresh canonical confirmation evaluations (default: the "
        "E15-locked 30; E17 registers two per member, uncapped)",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--confirm",
        action="store_true",
        help="run fresh canonical confirmations (needs the pinned image)",
    )
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument(
        "--constraint-override",
        action="append",
        default=None,
        metavar="NAME=VALUE",
        help="E18: override a MHDStableQIStellarator constraint threshold "
        "(repeatable); applies to both the feasibility filter (the archive's "
        "own residuals, computed under whatever overrides built it) and "
        "fresh --confirm evaluations, which reuse the SAME overrides so "
        "confirmations are measured under the definition the front itself "
        "was built under. Valid names: "
        "constellaration_methods.problems.overrides.CONSTRAINT_OVERRIDE_NAMES",
    )
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = run(args)
    print(json.dumps({k: v for k, v in report.items() if k != "members"}, indent=2))
    return report


if __name__ == "__main__":
    main()
