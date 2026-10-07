#!/usr/bin/env python3
"""E20 (p3-construction-search-plan-2026-09-25.md, Mechanism step 2): canonical
(VMEC/forward_model) evaluation of a construction-search bank's best
candidates -- once each for the top `--top`, then worst-of-three (two more
evaluations in fresh processes) for the best `--confirm-top` by V.

Reuses (extend-before-create): `evaluation.score_mhd_stable`'s
`report_boundary_files` -- the exact five-residual/V/raw/feasible canonical
evaluator the leaderboard-fidelity checks already use (`--boundary FILE...
--out`) -- rather than a second forward_model call site. This script writes
each bank_top.json boundary dict to its own file (same field set the family
files use: r_cos/r_sin/z_cos/z_sin/n_field_periods/is_stellarator_symmetric,
see `p3_construction_search._export_4x4_boundary`) and calls that function
directly. Also reuses `analysis.p3_design_diversity_probe`'s `_git_commit`
(reads `.git` directly, no `git` binary needed) for REPORT.json provenance.

Usage (inside the pinned image, e.g. cores 2-13):

    podman run --rm -v "$(pwd)":/workspace:Z -w /workspace --network none \\
        -v "$(pwd)/.git":/workspace/.git:ro,Z \\
        --pids-limit 8192 --ulimit nproc=16384:16384 --entrypoint taskset \\
        localhost/constellaration-methods:dev -c 2-13 \\
        cm-p3-construction-bank-eval \\
        results/p3-e20-construction-search \\
        --top 48 --confirm-top 12 --workers 12 --out results/p3-e20-stage2

(the `.git` mount is read-only and optional -- see
`p3_construction_search.py`'s docstring for why)

`--confirm-top 0` skips the worst-of-three confirmation step entirely (used
by the smoke run) -- REPORT.json then reports the once-only canonical V for
the top `--top` rows and no stage-2 label (labelling needs a confirmed,
worst-of-three V; a single canonical eval is not that).

E20b (p3-construction-search-plan-2026-09-25.md, "E20b -- aspect scan of the
construction bank") adds two things on top of the unchanged E20 stage-2 path:

- `--minor-radius R`: re-export each selected construction's boundary from
  its stored `vector`/`n_field_periods` at radius `R` via `p3_construction_
  search._export_4x4_boundary` (extend-before-create -- the same export
  call stage 1 already uses to build `bank_top.json`'s stored boundaries,
  imported here rather than duplicated) and evaluate THAT instead of the
  stored (radius-0.1) boundary. Omitted (the default), behaviour is
  byte-identical to E20's stage 2. `reexport_at_minor_radius` takes an
  injectable `export_fn` so the re-export bookkeeping is host-testable
  without the pinned image's pyQIC/constellaration stack -- same pattern as
  `p3_construction_search.export_top_with_placement`'s injected `export_fn`.
- `--select-by {proxy_v,label_list}`: `proxy_v` (default, unchanged) is
  `load_top_n`'s ranked top `--top`. `label_list` (with `--labels`, a
  comma-separated string) instead selects an exact, ordered set of labels --
  for reproducing the identical 48 constructions across multiple
  `--minor-radius` runs even though, in practice, `proxy_v` ranking already
  does this (each row's `proxy_v` is computed once at search time and
  stored in `bank_top.json`; it does not depend on the re-export radius).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from constellaration_methods.analysis.p3_design_diversity_probe import _git_commit
from constellaration_methods.construction import p3_construction_proxies as pcp

CANONICAL_ROW_NAMES = ("iota", "qi", "mirror", "flux", "well")
FEASIBILITY_TOLERANCE = 0.01  # matches problems._DEFAULT_RELATIVE_TOLERANCE

# E20b twist bins (registration: "< 0.14, 0.14-0.25, >= 0.25"), on
# near-axis |iota|/nfp.
TWIST_BIN_MID = 0.14
TWIST_BIN_HIGH = 0.25


def load_top_n(bank_dir: Path, n: int) -> list[dict[str, Any]]:
    """`bank_top.json` rows, sorted by proxy_v ascending (defensive -- the
    search already writes them sorted), truncated to the top `n`."""
    rows = json.loads((bank_dir / "bank_top.json").read_text())
    rows = sorted(rows, key=lambda r: r["proxy_v"])
    return rows[:n]


def select_rows(
    bank_dir: Path,
    *,
    top: int,
    select_by: str = "proxy_v",
    labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    """E20b `--select-by`: which `bank_top.json` rows to evaluate.
    `"proxy_v"` (default) is `load_top_n`'s ranked top `top`, byte-identical
    to E20's stage-2 selection. `"label_list"` instead returns exactly
    `labels`, in the order given (raises if `labels` is empty/None, or if
    any label is not in the bank -- fail fast rather than silently
    evaluating a shorter or differently-ordered set than requested)."""
    if select_by == "label_list":
        if not labels:
            raise ValueError("--select-by label_list requires --labels")
        by_label = {r["label"]: r for r in json.loads((bank_dir / "bank_top.json").read_text())}
        missing = [label for label in labels if label not in by_label]
        if missing:
            raise ValueError(f"labels not found in bank_top.json: {missing}")
        return [by_label[label] for label in labels]
    if select_by != "proxy_v":
        raise ValueError(f"unknown --select-by {select_by!r}; choices: proxy_v, label_list")
    return load_top_n(bank_dir, top)


def reexport_at_minor_radius(
    rows: list[dict[str, Any]],
    minor_radius: float,
    *,
    export_fn=None,
) -> list[dict[str, Any]]:
    """E20b `--minor-radius`: replace each row's stored (radius-0.1)
    `boundary` with a fresh export from its `vector`/`n_field_periods` at
    `minor_radius`. Reuses `p3_construction_search._export_4x4_boundary`
    (extend-before-create) via the injectable `export_fn`, which defaults to
    that real export but lets tests supply a fake one -- the real path needs
    the pinned image's pyQIC/constellaration stack, same split as
    `p3_construction_search.export_top_with_placement`. Returns new row
    dicts (input rows are not mutated) with `boundary` replaced and
    `minor_radius` recorded."""
    if export_fn is None:
        from constellaration_methods.construction import p3_construction_search as pcs

        def export_fn(vector: list[float], nfp: int) -> dict[str, Any]:
            return pcs._export_4x4_boundary(vector, nfp, minor_radius=minor_radius)

    out = []
    for r in rows:
        boundary = export_fn(r["vector"], r["n_field_periods"])
        out.append({**r, "boundary": boundary, "minor_radius": minor_radius})
    return out


def twist_bin(near_axis_iota_over_nfp: float) -> str:
    """E20b registration twist bins: '< 0.14', '0.14-0.25', '>= 0.25' on
    near-axis |iota|/nfp. Bounds are inclusive on the lower edge of each
    higher bin (0.14 falls in '0.14-0.25', 0.25 falls in '>=0.25')."""
    if near_axis_iota_over_nfp < TWIST_BIN_MID:
        return "<0.14"
    if near_axis_iota_over_nfp < TWIST_BIN_HIGH:
        return "0.14-0.25"
    return ">=0.25"


def write_boundary_files(rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for r in rows:
        p = out_dir / f"{r['label']}.boundary.json"
        p.write_text(json.dumps(r["boundary"]))
        paths.append(p)
    return paths


def _evaluate_boundary_file(path_str: str) -> dict[str, Any]:
    """One canonical evaluation of one boundary file, in whatever process
    calls this (top-level + picklable so `ProcessPoolExecutor(mp_context=
    spawn)` runs it in a genuinely fresh process for the worst-of-three
    confirmation)."""
    from constellaration_methods.evaluation import score_mhd_stable as sms

    throwaway = Path(f"/tmp/e20_stage2_eval_{Path(path_str).stem}.json")
    rows = sms.report_boundary_files([path_str], throwaway)
    return rows[0]


def canonical_once(paths: list[Path]) -> list[dict[str, Any]]:
    """One canonical evaluation per path, sequentially in this process --
    the registration's "top 48 ... evaluated once each" stage."""
    return [_evaluate_boundary_file(str(p)) for p in paths]


def rows_satisfied_at_seed(residuals: dict[str, float]) -> dict[str, bool]:
    """Which canonical rows are satisfied (residual <= 0) at the once-only
    evaluation -- reported alongside the worst-of-three V, not folded into
    it."""
    return {name: residuals.get(name, float("inf")) <= 0.0 for name in CANONICAL_ROW_NAMES}


def build_per_row_comparison(
    rows: list[dict[str, Any]],
    once: list[dict[str, Any]],
    minor_radius: float | None,
) -> list[dict[str, Any]]:
    """E20b: one near-axis-vs-canonical comparison record per selected row
    (`rows[i]`/`once[i]` pair, the same order `canonical_once` evaluated
    them in). A row whose canonical evaluation raised (`score_mhd_stable.
    report_boundary_files`'s `"error"` key, no `"V"`/`"raw"`) is
    `convergent=False` with `error_class` set from its truncated error
    string, matching how `score_mhd_stable.report_boundary_files` itself
    distinguishes a failed forward-model call from a scored one."""
    out = []
    for r, once_row in zip(rows, once):
        proxies = r.get("proxies", {})
        near_axis_iota = proxies.get("iota_over_nfp")
        convergent = "error" not in once_row
        raw = once_row.get("raw", {}) if convergent else {}
        out.append(
            {
                "label": r["label"],
                "n_field_periods": r.get("n_field_periods"),
                "minor_radius": minor_radius,
                "near_axis_iota_over_nfp": near_axis_iota,
                "near_axis_mirror": proxies.get("mirror"),
                "near_axis_qi_residual": proxies.get("qi_residual"),
                "convergent": convergent,
                "error_class": None if convergent else once_row.get("error"),
                "canonical_V": once_row.get("V") if convergent else None,
                "canonical_iota_over_nfp": raw.get("iota_over_nfp"),
                "canonical_mirror": raw.get("edge_mirror"),
                "canonical_log10_qi": raw.get("log10_qi"),
                "twist_bin": (twist_bin(near_axis_iota) if near_axis_iota is not None else "n/a"),
            }
        )
    return out


def convergence_fraction_by_twist_bin(
    per_row: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """E20b summary table: n, n_convergent and convergence_fraction per
    near-axis twist bin (`build_per_row_comparison`'s `twist_bin` field)."""
    bins: dict[str, dict[str, Any]] = {}
    for rec in per_row:
        b = bins.setdefault(rec["twist_bin"], {"n": 0, "n_convergent": 0})
        b["n"] += 1
        if rec["convergent"]:
            b["n_convergent"] += 1
    for b in bins.values():
        b["convergence_fraction"] = b["n_convergent"] / b["n"] if b["n"] else None
    return bins


def convergent_agreement_summary(per_row: list[dict[str, Any]]) -> dict[str, Any]:
    """E20b summary, over convergent rows only: median ratio canonical /
    near-axis for |iota|/nfp and mirror, and median canonical log10 qi.
    `None` for any median with no eligible rows (e.g. nothing converged)."""
    convergent = [r for r in per_row if r["convergent"]]
    iota_ratios = [
        abs(r["canonical_iota_over_nfp"]) / r["near_axis_iota_over_nfp"]
        for r in convergent
        if r.get("near_axis_iota_over_nfp") and r.get("canonical_iota_over_nfp") is not None
    ]
    mirror_ratios = [
        r["canonical_mirror"] / r["near_axis_mirror"]
        for r in convergent
        if r.get("near_axis_mirror") and r.get("canonical_mirror") is not None
    ]
    log10_qis = [
        r["canonical_log10_qi"] for r in convergent if r.get("canonical_log10_qi") is not None
    ]
    return {
        "n_convergent": len(convergent),
        "median_iota_ratio_canonical_over_near_axis": (
            float(np.median(iota_ratios)) if iota_ratios else None
        ),
        "median_mirror_ratio_canonical_over_near_axis": (
            float(np.median(mirror_ratios)) if mirror_ratios else None
        ),
        "median_canonical_log10_qi": (float(np.median(log10_qis)) if log10_qis else None),
    }


def confirm_worst_of_three(
    label: str, path: Path, first_v: float, *, workers: int
) -> dict[str, Any]:
    """Two more canonical evaluations of the same boundary file, each in a
    fresh spawned process, plus the already-computed `first_v` -- worst
    (max) of the three is the confirmed V."""
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(2, workers), mp_context=ctx) as pool:
        extra = list(pool.map(_evaluate_boundary_file, [str(path), str(path)]))
    vs = [first_v] + [r.get("V", float("inf")) for r in extra]
    return {
        "label": label,
        "v_runs": vs,
        "worst_of_three_v": max(vs),
    }


LABEL_TIERS_BEST_FIRST = ("goal", "substantial", "material", "null")


def best_label_row(confirmed: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    """The confirmed row that achieved the best label under `key`, best =
    earliest in `LABEL_TIERS_BEST_FIRST`; ties broken by lower worst-of-three
    V. None when nothing was confirmed."""
    if not confirmed:
        return None
    return min(
        confirmed,
        key=lambda c: (LABEL_TIERS_BEST_FIRST.index(c[key]), c["worst_of_three_v"]),
    )


def summarise_confirmed(confirmed: list[dict[str, Any]]) -> dict[str, Any]:
    """Report-level fields over the confirmed rows. The registration bars
    are EXISTENTIAL (material if ANY confirmed seed meets the bar, and so on
    up the tiers), so the report-level label is the best label achieved by
    ANY confirmed row, and `stage2_label_determined_by` names that row.
    `best_confirmed_*` (the min-V row) is kept as a separate field: it is
    the best V, not necessarily the row that set the label."""
    best_confirmed = min(confirmed, key=lambda c: c["worst_of_three_v"], default=None)
    by_stage2 = best_label_row(confirmed, "stage2_label")
    return {
        "best_confirmed_label": best_confirmed["label"] if best_confirmed else None,
        "best_confirmed_worst_of_three_v": (
            best_confirmed["worst_of_three_v"] if best_confirmed else None
        ),
        "stage2_label": by_stage2["stage2_label"] if by_stage2 else None,
        "stage2_label_determined_by": by_stage2["label"] if by_stage2 else None,
    }


def run_stage2(
    *,
    bank_dir: Path,
    top: int,
    confirm_top: int,
    workers: int,
    out_dir: Path,
    minor_radius: float | None = None,
    select_by: str = "proxy_v",
    labels: list[str] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = select_rows(bank_dir, top=top, select_by=select_by, labels=labels)
    if minor_radius is not None:
        rows = reexport_at_minor_radius(rows, minor_radius)
    boundary_dir = out_dir / "boundaries"
    paths = write_boundary_files(rows, boundary_dir)

    print(f"[E20 stage2] canonical-once eval of top {len(rows)}", flush=True)
    once = canonical_once(paths)
    jsonl_path = out_dir / "canonical_once.jsonl"
    with open(jsonl_path, "w") as fh:
        for r, row in zip(rows, once):
            rec = {"label": r["label"], **row}
            if minor_radius is not None:
                rec["minor_radius"] = minor_radius
                rec["near_axis_proxies"] = r.get("proxies")
            fh.write(json.dumps(rec) + "\n")

    ranked = sorted(zip(rows, once, paths), key=lambda t: t[1].get("V", float("inf")))
    best_once_row, best_once_eval, _ = ranked[0] if ranked else (None, None, None)
    confirmed: list[dict[str, Any]] = []
    for r, once_row, path in ranked[:confirm_top]:
        print(
            f"[E20 stage2] confirming {r['label']} (once V={once_row.get('V')})",
            flush=True,
        )
        c = confirm_worst_of_three(
            r["label"], path, once_row.get("V", float("inf")), workers=workers
        )
        c["rows_satisfied_at_seed"] = rows_satisfied_at_seed(once_row.get("residuals", {}))
        c["feasible"] = c["worst_of_three_v"] <= FEASIBILITY_TOLERANCE
        c["stage2_label"] = pcp.stage2_label(c["worst_of_three_v"], feasible=c["feasible"])
        confirmed.append(c)

    report = {
        "status": "PROVISIONAL (no lock snapshot yet)",
        "n_top_evaluated_once": len(rows),
        "n_confirm_top": confirm_top,
        "canonical_once_best_v": best_once_eval.get("V") if best_once_eval else None,
        "canonical_once_best_label": best_once_row["label"] if best_once_row else None,
        "confirmed": confirmed,
        **summarise_confirmed(confirmed),
        "git_commit": _git_commit(),
    }
    if minor_radius is not None:
        per_row = build_per_row_comparison(rows, once, minor_radius)
        report["minor_radius"] = minor_radius
        report["select_by"] = select_by
        report["per_row"] = per_row
        report["convergence_by_twist_bin"] = convergence_fraction_by_twist_bin(per_row)
        report["convergent_agreement_summary"] = convergent_agreement_summary(per_row)
    (out_dir / "REPORT.json").write_text(json.dumps(report, indent=2))
    print("\n=== E20 stage2 REPORT ===")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("bank_dir", type=Path)
    parser.add_argument("--top", type=int, default=48)
    parser.add_argument("--confirm-top", type=int, default=12)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--minor-radius",
        type=float,
        default=None,
        help="E20b: re-export each selected construction's boundary from its "
        "stored vector/n_field_periods at this radius (via "
        "p3_construction_search._export_4x4_boundary) and evaluate that "
        "instead of the stored boundary. Omitted: unchanged E20 behaviour.",
    )
    parser.add_argument(
        "--select-by",
        choices=["proxy_v", "label_list"],
        default="proxy_v",
        help="E20b: 'proxy_v' (default, unchanged) ranks bank_top.json by "
        "proxy_v and takes the top --top. 'label_list' (with --labels) "
        "selects exactly the given labels, in order.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Comma-separated construction labels, required with --select-by label_list.",
    )
    args = parser.parse_args()
    labels = args.labels.split(",") if args.labels else None
    run_stage2(
        bank_dir=args.bank_dir,
        top=args.top,
        confirm_top=args.confirm_top,
        workers=args.workers,
        out_dir=args.out,
        minor_radius=args.minor_radius,
        select_by=args.select_by,
        labels=labels,
    )


if __name__ == "__main__":
    main()
