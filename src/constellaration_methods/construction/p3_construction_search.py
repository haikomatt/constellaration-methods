#!/usr/bin/env python3
"""E20 (p3-construction-search-plan-2026-09-25.md, Mechanism step 1): the
construction search itself -- the H2 engine's near-axis configuration vector
and multi-start Latin-hypercube search, driven by the P3 proxy objective in
`construction.p3_construction_proxies` instead of H2's own QI-only objective.

Reuses (extend-before-create):
  - `engines.near_axis_search`: `NearAxisSpace`/`CONTINUOUS` (the box + vector
    layout), `config_arrays` (vector -> near-axis arrays, identical to
    `generate`'s own `from_x`), `latin_hypercube_starts` (the same start
    distribution H2 uses, factored out to be shared by both callers).
  - `construction.p3_construction_proxies`: the proxy rows, scalarisation, and
    D1 `nearest_family_distance`.
  - `near_axis_search_run.screen_config`: the boundary-export call
    (`near_axis_configuration_to_plasma_boundary` +
    `NearAxisToPlasmaBoundarySettings`) and `smooth_and_set_max_mode_numbers`
    for the top-200 bank's exported 4x4 boundary dicts.
  - `analysis.p3_design_diversity_probe`'s `_git_commit`/`_resolve_ref`:
    reads `.git` directly (stdlib file I/O, follows a worktree checkout's
    gitdir/commondir indirection) instead of shelling out to a `git` binary
    -- the pinned image this driver runs in has none installed, so a
    `subprocess.run(["git", ...])` provenance field would silently read
    "unknown" on every real in-image run.

Deviation from a literal "CMA-ES/NGOpt" reading of the registration: H2's own
engine (`near_axis_search.multi_start_search`) is Latin-hypercube multi-start
+ `scipy.optimize.minimize` (Nelder-Mead by default), NOT CMA-ES/NGOpt -- there
is no CMA-ES/NGOpt loop anywhere in this near-axis engine to reuse (checked
with a case-insensitive grep for "cma" and "ngopt" across the near-axis engine
and its runner: no hits). Per "Extend Before Create" this driver reuses the
engine that actually exists rather than bolting on a second optimiser
backend.

One "evaluated configuration" here = one Latin-hypercube start polished to a
local optimum against the proxy objective (the same granularity
`near_axis_search.multi_start_search` already reports as "n_starts"), not one
raw function evaluation inside the local solve -- `bank.jsonl` gets one row
per polished start. Each local solve is capped at `_LOCAL_MAXFEV` function
evaluations (default 80): the registration frames this stage as "cheap,
CPU-light", and an uncapped 6-D Nelder-Mead solve (scipy default ~1200
evaluations) at the ~1.4s-per-evaluation proxy oracle would blow the
3-hour/5,000-configuration budget by roughly two orders of magnitude.

Runs inside the pinned image, e.g. cores 2-13 (12 workers):

    podman run --rm -v "$(pwd)":/workspace:Z -w /workspace --network none \\
        -v "$(pwd)/.git":/workspace/.git:ro,Z \\
        --pids-limit 8192 --ulimit nproc=16384:16384 --entrypoint taskset \\
        localhost/constellaration-methods:dev -c 2-13 \\
        cm-p3-construction-search \\
        --budget-configs 5000 --total-hours 3 --workers 12 \\
        --out results/p3-e20-construction-search

The `.git` mount is read-only and optional: it lets `_git_commit` (see reuse
note above) read this repo's own commit hash from inside the container for
REPORT.json's `git_commit` field. Without it, `_git_commit` falls back to
"unknown" -- harmless, just less provenance.
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from constellaration_methods.analysis.p3_design_diversity_probe import _git_commit
from constellaration_methods.construction import p3_construction_proxies as pcp
from constellaration_methods.engines import near_axis_search as nas

REPO = Path(__file__).resolve().parents[3]

_LOCAL_MAXFEV = 80

# Named spaces this driver can draw Latin-hypercube starts from. "default"
# is byte-identical to the pre-E20 behaviour (the only space this script
# used before the E20 diagnostic); "p3-wide" is the diagnostic's widened
# box (`near_axis_search.p3_wide_space`) -- opt-in via `--space p3-wide`,
# never the default.
_SPACES: dict[str, Any] = {
    "default": lambda: nas.NearAxisSpace(n_field_periods=(3, 4)),
    "p3-wide": lambda: nas.p3_wide_space(),
}


def _resolve_space(name: str) -> nas.NearAxisSpace:
    try:
        return _SPACES[name]()
    except KeyError:
        raise ValueError(f"unknown --space {name!r}; choices: {sorted(_SPACES)}") from None


def _physics():
    """Lazily import the physics stack -- keeps this module host-importable
    for the CLI-parsing / argument-shape tests."""
    from constellaration.geometry import surface_rz_fourier as srf
    from constellaration.mhd import near_axis_configuration as nac

    return nac, srf


def make_observe_fn(*, minor_radius: float):
    """vector + nfp -> raw near-axis quantities the proxies need. See
    `p3_construction_proxies`'s module docstring for the Phase 0 evidence
    behind each attribute. A raising build (pyQIC can fail for some parameter
    corners, same as H2's own `near_axis_search_run.make_observe_fn`) comes
    back as the worst-possible row so the optimiser is repelled, not crashed."""
    nac, srf = _physics()

    def observe(vector: np.ndarray, nfp: int) -> dict[str, float]:
        try:
            cfg = nac.NearAxisConfiguration(**nas.config_arrays(vector, nfp))
            st = nac.near_axis_configuration_to_pyqic(cfg)
            boundary = nac.near_axis_configuration_to_plasma_boundary(
                cfg, nac.NearAxisToPlasmaBoundarySettings(minor_radius=minor_radius)
            )
            minor = float(srf.evaluate_minor_radius(boundary))
            major_radius = float(np.asarray(cfg.r_cos)[0])
            aspect_ratio = major_radius / minor if minor > 0 else float("inf")
            return {
                "iota_over_nfp": abs(float(st.iota)) / nfp if nfp else abs(float(st.iota)),
                "mirror": float(st.B0_well_depth),
                "qi_residual": float(st.min_geo_qi_consistency(order=1)),
                "aspect_ratio": aspect_ratio,
            }
        except Exception:  # noqa: BLE001 -- a failed build is a repelled point
            return {
                "iota_over_nfp": 0.0,
                "mirror": 1.0,
                "qi_residual": float("inf"),
                "aspect_ratio": float("inf"),
            }

    return observe


def _proxy_v(observables: dict[str, float], nfp: int) -> float:
    return pcp.evaluate_proxies(n_field_periods=nfp, **observables).proxy_v


def _polish_one_start(task: dict[str, Any]) -> dict[str, Any]:
    """Worker body (top-level so it is picklable for `ProcessPoolExecutor`
    with the `spawn` context): polish one Latin-hypercube start against the
    proxy objective, capped at `_LOCAL_MAXFEV` evaluations, and return one
    bank row (provenance included)."""
    from scipy import optimize

    space = _resolve_space(task.get("space_name", "default"))
    observe = make_observe_fn(minor_radius=task["minor_radius"])
    nfp = task["nfp"]
    bounds = space.bounds()
    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])

    def _fun(x: np.ndarray) -> float:
        return _proxy_v(observe(x, nfp), nfp)

    x0 = np.asarray(task["vector0"])
    try:
        res = optimize.minimize(
            _fun,
            x0,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxfev": _LOCAL_MAXFEV, "maxiter": _LOCAL_MAXFEV},
        )
        x = np.clip(res.x, lows, highs)
    except Exception:  # noqa: BLE001 -- a solver blow-up keeps the start point
        x = np.clip(x0, lows, highs)

    observables = observe(x, nfp)
    row = pcp.evaluate_proxies(n_field_periods=nfp, **observables)
    return {
        "label": task["label"],
        "n_field_periods": nfp,
        "vector": [float(v) for v in x],
        "proxies": {
            "iota_over_nfp": observables["iota_over_nfp"],
            "mirror": observables["mirror"],
            "qi_residual": observables["qi_residual"],
            "aspect_ratio": observables["aspect_ratio"],
            "violations": row.violations,
        },
        "proxy_v": row.proxy_v,
        "provenance": {
            "worker_pid": os.getpid(),
            "start_vector0": [float(v) for v in x0],
            "minor_radius": task["minor_radius"],
        },
    }


def _own_family() -> list[dict[str, Any]]:
    """`results/p3-e17-front/*.boundary.json` (not phireflected) +
    `davidkh_lineage_FEASIBLE_E13_PT3.boundary.json` -- our lineage. These
    result files are campaign artifacts, not shipped with this repository;
    callers that have not populated `results/` will simply get an empty
    family (see `nearest_family_distance`'s "n/a" handling)."""
    family = []
    for p in sorted(glob.glob(str(REPO / "results/p3-e17-front/*.boundary.json"))):
        if "phireflected" in p:
            continue
        family.append(json.loads(Path(p).read_text()))
    pt3 = REPO / "results/p3-close-out/davidkh_lineage_FEASIBLE_E13_PT3.boundary.json"
    if pt3.exists():
        family.append(json.loads(pt3.read_text()))
    return family


def _leader_family() -> list[dict[str, Any]]:
    """`results/p3-close-out/board/*.json`: `boundary_json` is a JSON-string
    list of boundary dicts (each itself a JSON string). Same note as
    `_own_family` above -- these are campaign result files, not shipped
    here."""
    family = []
    for p in sorted(glob.glob(str(REPO / "results/p3-close-out/board/*.json"))):
        d = json.loads(Path(p).read_text())
        for s in json.loads(d["boundary_json"]):
            family.append(json.loads(s) if isinstance(s, str) else s)
    return family


def _export_4x4_boundary(vector: list[float], nfp: int, *, minor_radius: float) -> dict:
    """The candidate's boundary, smoothed to a 4x4 mode budget, as a full
    `SurfaceRZFourier`-loadable dict -- same field set as the family files
    (`results/p3-e17-front/*.boundary.json`: r_cos, r_sin, z_cos, z_sin,
    n_field_periods, is_stellarator_symmetric), so both `nearest_family_
    distance` (D1) and stage 2's `score_mhd_stable.report_boundary_files`
    (which loads via `SurfaceRZFourier.model_validate_json`) can consume it
    directly.

    `smooth_and_set_max_mode_numbers` (and `near_axis_configuration_to_
    plasma_boundary` before it) leaves `r_sin`/`z_cos` as `None` for a
    stellarator-symmetric surface -- and, checked directly against the
    family files themselves (`results/p3-e17-front/G0_19.boundary.json`'s
    `r_sin`/`z_cos` are JSON `null`, not zero arrays), that IS the correct
    on-disk shape: `SurfaceRZFourier`'s own pydantic validator requires
    `r_sin`/`z_cos` to be `None` when `is_stellarator_symmetric` is `True`
    (confirmed in-container: substituting zero arrays instead is REJECTED
    by that validator, not merely redundant). So `None` is passed through
    unchanged here, not substituted."""
    nac, _ = _physics()
    cfg = nac.NearAxisConfiguration(**nas.config_arrays(np.asarray(vector), nfp))
    full = nac.near_axis_configuration_to_plasma_boundary(
        cfg, nac.NearAxisToPlasmaBoundarySettings(minor_radius=minor_radius)
    )
    smoothed = nac.smooth_and_set_max_mode_numbers(full, 4, 4)
    return {
        "r_cos": np.asarray(smoothed.r_cos, dtype=float).tolist(),
        "r_sin": smoothed.r_sin,
        "z_cos": smoothed.z_cos,
        "z_sin": np.asarray(smoothed.z_sin, dtype=float).tolist(),
        "n_field_periods": int(nfp),
        "is_stellarator_symmetric": True,
    }


def build_tasks(
    *,
    budget_configs: int,
    minor_radius: float,
    random_seed: int,
    space_name: str = "default",
) -> list[dict[str, Any]]:
    """Split `budget_configs` Latin-hypercube starts across nfp in {3, 4}
    (the registration's nfp restriction), same start distribution H2 uses
    (`near_axis_search.latin_hypercube_starts`). `space_name` selects the
    box (`_SPACES`); defaults to "default", byte-identical to this
    function's pre-E20-widening behaviour."""
    space = _resolve_space(space_name)
    nfps = list(space.n_field_periods)
    per_nfp = max(1, budget_configs // len(nfps))
    rng = np.random.RandomState(random_seed)
    tasks: list[dict[str, Any]] = []
    for nfp in nfps:
        starts = nas.latin_hypercube_starts(space, per_nfp, rng)
        for i in range(per_nfp):
            tasks.append(
                {
                    "label": f"e20_nfp{nfp}_{i:05d}",
                    "nfp": nfp,
                    "vector0": starts[i].tolist(),
                    "minor_radius": minor_radius,
                    "space_name": space_name,
                }
            )
    return tasks[:budget_configs]


def export_top_with_placement(
    top_rows: list[dict[str, Any]],
    *,
    export_fn,
    own_family: list[dict[str, Any]],
    leader_family: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Export each row's boundary (`export_fn(vector, nfp) -> boundary
    dict`, injected so this is host-testable with a fake exporter) and
    attach its D1 placement. A row whose export raises is SKIPPED, not
    fatal: the same near-axis parameter corners that produce a bad proxy
    can also fail `Frenet_to_cylindrical`'s root solve (the exact failure
    `near_axis_search_run.screen_config` already tolerates as
    "boundary_failed" rather than crashing the whole screen). Returns
    (rows_with_placement, n_export_failed)."""
    out: list[dict[str, Any]] = []
    n_failed = 0
    for r in top_rows:
        try:
            boundary = export_fn(r["vector"], r["n_field_periods"])
        except Exception as e:  # noqa: BLE001 -- see docstring
            n_failed += 1
            print(
                f"[E20] boundary export failed for {r['label']}: "
                f"{type(e).__name__}: {str(e)[:120]}",
                flush=True,
            )
            continue
        d_leader = pcp.nearest_family_distance(boundary, leader_family)
        d_own = pcp.nearest_family_distance(boundary, own_family)
        out.append(
            {
                **r,
                "boundary": boundary,
                "d1_to_leader_family": d_leader if d_leader is not None else "n/a",
                "d1_to_own_family": d_own if d_own is not None else "n/a",
            }
        )
    return out, n_failed


def run_search(
    *,
    budget_configs: int,
    total_hours: float,
    workers: int,
    minor_radius: float,
    random_seed: int,
    out_dir: Path,
    space_name: str = "default",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bank_path = out_dir / "bank.jsonl"
    tasks = build_tasks(
        budget_configs=budget_configs,
        minor_radius=minor_radius,
        random_seed=random_seed,
        space_name=space_name,
    )
    print(
        f"[E20] {len(tasks)} configs ({workers} workers, "
        f"cap {total_hours}h, local maxfev={_LOCAL_MAXFEV})",
        flush=True,
    )

    t0 = time.time()
    rows: list[dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    with (
        ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool,
        open(bank_path, "w") as fh,
    ):
        for i, row in enumerate(pool.map(_polish_one_start, tasks)):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            rows.append(row)
            if (i + 1) % 20 == 0 or (i + 1) == len(tasks):
                print(
                    f"[{i + 1}/{len(tasks)}] {row['label']} proxy_v={row['proxy_v']:.4f}",
                    flush=True,
                )
            if (time.time() - t0) > total_hours * 3600:
                print("[E20] wall cap reached", flush=True)
                break
    elapsed = time.time() - t0

    rows.sort(key=lambda r: r["proxy_v"])
    top = rows[:200]
    own_family = _own_family()
    leader_family = _leader_family()
    top_with_placement, n_export_failed = export_top_with_placement(
        top,
        export_fn=lambda vector, nfp: _export_4x4_boundary(vector, nfp, minor_radius=minor_radius),
        own_family=own_family,
        leader_family=leader_family,
    )
    (out_dir / "bank_top.json").write_text(json.dumps(top_with_placement, indent=2))

    leader_ds = [
        r["d1_to_leader_family"] for r in top_with_placement if r["d1_to_leader_family"] != "n/a"
    ]
    own_ds = [r["d1_to_own_family"] for r in top_with_placement if r["d1_to_own_family"] != "n/a"]
    report = {
        "n_configs_evaluated": len(rows),
        "n_configs_requested": len(tasks),
        "workers": workers,
        "elapsed_s": elapsed,
        "configs_per_minute": (len(rows) / elapsed * 60.0) if elapsed > 0 else None,
        "n_bank_top_export_failed": n_export_failed,
        "n_bank_top_exported": len(top_with_placement),
        "best_proxy_v": rows[0]["proxy_v"] if rows else None,
        "best_proxies": rows[0]["proxies"] if rows else None,
        "placement_median_d1_to_leader_family": (
            float(np.median(leader_ds)) if leader_ds else "n/a"
        ),
        "placement_median_d1_to_own_family": float(np.median(own_ds)) if own_ds else "n/a",
        "d_rand_reference": 0.388,
        "survey_seed_reference": 0.62,
        "well_row": "DROPPED -- not available; see "
        "construction.p3_construction_proxies module docstring (order='r2' "
        "not exposed by the reused near-axis engine; the survey's "
        "forward_model well estimate is too expensive for the per-candidate "
        "proxy oracle).",
        "space_name": space_name,
        "git_commit": _git_commit(),
    }
    (out_dir / "REPORT.json").write_text(json.dumps(report, indent=2))
    print("\n=== E20 construction search REPORT ===")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--budget-configs", type=int, default=5000)
    parser.add_argument("--total-hours", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--minor-radius", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=20260925)
    parser.add_argument("--space", choices=sorted(_SPACES), default="default")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "p3-e20-construction-search")
    args = parser.parse_args()
    run_search(
        budget_configs=args.budget_configs,
        total_hours=args.total_hours,
        workers=args.workers,
        minor_radius=args.minor_radius,
        random_seed=args.random_seed,
        out_dir=args.out,
        space_name=args.space,
    )


if __name__ == "__main__":
    main()
