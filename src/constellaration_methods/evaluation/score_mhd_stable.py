"""Score our best canonically-feasible designs on the MHDStableQIStellarator
(P3) leaderboard metric.

Unlike SimpleToBuildQIStellarator, `MHDStableQIStellarator.evaluate()` takes
a **list** of boundaries and scores the hypervolume of the Pareto front of
the feasible ones (objectives: maximize
`minimum_normalized_magnetic_gradient_scale_length`, minimize `aspect_ratio`;
reference point (1.0, 20.0)). To get a per-design score comparable to a
single-boundary leaderboard submission, each design here is evaluated alone,
`problem.evaluate([boundary])` -- a batch of one. `evaluate([x]).score` is
then exactly the hypervolume of that single point if feasible, else 0.0 (see
`problems.MHDStableQIStellarator._score`), which is the EXACT call the
leaderboard makes for a solo submission (same primitive the P2 scoring
script uses for the P2 metric). P3's score is unbounded (not the
/20-normalised simple_to_build score), so there is no 0-1 sanity ceiling
here -- the only sanity check available is that `evaluate()` runs the
identical code path.

Designs evaluated by default:
  1. `submitted` -- the submitted P2 entry (docs/results/submission/
     best_feasible_simple_to_build.json in this repo).
  2. any further canonically-feasible bank rows named in `--other-call-index`
     from a JSONL bank file the caller supplies with `--bank`; this repo does
     not ship that bank (it is campaign working data, not a published
     result) so the default is an empty list.

None of this is a submission -- P3 evaluate() only, no leaderboard POST.

Usage (inside the pinned image, cores 14-15 ONLY on the original campaign's
box; any two cores are fine elsewhere):
    podman run --rm -v "$(pwd)":/workspace:Z -w /workspace \\
        localhost/constellaration-methods:test \\
        nice -n 10 taskset -c 14,15 python -m constellaration_methods.evaluation.score_mhd_stable
"""

from __future__ import annotations

import json
import time
from pathlib import Path

DEFAULT_BANK = Path("results") / "family-study" / "climb" / "c8_climb_bank.jsonl"
DEFAULT_SUBMITTED = Path("docs") / "results" / "submission" / "best_feasible_simple_to_build.json"
DEFAULT_OUT_JSON = Path("results") / "mhd-eval" / "p3_scores.json"
DEFAULT_OUT_JSONL = Path("results") / "mhd-eval" / "p3_scores.jsonl"


def load_bank(bank_path: Path) -> dict[int, dict]:
    rows = [json.loads(line) for line in open(bank_path) if line.strip()]
    return {r["call_index"]: r for r in rows}


def report_boundary_files(paths: list[str], out_path: Path) -> list[dict]:
    """Per-constraint P3 residuals, raw metrics and the would-be solo score of
    arbitrary boundary JSON files (leaderboard loader, one high-fidelity
    forward-model call each). Diagnostic: which constraints an external
    design violates."""
    import numpy as np
    from constellaration import forward_model, problems
    from constellaration.geometry import surface_rz_fourier

    problem = problems.MHDStableQIStellarator()
    names = ["iota", "qi", "mirror", "flux", "well"]
    rows = []
    for path in paths:
        t0 = time.perf_counter()
        rec = {"file": path}
        try:
            boundary = surface_rz_fourier.SurfaceRZFourier.model_validate_json(
                Path(path).read_text()
            )
            metrics, _ = forward_model.forward_model(
                boundary=boundary,
                settings=forward_model.ConstellarationSettings.default_high_fidelity(),
            )
            r = np.asarray(problem._normalized_constraint_violations(metrics), float)
            rec.update(
                residuals=dict(zip(names, [float(v) for v in r])),
                V=float(r.max()),
                feasible=bool(problem.is_feasible(metrics)),
                solo_score_if_feasible=float(problem._score([metrics])),
                raw=dict(
                    iota_over_nfp=float(metrics.edge_rotational_transform_over_n_field_periods),
                    log10_qi=float(np.log10(metrics.qi)),
                    edge_mirror=float(metrics.edge_magnetic_mirror_ratio),
                    flux_compression=float(metrics.flux_compression_in_regions_of_bad_curvature),
                    vacuum_well=float(metrics.vacuum_well),
                    aspect_ratio=float(metrics.aspect_ratio),
                    min_len=float(metrics.minimum_normalized_magnetic_gradient_scale_length),
                ),
                mpol=int(np.asarray(boundary.r_cos).shape[0] - 1),
                ntor=int((np.asarray(boundary.r_cos).shape[1] - 1) // 2),
                nfp=int(boundary.n_field_periods),
            )
        except Exception as e:  # noqa: BLE001 -- never silently drop a design
            rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        rec["elapsed_s"] = round(time.perf_counter() - t0, 1)
        print(json.dumps(rec), flush=True)
        rows.append(rec)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=2))
    return rows


def evaluate_and_record(
    problem, tol: float, results: list[dict], label: str, boundary, extra: dict
) -> dict:
    """Evaluate one boundary as a solo submission and append the record to
    `results`. Split out from `main()` so it can be driven by a fake
    `problem`/boundary pair in tests -- no VMEC required to exercise the
    report-assembly logic."""
    t0 = time.perf_counter()
    rec = {"cohort": label, **extra}
    try:
        ev = problem.evaluate([boundary])  # batch of one -- solo-submission score
        length_obj, aspect_obj = ev.objectives[0]
        rec.update(
            {
                "score": float(ev.score),
                "min_normalized_magnetic_gradient_scale_length": float(length_obj[0]),
                "aspect_ratio": float(aspect_obj[0]),
                "feasibility_violation": float(ev.feasibility[0]),
                "feasible": bool(ev.feasibility[0] <= tol),
            }
        )
        print(
            f"[{label}] score={rec['score']:.4f} "
            f"min_len={rec['min_normalized_magnetic_gradient_scale_length']:.4f} "
            f"aspect={rec['aspect_ratio']:.4f} "
            f"feas_violation={rec['feasibility_violation']:.6f} "
            f"feasible={rec['feasible']}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001 -- never silently drop a design
        rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        print(f"[{label}] ERROR {rec['error']}", flush=True)
    rec["elapsed_s"] = time.perf_counter() - t0
    results.append(rec)
    return rec


def build_arg_parser():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--boundary",
        nargs="+",
        default=None,
        help="boundary JSON files: report P3 residuals/raw/solo score and exit",
    )
    ap.add_argument("--out", default="results/p3-probe/report.json")
    ap.add_argument("--submitted", default=str(DEFAULT_SUBMITTED))
    ap.add_argument("--bank", default=str(DEFAULT_BANK))
    ap.add_argument(
        "--other-call-index",
        type=int,
        nargs="*",
        default=[],
        help="further call_index rows from --bank to score alongside --submitted",
    )
    ap.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    ap.add_argument("--out-jsonl", default=str(DEFAULT_OUT_JSONL))
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.boundary:
        report_boundary_files(args.boundary, Path(args.out))
        return

    import numpy as np
    from constellaration import problems
    from constellaration.geometry import surface_rz_fourier

    from constellaration_methods.engines.alm_ngopt_optimizer import prepare_initial_boundary

    problem = problems.MHDStableQIStellarator()
    # module constant, not exported: same tolerance problem.is_feasible() uses
    tol = getattr(problems, "_DEFAULT_RELATIVE_TOLERANCE", 0.01)
    results: list[dict] = []
    out_json = Path(args.out_json)
    out_jsonl = Path(args.out_jsonl)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if out_jsonl.exists():
        out_jsonl.unlink()

    def record(label: str, boundary, extra: dict) -> None:
        rec = evaluate_and_record(problem, tol, results, label, boundary, extra)
        out_json.write_text(json.dumps(results, indent=2))
        with out_jsonl.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    submitted_path = Path(args.submitted)
    raw = json.loads(submitted_path.read_text())
    boundary = surface_rz_fourier.SurfaceRZFourier.model_validate_json(submitted_path.read_text())
    record(
        "submitted",
        boundary,
        {
            "id": submitted_path.stem,
            "source_file": str(submitted_path),
            "nfp": raw["n_field_periods"],
        },
    )

    if args.other_call_index:
        bank = load_bank(Path(args.bank))
        for ci in args.other_call_index:
            r = bank.get(ci)
            if r is None:
                rec = {
                    "cohort": "other",
                    "id": f"bank_call{ci}",
                    "call_index": ci,
                    "error": "call_index not found in bank",
                    "elapsed_s": 0.0,
                }
                print(f"[bank_call{ci}] SKIP {rec['error']}", flush=True)
                results.append(rec)
                out_json.write_text(json.dumps(results, indent=2))
                with out_jsonl.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                continue
            bd = r["boundary"]
            mmax = int(r.get("max_poloidal_mode", np.asarray(bd["r_cos"]).shape[0] - 1))
            nmax = int(r.get("max_toroidal_mode", (np.asarray(bd["r_cos"]).shape[1] - 1) // 2))
            nfp = int(r["n_field_periods"])
            surface = prepare_initial_boundary(
                {"r_cos": bd["r_cos"], "z_sin": bd["z_sin"], "n_field_periods": nfp},
                mmax,
                nmax,
                nfp,
            )
            record(
                "other",
                surface,
                {
                    "id": f"bank_call{ci}",
                    "call_index": ci,
                    "simple_to_build_objective": r.get("objective"),
                    "canonical_margin": r.get("canonical_margin"),
                    "nfp": nfp,
                },
            )

    print(f"\nwrote {out_json}\nwrote {out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
