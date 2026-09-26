"""Verify and package a P1 (`GeometricalProblem`) run's best candidate as a
leaderboard submission, proving the saved file re-scores by reloading it
from disk and running the leaderboard's own
`problems.GeometricalProblem().evaluate(...).score` -- the exact call the
leaderboard makes.

This is the single accurate P1 packaging script for this repository. It
replaces three near-duplicate scripts the source campaign accumulated
(a "walk candidates.ndjson best-first" script written twice with small
drifts, and a separate "package the final result and write provenance"
script): this module does both -- select the best canonically-feasible
candidate from a log (`candidates.ndjson`/`trajectory.ndjson`/
`result.ndjson`-shaped, any file with one JSON object per line and a
boundary field), verify it against a fresh re-evaluation of the saved file,
and write a provenance record in the flat schema this repo's own
`docs/results/submission/PROVENANCE_own_geometrical.json` actually uses
(`problem`/`score`/`objective`/`feasibility`/`board_top_at_submission`/
`beats_board`/`lineage`/`mode_budget` -- reconciled against that file, not
the more elaborate nested schema an earlier draft of this script produced).

Usage (inside the repo's own pinned image):
    podman run --rm -v "$(pwd)":/workspace:Z -w /workspace \\
        localhost/constellaration-methods:test \\
        python -m constellaration_methods.evaluation.verify_p1 \\
        --records results/p1-own-lineage/result.ndjson \\
        --boundary-field optimized_boundary \\
        --lineage "own -- rotating-ellipse analytic seed, NO leader boundary"
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

# The board-top score this repo's own campaign measured itself against at
# submission time (see docs/PROVENANCE.md) -- overridable via --board-top
# for a caller re-running this against a different snapshot.
DEFAULT_BOARD_TOP = 0.9701409584192382
DEFAULT_MODE_BUDGET = "8x8 NFP=3"


def read_records(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def boundary_dict_from_record(
    record: dict[str, Any], boundary_field: str = "boundary"
) -> dict[str, Any]:
    """The `SurfaceRZFourier`-shaped JSON dict a record's boundary field
    packages to, in the exact schema the leaderboard's winner target files
    use. `n_field_periods` defaults to 3 (this repo's P1 pin) if the record
    doesn't carry its own."""
    bd = record[boundary_field]
    return {
        "r_cos": bd["r_cos"],
        "z_sin": bd["z_sin"],
        "r_sin": bd.get("r_sin"),
        "z_cos": bd.get("z_cos"),
        "n_field_periods": int(record.get("n_field_periods", bd.get("n_field_periods", 3))),
        "is_stellarator_symmetric": True,
    }


def find_best_feasible(
    records: list[dict[str, Any]],
    evaluate_fn: Callable[[dict[str, Any]], tuple[float, float, float]],
    boundary_field: str = "boundary",
    max_candidates: int = 10,
) -> tuple[int, dict[str, Any], dict[str, Any]] | None:
    """Walks `records` best-objective-first (ascending -- lower
    `max_elongation` is better for P1), evaluating each candidate's boundary
    via the injected `evaluate_fn` (real call: reconstruct the boundary and
    run `problems.GeometricalProblem().evaluate(...)`; test call: a fake
    returning canned (score, objective, feasibility)). Returns
    `(rank, boundary_dict, evaluation)` for the first candidate whose score
    is > 0 (feasible, per this benchmark's own scoring convention -- see
    `verify_saved_file`'s docstring), or `None` if none of the first
    `max_candidates` are feasible. Stops at the first hit rather than
    scanning everything: once one candidate has proven feasible, later
    (worse-objective) candidates can't improve on it.
    """
    with_boundary = [r for r in records if r.get("objective") is not None and r.get(boundary_field)]
    with_boundary.sort(key=lambda r: r["objective"])

    for rank, record in enumerate(with_boundary[:max_candidates]):
        boundary_dict = boundary_dict_from_record(record, boundary_field=boundary_field)
        score, objective, feasibility = evaluate_fn(boundary_dict)
        if score > 0.0:
            return (
                rank,
                boundary_dict,
                {
                    "search_objective": record["objective"],
                    "score": score,
                    "objective": objective,
                    "feasibility": feasibility,
                },
            )
    return None


def verify_saved_file(
    saved_path: Path,
    evaluate_fn: Callable[[dict[str, Any]], tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Reloads `saved_path` from disk (never reuses an in-memory object) and
    re-evaluates it via `evaluate_fn`, so the provenance record proves the
    file that would actually be submitted, not a cached in-run number."""
    boundary_dict = json.loads(saved_path.read_text())
    return evaluate_fn(boundary_dict)


def build_provenance(
    *,
    score: float,
    objective: float,
    feasibility: float,
    lineage: str,
    mode_budget: str = DEFAULT_MODE_BUDGET,
    board_top: float = DEFAULT_BOARD_TOP,
) -> dict[str, Any]:
    """The flat provenance schema this repo's actual
    `docs/results/submission/PROVENANCE_own_geometrical.json` uses."""
    return {
        "problem": "GeometricalProblem",
        "score": score,
        "objective": objective,
        "feasibility": feasibility,
        "board_top_at_submission": board_top,
        "beats_board": score > board_top,
        "lineage": lineage,
        "mode_budget": mode_budget,
    }


def _real_evaluate_fn():
    from constellaration import problems
    from constellaration.geometry import surface_rz_fourier

    problem = problems.GeometricalProblem()

    def evaluate(boundary_dict: dict[str, Any]) -> tuple[float, float, float]:
        boundary = surface_rz_fourier.SurfaceRZFourier.model_validate_json(
            json.dumps(boundary_dict)
        )
        ev = problem.evaluate(boundary)
        return float(ev.score), float(ev.objective), float(ev.feasibility)

    return evaluate


def build_arg_parser():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--records",
        required=True,
        help="candidates.ndjson / trajectory.ndjson / result.ndjson -- one JSON object per line",
    )
    ap.add_argument("--boundary-field", default="boundary")
    ap.add_argument("--max-candidates", type=int, default=10)
    ap.add_argument("--out-boundary", default="results/submission/best_own_geometrical.json")
    ap.add_argument(
        "--out-provenance", default="results/submission/PROVENANCE_own_geometrical.json"
    )
    ap.add_argument(
        "--lineage",
        required=True,
        help="one-line lineage description, e.g. 'own -- rotating-ellipse analytic "
        "seed, random_seed 20260907, NO leader boundary'",
    )
    ap.add_argument("--mode-budget", default=DEFAULT_MODE_BUDGET)
    ap.add_argument("--board-top", type=float, default=DEFAULT_BOARD_TOP)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    records = read_records(Path(args.records))
    evaluate_fn = _real_evaluate_fn()

    found = find_best_feasible(
        records,
        evaluate_fn,
        boundary_field=args.boundary_field,
        max_candidates=args.max_candidates,
    )
    if found is None:
        raise SystemExit(
            f"no feasible candidate among the first {args.max_candidates} "
            f"(best-objective-first) records in {args.records}"
        )
    rank, boundary_dict, evaluation = found

    out_boundary = Path(args.out_boundary)
    out_boundary.parent.mkdir(parents=True, exist_ok=True)
    out_boundary.write_text(json.dumps(boundary_dict, indent=2))
    print(f"PACKAGED rank {rank}: wrote {out_boundary}", flush=True)

    # Prove the SAVED FILE scores as expected, not the in-memory boundary.
    score, objective, feasibility = verify_saved_file(out_boundary, evaluate_fn)
    provenance = build_provenance(
        score=score,
        objective=objective,
        feasibility=feasibility,
        lineage=args.lineage,
        mode_budget=args.mode_budget,
        board_top=args.board_top,
    )
    out_provenance = Path(args.out_provenance)
    out_provenance.write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2), flush=True)
    print(f"wrote {out_provenance}", flush=True)


if __name__ == "__main__":
    main()
