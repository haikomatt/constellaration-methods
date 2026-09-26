#!/usr/bin/env python3
"""Deep-margin-first canonical confirmation of candidates persisted by
`engines.alm_ngopt_optimizer.CandidateTracker` to a run's `candidates.ndjson`.

An overnight optimizer run typically canonically-checks only the single
strict-best-objective point, and that point can flip to
`canonical_non_convergent` under high-fidelity re-solve. But a large
fraction of feasible calls along the way can sit at near-zero constraint
violation and never get coefficient-logged, so a single flipped verdict is
about one arbitrary tiebreak pick, not about whether deep-margin
feasibility survives high fidelity. This script answers that question
directly: rank `candidates.ndjson` deep-margin-first, canonically confirm
the top-K, and write verdicts (with the real exception text on failure,
not just a boolean) to `confirmations.ndjson`.

Reuses `evaluation.rescore_trajectory.rescore_step` for the actual
canonical high-fidelity re-solve -- do not reinvent that check; canonical
confirmation is mandatory before anything is called feasible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from constellaration import forward_model, problems

from constellaration_methods.evaluation import rescore_trajectory


def read_candidates(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def select_top_k(
    candidates: list[dict[str, Any]],
    k: int = 10,
    objective_band: float = 0.05,
) -> list[dict[str, Any]]:
    """Deep-margin candidates first (`constraint_violation` ascending, the
    top `k`), plus any candidate within `objective_band` of the best
    objective found (surfaces near-optimal points even if not deep-margin),
    plus the single strict-best-objective candidate always included so the
    verdict is directly comparable to whichever one candidate a plain
    overnight run happened to check. Deduped by `call_index`; result may
    exceed `k` once the always-included extras are folded in -- that's
    intentional, not a bug.
    """
    if not candidates:
        return []

    by_violation = sorted(candidates, key=lambda c: c["constraint_violation"])
    best_objective_candidate = min(candidates, key=lambda c: c["objective"])
    best_objective = best_objective_candidate["objective"]

    selected: dict[Any, dict[str, Any]] = {}
    for c in by_violation[:k]:
        selected[c["call_index"]] = c
    for c in candidates:
        if c["objective"] <= best_objective + objective_band:
            selected[c["call_index"]] = c
    selected[best_objective_candidate["call_index"]] = best_objective_candidate

    return sorted(selected.values(), key=lambda c: c["constraint_violation"])


def select_band_spread(
    candidates: list[dict[str, Any]],
    band: tuple[float, float],
    n: int,
) -> list[dict[str, Any]]:
    """Evenly-spaced spread of `n` candidates across the objective band
    `(lo, hi)` (exclusive), deepest margin first among equal objectives.
    Spreading by objective (rather than top-k by margin) localizes a
    convergent wall inside a band with no canonical verdicts yet, where
    coverage of the band beats depth at one end.
    """
    lo, hi = band
    in_band = sorted(
        (c for c in candidates if lo < c["objective"] < hi),
        key=lambda c: (c["objective"], c["constraint_violation"]),
    )
    if len(in_band) <= n:
        return in_band
    idxs = sorted(set(np.linspace(0, len(in_band) - 1, n).round().astype(int).tolist()))
    return [in_band[i] for i in idxs]


def select_call_indices(
    candidates: list[dict[str, Any]], call_indices: list[int]
) -> list[dict[str, Any]]:
    """Explicit worklist selection -- the same shape `--print-selection`
    emits (a plain list of `call_index` values), so a discovery loop's
    elite-selection phase can hand this script a worklist directly instead
    of re-deriving a selection with `select_top_k`/`select_band_spread`.
    Order follows `call_indices`; indices absent from `candidates` are
    silently skipped rather than raising, since a worklist may be replayed
    against a `candidates` file that has since been re-merged/renumbered."""
    by_index = {c["call_index"]: c for c in candidates}
    return [by_index[i] for i in call_indices if i in by_index]


def confirm_candidates(
    candidates_path: Path,
    out_path: Path,
    k: int = 10,
    objective_band: float = 0.05,
    problem: problems.GeometricalProblem | None = None,
    only_call_index: int | None = None,
    call_indices: list[int] | None = None,
) -> dict[str, Any]:
    problem = problem or problems.GeometricalProblem()
    # Match `problems.SingleObjectiveProblem.evaluate()`'s own convention:
    # `default_high_fidelity_skip_qi()` when the problem doesn't need QI
    # (true for `GeometricalProblem`).
    needs_qi = getattr(problem, "_does_it_require_qi", True)
    high_fidelity_settings = (
        forward_model.ConstellarationSettings.default_high_fidelity()
        if needs_qi
        else forward_model.ConstellarationSettings.default_high_fidelity_skip_qi()
    )

    candidates = read_candidates(candidates_path)
    if only_call_index is not None:
        selected = [c for c in candidates if c["call_index"] == only_call_index]
    elif call_indices is not None:
        selected = select_call_indices(candidates, call_indices)
    else:
        selected = select_top_k(candidates, k=k, objective_band=objective_band)

    n_confirmed_feasible = 0
    n_flipped_non_convergent = 0
    n_still_infeasible = 0
    with open(out_path, "w") as f:
        for candidate in selected:
            verdict = rescore_trajectory.rescore_step(candidate, problem, high_fidelity_settings)
            f.write(json.dumps(verdict) + "\n")
            if verdict["canonical_non_convergent"]:
                n_flipped_non_convergent += 1
            elif verdict["canonical_feasible"]:
                n_confirmed_feasible += 1
            else:
                n_still_infeasible += 1

    return {
        "n_candidates_considered": len(candidates),
        "n_confirmed": len(selected),
        "n_confirmed_feasible": n_confirmed_feasible,
        "n_flipped_non_convergent": n_flipped_non_convergent,
        "n_still_infeasible": n_still_infeasible,
        "out_path": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="path to a candidates.ndjson written by engines.alm_ngopt_optimizer.CandidateTracker",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="defaults to confirmations.ndjson next to --candidates",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--objective-band",
        type=float,
        default=0.05,
        help="also confirm any candidate within this much of the best "
        "objective found, even if not deep-margin",
    )
    parser.add_argument(
        "--only",
        type=int,
        default=None,
        metavar="CALL_INDEX",
        help="confirm exactly this one candidate (for a parallel per-candidate driver)",
    )
    parser.add_argument(
        "--band",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="with --spread: select an even spread across this objective "
        "band instead of top-k (gap-band wall-localization sweep)",
    )
    parser.add_argument(
        "--spread",
        type=int,
        default=48,
        help="number of candidates to spread-select across --band",
    )
    parser.add_argument(
        "--print-selection",
        action="store_true",
        help="print the selected call_index list (one per line) and exit "
        "without confirming anything -- a parallel driver uses this to "
        "build its worklist",
    )
    parser.add_argument(
        "--call-indices-file",
        type=Path,
        default=None,
        help="confirm exactly this explicit worklist of call_index values "
        "(one per line, same shape --print-selection emits) instead of "
        "computing a top-k/band selection",
    )
    args = parser.parse_args()
    out_path = args.out or args.candidates.parent / "confirmations.ndjson"

    if args.print_selection:
        candidates = read_candidates(args.candidates)
        if args.band is not None:
            selected = select_band_spread(candidates, tuple(args.band), args.spread)
        else:
            selected = select_top_k(candidates, k=args.k, objective_band=args.objective_band)
        for c in selected:
            print(c["call_index"])
        return

    call_indices = None
    if args.call_indices_file is not None:
        text = args.call_indices_file.read_text()
        call_indices = [int(line) for line in text.splitlines() if line.strip()]

    summary = confirm_candidates(
        args.candidates,
        out_path,
        k=args.k,
        objective_band=args.objective_band,
        only_call_index=args.only,
        call_indices=call_indices,
    )
    print(json.dumps(summary, indent=2))
    print(
        "\nDo any deep-margin (violation ~= 0) candidates come back "
        "canonical_feasible=True? Check "
        f"{out_path} directly -- ranked deep-margin-first."
    )


if __name__ == "__main__":
    main()
