"""Package our best canonically-feasible design as a leaderboard submission
for SimpleToBuildQIStellarator, in the EXACT format the winner target files
use (SurfaceRZFourier JSON: r_cos/z_sin/r_sin/z_cos/n_field_periods/
is_stellarator_symmetric), and PROVE the saved file re-scores to the expected
score by reloading it from disk and running the leaderboard's own
`problem.evaluate(...).score`. Container-side (needs constellaration).

This repo's own campaign's best OWN-lineage canonically-feasible design was
the c8 climb bank's call 30548 (objective 11.6433 -> score 0.5822, canonical
margin 0.0072, 8x8 NFP=3): the climb-lifted design (4x4 -> 8x8, higher-mode
content), beating an earlier 4x4 bank best (11.124 / 0.5562). See
docs/results/submission/PROVENANCE.json for the record this script produced.
This repo does not ship the working `c8_climb_bank.jsonl` (campaign working
data, not a published result) -- `--bank`/`--call-index` let a caller re-run
this against their own bank.

Usage (inside the repo's own pinned image):
    podman run --rm -v "$(pwd)":/workspace:Z -w /workspace \\
        localhost/constellaration-methods:test \\
        python -m constellaration_methods.evaluation.package_submission
"""

from __future__ import annotations

import json
from pathlib import Path

from constellaration_methods.engines.alm_ngopt_optimizer import prepare_initial_boundary

DEFAULT_BANK = Path("results") / "family-study" / "climb" / "c8_climb_bank.jsonl"
DEFAULT_CALL_INDEX = 30548
DEFAULT_EXPECTED_SCORE = 0.5822
DEFAULT_OUT_DIR = Path("results") / "submission"


def build_arg_parser():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", default=str(DEFAULT_BANK))
    ap.add_argument("--call-index", type=int, default=DEFAULT_CALL_INDEX)
    ap.add_argument("--expected-score", type=float, default=DEFAULT_EXPECTED_SCORE)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    bank = Path(args.bank)
    out_dir = Path(args.out_dir)

    import numpy as np
    from constellaration import problems
    from constellaration.geometry import surface_rz_fourier

    rec = next(
        json.loads(line)
        for line in open(bank)
        if line.strip() and json.loads(line)["call_index"] == args.call_index
    )
    bd = rec["boundary"]
    mmax = int(rec.get("max_poloidal_mode", np.asarray(bd["r_cos"]).shape[0] - 1))
    nmax = int(rec.get("max_toroidal_mode", (np.asarray(bd["r_cos"]).shape[1] - 1) // 2))
    nfp = int(rec["n_field_periods"])
    surface = prepare_initial_boundary(
        {"r_cos": bd["r_cos"], "z_sin": bd["z_sin"], "n_field_periods": nfp},
        mmax,
        nmax,
        nfp,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    sub_path = out_dir / "best_feasible_simple_to_build.json"
    sub_path.write_text(surface.model_dump_json())

    # Prove the SAVED FILE scores as expected (reload from disk, do not reuse
    # the in-memory object).
    problem = problems.SimpleToBuildQIStellarator()
    reloaded = surface_rz_fourier.SurfaceRZFourier.model_validate_json(sub_path.read_text())
    ev = problem.evaluate(reloaded)
    score = float(ev.score)
    objective = float(ev.objective)
    feasibility = float(ev.feasibility)
    feasible = bool(problem.is_feasible(ev.metrics)) if hasattr(ev, "metrics") else None
    ok = abs(score - args.expected_score) < 5e-4

    provenance = {
        "problem": "SimpleToBuildQIStellarator",
        "submission_file": str(sub_path),
        "source": {
            "bank": str(bank),
            "call_index": args.call_index,
            "random_seed": rec.get("random_seed"),
            "mode_budget": f"{mmax}x{nmax}",
            "n_field_periods": nfp,
        },
        "verified_on_saved_file": {
            "score": score,
            "objective": objective,
            "feasibility_violation": feasibility,
            "feasible": feasible,
            "expected_score": args.expected_score,
            "matches_expected": ok,
        },
        "leaderboard_context": {
            "current_top_score": 0.6361,
            "current_bottom_top8_score": 0.5371,
            "note": "clears the current bottom-of-top-8 bar on the on-disk "
            "snapshot; score = objective/20 for feasible designs",
        },
    }
    (out_dir / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2))

    print(json.dumps(provenance["verified_on_saved_file"], indent=2), flush=True)
    print(
        f"\n{'OK' if ok else 'MISMATCH'}: saved {sub_path} scores {score:.4f} "
        f"(expected {args.expected_score})",
        flush=True,
    )
    if not ok:
        raise SystemExit("submission score does not match expected -- do NOT submit")


if __name__ == "__main__":
    main()
