"""E18 item 3 -- recompute the V-vs-evaluations curve from a run's own
archive.jsonl after the fact, so a run that predates this tooling (e.g.
results/p3-e17-nsga2) gets one too, and tabulate it for the write-up.

Pure/no VMEC: `p3_front_from_archive.v_curve_from_rows` reads only the
residuals ('r') already written to every archive row -- the same field
`b3_nsga2_mhd.py`'s seeded runner now logs `best_v_so_far`/`n_feasible_so_far`
from per generation, and REPORT.json's own `v_curve` from at the end of a
run. This script is the offline counterpart of that same computation, not a
reimplementation of it.

Usage:
    python3 scripts/p3_curve_from_archive.py results/p3-e17-nsga2 \\
        --out results/p3-e17-nsga2/v_curve.json --every 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from constellaration_methods.fronts.p3_front_from_archive import v_curve_from_rows


def load_archive_rows(archive_dir) -> list:
    """Every row of `archive_dir/archive.jsonl`, in file order."""
    path = Path(archive_dir) / "archive.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def render_table(curve: list) -> str:
    """A compact, human-readable table of `[evaluations, best_v_so_far]`
    rows for the write-up; `None` (no convergent evaluation yet) prints as
    'n/a' rather than the string 'None'."""
    lines = ["evaluations | best_v_so_far", "------------|---------------"]
    for evaluations, best_v in curve:
        v_str = "n/a" if best_v is None else f"{best_v:.6f}"
        lines.append(f"{evaluations:>11} | {v_str}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive_dir", help="e.g. results/p3-e17-nsga2")
    ap.add_argument("--out", required=True, help="output JSON file (the curve)")
    ap.add_argument("--every", type=int, default=100)
    return ap


def main(argv=None) -> list:
    args = build_parser().parse_args(argv)
    rows = load_archive_rows(args.archive_dir)
    curve = v_curve_from_rows(rows, every=args.every)
    Path(args.out).write_text(json.dumps(curve, indent=2))
    print(render_table(curve))
    return curve


if __name__ == "__main__":
    main()
