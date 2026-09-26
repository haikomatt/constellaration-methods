#!/usr/bin/env python3
"""Strip boundary-coordinate fields out of a P3 result artefact.

Every P3 (MHDStableQIStellarator) result this repo publishes descends from
a public leaderboard entrant's boundary (see PROVENANCE.md) and is measurement
and method only -- no P3 boundary coordinates are redistributed. This script
recursively removes the keys that carry Fourier boundary coefficients (`x`,
`boundary`, `boundary_dict`, `r_cos`, `z_sin`, `r_sin`, `z_cos`) from any
nested JSON structure -- dicts, lists of dicts, lists of lists -- leaving
every other field (residuals, hypervolumes, tags, counts, timings) intact.

Used to copy P3 REPORT.json/curve.json files into docs/results/p3/: never
copy one of those files directly, always pipe it through this script first.
`tools/check_no_boundaries.py` is the corresponding CI-style check that a
copy was actually stripped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

BOUNDARY_KEYS = frozenset({"x", "boundary", "boundary_dict", "r_cos", "z_sin", "r_sin", "z_cos"})


def strip_boundaries(obj: Any) -> Any:
    """Returns a deep copy of `obj` with every dict key in `BOUNDARY_KEYS`
    removed, recursively, at any nesting depth (dicts nested in lists nested
    in dicts, etc.)."""
    if isinstance(obj, dict):
        return {
            key: strip_boundaries(value) for key, value in obj.items() if key not in BOUNDARY_KEYS
        }
    if isinstance(obj, list):
        return [strip_boundaries(item) for item in obj]
    return obj


def strip_file(src: Path, dst: Path) -> None:
    """Reads a JSON file, strips boundary keys, and writes the result to
    `dst` (creating parent directories as needed)."""
    data = json.loads(src.read_text())
    stripped = strip_boundaries(data)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(stripped, indent=2) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path, help="source JSON file (e.g. a P3 REPORT.json)")
    ap.add_argument("dst", type=Path, help="destination path for the stripped copy")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    strip_file(args.src, args.dst)
    print(f"wrote {args.dst} (boundary keys stripped)")


if __name__ == "__main__":
    main()
