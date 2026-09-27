#!/usr/bin/env python3
"""CI-style check: fail if any file under docs/results/p3/ contains a
Fourier boundary array.

The corresponding stripper is `tools/strip_boundaries.py` -- this script
verifies its own invariant held for everything actually committed, so a
P3 artefact copied in without going through the stripper (or a stripper
regression) fails the build loudly instead of silently leaking a boundary.

Checks every `*.json` file under the given root (default docs/results/p3/)
for any of the banned keys, at any nesting depth, via the same recursive
walk `strip_boundaries.strip_boundaries` uses. Exits non-zero and prints
every offending file/key pair if any are found.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from strip_boundaries import BOUNDARY_KEYS

# Explicit allowlist: paths (relative to the check root, forward-slash separated
# directory prefixes) where boundary-coordinate keys are EXPECTED and permitted.
# Every entry here is an own-lineage, clean-provenance P3 design (near-axis QI
# database seed, CC BY 4.0, no leaderboard boundary anywhere in its lineage --
# see PROVENANCE.md) that is submittable at Matt's discretion, unlike every
# davidkh-lineage P3 artefact elsewhere under docs/results/p3/, which stays
# stripped. Do not add a path here for a davidkh-derived result.
BOUNDARY_ALLOWED_PREFIXES = (
    "E22/own-lineage/",  # E22: first own-lineage P3-feasible boundary + its reflection
    "E23.0/own-lineage/",  # E23.0: the E22-archive baseline export (same lineage)
    "E22b/own-lineage/",  # E22b: repeatability link 1/2, own-lineage boundary + reflection
    "E22c/own-lineage/",  # E22c: repeatability link 2/2, own-lineage boundary + reflection
    "E23/own-lineage/",  # E23: the 53-member own-lineage front, confirmed hypervolume 89.71
)


def is_allowlisted(path: Path, root: Path) -> bool:
    """True if `path` (a file under `root`) falls under one of the explicit
    own-lineage exemption prefixes in BOUNDARY_ALLOWED_PREFIXES."""
    rel = path.relative_to(root).as_posix()
    return any(rel.startswith(prefix) for prefix in BOUNDARY_ALLOWED_PREFIXES)


def find_boundary_keys(obj: Any, path: str = "$") -> list[str]:
    """Returns the JSON-path-ish locations of every banned key found in
    `obj`, recursively. Empty list means clean."""
    hits: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f"{path}.{key}"
            if key in BOUNDARY_KEYS:
                hits.append(here)
            hits.extend(find_boundary_keys(value, here))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            hits.extend(find_boundary_keys(item, f"{path}[{i}]"))
    return hits


def check_tree(root: Path) -> dict[str, list[str]]:
    """Returns {file_path: [offending JSON paths]} for every JSON file under
    `root` that contains a banned key. Empty dict means the whole tree is
    clean. A missing `root` is treated as clean (nothing to check), not an
    error -- a repo checkout may legitimately have no P3 results yet."""
    offenders: dict[str, list[str]] = {}
    if not root.exists():
        return offenders
    for path in sorted(root.rglob("*.json")):
        if is_allowlisted(path, root):
            continue
        data = json.loads(path.read_text())
        hits = find_boundary_keys(data)
        if hits:
            offenders[str(path)] = hits
    return offenders


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("docs/results/p3"))
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    offenders = check_tree(args.root)
    if not offenders:
        print(f"OK: no boundary-coordinate keys found under {args.root}")
        return
    for path, hits in offenders.items():
        print(f"BOUNDARY LEAK: {path}")
        for hit in hits:
            print(f"    {hit}")
    raise SystemExit(f"{len(offenders)} file(s) under {args.root} contain boundary coordinates")


if __name__ == "__main__":
    main()
