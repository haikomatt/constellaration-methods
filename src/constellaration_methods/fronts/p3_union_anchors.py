#!/usr/bin/env python3
"""E25: merge verified own-lineage
feasible points into ONE E15-format anchors REPORT for `b3_nsga2_mhd.py --anchors`.

Inputs are restoration-runner candidate files (x, r, raw, mpol/ntor/nfp/scaling) --
the E22/E22b/E22c feasible candidates and any E24r one -- all in one representation
(enforced: mpol, ntor, nfp, scaling must agree; a mismatch is a hard error, never a
silent mix). Output members carry tag, x, archived_r/confirmed_r (the candidate's r),
confirmed_aspect_ratio / confirmed_min_len (from raw), retained=True, strict_nominal.

    python scripts/p3_union_anchors.py --candidate A.json B.json ... --out REPORT.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _default_solo_hv(aspect: float, min_len: float) -> float:
    from constellaration_methods.fronts.p3_front_from_archive import hypervolume

    return hypervolume([{"aspect_ratio": aspect, "min_len": min_len}])


def fixed_major_radius(boundary_dict: dict) -> float:
    """r_cos(m=0, n=0): the one boundary entry outside the engine's mask, so it is part of
    the representation, not of x. Two anchors with different values are the same x drawn
    as different shapes (E25, 2026-10-26)."""
    r_cos = boundary_dict["r_cos"]
    return float(r_cos[0][(len(r_cos[0]) - 1) // 2])


MAJOR_RADIUS_TOL = 1e-12


def union_anchors(candidates: list[dict], solo_hv_fn=None) -> dict:
    """`solo_hv_fn(aspect, min_len) -> float` is injectable for tests; the
    default is the library primitive via `p3_front_from_archive.hypervolume`
    (the builder's `load_anchors` reads `solo_hypervolume`; its absence killed
    the first E25 launch, 2026-10-03)."""
    solo_hv_fn = solo_hv_fn or _default_solo_hv
    reps = {
        (int(c["mpol"]), int(c["ntor"]), int(c["nfp"]), float(c.get("scaling", 1.0)))
        for c in candidates
    }
    if len(reps) != 1:
        raise ValueError(f"anchors span more than one representation: {sorted(reps)}")
    radii = {
        (c.get("tag") or c.get("note", "?")): fixed_major_radius(c["boundary_dict"])
        for c in candidates
        if c.get("boundary_dict")
    }
    if radii and max(radii.values()) - min(radii.values()) > MAJOR_RADIUS_TOL:
        raise ValueError(
            "anchors differ in the fixed major radius r_cos(0,0), so they are not one "
            f"representation: {radii}"
        )
    members = []
    for c in candidates:
        if not c.get("raw"):
            raise ValueError(f"candidate {c.get('note', '?')!r} has no raw metrics")
        r = [float(v) for v in c["r"]]
        members.append(
            {
                "tag": c.get("tag") or c.get("note", "anchor")[:80],
                "x": [float(v) for v in c["x"]],
                "archived_r": r,
                "confirmed_r": r,
                "archived_V": max(r),
                "confirmed_V": max(r),
                "archived_aspect_ratio": float(c["raw"]["aspect_ratio"]),
                "archived_min_len": float(c["raw"]["min_len"]),
                "confirmed_aspect_ratio": float(c["raw"]["aspect_ratio"]),
                "confirmed_min_len": float(c["raw"]["min_len"]),
                "retained": True,
                "solo_hypervolume": float(
                    solo_hv_fn(float(c["raw"]["aspect_ratio"]), float(c["raw"]["min_len"]))
                ),
                "strict_nominal": bool(max(r) <= 0.0),
                "feasible": bool(max(r) <= 0.01),
            }
        )
    (mpol, ntor, nfp, scaling) = next(iter(reps))
    return {
        "n_candidates": len(members),
        "n_nondominated": len(members),
        "n_retained": len(members),
        "members": members,
        "confirmed_front_tags": [m["tag"] for m in members],
        "representation": {"mpol": mpol, "ntor": ntor, "nfp": nfp, "scaling": scaling},
        "note": (
            "union of verified own-lineage feasible candidates (E25 anchors); "
            "hypervolume not computed here"
        ),
    }


def _default_boundary_fn(cand: dict) -> dict:
    from constellaration_methods.restoration.p3_promote_candidate import boundary_dict_of_candidate

    return boundary_dict_of_candidate(cand)


def write_anchor_boundaries(candidates: list[dict], out_dir: Path, boundary_fn=None) -> list[Path]:
    """Write each candidate's leaderboard-format boundary to `<safe tag>.boundary.json` in
    `out_dir`, the name and place `p3_front_builder.load_anchor_x` reads (missing files
    killed the E25 re-launch, 2026-10-22). An existing file with a different boundary is
    a hard error, never silently replaced. `boundary_fn` is injectable for tests."""
    from constellaration_methods.fronts.p3_front_from_archive import _safe_name

    boundary_fn = boundary_fn or _default_boundary_fn
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for c in candidates:
        tag = c.get("tag") or c.get("note", "anchor")[:80]
        path = out_dir / f"{_safe_name(tag)}.boundary.json"
        text = json.dumps(boundary_fn(c), indent=1)
        if path.exists() and path.read_text() != text:
            raise ValueError(f"{path} exists and differs from the boundary of {tag!r}")
        path.write_text(text)
        written.append(path)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--candidate", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cands = []
    for p in a.candidate:
        c = json.loads(p.read_text())
        c.setdefault("tag", f"anchor:{p.parent.name}/{p.stem}")
        cands.append(c)
    rep = union_anchors(cands)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1))
    paths = write_anchor_boundaries(cands, a.out.parent)
    print(f"wrote {len(paths)} anchor boundaries beside the report")
    print(f"wrote {a.out}: {rep['n_retained']} anchors")


if __name__ == "__main__":
    main()
