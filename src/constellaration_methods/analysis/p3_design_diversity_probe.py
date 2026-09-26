"""P3 design-space diversity probe (D1) -- DRAFT measurement, not locked
(vault `p3-design-diversity-probe-2026-09-23.md`). Answers Proxima's open
question: do distinct regions of design space map to similar metric
values ("near-equivalent" designs), or is the benchmark-feasible region
rigid? A measurement only -- no labels are pulled from it, nothing is
optimised, and the reading it prints is explicitly PROVISIONAL because
the registration has not been snapshotted/locked.

Algorithms live in `constellaration_methods.analysis.boundary_diversity`
(host-testable, pure numpy); this module is the CLI: load the three data
sources onto one comparable representation, canonicalise, run the
registration's Method steps, and write the report.

**idx -> boundary rule** (Phase 0 evidence, `results/atlas-probe/`):
`boundaries_table.csv` has one row per (slug, idx); `bundles/<slug>.json`
is a flat JSON list, and idx is a 0-based index into THAT list, each
element itself a JSON-encoded string (not a dict) of a leaderboard-format
boundary (r_cos, z_sin, r_sin=null, z_cos=null, n_field_periods,
is_stellarator_symmetric). Confirmed against `scripts/atlas_probe_analyze
.load_bundle_pair`, which uses the identical `bundle_data[i]` + conditional
`json.loads` pattern -- reused here rather than re-derived. The CSV's
`L_grad_B` column and the archive/board `min_len` field name the SAME
physical quantity (`minimum_normalized_magnetic_gradient_scale_length`,
confirmed against `atlas_probe_analyze.METRIC_KEY_MAP`), just spelled
differently across scripts in this repo.

**Board files' `feasibilities`** (`results/p3-close-out/board/*.json`,
copied from the scratchpad so this script does not depend on /tmp): a
per-boundary list, one float each -- a per-member violation scalar (0.0 =
exactly on the constraint boundary, > 0 = infeasible by that margin), NOT
the five-metric residual vector used elsewhere in this repo. The contest
itself scores a member feasible when this violation is within its own
benchmark tolerance, `BENCHMARK_TOL = 0.01` (evidence: davidkh's file
(2026-07-10) has overall `feasibility` 0.0087 and `score` 187.9 with all
200 boundaries scored by the leaderboard, and its per-boundary
`feasibilities` are all <= 0.0098, i.e. within tolerance though none is
exactly 0.0; canyon-ant's file (2026-09-09) has overall `feasibility`
0.0029 with all 43 boundaries scored, six of them exactly 0.0 and the
rest up to 0.0085, again all within tolerance). An earlier version of this
loader required `feasibilities[i] <= 0` exactly, which is wrong -- it
zeroed out davidkh entirely (0/200 "feasible") and dropped 37 of
canyon-ant's 43 boundaries, losing most of the non-phanerozoic data for
the cross-lineage statistics. This script now treats
`feasibilities[i] <= BENCHMARK_TOL` as feasible, matching the contest's
own scoring rule (regression-ratchet test:
`tests/analysis/test_p3_design_diversity_probe.py
::test_load_board_feasibility_is_within_benchmark_tolerance_not_only_zero`).

**Locked run parameters** (the registered run; see `parse_args` for the
per-flag rationale): `--grid 32 --max-cell-members 60 --objective-tol 0.05
--n-random 10000 --cell 0.1 --seed 0`. These are now the CLI defaults, so
the registered run is reproducible from the command line below with no
tuning flags at all -- every parameter value, plus this script's git
commit, is also recorded in the written REPORT.json.

Usage (host, if numpy/scipy/matplotlib are all it needs -- true for
--atlas/--board; --ours needs jax + `alm_ngopt_optimizer`, only present in
the pinned image):
    python3 -m constellaration_methods.analysis.p3_design_diversity_probe \\
        --atlas results/atlas-probe --board results/p3-close-out/board \\
        --ours results/p3-e17-nsga2 \\
        --ours-representation results/p3-e13-davidkh-7x7/start_serialised.json \\
        --out results/p3-diversity-d1

Inside the pinned image (cores 14-15 ONLY -- this is a CPU-only measurement
probe, no VMEC, never touches cores 2-13):
    podman run --rm \\
        -v "$(pwd)":/workspace:Z -w /workspace \\
        -v "<repo>/results":"<repo>/results":Z \\
        localhost/constellaration-methods:test \\
        taskset -c 14,15 python -m constellaration_methods.analysis.p3_design_diversity_probe ...
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from constellaration_methods.analysis import boundary_diversity as bd
from constellaration_methods.engines.boundary_grid import effective_modes

# This module now lives at src/constellaration_methods/analysis/, so its
# own repo root is four levels up (analysis -> constellaration_methods ->
# src -> repo root) -- kept only as the default for `_git_commit` below,
# not to feed a sys.path hack (this package is imported normally).
REPO = Path(__file__).resolve().parents[3]

# --------------------------------------------------------------------------- #
# feasibility formulae (registration step 3; identical to results/p3-e17-
# nsga2/archive.jsonl's own "r" vectors -- checked numerically in Phase 0)
# --------------------------------------------------------------------------- #

BENCHMARK_TOL = 0.01

BOARD_FEASIBILITY_RULE = (
    "A board-file boundary is feasible iff feasibilities[i] <= BENCHMARK_TOL "
    f"({BENCHMARK_TOL}), matching the contest's own scoring tolerance -- not "
    "only feasibilities[i] == 0.0 (see module docstring)."
)


def _resolve_ref(gitdir: Path, commondir: Path, ref: str) -> str | None:
    """`ref` (e.g. `refs/heads/main`) -> commit SHA, checking the loose ref
    file first (in both `gitdir` and `commondir` -- worktree-private refs
    like `refs/bisect` live under `gitdir`, but ordinary branch refs live
    under the shared `commondir`), then `commondir/packed-refs`."""
    for base in (gitdir, commondir):
        try:
            return (base / ref).read_text().strip() or None
        except OSError:
            continue
    try:
        for line in (commondir / "packed-refs").read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref:
                return parts[0]
    except OSError:
        pass
    return None


def _git_commit(repo_root: Path = REPO) -> str:
    """The git commit of the script's own repo (HEAD), recorded in
    REPORT.json so the run is traceable to the exact code that produced
    it. Reads `.git` directly (stdlib file I/O) rather than shelling out
    to a `git` binary: the pinned `phase1` podman image this script runs
    in has NO `git` binary installed at all (confirmed against
    `src/qi_discovery.py`'s `_current_git_commit`, which documents the
    same finding for the orchestrator's manifest provenance field), so a
    `subprocess.run(["git", ...])` implementation would silently report
    'unknown' on every real run.

    Unlike `qi_discovery._current_git_commit`, this ALSO follows a git
    *worktree* checkout: this script's own repo (`REPO`) is normally a
    linked worktree (e.g. a linked git worktree), so
    `REPO/.git` is a one-line pointer file (`gitdir: <path>`), not a
    directory -- and that pointed-to per-worktree git dir has its own
    `HEAD` but shares `refs/` with the main checkout via its `commondir`
    file. Falls back to 'unknown' (never raises) for any shape this
    parsing doesn't recognise."""
    dotgit = repo_root / ".git"
    try:
        if dotgit.is_dir():
            gitdir = dotgit
        else:
            pointer = dotgit.read_text().strip()
            if not pointer.startswith("gitdir:"):
                return "unknown"
            gitdir = Path(pointer.split(":", 1)[1].strip())
        try:
            commondir = (gitdir / (gitdir / "commondir").read_text().strip()).resolve()
        except OSError:
            commondir = gitdir  # non-worktree repo: refs live in gitdir itself

        head = (gitdir / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head or "unknown"  # detached HEAD: the line itself is the SHA
        ref = head.split(":", 1)[1].strip()
        commit = _resolve_ref(gitdir, commondir, ref)
        return commit if commit is not None else "unknown"
    except OSError:
        return "unknown"


def atlas_residuals(
    iota_pp: float, qi: float, edge_mirror: float, flux_comp: float, vacuum_well: float
) -> list[float]:
    """The five normalised residuals, registration formulae. `qi` <= 0 is
    treated as an immediate fail (log10 undefined) rather than raising --
    Phase 0 found no such rows in the atlas table, but a loader must not
    crash on a future dataset that has one."""
    log10_qi = math.log10(qi) if qi > 0 else math.inf
    return [
        (0.25 - abs(iota_pp)) / 0.25,
        (log10_qi + 3.5) / 3.5,
        (edge_mirror - 0.25) / 0.25,
        (flux_comp - 0.9) / 0.9,
        -vacuum_well / 0.1,
    ]


# --------------------------------------------------------------------------- #
# record loading: one common shape across all three sources
# --------------------------------------------------------------------------- #


def load_atlas(atlas_dir: Path) -> list[dict[str, Any]]:
    """`boundaries_table.csv` + `bundles/<slug>.json` -- see module
    docstring for the idx -> boundary rule."""
    csv_path = atlas_dir / "boundaries_table.csv"
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    by_slug: dict[str, list[dict]] = {}
    for row in rows:
        by_slug.setdefault(row["slug"], []).append(row)

    records = []
    for slug, slug_rows in by_slug.items():
        bundle = json.loads((atlas_dir / "bundles" / f"{slug}.json").read_text())
        for row in slug_rows:
            idx = int(row["idx"])
            entry = bundle[idx]
            boundary = json.loads(entry) if isinstance(entry, str) else entry
            residuals = atlas_residuals(
                float(row["iota_pp"]),
                float(row["qi"]),
                float(row["edge_mirror"]),
                float(row["flux_comp"]),
                float(row["vacuum_well"]),
            )
            records.append(
                {
                    "source": "phanerozoic",
                    "id": f"phanerozoic:{slug}:{idx}",
                    "nfp": int(boundary["n_field_periods"]),
                    "r_cos": np.asarray(boundary["r_cos"], dtype=float),
                    "z_sin": np.asarray(boundary["z_sin"], dtype=float),
                    "aspect_ratio": float(row["aspect_ratio"]),
                    "min_len": float(row["L_grad_B"]),
                    "feasible": max(residuals) <= BENCHMARK_TOL,
                    "feasible_reason": "table_residuals",
                }
            )
    return records


def load_board(board_dir: Path) -> list[dict[str, Any]]:
    """One record per boundary in each `*_mhd_stable_results.json` file
    under `board_dir` -- see module docstring for the `feasibilities`
    convention."""
    records = []
    for path in sorted(board_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        user = payload["user"]
        boundaries = json.loads(payload["boundary_json"])
        objectives = json.loads(payload["objectives"])
        feasibilities = json.loads(payload["feasibilities"]) if "feasibilities" in payload else None
        for i, (b_str, obj) in enumerate(zip(boundaries, objectives)):
            boundary = json.loads(b_str)
            aspect_ratio = float(obj[0][0])
            min_len = float(obj[1][0])
            if feasibilities is not None:
                feasible = bool(feasibilities[i] <= BENCHMARK_TOL)
                reason = "file_feasibilities"
            else:
                feasible = None
                reason = "unknown_excluded"
            records.append(
                {
                    "source": user,
                    "id": f"{user}:{i}",
                    "nfp": int(boundary["n_field_periods"]),
                    "r_cos": np.asarray(boundary["r_cos"], dtype=float),
                    "z_sin": np.asarray(boundary["z_sin"], dtype=float),
                    "aspect_ratio": aspect_ratio,
                    "min_len": min_len,
                    "feasible": feasible,
                    "feasible_reason": reason,
                }
            )
    return records


def load_ours(ours_dir: Path, representation_path: Path) -> list[dict[str, Any]]:
    """`results/p3-e17-nsga2/archive.jsonl` rows, boundary rebuilt from
    `x` via the davidkh-7x7 representation
    (`p3_promote_candidate.boundary_dict_of_candidate`, the same function
    `p3_front_from_archive.py` uses) -- needs jax, so only feasible rows
    are rebuilt (the rest never enter the near-equivalent/clustering
    steps, which both require feasibility)."""
    from constellaration_methods.restoration.p3_promote_candidate import (  # local: jax-dependent
        boundary_dict_of_candidate,
    )

    representation = json.loads(representation_path.read_text())
    rep_base = dict(
        boundary_dict=representation["boundary_dict"],
        mpol=int(representation["mpol"]),
        ntor=int(representation["ntor"]),
        nfp=int(representation["nfp"]),
        scaling=float(representation.get("scaling", 1.0)),
    )
    rows = [
        json.loads(line)
        for line in (ours_dir / "archive.jsonl").read_text().splitlines()
        if line.strip()
    ]
    records = []
    for row in rows:
        if not row.get("feasible"):
            continue
        cand = dict(rep_base, x=row["x"])
        boundary = boundary_dict_of_candidate(cand)
        records.append(
            {
                "source": "ours",
                "id": f"ours:{row['tag']}",
                "nfp": int(boundary["n_field_periods"]),
                "r_cos": np.asarray(boundary["r_cos"], dtype=float),
                "z_sin": np.asarray(boundary["z_sin"], dtype=float),
                "aspect_ratio": float(row["raw"]["aspect_ratio"]),
                "min_len": float(row["raw"]["min_len"]),
                "feasible": True,
                "feasible_reason": "archive_max_r_le_tol",
            }
        )
    return records


# --------------------------------------------------------------------------- #
# canonicalisation pass: pad/normalise, minor radius, tail-energy proxy
# --------------------------------------------------------------------------- #


def canonicalize_records(records: list[dict[str, Any]], target_mpol: int, target_ntor: int) -> None:
    """In place: adds `r_pad`, `z_pad` (padded, R00-normalised),
    `minor_radius`, and `tail_energy_fraction` (registration step 6, proxy
    2 -- `boundary_grid.effective_modes`'s m>4-or-|n|>4 tail fraction,
    reused as-is) to every record."""
    for rec in records:
        r_pad, z_pad = bd.pad_and_normalize(rec["r_cos"], rec["z_sin"], target_mpol, target_ntor)
        rec["r_pad"] = r_pad
        rec["z_pad"] = z_pad
        rec["minor_radius"] = bd.minor_radius(r_pad, z_pad, rec["nfp"])
        _, _, tail_frac = effective_modes(r_pad, z_pad, tol=1e-12, m_cut=4, n_cut=4)
        rec["tail_energy_fraction"] = tail_frac


def compute_curvature(records: list[dict[str, Any]], n_theta: int, n_phi: int) -> None:
    """In place, feasible records only: adds `max_curvature` (registration
    step 6, proxy 1). Infeasible records never enter the near-equivalent
    or clustering steps, so their curvature is never used -- skip the
    grid-derivative work for them."""
    for rec in records:
        if not rec["feasible"]:
            continue
        theta, phi = bd.make_grid(rec["nfp"], n_theta, n_phi)
        fields = bd.evaluate_surface_with_derivatives(
            rec["r_pad"], rec["z_pad"], rec["nfp"], theta, phi
        )
        rec["max_curvature"] = bd.principal_curvature_max(fields, rec["minor_radius"])


# --------------------------------------------------------------------------- #
# pairwise physical distance -- shared by near-equivalent pairs, the
# reference distribution, and clustering
# --------------------------------------------------------------------------- #


def make_physical_distance_fn(records: list[dict[str, Any]], n_theta: int, n_phi: int):
    """A `distance_fn(i, j)` closure over global record indices; the
    (theta, phi) grid is built once per nfp, not once per pair."""
    grids: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def distance_fn(i: int, j: int) -> float:
        rec_a, rec_b = records[i], records[j]
        nfp = rec_a["nfp"]
        if nfp not in grids:
            grids[nfp] = bd.make_grid(nfp, n_theta, n_phi)
        theta, phi = grids[nfp]
        return bd.physical_distance(
            rec_a["r_pad"],
            rec_a["z_pad"],
            rec_b["r_pad"],
            rec_b["z_pad"],
            nfp,
            theta,
            phi,
            rec_a["minor_radius"],
            rec_b["minor_radius"],
        )

    return distance_fn


def lineage_split(source_a: str, source_b: str) -> str:
    """ "within" (both phanerozoic), "cross" (different sources), or
    "other" (same non-phanerozoic source, e.g. two "ours" rows) --
    registration: "split every statistic within-lineage (phanerozoic
    only) and cross-lineage (pairs whose members come from different
    sources)"."""
    if source_a == source_b == "phanerozoic":
        return "within"
    if source_a != source_b:
        return "cross"
    return "other"


# --------------------------------------------------------------------------- #
# statistics: near-equivalent pairs, reference distribution, reading
# --------------------------------------------------------------------------- #


def compute_pair_statistics(
    records: list[dict[str, Any]],
    *,
    objective_tol: float,
    max_per_cell: int,
    n_theta: int,
    n_phi: int,
    n_random: int,
    seed: int,
) -> dict[str, Any]:
    """Registration steps 3-5 (pairs, reference distribution, fraction
    statistic). Returns the report fragment plus the raw pair rows (for
    `pairs.csv`) and the reference-distance sample (for the histogram)."""
    aspect = np.array([r["aspect_ratio"] for r in records])
    min_len = np.array([r["min_len"] for r in records])
    feasible = np.array([bool(r["feasible"]) for r in records])
    nfp = np.array([r["nfp"] for r in records])
    distance_fn = make_physical_distance_fn(records, n_theta, n_phi)

    candidate_pairs = bd.near_equivalent_pairs(
        aspect, min_len, feasible, nfp, objective_tol, max_per_cell=max_per_cell
    )

    pair_rows = []
    for i, j in candidate_pairs:
        split = lineage_split(records[i]["source"], records[j]["source"])
        pair_rows.append(
            {
                "i": i,
                "j": j,
                "id_i": records[i]["id"],
                "id_j": records[j]["id"],
                "source_i": records[i]["source"],
                "source_j": records[j]["source"],
                "lineage_split": split,
                "d_aspect": abs(aspect[i] - aspect[j]),
                "d_min_len": abs(min_len[i] - min_len[j]),
                "physical_distance": distance_fn(i, j),
                "coefficient_distance": bd.coefficient_distance(
                    records[i]["r_pad"],
                    records[i]["z_pad"],
                    records[j]["r_pad"],
                    records[j]["z_pad"],
                ),
            }
        )

    rng = np.random.default_rng(seed)
    reference_pairs = bd.sample_reference_pairs(feasible, nfp, rng, n_random)
    reference_distances = np.array([distance_fn(i, j) for i, j in reference_pairs])
    d_rand = float(np.median(reference_distances)) if len(reference_distances) else None
    d_10 = float(np.percentile(reference_distances, 10)) if len(reference_distances) else None

    fractions = {}
    for split in ("within", "cross"):
        split_distances = [
            row["physical_distance"] for row in pair_rows if row["lineage_split"] == split
        ]
        n_pairs = len(split_distances)
        frac = (
            float(np.mean(np.array(split_distances) >= d_rand))
            if n_pairs and d_rand is not None
            else None
        )
        fractions[split] = {"n_pairs": n_pairs, "fraction_ge_d_rand": frac}

    reading = "Inconclusive"
    if fractions["within"]["fraction_ge_d_rand"] is not None:
        reading = bd.classify_reading(
            fractions["within"]["fraction_ge_d_rand"], fractions["within"]["n_pairs"]
        )

    return {
        "pair_rows": pair_rows,
        "reference_distances": reference_distances,
        "d_rand": d_rand,
        "d_10": d_10,
        "fractions": fractions,
        "reading": reading,
        "n_candidate_pairs": len(candidate_pairs),
        "n_reference_pairs": len(reference_pairs),
    }


# --------------------------------------------------------------------------- #
# clustering: per objective cell, within-lineage and cross-lineage
# --------------------------------------------------------------------------- #


def compute_clusters(
    records: list[dict[str, Any]],
    *,
    cell: float,
    max_per_cell: int,
    n_theta: int,
    n_phi: int,
    threshold: float,
) -> list[dict[str, Any]]:
    """Registration step 5's per-cell single-linkage clustering (cells
    with >= 10 feasible members) plus step 6's per-cluster curvature/
    tail-fraction spread. Two populations, per the registration's
    within/cross split:
    - within-lineage: phanerozoic-only records bucketed into cells.
    - cross-lineage: the full combined pool bucketed into cells, kept
      only where a cell mixes >= 2 distinct sources -- a same-source cell
      says nothing about cross-entrant agreement.
    Returns one row per (split, cell) with the cluster count and, for
    each cluster in that cell, the curvature/tail-fraction spread.
    """
    aspect = np.array([r["aspect_ratio"] for r in records])
    min_len = np.array([r["min_len"] for r in records])
    feasible = np.array([bool(r["feasible"]) for r in records])
    nfp = np.array([r["nfp"] for r in records])
    source = np.array([r["source"] for r in records], dtype=object)
    distance_fn = make_physical_distance_fn(records, n_theta, n_phi)

    phanerozoic_mask = feasible & (source == "phanerozoic")
    rows = []
    rows += _cluster_population(
        records,
        phanerozoic_mask,
        aspect,
        min_len,
        nfp,
        source,
        cell,
        max_per_cell,
        threshold,
        distance_fn,
        split="within",
    )
    rows += _cluster_population(
        records,
        feasible,
        aspect,
        min_len,
        nfp,
        source,
        cell,
        max_per_cell,
        threshold,
        distance_fn,
        split="cross",
        require_mixed_sources=True,
    )
    return rows


def _cluster_population(
    records,
    population_mask,
    aspect,
    min_len,
    nfp,
    source,
    cell,
    max_per_cell,
    threshold,
    distance_fn,
    *,
    split,
    require_mixed_sources=False,
) -> list[dict[str, Any]]:
    # bucket_by_cell skips any index where the feasibility flag is False
    # before it ever reads x/y for that index, so `aspect`/`min_len` can
    # be passed as-is (no need to mask out the non-population rows).
    buckets = bd.bucket_by_cell(
        aspect, min_len, population_mask, nfp, cell, max_per_cell=max_per_cell
    )
    rows = []
    for (grp, cx, cy), members in buckets.items():
        if len(members) < 10:
            continue
        if require_mixed_sources and len({source[i] for i in members}) < 2:
            continue
        clusters = bd.single_linkage_clusters(members, distance_fn, threshold)
        curvatures = [records[i].get("max_curvature") for i in members]
        tail_fracs = [records[i]["tail_energy_fraction"] for i in members]
        rows.append(
            {
                "lineage_split": split,
                "nfp": grp,
                "cell_aspect": cx,
                "cell_min_len": cy,
                "n_members": len(members),
                "n_sources": len({source[i] for i in members}),
                "n_clusters": len(clusters),
                "cluster_sizes": sorted((len(c) for c in clusters), reverse=True),
                "curvature_min": float(np.min(curvatures)),
                "curvature_max": float(np.max(curvatures)),
                "curvature_std": float(np.std(curvatures)),
                "tail_fraction_min": float(np.min(tail_fracs)),
                "tail_fraction_max": float(np.max(tail_fracs)),
                "tail_fraction_std": float(np.std(tail_fracs)),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# output: REPORT.json, pairs.csv, clusters.csv, two PNGs
# --------------------------------------------------------------------------- #


def write_pairs_csv(path: Path, pair_rows: list[dict[str, Any]]) -> None:
    fields = [
        "i",
        "j",
        "id_i",
        "id_j",
        "source_i",
        "source_j",
        "lineage_split",
        "d_aspect",
        "d_min_len",
        "physical_distance",
        "coefficient_distance",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in pair_rows:
            writer.writerow({k: row[k] for k in fields})


def write_clusters_csv(path: Path, cluster_rows: list[dict[str, Any]]) -> None:
    fields = [
        "lineage_split",
        "nfp",
        "cell_aspect",
        "cell_min_len",
        "n_members",
        "n_sources",
        "n_clusters",
        "cluster_sizes",
        "curvature_min",
        "curvature_max",
        "curvature_std",
        "tail_fraction_min",
        "tail_fraction_max",
        "tail_fraction_std",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in cluster_rows:
            out = dict(row)
            out["cluster_sizes"] = json.dumps(out["cluster_sizes"])
            writer.writerow({k: out[k] for k in fields})


def plot_distance_histogram(
    path: Path, pair_rows: list[dict[str, Any]], reference_distances: np.ndarray
) -> None:
    near_eq = np.array([row["physical_distance"] for row in pair_rows])
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=110)
    if len(reference_distances):
        ax.hist(
            reference_distances,
            bins=40,
            alpha=0.5,
            density=True,
            label=f"random feasible pairs (n={len(reference_distances)})",
            color="#888888",
        )
    if len(near_eq):
        ax.hist(
            near_eq,
            bins=40,
            alpha=0.6,
            density=True,
            label=f"near-equivalent pairs (n={len(near_eq)})",
            color="#2bb35f",
        )
    ax.set_xlabel("physical distance (units of mean minor radius)")
    ax.set_ylabel("density")
    ax.set_title("P3 diversity probe (D1, PROVISIONAL): near-equivalent vs random pairs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_cluster_count_distribution(path: Path, cluster_rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), dpi=110, sharey=True)
    for ax, split in zip(axes, ("within", "cross")):
        counts = [row["n_clusters"] for row in cluster_rows if row["lineage_split"] == split]
        if counts:
            bins = np.arange(1, max(counts) + 2) - 0.5
            ax.hist(counts, bins=bins, color="#2b6fb3" if split == "within" else "#b3702b")
        ax.set_title(f"{split}-lineage (n_cells={len(counts)})", fontsize=9)
        ax.set_xlabel("clusters per cell (single-linkage @ d_10)")
    axes[0].set_ylabel("number of cells")
    fig.suptitle("P3 diversity probe (D1, PROVISIONAL): per-cell cluster counts")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summarize_cluster_sizes(cluster_rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    counts = [row["n_clusters"] for row in cluster_rows if row["lineage_split"] == split]
    if not counts:
        return {"n_cells": 0}
    return {
        "n_cells": len(counts),
        "min": int(min(counts)),
        "max": int(max(counts)),
        "mean": float(np.mean(counts)),
        "median": float(np.median(counts)),
        "histogram": {str(k): int(v) for k, v in zip(*np.unique(counts, return_counts=True))},
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--atlas", required=True, type=Path)
    p.add_argument("--board", required=True, type=Path)
    p.add_argument("--ours", required=True, type=Path)
    p.add_argument("--ours-representation", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--objective-tol",
        type=float,
        default=0.05,
        help=(
            "L-infinity tolerance in objective space (|d aspect| and "
            "|d min_len|, each <= this) for the near-equivalent pair search "
            "(registration step 3). Locked default: 0.05."
        ),
    )
    p.add_argument(
        "--grid",
        type=int,
        default=32,
        help=(
            "n_theta = n_phi for the (theta, phi) surface grid used by "
            "curvature and physical-distance evaluation. Locked default: 32."
        ),
    )
    p.add_argument(
        "--n-random",
        type=int,
        default=10_000,
        help=(
            "Number of same-nfp, feasible-only reference pairs sampled for "
            "the d_rand / d_10 reference distribution. Locked default: 10000."
        ),
    )
    p.add_argument(
        "--cell",
        type=float,
        default=0.1,
        help=(
            "Objective-space cell width for per-cell clustering (registration "
            "step 5), independent of --objective-tol's near-equivalent-pair "
            "cell. Locked default: 0.1."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Seed for the reference-pair RNG (numpy Generator via "
            "np.random.default_rng) -- the only source of randomness in this "
            "script. Locked default: 0."
        ),
    )
    p.add_argument(
        "--max-cell-members",
        type=int,
        default=60,
        help=(
            "Deterministic subsample cap per objective cell (near-equivalent "
            "search and clustering), keeping pairwise work bounded on dense "
            "fronts. Subsampling rule (see bucket_by_cell in "
            "constellaration_methods/analysis/boundary_diversity.py): a cell "
            "with n > this many feasible members keeps every "
            "ceil(n / max_per_cell)-th member in input order (a fixed stride "
            "over the cell's original ordering, so the kept subset is "
            "identical run to run for the same inputs) -- no random sampling "
            "is involved. Locked default: 60."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    t0 = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)

    print("loading atlas...", flush=True)
    atlas_records = load_atlas(args.atlas)
    print(f"  {len(atlas_records)} atlas boundaries", flush=True)

    print("loading board...", flush=True)
    board_records = load_board(args.board)
    print(f"  {len(board_records)} board boundaries", flush=True)

    print("loading ours (rebuilding feasible boundaries from x)...", flush=True)
    ours_records = load_ours(args.ours, args.ours_representation)
    print(f"  {len(ours_records)} ours boundaries", flush=True)

    records = atlas_records + board_records + ours_records
    target_mpol = max(r["r_cos"].shape[0] - 1 for r in records)
    target_ntor = max((r["r_cos"].shape[1] - 1) // 2 for r in records)
    print(f"common grid: mpol={target_mpol} ntor={target_ntor}", flush=True)

    print(
        "canonicalising (pad + R00-normalise + minor radius + tail fraction)...",
        flush=True,
    )
    canonicalize_records(records, target_mpol, target_ntor)

    print("computing curvature for feasible records...", flush=True)
    compute_curvature(records, args.grid, args.grid)

    n_feasible = sum(1 for r in records if r["feasible"])
    n_unknown = sum(1 for r in records if r["feasible"] is None)
    print(f"  {n_feasible} feasible, {n_unknown} unknown-excluded", flush=True)

    print("near-equivalent pairs + reference distribution...", flush=True)
    pair_stats = compute_pair_statistics(
        records,
        objective_tol=args.objective_tol,
        max_per_cell=args.max_cell_members,
        n_theta=args.grid,
        n_phi=args.grid,
        n_random=args.n_random,
        seed=args.seed,
    )
    print(
        f"  {pair_stats['n_candidate_pairs']} candidate pairs, "
        f"d_rand={pair_stats['d_rand']}, d_10={pair_stats['d_10']}",
        flush=True,
    )

    print("per-cell clustering...", flush=True)
    cluster_rows = compute_clusters(
        records,
        cell=args.cell,
        max_per_cell=args.max_cell_members,
        n_theta=args.grid,
        n_phi=args.grid,
        threshold=pair_stats["d_10"] if pair_stats["d_10"] is not None else args.cell,
    )
    print(f"  {len(cluster_rows)} qualifying cells", flush=True)

    write_pairs_csv(args.out / "pairs.csv", pair_stats["pair_rows"])
    write_clusters_csv(args.out / "clusters.csv", cluster_rows)
    plot_distance_histogram(
        args.out / "distance_histogram.png",
        pair_stats["pair_rows"],
        pair_stats["reference_distances"],
    )
    plot_cluster_count_distribution(args.out / "cluster_counts.png", cluster_rows)

    elapsed_s = time.monotonic() - t0
    report = {
        "PROVISIONAL": True,
        "note": (
            "DRAFT measurement, not locked. No labels were pulled from this "
            "run and nothing was optimised; the reading below is provisional "
            "because docs/preregistration-snapshots has no D1 lock snapshot."
        ),
        "parameters": {
            "atlas": str(args.atlas),
            "board": str(args.board),
            "ours": str(args.ours),
            "ours_representation": str(args.ours_representation),
            "objective_tol": args.objective_tol,
            "grid": args.grid,
            "n_random": args.n_random,
            "cell": args.cell,
            "seed": args.seed,
            "max_cell_members": args.max_cell_members,
            "target_mpol": target_mpol,
            "target_ntor": target_ntor,
            "benchmark_tol": BENCHMARK_TOL,
            "board_feasibility_rule": BOARD_FEASIBILITY_RULE,
        },
        "git_commit": _git_commit(),
        "counts": {
            "atlas": len(atlas_records),
            "board": len(board_records),
            "ours": len(ours_records),
            "total": len(records),
            "feasible": n_feasible,
            "unknown_excluded": n_unknown,
            "by_source": {
                src: sum(1 for r in records if r["source"] == src)
                for src in sorted({r["source"] for r in records})
            },
            "feasible_by_source": {
                src: sum(1 for r in records if r["source"] == src and r["feasible"])
                for src in sorted({r["source"] for r in records})
            },
        },
        "reference_distribution": {
            "n_pairs": pair_stats["n_reference_pairs"],
            "d_rand_median": pair_stats["d_rand"],
            "d_10_percentile": pair_stats["d_10"],
        },
        "near_equivalent_pairs": {
            "n_candidate_pairs": pair_stats["n_candidate_pairs"],
            "within_lineage": pair_stats["fractions"]["within"],
            "cross_lineage": pair_stats["fractions"]["cross"],
        },
        "reading": {
            "value": pair_stats["reading"],
            "basis": "within-lineage (phanerozoic-only) near-equivalent pairs",
            "thresholds": {"degenerate_ge": 0.20, "rigid_lt": 0.05, "min_pairs": 100},
            "PROVISIONAL": True,
        },
        "clustering": {
            "within_lineage": summarize_cluster_sizes(cluster_rows, "within"),
            "cross_lineage": summarize_cluster_sizes(cluster_rows, "cross"),
        },
        "runtime_seconds": elapsed_s,
    }
    (args.out / "REPORT.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out / 'REPORT.json'} in {elapsed_s:.1f}s", flush=True)
    print(
        f"PROVISIONAL reading: {pair_stats['reading']} "
        "(not locked -- see docs/preregistration-snapshots)",
        flush=True,
    )


if __name__ == "__main__":
    main()
