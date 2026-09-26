"""P3 design-space diversity probe (D1): pure-numpy tests for the core
algorithms in `constellaration_methods.analysis.boundary_diversity` (vault
`p3-design-diversity-probe-2026-09-23.md`, Method section). No VMEC, no
jax, no I/O -- host-runnable per the repo's no-physics-in-unit-tests
discipline. The CLI script
(`constellaration_methods.analysis.p3_design_diversity_probe`) is mostly
wired onto the real datasets by the single real run, not by this suite --
except its board-file feasibility rule (`load_board`), which gets a
regression-ratchet unit test below on a synthetic fixture: real board
files are scored by the contest as feasible when the violation is within
tolerance 0.01, not only at exactly 0.0 (davidkh's file has overall
`feasibility` 0.0087 with a scored 200-member leaderboard entry -- a
loader that only accepts exactly-0.0 rows silently drops it).
"""

import json

import numpy as np
import pytest

from constellaration_methods.analysis.boundary_diversity import (
    bucket_by_cell,
    classify_reading,
    coefficient_distance,
    make_grid,
    minor_radius,
    near_equivalent_pairs,
    pad_and_normalize,
    physical_distance,
    principal_curvature_max,
    sample_reference_pairs,
    single_linkage_cluster_count,
    single_linkage_clusters,
)
from constellaration_methods.analysis.boundary_symmetry import symmetry_variants
from constellaration_methods.analysis.p3_design_diversity_probe import (
    _git_commit,
    load_board,
)


def _random_boundary(seed=0, mpol=4, ntor=4, r00=1.0):
    rng = np.random.default_rng(seed)
    m_rows, n_cols = mpol + 1, 2 * ntor + 1
    r_cos = np.zeros((m_rows, n_cols))
    z_sin = np.zeros((m_rows, n_cols))
    r_cos[0, ntor] = r00
    r_cos[1:, :] = rng.normal(scale=0.05, size=(m_rows - 1, n_cols))
    z_sin[1:, :] = rng.normal(scale=0.05, size=(m_rows - 1, n_cols))
    return r_cos, z_sin


def _circular_torus(a, r0=1.0):
    """mpol=1, ntor=0: R = r0 + a*cos(theta), Z = a*sin(theta)."""
    r_cos = np.zeros((2, 1))
    z_sin = np.zeros((2, 1))
    r_cos[0, 0] = r0
    r_cos[1, 0] = a
    z_sin[1, 0] = a
    return r_cos, z_sin


# --------------------------------------------------------------------------- #
# symmetry-quotient: every group image is distance 0
# --------------------------------------------------------------------------- #


def test_all_eight_symmetry_images_are_zero_coefficient_distance():
    r_cos, z_sin = _random_boundary()
    r_pad, z_pad = pad_and_normalize(r_cos, z_sin, target_mpol=7, target_ntor=7)

    variants = symmetry_variants(r_pad, z_pad)
    assert len(variants) == 8  # the full discrete group, identity included

    for r_v, z_v, tag in variants:
        d = coefficient_distance(r_pad, z_pad, r_v, z_v)
        assert d == pytest.approx(0.0, abs=1e-10), f"variant {tag} not distance 0"


def test_all_eight_symmetry_images_are_zero_physical_distance():
    r_cos, z_sin = _random_boundary()
    r_pad, z_pad = pad_and_normalize(r_cos, z_sin, target_mpol=7, target_ntor=7)
    theta, phi = make_grid(nfp=3, n_theta=32, n_phi=32)
    a_minor = minor_radius(r_pad, z_pad, nfp=3)

    for r_v, z_v, tag in symmetry_variants(r_pad, z_pad):
        d = physical_distance(r_pad, z_pad, r_v, z_v, 3, theta, phi, a_minor, a_minor)
        assert d == pytest.approx(0.0, abs=1e-8), f"variant {tag} not distance 0"


def test_enantiomer_twin_deduplicates_at_zero_distance():
    """A design and its handedness-flipped (toroidal-reversal) twin --
    the scenario the registration calls "enantiomer twins must be
    deduplicated" -- must register as the SAME design (distance 0), not
    as two independent near-equivalent-but-distinct entries."""
    r_cos, z_sin = _random_boundary(seed=7)
    r_pad, z_pad = pad_and_normalize(r_cos, z_sin, target_mpol=7, target_ntor=7)
    enantiomer = next((rv, zv) for rv, zv, tag in symmetry_variants(r_pad, z_pad) if tag["flip"])
    r_en, z_en = enantiomer

    assert coefficient_distance(r_pad, z_pad, r_en, z_en) == pytest.approx(0.0, abs=1e-10)
    theta, phi = make_grid(nfp=3, n_theta=32, n_phi=32)
    a_minor = minor_radius(r_pad, z_pad, nfp=3)
    assert physical_distance(
        r_pad, z_pad, r_en, z_en, 3, theta, phi, a_minor, a_minor
    ) == pytest.approx(0.0, abs=1e-8)


# --------------------------------------------------------------------------- #
# hand-computable physical distance: two circular tori
# --------------------------------------------------------------------------- #


def test_physical_distance_of_two_circular_tori_matches_hand_calculation():
    """R_A = 1 + 0.2 cos(theta), R_B = 1 + 0.3 cos(theta) (Z analogous):
    pointwise separation is exactly |0.2 - 0.3| = 0.1 at every (theta,
    phi) -- the tori are axisymmetric, so RMS = 0.1 exactly. Normalising
    by the mean minor radius (0.2 + 0.3)/2 = 0.25 gives 0.1/0.25 = 0.4."""
    r_a, z_a = _circular_torus(0.2)
    r_b, z_b = _circular_torus(0.3)
    theta, phi = make_grid(nfp=3, n_theta=64, n_phi=64)
    a_minor = minor_radius(r_a, z_a, nfp=3)
    b_minor = minor_radius(r_b, z_b, nfp=3)

    assert a_minor == pytest.approx(0.2)
    assert b_minor == pytest.approx(0.3)

    d = physical_distance(r_a, z_a, r_b, z_b, 3, theta, phi, a_minor, b_minor)
    assert d == pytest.approx(0.4, abs=1e-9)


# --------------------------------------------------------------------------- #
# curvature: circular torus closed form
# --------------------------------------------------------------------------- #


def test_curvature_of_circular_torus_matches_analytic_max_principal_curvature():
    """A circular torus of minor radius a has principal curvatures 1/a
    (constant, the minor-circle direction) and cos(theta)/(R0 + a cos
    theta) (the toroidal direction, bounded by 1/(R0+a) < 1/a). So the max
    principal curvature over the whole surface is exactly 1/a, and in
    units of 1/minor_radius (this module's convention) that is exactly
    1.0 -- independent of aspect ratio."""
    from constellaration_methods.analysis import boundary_diversity as bd

    for a, r0 in [(0.1, 1.0), (0.3, 1.0), (0.5, 1.0), (0.2, 2.0)]:
        r_cos, z_sin = _circular_torus(a, r0=r0)
        theta, phi = make_grid(nfp=3, n_theta=64, n_phi=64)
        a_minor = minor_radius(r_cos, z_sin, nfp=3)
        fields = bd.evaluate_surface_with_derivatives(r_cos, z_sin, 3, theta, phi)
        kappa = principal_curvature_max(fields, a_minor)
        assert kappa == pytest.approx(1.0, rel=1e-6), (a, r0, kappa)


# --------------------------------------------------------------------------- #
# near-equivalent pairing: tolerance + feasibility
# --------------------------------------------------------------------------- #


def test_near_equivalent_pairs_respects_objective_tolerance():
    x = np.array([1.0, 1.03, 1.20, 1.04])
    y = np.array([5.0, 5.02, 5.10, 5.90])
    feasible = np.array([True, True, True, True])
    group = np.array([3, 3, 3, 3])

    pairs = near_equivalent_pairs(x, y, feasible, group, tol=0.05)

    # (0,1): dx=0.03<=0.05, dy=0.02<=0.05 -> pair.
    # (0,2): dx=0.20>0.05 -> not a pair.
    # (0,3): dy=0.90>0.05 -> not a pair.
    # (1,3): dx=0.01<=0.05 but dy=0.88>0.05 -> not a pair.
    assert pairs == [(0, 1)]


def test_near_equivalent_pairs_excludes_infeasible_members():
    x = np.array([1.0, 1.01])
    y = np.array([5.0, 5.01])
    feasible = np.array([True, False])  # second member fails feasibility
    group = np.array([3, 3])

    pairs = near_equivalent_pairs(x, y, feasible, group, tol=0.05)

    assert pairs == []


def test_near_equivalent_pairs_excludes_different_nfp():
    x = np.array([1.0, 1.01])
    y = np.array([5.0, 5.01])
    feasible = np.array([True, True])
    group = np.array([3, 4])  # different nfp -- never compared

    pairs = near_equivalent_pairs(x, y, feasible, group, tol=0.05)

    assert pairs == []


def test_bucket_by_cell_deterministic_subsample_keeps_stride_members():
    x = np.arange(20, dtype=float) * 0.001  # all inside one 0.1-wide cell
    y = np.zeros(20)
    feasible = np.ones(20, dtype=bool)
    group = np.zeros(20, dtype=int)

    buckets = bucket_by_cell(x, y, feasible, group, cell=0.1, max_per_cell=5)

    assert len(buckets) == 1
    (kept,) = buckets.values()
    assert len(kept) == 5
    assert kept == sorted(kept)  # order preserved, deterministic


# --------------------------------------------------------------------------- #
# single-linkage clustering on a synthetic cell
# --------------------------------------------------------------------------- #


def test_single_linkage_cluster_count_on_synthetic_cell():
    # Positions on a line: two tight pairs (0,1) and (2,3), far from a
    # fifth singleton (4). Distance = |position difference|.
    positions = {0: 0.0, 1: 0.01, 2: 1.0, 3: 1.02, 4: 5.0}
    indices = list(positions)

    def distance_fn(i, j):
        return abs(positions[i] - positions[j])

    n_clusters = single_linkage_cluster_count(indices, distance_fn, threshold=0.05)

    assert n_clusters == 3  # {0,1}, {2,3}, {4}


def test_single_linkage_clusters_returns_the_actual_member_partition():
    positions = {0: 0.0, 1: 0.01, 2: 1.0, 3: 1.02, 4: 5.0}
    indices = list(positions)

    def distance_fn(i, j):
        return abs(positions[i] - positions[j])

    groups = single_linkage_clusters(indices, distance_fn, threshold=0.05)

    as_sets = {frozenset(g) for g in groups}
    assert as_sets == {frozenset({0, 1}), frozenset({2, 3}), frozenset({4})}


def test_single_linkage_cluster_count_all_within_threshold_is_one_cluster():
    positions = {0: 0.0, 1: 0.01, 2: 0.02}
    indices = list(positions)

    def distance_fn(i, j):
        return abs(positions[i] - positions[j])

    assert single_linkage_cluster_count(indices, distance_fn, threshold=0.05) == 1


# --------------------------------------------------------------------------- #
# reference distribution: same-nfp, feasible-only sampling
# --------------------------------------------------------------------------- #


def test_sample_reference_pairs_only_same_group_and_feasible():
    feasible = np.array([True, True, False, True, True])
    group = np.array([3, 3, 3, 4, 4])
    rng = np.random.default_rng(0)

    pairs = sample_reference_pairs(feasible, group, rng, n_pairs=200)

    assert len(pairs) == 200
    for i, j in pairs:
        assert i != j
        assert feasible[i] and feasible[j]
        assert group[i] == group[j]


# --------------------------------------------------------------------------- #
# reading classification on planted fractions
# --------------------------------------------------------------------------- #


def test_classify_reading_degenerate_above_threshold_with_enough_pairs():
    assert classify_reading(0.25, n_pairs=150) == "Degenerate"


def test_classify_reading_rigid_below_threshold_with_enough_pairs():
    assert classify_reading(0.02, n_pairs=150) == "Rigid"


def test_classify_reading_inconclusive_in_the_middle():
    assert classify_reading(0.10, n_pairs=150) == "Inconclusive"


def test_classify_reading_inconclusive_below_minimum_pair_count():
    # Would be "Degenerate" by fraction alone, but < 100 pairs.
    assert classify_reading(0.90, n_pairs=42) == "Inconclusive"


def test_classify_reading_boundary_values_are_inclusive_and_exclusive_correctly():
    assert classify_reading(0.20, n_pairs=100) == "Degenerate"  # >= is Degenerate
    # < is Rigid, so the 0.05 boundary itself is not Rigid (nor >= 0.20 Degenerate).
    assert classify_reading(0.05, n_pairs=100) == "Inconclusive"
    assert classify_reading(0.0499, n_pairs=100) == "Rigid"


# --------------------------------------------------------------------------- #
# board loader: feasibility is a benchmark-tolerance test, not an
# exactly-zero test (regression ratchet -- see module docstring)
# --------------------------------------------------------------------------- #


def _synthetic_boundary(nfp=3):
    r_cos = np.zeros((2, 3))
    z_sin = np.zeros((2, 3))
    r_cos[0, 1] = 1.0
    r_cos[1, 1] = 0.1
    z_sin[1, 1] = 0.1
    return {
        "r_cos": r_cos.tolist(),
        "z_sin": z_sin.tolist(),
        "r_sin": None,
        "z_cos": None,
        "n_field_periods": nfp,
        "is_stellarator_symmetric": True,
    }


def _write_synthetic_board_file(tmp_path, feasibilities):
    n = len(feasibilities)
    boundary_json = json.dumps([json.dumps(_synthetic_boundary()) for _ in range(n)])
    objectives = json.dumps([[[7.0 + 0.1 * i], [0.05 + 0.001 * i]] for i in range(n)])
    payload = {
        "submission_filename": "synthetic.json",
        "submission_time": "2026-09-23T00:00:00",
        "problem_type": "mhd_stable",
        "boundary_json": boundary_json,
        "evaluated": True,
        "user": "synthetic-user",
        "objectives": objectives,
        "feasibility": float(np.mean(feasibilities)),
        "score": 0.0,
        "result_filename": "synthetic_results.json",
        "feasibilities": json.dumps(feasibilities),
        "objective": 0.0,
        "minimize_objective": True,
    }
    path = tmp_path / "synthetic_mhd_stable_results.json"
    path.write_text(json.dumps(payload))
    return path


def test_load_board_feasibility_is_within_benchmark_tolerance_not_only_zero(tmp_path):
    """Fixture modelled on the real board files (Phase 0: davidkh's file
    has overall `feasibility` 0.0087 and a scored 200-member leaderboard
    entry; canyon-ant's has 6 exact-0.0 rows out of 43 but a scored 43-
    member entry) -- the contest scores a member feasible when its
    violation is within tolerance 0.01, so the loader must accept
    `feasibilities[i] <= 0.01`, not only `== 0.0`."""
    # 0.0, 0.0087, 0.01 are feasible; 0.0101 is not.
    _write_synthetic_board_file(tmp_path, [0.0, 0.0087, 0.01, 0.0101])

    records = load_board(tmp_path)

    assert [r["feasible"] for r in records] == [True, True, True, False]
    assert all(r["feasible_reason"] == "file_feasibilities" for r in records)


# --------------------------------------------------------------------------- #
# _git_commit: stdlib-only (no `git` binary in the phase1 image), and must
# follow a worktree checkout's `.git` pointer file -- this script's own
# repo (REPO) is normally a linked worktree, unlike qi_discovery.REPO_ROOT
# --------------------------------------------------------------------------- #


def test_git_commit_resolves_for_this_real_worktree_repo():
    """This script's own repo is a git worktree (`.git` is a pointer file,
    not a directory) -- the case `qi_discovery._current_git_commit`
    explicitly does not handle. Regression ratchet: a naive reuse of that
    function (or a `subprocess.run(["git", ...])` call, which silently
    fails in the git-less phase1 image) would report 'unknown' here."""
    import re

    commit = _git_commit()

    assert commit != "unknown"
    assert re.fullmatch(r"[0-9a-f]{40}", commit)


def test_git_commit_follows_a_synthetic_worktree_pointer_file(tmp_path):
    """Synthetic layout matching real `git worktree add` output: the
    worktree root's `.git` is `gitdir: <private gitdir>`; the private
    gitdir has its own `HEAD` and a `commondir` file pointing back at the
    main checkout's `.git`, where `refs/heads/<branch>` actually lives."""
    main_git = tmp_path / "main" / ".git"
    private_gitdir = main_git / "worktrees" / "wt"
    private_gitdir.mkdir(parents=True)
    (private_gitdir / "commondir").write_text("../..\n")
    (private_gitdir / "HEAD").write_text("ref: refs/heads/feature/x\n")
    sha = "d" * 40
    (main_git / "refs" / "heads" / "feature").mkdir(parents=True)
    (main_git / "refs" / "heads" / "feature" / "x").write_text(sha + "\n")

    worktree_root = tmp_path / "wt"
    worktree_root.mkdir()
    (worktree_root / ".git").write_text(f"gitdir: {private_gitdir}\n")

    assert _git_commit(worktree_root) == sha


def test_git_commit_returns_unknown_for_a_non_git_directory(tmp_path):
    assert _git_commit(tmp_path) == "unknown"
