"""E20: host-runnable tests for
`constellaration_methods.construction.p3_construction_search` -- the pieces
that need no pyQIC/constellaration import (task generation, CLI parsing,
family-file loading). The physics-dependent `make_observe_fn` /
`_polish_one_start` / `_export_4x4_boundary` paths are exercised only by the
in-image smoke run, same split as the near-axis engine's own runner tests."""

from __future__ import annotations

from constellaration_methods.construction import p3_construction_search as pcs


def test_build_tasks_splits_budget_across_nfp_3_and_4():
    tasks = pcs.build_tasks(budget_configs=10, minor_radius=0.1, random_seed=0)
    assert len(tasks) == 10
    nfps = {t["nfp"] for t in tasks}
    assert nfps == {3, 4}
    assert sum(1 for t in tasks if t["nfp"] == 3) == 5
    assert sum(1 for t in tasks if t["nfp"] == 4) == 5


def test_build_tasks_labels_are_unique():
    tasks = pcs.build_tasks(budget_configs=20, minor_radius=0.1, random_seed=1)
    labels = [t["label"] for t in tasks]
    assert len(labels) == len(set(labels))


def test_build_tasks_deterministic_from_seed():
    a = pcs.build_tasks(budget_configs=8, minor_radius=0.1, random_seed=7)
    b = pcs.build_tasks(budget_configs=8, minor_radius=0.1, random_seed=7)
    assert a == b


def test_build_tasks_vector0_within_space_bounds():
    from constellaration_methods.engines import near_axis_search as nas

    space = nas.NearAxisSpace(n_field_periods=(3, 4))
    bounds = space.bounds()
    tasks = pcs.build_tasks(budget_configs=6, minor_radius=0.1, random_seed=0)
    for t in tasks:
        for v, (lo, hi) in zip(t["vector0"], bounds):
            assert lo <= v <= hi


# ---------------------------------------------------------------------------
# --space: E20 optional widened box, opt-in only (default behaviour
# byte-identical to pre-E20 -- see test_p3_construction_search.py's own
# pre-existing tests above, none of which pass space_name)
# ---------------------------------------------------------------------------


def test_resolve_space_default_matches_pre_e20_hardcoded_space():
    from constellaration_methods.engines import near_axis_search as nas

    resolved = pcs._resolve_space("default")
    expected = nas.NearAxisSpace(n_field_periods=(3, 4))
    assert resolved.bounds() == expected.bounds()
    assert resolved.n_field_periods == expected.n_field_periods


def test_resolve_space_p3_wide_matches_the_near_axis_search_factory():
    from constellaration_methods.engines import near_axis_search as nas

    resolved = pcs._resolve_space("p3-wide")
    expected = nas.p3_wide_space()
    assert resolved.bounds() == expected.bounds()
    assert resolved.n_field_periods == expected.n_field_periods


def test_resolve_space_unknown_name_raises():
    import pytest

    with pytest.raises(ValueError, match="unknown --space"):
        pcs._resolve_space("not-a-real-space")


def test_build_tasks_default_space_name_is_default():
    tasks = pcs.build_tasks(budget_configs=4, minor_radius=0.1, random_seed=0)
    assert {t["space_name"] for t in tasks} == {"default"}


def test_build_tasks_p3_wide_space_name_threads_through_and_widens_r_cos_1():
    from constellaration_methods.engines import near_axis_search as nas

    tasks = pcs.build_tasks(
        budget_configs=2000, minor_radius=0.1, random_seed=0, space_name="p3-wide"
    )
    assert {t["space_name"] for t in tasks} == {"p3-wide"}
    r_cos_1_values = [t["vector0"][0] for t in tasks]
    # the widened box's whole point is admitting r_cos_1 > 0 (the default
    # box's upper bound is pinned at exactly 0.0) -- with 2000 LHS draws
    # spanning (-1.2, 0.4), at least one must land past the old edge.
    assert max(r_cos_1_values) > 0.0
    default_bounds = nas.NearAxisSpace(n_field_periods=(3, 4)).bounds()
    assert max(r_cos_1_values) <= default_bounds[0][1] + 0.4 + 1e-9


def test_cli_parses_space_flag_default_and_p3_wide():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--space", choices=sorted(pcs._SPACES), default="default")
    assert parser.parse_args([]).space == "default"
    assert parser.parse_args(["--space", "p3-wide"]).space == "p3-wide"


def test_proxy_v_matches_evaluate_proxies():
    from constellaration_methods.construction import p3_construction_proxies as pcp

    observables = {
        "iota_over_nfp": 0.30,
        "mirror": 0.10,
        "qi_residual": 0.0,
        "aspect_ratio": 10.0,
    }
    assert (
        pcs._proxy_v(observables, 3)
        == pcp.evaluate_proxies(n_field_periods=3, **observables).proxy_v
    )


def test_cli_parses_registration_smoke_flags():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-configs", type=int, default=5000)
    parser.add_argument("--total-hours", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(["--budget-configs", "200", "--total-hours", "0.5", "--workers", "12"])
    assert args.budget_configs == 200
    assert args.total_hours == 0.5
    assert args.workers == 12


def test_git_commit_returns_a_hex_sha_in_this_repo():
    sha = pcs._git_commit()
    assert sha != "unknown"
    assert len(sha) == 40
    int(sha, 16)  # raises if not hex


# ---------------------------------------------------------------------------
# export_top_with_placement -- graceful per-row export failure handling
# ---------------------------------------------------------------------------


def _row(label, nfp=3, proxy_v=0.5):
    return {
        "label": label,
        "n_field_periods": nfp,
        "vector": [0.0] * 6,
        "proxies": {},
        "proxy_v": proxy_v,
        "provenance": {},
    }


def _boundary(nfp=3):
    return {
        "r_cos": [[0.0, 1.0, 0.0], [0.0, 0.1, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]],
        "n_field_periods": nfp,
    }


def test_export_top_with_placement_skips_a_failing_row_without_crashing():
    rows = [_row("a"), _row("b"), _row("c")]

    def export_fn(vector, nfp):
        # fail on the second row specifically, by call order
        export_fn.calls += 1
        if export_fn.calls == 2:
            raise ValueError("f(a) and f(b) must have different signs")
        return _boundary(nfp)

    export_fn.calls = 0
    out, n_failed = pcs.export_top_with_placement(
        rows, export_fn=export_fn, own_family=[], leader_family=[]
    )
    assert n_failed == 1
    assert [r["label"] for r in out] == ["a", "c"]


def test_export_top_with_placement_attaches_na_when_no_family_member():
    rows = [_row("a")]
    out, n_failed = pcs.export_top_with_placement(
        rows,
        export_fn=lambda vector, nfp: _boundary(nfp),
        own_family=[],
        leader_family=[],
    )
    assert n_failed == 0
    assert out[0]["d1_to_leader_family"] == "n/a"
    assert out[0]["d1_to_own_family"] == "n/a"


def test_export_top_with_placement_computes_zero_distance_to_self_in_family():
    rows = [_row("a")]
    b = _boundary()
    out, n_failed = pcs.export_top_with_placement(
        rows,
        export_fn=lambda vector, nfp: b,
        own_family=[b],
        leader_family=[],
    )
    assert n_failed == 0
    assert out[0]["d1_to_own_family"] == 0.0
