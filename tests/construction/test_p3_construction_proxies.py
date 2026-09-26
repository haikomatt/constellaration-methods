"""E20: TDD for the pure P3 construction-search proxies in
`constellaration_methods.construction.p3_construction_proxies`. Host-testable:
pure numpy/dataclasses, no constellaration/pyQIC import (see module
docstring's Phase 0 evidence)."""

from __future__ import annotations

import pytest

from constellaration_methods.construction import p3_construction_proxies as pcp

# ---------------------------------------------------------------------------
# per-row proxies
# ---------------------------------------------------------------------------


def test_iota_violation_zero_when_at_bound():
    assert pcp.iota_violation(0.25) == 0.0


def test_iota_violation_zero_above_bound():
    assert pcp.iota_violation(0.4) == 0.0


def test_iota_violation_positive_below_bound():
    # shortfall 0.05 normalised by 0.25 -> 0.2
    assert pcp.iota_violation(0.20) == pytest.approx(0.2)


def test_mirror_violation_zero_when_at_or_below_bound():
    assert pcp.mirror_violation(0.25) == 0.0
    assert pcp.mirror_violation(0.10) == 0.0


def test_mirror_violation_positive_above_bound():
    # excess 0.05 normalised by 0.25 -> 0.2
    assert pcp.mirror_violation(0.30) == pytest.approx(0.2)


def test_qi_violation_zero_at_zero_residual():
    assert pcp.qi_violation(0.0) == 0.0


def test_qi_violation_scales_by_max_qi_residual():
    # at exactly the H2 viable threshold, violation == 1.0
    assert pcp.qi_violation(pcp.MAX_QI_RESIDUAL) == 1.0


def test_qi_violation_infinite_on_failed_build():
    assert pcp.qi_violation(float("inf")) == float("inf")


def test_qi_violation_infinite_on_nan():
    assert pcp.qi_violation(float("nan")) == float("inf")


def test_aspect_violation_zero_inside_band():
    assert pcp.aspect_violation(10.0) == 0.0
    assert pcp.aspect_violation(8.0) == 0.0
    assert pcp.aspect_violation(12.0) == 0.0


def test_aspect_violation_positive_below_band():
    # 8 - 6 = 2, normalised by 8 -> 0.25
    assert pcp.aspect_violation(6.0) == 0.25


def test_aspect_violation_positive_above_band():
    # 18 - 12 = 6, normalised by 12 -> 0.5
    assert pcp.aspect_violation(18.0) == 0.5


def test_nfp_violation_zero_for_allowed():
    assert pcp.nfp_violation(3) == 0.0
    assert pcp.nfp_violation(4) == 0.0


def test_nfp_violation_one_for_disallowed():
    assert pcp.nfp_violation(2) == 1.0
    assert pcp.nfp_violation(5) == 1.0


# ---------------------------------------------------------------------------
# scalarisation
# ---------------------------------------------------------------------------


def test_evaluate_proxies_all_satisfied_gives_zero_v():
    row = pcp.evaluate_proxies(
        iota_over_nfp=0.30,
        mirror=0.10,
        qi_residual=0.0,
        aspect_ratio=10.0,
        n_field_periods=3,
    )
    assert row.proxy_v == 0.0
    assert row.well_available is False
    assert "well" not in row.violations


def test_evaluate_proxies_v_is_max_of_violations():
    row = pcp.evaluate_proxies(
        iota_over_nfp=0.20,  # violation 0.2
        mirror=0.30,  # violation 0.2
        qi_residual=0.0,  # violation 0.0
        aspect_ratio=6.0,  # violation 0.25 -- the binding row
        n_field_periods=3,
    )
    assert row.proxy_v == 0.25
    assert row.violations["aspect"] == 0.25


def test_evaluate_proxies_drops_well_row():
    row = pcp.evaluate_proxies(
        iota_over_nfp=0.30,
        mirror=0.10,
        qi_residual=0.0,
        aspect_ratio=10.0,
        n_field_periods=4,
    )
    assert set(row.violations) == {"iota", "mirror", "qi", "aspect", "nfp"}


# ---------------------------------------------------------------------------
# stage-2 label thresholds (planted V values)
# ---------------------------------------------------------------------------


def test_stage2_label_goal_when_feasible_regardless_of_v():
    assert pcp.stage2_label(0.9, feasible=True) == "goal"


def test_stage2_label_substantial_at_or_below_0_10():
    assert pcp.stage2_label(0.10, feasible=False) == "substantial"
    assert pcp.stage2_label(0.05, feasible=False) == "substantial"


def test_stage2_label_material_between_0_10_and_0_20():
    assert pcp.stage2_label(0.20, feasible=False) == "material"
    assert pcp.stage2_label(0.15, feasible=False) == "material"


def test_stage2_label_null_above_0_20():
    assert pcp.stage2_label(0.21, feasible=False) == "null"
    assert pcp.stage2_label(0.49, feasible=False) == "null"


# ---------------------------------------------------------------------------
# D1 placement distance
# ---------------------------------------------------------------------------


def _circular_torus_boundary(nfp: int, aspect: float) -> dict:
    # axisymmetric (n=0 column only): R = 1 + a*cos(theta), Z = a*sin(theta)
    a = 1.0 / aspect
    r_cos = [[0.0, 1.0, 0.0], [0.0, a, 0.0]]
    z_sin = [[0.0, 0.0, 0.0], [0.0, a, 0.0]]
    return {"r_cos": r_cos, "z_sin": z_sin, "n_field_periods": nfp}


def test_nearest_family_distance_zero_for_self():
    b = _circular_torus_boundary(nfp=3, aspect=10.0)
    d = pcp.nearest_family_distance(b, [b])
    assert d == pytest.approx(0.0, abs=1e-9)


def test_nearest_family_distance_none_when_no_same_nfp_member():
    b = _circular_torus_boundary(nfp=3, aspect=10.0)
    other = _circular_torus_boundary(nfp=4, aspect=10.0)
    assert pcp.nearest_family_distance(b, [other]) is None


def test_nearest_family_distance_picks_the_closer_member():
    b = _circular_torus_boundary(nfp=3, aspect=10.0)
    near = _circular_torus_boundary(nfp=3, aspect=10.5)
    far = _circular_torus_boundary(nfp=3, aspect=20.0)
    d_near = pcp.nearest_family_distance(b, [far, near])
    d_far_only = pcp.nearest_family_distance(b, [far])
    assert d_near < d_far_only


def test_bank_row_json_round_trip():
    row = pcp.BankRow(
        label="e20_nfp3_000",
        n_field_periods=3,
        vector=[-0.15, 0.5, 0.0, 0.0, 0.8, 0.12],
        proxies={"iota": 0.0, "mirror": 0.0, "qi": 0.0, "aspect": 0.0, "nfp": 0.0},
        proxy_v=0.0,
        provenance={"git_commit": "deadbeef", "worker": 3},
    )
    d = row.to_json_dict()
    restored = pcp.BankRow.from_json_dict(d)
    assert restored == row
