"""Tests for scripts/p3_front_builder.py -- E16 (vault
p3-reopen-three-experiments-2026-09-16, "bounded front-building from the
nine verified anchors"). Pure-rule tests need no VMEC; `nondominated`,
`hypervolume` and the toy end-to-end run import `constellaration.problems`
lazily and so need the pinned image (run inside the container, not on host).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from constellaration_methods.fronts.p3_front_builder import (
    FrontModel,
    anchor_rows_from_report,
    apply_smoke_ceilings,
    best_t_for_front,
    build_arg_parser,
    evaluate_row7,
    front_accept,
    front_step,
    hv_gain_meets_threshold,
    is_benchmark_feasible,
    load_anchors,
    normalise_objectives,
    order_anchors,
    run_from_anchors,
    scalarise,
    worse_of_three,
)
from constellaration_methods.restoration.p3_feasibility_restoration import (
    RestorationLPInfeasible,
)


# --------------------------------------------------------------------------- #
# scalarisation normalisation
# --------------------------------------------------------------------------- #
def test_normalise_objectives_at_the_anchor_is_minus_one_plus_one():
    obj_a, obj_l = normalise_objectives(10.0, 10.0, 10.0, 10.0)
    assert obj_a == pytest.approx(-1.0)
    assert obj_l == pytest.approx(1.0)


def test_normalise_objectives_improvement_moves_both_components_up():
    # lower aspect (improvement, minimised) and higher min_len (improvement,
    # maximised) must both raise their normalised component
    obj_a, obj_l = normalise_objectives(9.0, 11.0, 10.0, 10.0)
    assert obj_a > -1.0
    assert obj_l > 1.0


def test_scalarise_weight_cycle():
    assert scalarise(10.0, 10.0, 10.0, 10.0, (1.0, 0.0)) == pytest.approx(-1.0)
    assert scalarise(10.0, 10.0, 10.0, 10.0, (0.0, 1.0)) == pytest.approx(1.0)
    assert scalarise(10.0, 10.0, 10.0, 10.0, (1.0, 1.0)) == pytest.approx(0.0)


def test_scalarise_rewards_improvement_regardless_of_scalarisation():
    base = scalarise(10.0, 10.0, 10.0, 10.0, (1.0, 1.0))
    improved = scalarise(9.5, 10.5, 10.0, 10.0, (1.0, 1.0))
    assert improved > base


# --------------------------------------------------------------------------- #
# evaluate_row7
# --------------------------------------------------------------------------- #
def test_evaluate_row7_builds_seven_component_rows():
    rs = [[0.1, -0.1, 0.2, -0.2, 0.0], None, [0.0] * 5]
    raws = [
        {"raw": {"aspect_ratio": 9.0, "min_len": 11.0}},
        {"raw": {"aspect_ratio": 8.0, "min_len": 12.0}},
        {"raw": {}},  # missing aspect/min_len -> incomplete
    ]
    rows = evaluate_row7(rs, raws)
    assert rows[0] is not None and len(rows[0]) == 7
    assert rows[0][5] == pytest.approx(9.0) and rows[0][6] == pytest.approx(11.0)
    assert rows[1] is None  # non-convergent (r is None)
    assert rows[2] is None  # incomplete raw


# --------------------------------------------------------------------------- #
# anchor_rows_from_report / order_anchors
# --------------------------------------------------------------------------- #
def test_anchor_rows_from_report_keeps_only_retained_members():
    report = {
        "members": [
            {
                "tag": "a",
                "retained": True,
                "confirmed_r": [0.0, 0.0, 0.0, 0.0, 0.0],
                "confirmed_aspect_ratio": 10.5,
                "confirmed_min_len": 10.5,
                "solo_hypervolume": 109.0,
            },
            {"tag": "b", "retained": False},
        ]
    }
    rows = anchor_rows_from_report(report)
    assert [r["tag"] for r in rows] == ["a"]
    assert rows[0]["aspect"] == pytest.approx(10.5)


def test_order_anchors_ascending_by_solo_hypervolume():
    rows = [
        {"tag": "high", "solo_hypervolume": 110.0},
        {"tag": "low", "solo_hypervolume": 108.0},
        {"tag": "mid", "solo_hypervolume": 109.0},
    ]
    ordered = order_anchors(rows)
    assert [r["tag"] for r in ordered] == ["low", "mid", "high"]


# --------------------------------------------------------------------------- #
# load_anchors -- E15 (top-level "members") vs E16 (top-level "confirmed_front")
# report layouts (E17 bug: seeded mode crashed with "IndexError: tuple index
# out of range" at anchor_xs.shape[1] because load_anchors silently returned
# zero anchors for an E16 report -- report.get("members", []) is always empty
# for that layout). Needs the engine (candidate_from_boundary_dict ->
# masked_coefficients_and_scale) to resolve each anchor's x from its exported
# boundary, so run inside the pinned image.
# --------------------------------------------------------------------------- #
def _write_boundary_and_representation(tmp_path: Path) -> tuple[dict, list]:
    """A small valid stellarator-symmetric boundary (same shape as
    test_p3_promote.py's own fixtures) plus the residuals used to build it,
    written as an exported anchor boundary file. Returns (representation,
    r) -- the caller writes the boundary.json into each report's own dir."""
    mpol, ntor, nfp = 2, 2, 3
    r_cos = np.zeros((mpol + 1, 2 * ntor + 1))
    z_sin = np.zeros_like(r_cos)
    r_cos[0, ntor] = 1.0
    r_cos[1, ntor] = 0.15
    z_sin[1, ntor] = 0.15
    r_cos[1, ntor + 1] = 0.02
    z_sin[1, ntor + 1] = 0.02
    bd = {
        "r_cos": r_cos.tolist(),
        "z_sin": z_sin.tolist(),
        "r_sin": np.zeros_like(r_cos).tolist(),
        "z_cos": np.zeros_like(z_sin).tolist(),
        "n_field_periods": nfp,
        "is_stellarator_symmetric": True,
    }
    r = [0.001, -0.002, 0.003, -0.004, 0.005]
    representation = dict(boundary_dict=bd, mpol=mpol, ntor=ntor, nfp=nfp, scaling=1.0)
    return representation, r, bd


def test_load_anchors_e16_layout_matches_equivalent_e15_layout(tmp_path):
    pytest.importorskip("jax")
    representation, r, bd = _write_boundary_and_representation(tmp_path)
    rep_path = tmp_path / "start_serialised.json"
    rep_path.write_text(json.dumps(representation))

    e15_dir = tmp_path / "e15"
    e15_dir.mkdir()
    (e15_dir / "P_L3_t1.0.boundary.json").write_text(json.dumps(bd))
    e15_report = {
        "members": [
            {
                "tag": "P:L3:t1.0",
                "retained": True,
                "confirmed_r": r,
                "confirmed_aspect_ratio": 10.5,
                "confirmed_min_len": 10.4,
                "solo_hypervolume": 109.0,
            }
        ]
    }
    (e15_dir / "REPORT.json").write_text(json.dumps(e15_report))

    e16_dir = tmp_path / "e16"
    e16_dir.mkdir()
    (e16_dir / "P_L3_t1.0.boundary.json").write_text(json.dumps(bd))
    e16_report = {
        "confirmed_front": [{"tag": "P:L3:t1.0", "r": r, "aspect_ratio": 10.5, "min_len": 10.4}],
        "n_front_members": 1,
    }
    (e16_dir / "REPORT.json").write_text(json.dumps(e16_report))

    e15_anchors, e15_rep = load_anchors(e15_dir / "REPORT.json", rep_path)
    e16_anchors, e16_rep = load_anchors(e16_dir / "REPORT.json", rep_path)

    assert e15_rep == e16_rep
    assert len(e15_anchors) == len(e16_anchors) == 1
    a15, a16 = e15_anchors[0], e16_anchors[0]
    assert a15["tag"] == a16["tag"] == "P:L3:t1.0"
    assert a15["aspect"] == pytest.approx(a16["aspect"])
    assert a15["min_len"] == pytest.approx(a16["min_len"])
    assert np.allclose(a15["x"], a16["x"])


def test_load_anchors_unknown_layout_raises_value_error_naming_keys(tmp_path):
    report_path = tmp_path / "REPORT.json"
    report_path.write_text(json.dumps({"foo": 1, "bar": 2}))
    with pytest.raises(ValueError) as excinfo:
        load_anchors(report_path, tmp_path / "start_serialised.json")
    assert "foo" in str(excinfo.value) and "bar" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# worse_of_three
# --------------------------------------------------------------------------- #
def test_worse_of_three_picks_the_largest_max_residual():
    rs = [[0.001, 0.002], [0.009, 0.001], [0.005, 0.005]]
    raws = ["a", "b", "c"]
    r, raw = worse_of_three(rs, raws)
    assert r == [0.009, 0.001]
    assert raw == "b"


def test_worse_of_three_skips_none():
    rs = [None, [0.002, 0.001], None]
    raws = ["a", "b", "c"]
    r, raw = worse_of_three(rs, raws)
    assert r == [0.002, 0.001] and raw == "b"


def test_worse_of_three_all_none():
    assert worse_of_three([None, None], ["a", "b"]) == (None, None)


# --------------------------------------------------------------------------- #
# is_benchmark_feasible / hv_gain_meets_threshold / front_accept
# --------------------------------------------------------------------------- #
def test_is_benchmark_feasible():
    assert is_benchmark_feasible([0.01, -0.5, 0.0, -0.2, -0.3])
    assert not is_benchmark_feasible([0.0101, -0.5, 0.0, -0.2, -0.3])
    assert not is_benchmark_feasible(None)
    assert not is_benchmark_feasible([0.0, float("nan"), 0.0, 0.0, 0.0])


def test_hv_gain_meets_threshold():
    assert hv_gain_meets_threshold(111.30, 111.31, threshold=0.01)
    assert not hv_gain_meets_threshold(111.30, 111.305, threshold=0.01)


def test_front_accept_requires_both_feasibility_and_gain():
    r = [0.0, 0.0, 0.0, 0.0, 0.0]
    assert front_accept(r, 111.0, 111.02)
    assert not front_accept(r, 111.0, 111.005)  # gain too small
    assert not front_accept([0.02, 0, 0, 0, 0], 111.0, 111.5)  # infeasible


# --------------------------------------------------------------------------- #
# front_step
# --------------------------------------------------------------------------- #
def _model(J_row3=0.0, J_row5=-1.0, well=-0.05):
    r0 = np.array([0.0, 0.0, 0.0, 0.0, well, 10.0, 10.0])
    J = np.zeros((7, 1))
    J[3, 0] = J_row3
    J[5, 0] = J_row5
    return FrontModel(r0=r0, J=J, C=None, aspect_anchor=10.0, min_len_anchor=10.0)


def test_front_step_improves_within_bounds_when_unconstrained():
    # flux does not move with d at all -> the trust box binds, not the residual
    model = _model(J_row3=0.0, J_row5=-1.0)
    d, gain = front_step(model, (1.0, 0.0), delta=0.01, tol_model=0.008)
    assert d[0] == pytest.approx(0.01, abs=1e-6)
    assert gain > 0.0


def test_front_step_residual_bound_binds():
    # flux rises 1:1 with d and the box is generous (delta=1.0): the model
    # must stop at the tol_model bound, not walk to the box edge
    model = _model(J_row3=1.0, J_row5=-1.0)
    d, gain = front_step(model, (1.0, 0.0), delta=1.0, tol_model=0.008)
    assert d[0] == pytest.approx(0.008, abs=1e-4)
    assert gain > 0.0
    assert gain < 1.0 * (1.0 - (-1.0)) / 10.0  # far short of the box-unconstrained gain


def test_front_step_well_cap_applies_once_reached():
    # well already <= 0 (reached): a step that would push it positive must
    # be refused even though delta would otherwise allow it
    r0 = np.array([0.0, 0.0, 0.0, 0.0, -0.001, 10.0, 10.0])
    J = np.zeros((7, 1))
    J[4, 0] = 1.0  # well rises with d
    J[5, 0] = -1.0
    model = FrontModel(r0=r0, J=J, C=None, aspect_anchor=10.0, min_len_anchor=10.0)
    d, gain = front_step(model, (1.0, 0.0), delta=1.0, tol_model=0.008)
    # predicted well residual must stay <= 0, i.e. d <= 0.001
    assert d[0] <= 0.001 + 1e-6


def test_front_step_infeasible_solve_raises():
    # no direction can improve the objective without breaking an
    # ALREADY-VIOLATED residual bound with zero slack available anywhere
    r0 = np.array([0.02, 0.0, 0.0, 0.0, -0.05, 10.0, 10.0])
    J = np.zeros((7, 1))
    J[0, 0] = 1.0  # the only coordinate makes the already-violated row worse
    J[5, 0] = -1.0
    model = FrontModel(r0=r0, J=J, C=None, aspect_anchor=10.0, min_len_anchor=10.0)
    # the bound (0.008) is already exceeded at d=0 and only gets worse for d>0;
    # for d<0 aspect gets worse (fine, no constraint there) but row 0 improves --
    # so pick a model where row 0 can *never* satisfy the bound within the box
    with pytest.raises(RestorationLPInfeasible):
        front_step(model, (1.0, 0.0), delta=1e-9, tol_model=-1.0)  # impossible bound


# --------------------------------------------------------------------------- #
# best_t_for_front
# --------------------------------------------------------------------------- #
def test_best_t_for_front_picks_the_admissible_maximiser():
    # A,B,C describe [r0..r3(4 rows), s] via A t^2 + B t + C; s increases with
    # t but row 0 crosses tol at t=1.0, so only t<=~0.5 should qualify
    C = np.array([-0.1, -0.1, -0.1, -0.1, 0.0])  # r0..r3 start well inside, s starts at 0
    B = np.array([0.0, 0.0, 0.0, 0.0, 1.0])  # only s moves linearly
    A = np.array([0.2, 0.0, 0.0, 0.0, 0.0])  # r0 grows quadratically -> crosses tol
    t_grid = np.linspace(0.1, 1.5, 29)
    t_star, s_star = best_t_for_front(A, B, C, t_grid, tol=0.01)
    assert t_star is not None
    r0_at_t = A[0] * t_star**2 + C[0]
    assert r0_at_t <= 0.01 + 1e-9
    # the true unconstrained max (t=1.5) must NOT be picked (r0 there is 0.35 > tol)
    assert t_star < 1.5


def test_best_t_for_front_none_when_nothing_qualifies():
    C = np.array([0.02, 0.0])  # already over tol at t=0
    B = np.array([0.0, 1.0])
    A = np.array([0.0, 0.0])
    t_star, s_star = best_t_for_front(A, B, C, np.linspace(0.1, 1.5, 5), tol=0.01)
    assert t_star is None and s_star is None


def test_best_t_for_front_respects_preserved_rows():
    # row 1 satisfied at r0 (<= tol) must stay satisfied even if the grid
    # would otherwise pick a t that breaks it
    C = np.array([-0.1, -0.005, 0.0])
    B = np.array([0.0, 0.0, 1.0])
    A = np.array([0.0, 0.02, 0.0])  # row 1 grows and crosses tol quickly
    t_star, _ = best_t_for_front(A, B, C, np.linspace(0.1, 1.5, 29), tol=0.01, r0=[-0.1, -0.005])
    row1_at_t = A[1] * t_star**2 + B[1] * t_star + C[1]
    assert row1_at_t <= 0.01 + 1e-9


# --------------------------------------------------------------------------- #
# ceilings
# --------------------------------------------------------------------------- #
def test_default_ceilings_match_the_preregistration():
    args = build_arg_parser().parse_args(
        ["--anchors", "a.json", "--representation", "r.json", "--out-dir", "out"]
    )
    apply_smoke_ceilings(args)
    assert args.total_hours == pytest.approx(60.0)
    assert args.max_opt_evals == 3000
    assert args.max_verif_evals == 60
    assert args.max_models == 6


def test_smoke_forces_its_own_ceilings_even_if_others_given():
    args = build_arg_parser().parse_args(
        [
            "--anchors",
            "a.json",
            "--representation",
            "r.json",
            "--total-hours",
            "60",
            "--max-opt-evals",
            "3000",
            "--smoke",
        ]
    )
    apply_smoke_ceilings(args)
    assert args.total_hours == pytest.approx(2.0)
    assert args.max_opt_evals == 600


def test_resolve_out_dir_defaults_to_the_smoke_path():
    from constellaration_methods.fronts.p3_front_builder import resolve_out_dir

    args = build_arg_parser().parse_args(
        ["--anchors", "a.json", "--representation", "r.json", "--smoke"]
    )
    assert resolve_out_dir(args) == "results/p3-e16-smoke"


def test_resolve_out_dir_required_without_smoke():
    from constellaration_methods.fronts.p3_front_builder import resolve_out_dir

    args = build_arg_parser().parse_args(["--anchors", "a.json", "--representation", "r.json"])
    with pytest.raises(SystemExit):
        resolve_out_dir(args)


# --------------------------------------------------------------------------- #
# toy end-to-end run (payload 'fake_front'; needs constellaration for
# hypervolume/nondominated -- run inside the pinned image)
# --------------------------------------------------------------------------- #
def test_runner_toy_end_to_end_produces_a_confirmed_front_gain(tmp_path):
    pytest.importorskip(
        "constellaration",
        reason="needs the pinned constellaration package; run inside the repo's Docker image",
    )
    from types import SimpleNamespace

    representation = {
        "boundary_dict": {},
        "mpol": 1,
        "ntor": 1,
        "nfp": 1,
        "scaling": 1.0,
        "fake_front": True,
    }
    anchor = {
        "tag": "anchor0",
        "r": [-0.05, -0.05, -0.05, -0.05, -0.05],
        "aspect": 10.0,
        "min_len": 10.0,
        "solo_hypervolume": 110.0,
        "x": [0.0, 0.0, 0.0],
    }
    args = SimpleNamespace(
        workers=2,
        h=0.01,
        delta0=0.0025,
        delta_cap=0.01,
        delta_floor=1e-5,
        tol=0.01,
        tol_model=0.008,
        hv_gain_threshold=0.01,
        pass_gain_threshold=0.05,
        total_hours=2.0,
        max_opt_evals=600,
        max_verif_evals=60,
        max_models=6,
        seed=0,
        smoke=True,
    )
    report = run_from_anchors([anchor], representation, tmp_path / "out", args)
    assert report["stop_reason"] == "smoke_complete"
    assert len(report["confirmed_front"]) >= 1
    assert report["hypervolume_gain"] > 0.0
    assert (tmp_path / "out" / "REPORT.json").exists()
    # at least one member beyond the original anchor must have been confirmed
    assert any(m["tag"] != "anchor0" for m in report["confirmed_front"])
