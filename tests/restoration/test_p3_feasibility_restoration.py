"""Pure-logic tests for scripts/p3_feasibility_restoration.py (pre-registered
in the vault: p3-feasibility-restoration-plan-2026-09-15). No VMEC: the
canonical residual function is injected; these tests use a linear toy
r(x) = A x + b so the Jacobian, the LP step and the acceptance rule have
known answers.
"""

from pathlib import Path

import numpy as np
import pytest

from constellaration_methods.restoration.p3_feasibility_restoration import (
    Budget,
    TrustRegion,
    accept_step,
    central_difference_jacobian,
    classify_outcome,
    fd_stability,
    probe_points,
    restoration_step,
)

A = np.array([[1.0, 0.0, 2.0], [0.0, -1.0, 1.0], [0.5, 0.5, 0.0]])  # 3 residuals, 3 coords
B = np.array([0.2, 0.1, -0.05])  # residuals 0,1 violated; 2 satisfied


def r(x):
    return A @ x + B


def test_probe_points_central_differences():
    x0 = np.zeros(3)
    pts, meta = probe_points(x0, h=0.1)
    assert pts.shape == (6, 3)
    assert meta[0] == (0, +1) and meta[1] == (0, -1)
    assert pts[0][0] == pytest.approx(0.1) and pts[1][0] == pytest.approx(-0.1)


def test_jacobian_recovers_linear_model():
    x0 = np.zeros(3)
    pts, meta = probe_points(x0, h=0.1)
    rs = [r(p) for p in pts]
    J, ok_cols = central_difference_jacobian(rs, meta, n=3, h=0.1)
    assert np.allclose(J, A)
    assert ok_cols == [True, True, True]


def test_jacobian_marks_nonconvergent_column():
    x0 = np.zeros(3)
    pts, meta = probe_points(x0, h=0.1)
    rs = [r(p) for p in pts]
    rs[2] = None  # coordinate 1, + side failed
    J, ok_cols = central_difference_jacobian(rs, meta, n=3, h=0.1)
    assert ok_cols == [True, False, True]
    assert np.all(np.isnan(J[:, 1])) and np.allclose(J[:, [0, 2]], A[:, [0, 2]])


def test_fd_stability_linear_is_stable_and_noisy_is_not():
    stable, disc = fd_stability(A, A * 1.05, np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]]))
    assert stable and disc == pytest.approx(0.05)
    stable, disc = fd_stability(A, A * 1.5, np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]]))
    assert not stable and disc == pytest.approx(0.5)


def test_restoration_step_reduces_max_violation_and_keeps_satisfied():
    x0 = np.zeros(3)
    d, t_pred = restoration_step(r(x0), A, delta=0.1, tol=0.01)
    assert np.all(np.abs(d) <= 0.1 + 1e-9)
    r_new = r(x0 + d)
    assert r_new.max() < r(x0).max()
    assert t_pred == pytest.approx(r_new.max(), abs=1e-6)
    assert r_new[2] <= 0.01 + 1e-9  # satisfied constraint preserved


def test_accept_step_rule():
    tol = 0.01
    assert accept_step(np.array([0.2, 0.1, -0.05]), np.array([0.15, 0.1, -0.05]), tol)
    # max improved but a satisfied constraint became violated -> reject
    assert not accept_step(np.array([0.2, 0.1, -0.05]), np.array([0.15, 0.1, 0.02]), tol)
    # no improvement -> reject
    assert not accept_step(np.array([0.2, 0.1, -0.05]), np.array([0.2, 0.1, -0.05]), tol)
    # non-convergent trial -> reject
    assert not accept_step(np.array([0.2, 0.1, -0.05]), None, tol)


def test_trust_region_shrinks_on_reject_grows_on_accept_capped():
    tr = TrustRegion(delta=0.05, cap=0.1)
    tr.reject()
    assert tr.delta == pytest.approx(0.025)
    tr.accept()
    tr.accept()
    tr.accept()
    assert tr.delta == pytest.approx(0.1)


def test_budget_hard_cap():
    b = Budget(cap=5)
    b.charge(3)
    assert not b.exhausted and b.remaining == 2
    b.charge(2)
    assert b.exhausted
    with pytest.raises(RuntimeError):
        b.charge(1)


def test_classify_outcome_all_four_branches():
    base = dict(
        stable=True,
        nonconv_frac=0.1,
        final_max=0.15,
        final_well=0.01,
        accepted_steps=[{"d_mirror": -0.02, "d_qi": -0.01, "d_max": -0.03, "others_worse": False}],
    )
    assert classify_outcome(**{**base, "final_max": 0.09}) == "practical_success"
    assert classify_outcome(**base) == "mechanistic_positive"
    assert classify_outcome(**{**base, "accepted_steps": []}) == "local_negative"
    assert classify_outcome(**{**base, "stable": False}) == "inconclusive"
    assert classify_outcome(**{**base, "nonconv_frac": 0.4}) == "inconclusive"
    # practical success requires well >= 0
    assert (
        classify_outcome(**{**base, "final_max": 0.09, "final_well": -0.001}) != "practical_success"
    )


# --- amendment after the first run (2026-09-15): per-constraint stability,
# linearise-subset with acceptance-only checks, Jacobian reuse -------------


def test_fd_stability_is_per_constraint_and_row_normalised():
    from constellaration_methods.restoration.p3_feasibility_restoration import fd_stability_rows

    J1 = np.array([[1.0, 0.0, 2.0], [0.001, 0.0, 0.0]])  # row 1 near-zero everywhere
    J2 = np.array([[1.02, 0.0, 2.0], [0.02, 0.0, 0.0]])  # row 1 changes a lot relatively
    disc = fd_stability_rows(J1, J2, np.zeros((0, 2)), np.zeros((0, 2)))
    assert disc[0] < 0.05  # stable row
    assert disc[1] > 0.25  # unstable row, flagged on its own
    # a near-zero ELEMENT inside a stable row must not blow up the row verdict
    J3 = np.array([[1.0, 1e-6, 2.0]])
    J4 = np.array([[1.0, 1e-3, 2.0]])
    assert fd_stability_rows(J3, J4, np.zeros((0, 1)), np.zeros((0, 1)))[0] < 0.01


def test_restoration_step_excluded_rows_are_not_linearised():
    x0 = np.zeros(3)
    d, _ = restoration_step(r(x0), A, delta=0.1, tol=0.01, linearise=[0, 1])
    # row 2 (excluded) puts no constraint on d; row 0/1 still reduced
    assert r(x0 + d)[:2].max() < r(x0)[:2].max()


def test_accept_step_checks_excluded_rows_do_not_worsen():
    tol = 0.01
    r_old = np.array([0.2, 0.1, 0.09])
    # max improved; excluded row 2 worsened by more than tol -> reject
    assert not accept_step(r_old, np.array([0.15, 0.1, 0.11]), tol, watch=[2])
    # worsened within tol -> accept
    assert accept_step(r_old, np.array([0.15, 0.1, 0.095]), tol, watch=[2])


def test_jacobian_from_log(tmp_path):
    import json

    from constellaration_methods.restoration.p3_feasibility_restoration import jacobian_from_log

    x0 = np.zeros(3)
    pts, meta = probe_points(x0, h=0.1)
    with open(tmp_path / "evaluations.jsonl", "w") as fh:
        fh.write(json.dumps({"tag": "start", "x": x0.tolist(), "r": r(x0).tolist()}) + "\n")
        for p, (j, s) in zip(pts, meta):
            fh.write(
                json.dumps(
                    {
                        "tag": f"J0:{j}:{'+' if s > 0 else '-'}",
                        "x": p.tolist(),
                        "r": r(p).tolist(),
                    }
                )
                + "\n"
            )
    J, ok, x_at, r_at = jacobian_from_log(tmp_path / "evaluations.jsonl", relin=0, n=3, h=0.1)
    assert np.allclose(J, A) and all(ok)
    assert np.allclose(x_at, x0) and np.allclose(r_at, r(x0))


def test_restoration_step_preserves_nonnegative_well_not_just_tolerance():
    # satisfied rows may not be pushed past 0 (raw well >= 0), even though the
    # benchmark tolerance would allow +tol (dry run 2026-09-15: the LP parked
    # well at +0.01, i.e. raw well slightly negative)
    x0 = np.zeros(3)
    d, _ = restoration_step(r(x0), A, delta=0.5, tol=0.01)
    assert r(x0 + d)[2] <= 0.0 + 1e-9


def test_others_worse_counts_already_violated_rows_too():
    # pre-registration: an accepted step must not worsen the OTHER constraints
    # (iota, flux, well) beyond tol -- whether or not they were already violated.
    # Run 2 step 3 worsened flux 0.095 -> 0.123 and was counted as clean.
    from constellaration_methods.restoration.p3_feasibility_restoration import others_worse

    r_old = np.array([0.05, 0.17, 0.18, 0.09, -0.06])
    assert others_worse(r_old, np.array([0.05, 0.15, 0.17, 0.12, -0.06]), tol=0.01)  # flux +0.03
    assert not others_worse(r_old, np.array([0.05, 0.15, 0.17, 0.095, -0.06]), tol=0.01)  # +0.005
    assert others_worse(r_old, np.array([0.05, 0.15, 0.17, 0.09, 0.02]), tol=0.01)  # well crosses


def test_mechanistic_positive_gain_counts_clean_joint_steps_only():
    # pre-registration: the steps that count must reduce BOTH mirror and QI
    # AND not worsen the others; the 0.02 total gain is over those steps.
    steps = [
        {"d_mirror": -0.009, "d_qi": -0.019, "d_max": -0.009, "others_worse": True},
        {"d_mirror": -0.017, "d_qi": -0.009, "d_max": -0.017, "others_worse": False},
        {"d_mirror": -0.009, "d_qi": 0.001, "d_max": -0.008, "others_worse": True},
    ]
    # total gain 0.034 but clean-step gain 0.017 -> not mechanistic_positive
    assert classify_outcome(True, 0.0, 0.144, 0.0035, steps) == "local_negative"
    steps[0]["others_worse"] = False  # now two clean steps, gain 0.026
    assert classify_outcome(True, 0.0, 0.144, 0.0035, steps) == "mechanistic_positive"


# --- paired Broyden vs full re-linearisation (reviewer round 6, 2026-09-15) --


def test_ratio_trust_region_policy():
    from constellaration_methods.restoration.p3_feasibility_restoration import RatioTrustRegion

    tr = RatioTrustRegion(delta=0.00125, cap=0.0025)
    tr.update(rho=0.1)  # poor agreement -> shrink
    assert tr.delta == pytest.approx(0.000625)
    tr.update(rho=0.5)  # moderate -> retain
    assert tr.delta == pytest.approx(0.000625)
    tr.update(rho=0.9)  # strong -> expand x1.5
    assert tr.delta == pytest.approx(0.0009375)
    for _ in range(5):
        tr.update(rho=0.95)
    assert tr.delta == pytest.approx(0.0025)  # capped
    tr.reject()
    assert tr.delta == pytest.approx(0.00125)


def test_cumulative_acceptance_anchored_to_start_and_measured_well():
    from constellaration_methods.restoration.p3_feasibility_restoration import (
        accept_step_cumulative,
    )

    r_start = np.array([0.062, 0.143, 0.152, 0.095, -0.038])
    r_old = r_start.copy()
    ok = np.array(
        [0.060, 0.135, 0.140, 0.100, -0.030]
    )  # max falls, flux +0.005 cumulative, well raw +0.003
    assert accept_step_cumulative(r_start, r_old, ok, tol=0.01)
    # cumulative flux drift beyond start + tol -> reject even though max falls
    bad_flux = np.array([0.060, 0.135, 0.140, 0.106, -0.030])
    assert not accept_step_cumulative(r_start, r_old, bad_flux, tol=0.01)
    # measured raw well negative (normalised residual > 0) -> reject
    bad_well = np.array([0.060, 0.135, 0.140, 0.095, 0.002])
    assert not accept_step_cumulative(r_start, r_old, bad_well, tol=0.01)
    # a later step is judged against the START, not the previous incumbent
    r_old2 = np.array([0.060, 0.135, 0.140, 0.104, -0.030])
    creep = np.array([0.060, 0.130, 0.135, 0.109, -0.030])  # +0.005 vs old but +0.014 vs start
    assert not accept_step_cumulative(r_start, r_old2, creep, tol=0.01)


def test_broyden_update_is_exact_on_linear_model_and_safeguarded():
    from constellaration_methods.restoration.p3_feasibility_restoration import broyden_update

    J = A + 0.3 * np.ones_like(A)  # wrong initial model
    x_old = np.zeros(3)
    dx = np.array([0.001, -0.002, 0.0005])
    dr = r(x_old + dx) - r(x_old)
    J1, applied = broyden_update(J, dx, dr)
    assert applied
    # after the update the model is exact along dx
    assert np.allclose(J1 @ dx, dr)
    # tiny step -> skipped
    _, applied = broyden_update(J, np.zeros(3) + 1e-9, np.zeros(3))
    assert not applied
    # a wild residual change that would blow a row up by > 50% -> skipped
    _, applied = broyden_update(A, dx, dr + np.array([50.0, 0.0, 0.0]))
    assert not applied


def test_drift_check_and_partial_refresh_selection():
    from constellaration_methods.restoration.p3_feasibility_restoration import (
        drift_rows,
        worst_columns,
    )

    J = A.copy()
    dirs = np.eye(3)[:2]
    measured = np.array([A @ d for d in dirs])
    disc = drift_rows(J, dirs, measured)
    assert np.all(disc < 1e-9)
    J_bad = A.copy()
    J_bad[0] *= 2.0  # row 0 drifted
    disc = drift_rows(J_bad, dirs, measured)
    assert disc[0] > 0.25 and disc[1] < 1e-9
    # the columns that most dominate row 0 of |J|
    assert worst_columns(J_bad, row=0, k=2) == [2, 0]


def test_start_candidate_loads_x_r_and_jacobian_pointer(tmp_path):
    import json

    from constellaration_methods.restoration.p3_feasibility_restoration import load_start_candidate

    x0 = np.zeros(3)
    pts, meta = probe_points(x0, h=0.1)
    log = tmp_path / "evaluations.jsonl"
    with open(log, "w") as fh:
        fh.write(json.dumps({"tag": "start", "x": x0.tolist(), "r": r(x0).tolist()}) + "\n")
        for p, (j, s) in zip(pts, meta):
            fh.write(
                json.dumps(
                    {
                        "tag": f"J2:{j}:{'+' if s > 0 else '-'}",
                        "x": p.tolist(),
                        "r": r(p).tolist(),
                    }
                )
                + "\n"
            )
    cand = tmp_path / "cand.json"
    cand.write_text(
        json.dumps(
            {
                "x": x0.tolist(),
                "r": r(x0).tolist(),
                "raw": {},
                "boundary_dict": {},
                "mpol": 4,
                "ntor": 4,
                "nfp": 3,
                "scaling": 1.0,
                "jacobian": {"log": str(log), "relin": 2, "h": 0.1},
            }
        )
    )
    x, rr, J, ok, rep = load_start_candidate(cand, n=3)
    assert np.allclose(x, x0) and np.allclose(rr, r(x0)) and np.allclose(J, A) and all(ok)
    assert rep["mpol"] == 4


def test_restoration_step_respects_cumulative_caps():
    # the LP must not propose what acceptance would reject: iota/flux/well are
    # capped at start + tol in the linear model as well
    x0 = np.zeros(3)
    r_start = r(x0)
    d, _ = restoration_step(
        r(x0), A, delta=0.5, tol=0.01, caps={2: r_start[2] + 0.01, 0: r_start[0] + 0.01}
    )
    pred = r(x0 + d)
    assert pred[2] <= r_start[2] + 0.01 + 1e-9
    assert pred[0] <= r_start[0] + 0.01 + 1e-9


def test_full_arm_keeps_trialling_when_relinearisation_does_not_fit():
    # Arm F stopped at 166/300 because a 160-eval refresh would not fit; it
    # should have continued trials on the existing model until the allowance
    # is spent (the stale model is still a model; acceptance is canonical).
    from constellaration_methods.restoration.p3_feasibility_restoration import relinearisation_fits

    assert relinearisation_fits(remaining=170, n=80)
    assert not relinearisation_fits(remaining=134, n=80)


def test_stall_counts_null_accepted_steps_and_radius_floor():
    # Arm B cycled at the radius floor accepting steps with gains in the 5th
    # decimal; those must count toward the stall, and reaching the floor must
    # end the arm rather than spend the allowance on null steps.
    from constellaration_methods.restoration.p3_feasibility_restoration import StallCounter

    st = StallCounter(max_in_row=3, min_gain=1e-4)
    st.record(accepted=True, gain=0.01)
    assert not st.stalled
    st.record(accepted=True, gain=0.00005)
    st.record(accepted=False, gain=None)
    st.record(accepted=True, gain=0.00002)
    assert st.stalled
    st2 = StallCounter(max_in_row=3, min_gain=1e-4)
    assert st2.at_floor(delta=1e-5, floor=1e-5)
    assert not st2.at_floor(delta=2e-5, floor=1e-5)


# --- reviewer round 7 (2026-09-15): monotone-V acceptance, no separate flux cap --


def test_monotone_v_acceptance_lets_flux_rise_under_a_falling_maximum():
    from constellaration_methods.restoration.p3_feasibility_restoration import (
        accept_step_monotone_v,
    )

    r_start = np.array([0.062, 0.143, 0.152, 0.095, -0.038])
    r_old = np.array([0.061, 0.141, 0.141, 0.100, -0.039])  # V = 0.141
    ok = np.array([0.058, 0.137, 0.136, 0.108, -0.036])  # V = 0.137, flux 0.108 (over the old cap)
    assert accept_step_monotone_v(r_start, r_old, ok, tol=0.01, eps=1e-4)
    # flux becoming the worst constraint above V_old - eps -> reject
    bad = np.array([0.058, 0.130, 0.130, 0.145, -0.036])
    assert not accept_step_monotone_v(r_start, r_old, bad, tol=0.01, eps=1e-4)
    # not a material improvement -> reject
    flat = np.array([0.061, 0.141, 0.14095, 0.100, -0.039])
    assert not accept_step_monotone_v(r_start, r_old, flat, tol=0.01, eps=1e-4)
    # raw well negative -> reject
    well_bad = np.array([0.058, 0.137, 0.136, 0.108, 0.001])
    assert not accept_step_monotone_v(r_start, r_old, well_bad, tol=0.01, eps=1e-4)
    # iota protection retained (anchored to start + tol)
    iota_bad = np.array([0.075, 0.137, 0.136, 0.108, -0.036])
    assert not accept_step_monotone_v(r_start, r_old, iota_bad, tol=0.01, eps=1e-4)
    # a constraint that reached tolerance must stay there
    r_old2 = np.array([0.005, 0.141, 0.141, 0.100, -0.039])
    lost = np.array([0.02, 0.137, 0.136, 0.108, -0.036])
    assert not accept_step_monotone_v(r_start, r_old2, lost, tol=0.01, eps=1e-4)


def test_lp_infeasible_is_reported_not_silently_relaxed():
    from constellaration_methods.restoration.p3_feasibility_restoration import (
        RestorationLPInfeasible,
        restoration_step,
    )

    # caps that exclude every step (cap below the current residual) must raise,
    # not fall back to an LP without the preservation rows
    x0 = np.zeros(3)
    with pytest.raises(RestorationLPInfeasible):
        restoration_step(r(x0), A, delta=0.001, tol=0.01, caps={0: r(x0)[0] - 1.0})


def test_broyden_update_with_a_frozen_nan_column_does_not_poison_the_model():
    # E4 (2026-09-16): one Jacobian column was NaN (non-convergent probe); the
    # update propagated NaN through J @ dx into every row and the safeguard let
    # a non-finite result through, so the LP saw an all-zero model.
    from constellaration_methods.restoration.p3_feasibility_restoration import broyden_update

    J = A.astype(float).copy()
    J[:, 1] = np.nan  # frozen column
    dx = np.array([0.001, 0.0, 0.0005])  # frozen coordinate not moved
    dr = A @ dx
    J1, applied = broyden_update(J, dx, dr)
    assert applied
    assert np.all(np.isnan(J1[:, 1]))  # frozen column stays frozen
    assert np.all(np.isfinite(np.delete(J1, 1, axis=1)))
    assert np.allclose(np.delete(J1, 1, axis=1) @ np.delete(dx, 1), dr)  # exact along dx
    # a non-finite dr must be refused, never applied
    _, applied2 = broyden_update(J, dx, np.array([np.nan, 0.0, 0.0]))
    assert not applied2


def test_restoration_step_per_coordinate_radius():
    # E6b: an anisotropic trust box, |d_j| <= delta * mult_j
    x0 = np.zeros(3)
    mult = np.array([1.0, 10.0, 0.5])
    d, _ = restoration_step(r(x0), A, delta=0.01, tol=0.01, radius_mult=mult)
    assert np.all(np.abs(d) <= 0.01 * mult + 1e-12)
    # and the box is actually used where it helps (some coordinate near its bound)
    assert np.max(np.abs(d) / (0.01 * mult)) > 0.99


def test_jacobian_from_archive_reuses_probe_rows_without_a_start_row(tmp_path):
    # E6b: the final runner's archive.jsonl has J0:{j}:{+|-} rows but no
    # 'start' row (the start lives in start_serialised.json)
    import json

    from constellaration_methods.restoration.p3_feasibility_restoration import jacobian_from_archive

    x0 = np.zeros(3)
    pts, meta = probe_points(x0, h=0.1)
    with open(tmp_path / "archive.jsonl", "w") as fh:
        for p, (j, s) in zip(pts, meta):
            fh.write(
                json.dumps(
                    {
                        "tag": f"J0:{j}:{'+' if s > 0 else '-'}",
                        "x": p.tolist(),
                        "r": r(p).tolist(),
                    }
                )
                + "\n"
            )
        fh.write(json.dumps({"tag": "JR:0:+", "x": [9, 9, 9], "r": [1, 1, 1]}) + "\n")
    J, ok = jacobian_from_archive(tmp_path / "archive.jsonl", "J0", n=3, h=0.1)
    assert np.allclose(J, A) and all(ok)


# ---- E7: diagonal second-order model from step-scale probes ----------------
Q = np.array([[4.0, 0.0, 1.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]])  # diagonal curvature per residual


def r_quad(x):
    return A @ x + B + 0.5 * Q @ (x**2)


def test_probe_points_accept_per_coordinate_steps():
    pts, meta = probe_points(np.zeros(3), 0.01, h_per_coord=[0.01, 0.5, 0.1])
    assert np.allclose(pts[2], [0.0, 0.5, 0.0]) and np.allclose(pts[3], [0.0, -0.5, 0.0])
    assert meta[2] == (1, +1) and meta[5] == (2, -1)


def test_central_difference_jacobian_with_per_coordinate_steps_and_curvature():
    from constellaration_methods.restoration.p3_feasibility_restoration import (
        central_difference_curvature,
    )

    h = np.array([0.01, 0.5, 0.1])
    x0 = np.zeros(3)
    pts, meta = probe_points(x0, 0.01, h_per_coord=h)
    res = [r_quad(p) for p in pts]
    J, ok = central_difference_jacobian(res, meta, n=3, h=h)
    assert np.allclose(J, A) and all(ok)  # central differences are exact for a quadratic
    C, okc = central_difference_curvature(res, meta, n=3, h=h, r0=r_quad(x0))
    assert np.allclose(C, Q) and all(okc)
    res[2] = None  # a non-convergent probe leaves that column NaN in both
    J2, ok2 = central_difference_jacobian(res, meta, n=3, h=h)
    C2, okc2 = central_difference_curvature(res, meta, n=3, h=h, r0=r_quad(x0))
    assert np.isnan(J2[:, 1]).all() and np.isnan(C2[:, 1]).all() and not ok2[1] and not okc2[1]


def test_quadratic_restoration_step_stops_at_the_bowl_where_the_lp_overshoots():
    # (rewritten 2026-09-18: the earlier toy's satisfied row happened to be the
    # binding row under the old 0 bound, so the LP never reached the bowl; here
    # the violated rows are pure bowls and the satisfied row is uncoupled)
    from constellaration_methods.restoration.p3_feasibility_restoration import (
        restoration_step_quadratic,
    )

    A2 = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    B2 = np.array([0.2, 0.1, -0.05])
    Q2 = np.array([[4.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]])

    def r2(x):
        return A2 @ x + B2 + 0.5 * Q2 @ (x**2)

    x0 = np.zeros(3)
    r0 = r2(x0)
    delta = 1.0  # a wide box: the linear model runs to the edge
    d_lin, _ = restoration_step(r0, A2, delta, tol=0.01)
    d_q, t_q, model_used = restoration_step_quadratic(r0, A2, Q2, delta, tol=0.01)
    true_lin = r2(x0 + d_lin).max()
    true_q = r2(x0 + d_q).max()
    assert true_q < true_lin  # the quadratic step is better where the model is exact
    assert abs(true_q - t_q) < 1e-6  # and its prediction is exact for a quadratic
    assert np.all(np.abs(d_q) <= delta + 1e-9)
    assert r2(x0 + d_q)[2] <= 0.01 + 1e-9  # satisfied row within tolerance
    assert model_used == "diagonal"  # no lowrank given: the only rung is diagonal


def test_quadratic_restoration_step_reduces_to_the_lp_without_curvature():
    from constellaration_methods.restoration.p3_feasibility_restoration import (
        restoration_step_quadratic,
    )

    r0 = r(np.zeros(3))
    d_lin, t_lin = restoration_step(r0, A, 0.01, tol=0.01)
    d_q, t_q, _ = restoration_step_quadratic(r0, A, np.zeros_like(A), 0.01, tol=0.01)
    assert abs(t_q - t_lin) < 1e-6


def test_quadratic_restoration_step_honours_caps_and_per_coordinate_radius():
    from constellaration_methods.restoration.p3_feasibility_restoration import (
        RestorationLPInfeasible,
        restoration_step_quadratic,
    )

    r0 = r_quad(np.zeros(3))
    mult = np.array([1.0, 10.0, 0.5])
    d, _, _ = restoration_step_quadratic(r0, A, Q, 0.01, tol=0.01, radius_mult=mult)
    assert np.all(np.abs(d) <= 0.01 * mult + 1e-9)
    import pytest as _pytest

    with _pytest.raises(RestorationLPInfeasible):
        restoration_step_quadratic(r0, A, Q, 0.01, tol=0.01, caps={0: -10.0})


def test_curvature_from_archive_rebuilds_c_from_probe_rows(tmp_path):
    import json

    from constellaration_methods.restoration.p3_feasibility_restoration import (
        curvature_from_archive,
    )

    h = np.array([0.01, 0.5, 0.1])
    x0 = np.zeros(3)
    pts, meta = probe_points(x0, 0.01, h_per_coord=h)
    with open(tmp_path / "archive.jsonl", "w") as fh:
        for p, (j, s) in zip(pts, meta):
            fh.write(
                json.dumps(
                    {
                        "tag": f"J0:{j}:{'+' if s > 0 else '-'}",
                        "x": p.tolist(),
                        "r": r_quad(p).tolist(),
                    }
                )
                + "\n"
            )
    C, ok = curvature_from_archive(tmp_path / "archive.jsonl", "J0", n=3, h=h, r0=r_quad(x0))
    assert np.allclose(C, Q) and all(ok)


# ---- review round 8 (2026-09-18): the step model must not be stricter than acceptance
def test_restoration_steps_bound_satisfied_rows_at_the_tolerance_not_zero():
    # review round 8: a satisfied row at +0.008 (inside tol 0.01) that is COUPLED to
    # the descent (it rises slowly as the violated row falls) must be allowed to use
    # the remaining tolerance; the old bound min(tol, 0) = 0 forced it below zero,
    # which here means climbing the violated row instead
    from constellaration_methods.restoration.p3_feasibility_restoration import (
        restoration_step_quadratic,
    )

    r0 = np.array([0.008, 0.2, -0.05])
    Aq = np.array([[0.05, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    d_lp, _ = restoration_step(r0, Aq, 0.05, tol=0.01)
    d_qp, _, _ = restoration_step_quadratic(r0, Aq, np.zeros_like(Aq), 0.05, tol=0.01)
    for d in (d_lp, d_qp):
        pred = r0 + Aq @ d
        assert pred[1] < r0[1] - 0.01  # the violated row falls
        assert pred[0] <= 0.01 + 1e-9  # the satisfied row may use the tolerance


def test_restoration_steps_keep_the_well_at_zero_once_reached():
    from constellaration_methods.restoration.p3_feasibility_restoration import (
        restoration_step_quadratic,
    )

    # 5 rows, well last and satisfied at -0.001: with the satisfied-row bound now
    # at tol, the raw-well requirement (residual <= 0) is the runner's cap
    # (well_cap_for), passed here explicitly (post-Green change, documented)
    r0 = np.array([0.1, 0.1, 0.1, 0.1, -0.001])
    J = np.zeros((5, 2))
    J[:4, 0] = -1.0  # x0 up lowers the violated rows
    J[4, 0] = 1.0  # ...and raises the well residual
    d, _ = restoration_step(r0, J, 0.05, tol=0.01, caps={4: 0.0})
    assert (r0 + J @ d)[4] <= 0.0 + 1e-9
    d, _, _ = restoration_step_quadratic(r0, J, np.zeros_like(J), 0.05, tol=0.01, caps={4: 0.0})
    assert (r0 + J @ d)[4] <= 0.0 + 1e-9


# --------------------------------------------------------------------------- #
# E19 item 1 -- subspace (off-diagonal) curvature: direction selection from
# prior-lineage archives, SVD fill, subspace probes, block extraction, and
# the quadratic step under a low-rank cross term.
# --------------------------------------------------------------------------- #
def _write_fake_archive(out_dir, x0, accepted_specs, recovery=None):
    """A minimal p3_final_restoration-shaped archive directory: REPORT.json
    (accepted list of short-tag/r pairs), start_serialised.json (x0),
    archive.jsonl (every probed/trialled point, keyed by the LONG tag with
    the ':d...:t...' suffix p3_final_restoration actually writes), and
    optionally events.jsonl with one 'recovery_restart' event. `accepted_specs`
    is a list of (short_tag, d) displacement vectors applied in sequence from
    x0 (or from the recovery point once `recovery` fires)."""
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "start_serialised.json").write_text(json.dumps({"x": list(x0)}))
    archive_rows = []
    accepted = []
    x = np.asarray(x0, float)
    recovery_tag, recovery_x = None, None
    if recovery is not None:
        recovery_x = np.asarray(recovery, float)
        recovery_tag = "P:T99:d0.001:t1.00"
        archive_rows.append({"tag": recovery_tag, "x": recovery_x.tolist(), "r": [0.0]})
    applied_recovery = False
    for step_index, (short_tag, d) in enumerate(accepted_specs):
        phase = short_tag.split(":", 1)[0]
        is_recovery_step = phase.startswith("R") and phase[1:].isdigit()
        if recovery is not None and is_recovery_step and not applied_recovery:
            applied_recovery = True
            base = recovery_x
        else:
            base = x
        x_after = base + np.asarray(d, float)
        full_tag = f"{short_tag}:d0.001:t1.00"
        r_row = [float(step_index)]  # arbitrary, only used to match by value
        archive_rows.append({"tag": full_tag, "x": x_after.tolist(), "r": r_row})
        accepted.append({"tag": short_tag, "r": r_row})
        x = x_after
    (out_dir / "archive.jsonl").write_text(
        "\n".join(json.dumps(row) for row in archive_rows) + "\n"
    )
    (out_dir / "REPORT.json").write_text(json.dumps({"accepted": accepted}))
    if recovery_tag is not None:
        (out_dir / "events.jsonl").write_text(
            json.dumps({"event": "recovery_restart", "tag": recovery_tag}) + "\n"
        )
    else:
        (out_dir / "events.jsonl").write_text("")


def test_accepted_step_displacements_chains_through_accepted_steps(tmp_path):
    from constellaration_methods.restoration.p3_feasibility_restoration import (
        accepted_step_displacements,
    )

    _write_fake_archive(
        tmp_path,
        x0=[0.0, 0.0, 0.0],
        accepted_specs=[("P:T0", [0.1, 0.0, 0.0]), ("P:T1", [0.0, 0.2, 0.0])],
    )
    ds = accepted_step_displacements(tmp_path)
    assert len(ds) == 2
    assert np.allclose(ds[0], [0.1, 0.0, 0.0])
    assert np.allclose(ds[1], [0.0, 0.2, 0.0])  # chained from P:T0's x_after


def test_accepted_step_displacements_uses_the_recovery_point_not_the_prior_step(
    tmp_path,
):
    from constellaration_methods.restoration.p3_feasibility_restoration import (
        accepted_step_displacements,
    )

    # after the recovery restart, the first R-tagged accepted step's "before"
    # point is the recovery point, not the previous (P-tagged) accepted step
    _write_fake_archive(
        tmp_path,
        x0=[0.0, 0.0, 0.0],
        accepted_specs=[("P:T0", [0.1, 0.0, 0.0]), ("R0:T5", [0.0, 0.0, 0.3])],
        recovery=[0.05, 0.0, 0.0],  # NOT the same as P:T0's x_after ([0.1, 0, 0])
    )
    ds = accepted_step_displacements(tmp_path)
    assert np.allclose(ds[0], [0.1, 0.0, 0.0])
    assert np.allclose(ds[1], [0.0, 0.0, 0.3])  # measured from the recovery point


def test_directions_from_archives_orthonormal_and_capped_at_k():
    from constellaration_methods.restoration.p3_feasibility_restoration import (
        directions_from_archives,
    )

    def make(tmp, x0, specs, recovery=None):
        _write_fake_archive(tmp, x0, specs, recovery)
        return tmp

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p1 = Path(td) / "a1"
        p2 = Path(td) / "a2"
        make(p1, [0.0, 0.0, 0.0, 0.0], [("P:T0", [1.0, 0.0, 0.0, 0.0])])
        make(
            p2,
            [0.0, 0.0, 0.0, 0.0],
            [
                ("P:T0", [0.0, 2.0, 0.0, 0.0]),
                ("P:T1", [1.0, 0.0, 0.0, 0.0]),  # duplicate direction of a1's step
            ],
        )
        U = directions_from_archives([p1, p2], n=4, k=5)
        assert U.shape[1] == 4
        assert U.shape[0] <= 3  # at most 2 distinct directions + no third source
        # orthonormal
        assert np.allclose(U @ U.T, np.eye(U.shape[0]), atol=1e-9)
        # capped: asking for k=1 keeps only the most-recent direction (a2's last step)
        U1 = directions_from_archives([p1, p2], n=4, k=1)
        assert U1.shape == (1, 4)
        assert np.allclose(np.abs(U1[0]), [1.0, 0.0, 0.0, 0.0], atol=1e-9)


def test_fill_with_svd_appends_orthonormal_top_singular_vectors():
    from constellaration_methods.restoration.p3_feasibility_restoration import fill_with_svd

    J = np.array([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    U0 = np.zeros((0, 3))
    U = fill_with_svd(U0, J, k=2)
    assert U.shape == (2, 3)
    assert np.allclose(U @ U.T, np.eye(2), atol=1e-9)
    # dominant singular direction (largest singular value, coordinate 0) first
    assert abs(U[0, 0]) > 0.99
    # given an existing direction, svd fill orthogonalises against it
    U_seed = np.array([[1.0, 0.0, 0.0]])
    U2 = fill_with_svd(U_seed, J, k=2)
    assert U2.shape == (2, 3)
    assert np.allclose(U2[0], U_seed[0])
    assert abs(U2[1] @ U_seed[0]) < 1e-9


def test_subspace_probe_points_counts_and_tags():
    from constellaration_methods.restoration.p3_feasibility_restoration import subspace_probe_points

    x0 = np.zeros(5)
    U = np.eye(3, 5)  # K=3 orthonormal directions in R^5
    pts, meta = subspace_probe_points(x0, U, h=0.1)
    K = 3
    assert pts.shape == (2 * K + K * (K - 1), 5)
    assert len(meta) == len(pts)
    diag_tags = [t for t in meta if t[0] == "diag"]
    pair_tags = [t for t in meta if t[0] == "pair"]
    assert len(diag_tags) == 2 * K
    assert len(pair_tags) == K * (K - 1)
    assert np.allclose(pts[0], [0.1, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(pts[1], [-0.1, 0.0, 0.0, 0.0, 0.0])
    # a pair probe is x0 +- h(u_i + u_j)
    pair_idx = meta.index(("pair", 0, 1, +1))
    assert np.allclose(pts[pair_idx], [0.1, 0.1, 0.0, 0.0, 0.0])


def test_subspace_curvature_from_probes_matches_a_known_quadratic():
    from constellaration_methods.restoration.p3_feasibility_restoration import (
        subspace_curvature_from_probes,
        subspace_probe_points,
    )

    # a single-residual quadratic with a KNOWN K x K Hessian block S_true in
    # the subspace spanned by U = I[:3] (coordinates 0,1,2 of R^4)
    S_true = np.array([[4.0, 1.5, -0.5], [1.5, 3.0, 0.7], [-0.5, 0.7, 2.0]])
    U = np.eye(3, 4)

    def f(x):
        xs = x[:3]
        return np.array([float(xs @ S_true @ xs) / 2.0])

    x0 = np.array([0.3, -0.2, 0.1, 0.0])
    r0 = f(x0)
    h = 1e-3
    pts, meta = subspace_probe_points(x0, U, h)
    residuals = [f(p) for p in pts]
    S = subspace_curvature_from_probes(residuals, meta, r0, U, h)
    assert S.shape == (1, 3, 3)
    assert np.allclose(S[0], S_true, atol=1e-6)


def test_subspace_curvature_from_probes_marks_missing_probe_entries_nan():
    from constellaration_methods.restoration.p3_feasibility_restoration import (
        subspace_curvature_from_probes,
        subspace_probe_points,
    )

    S_true = np.eye(3) * 2.0
    U = np.eye(3, 3)

    def f(x):
        return np.array([float(x @ S_true @ x) / 2.0])

    x0 = np.zeros(3)
    r0 = f(x0)
    h = 1e-3
    pts, meta = subspace_probe_points(x0, U, h)
    residuals = [f(p) for p in pts]
    # a non-convergent pair probe (0,1)+
    idx = meta.index(("pair", 0, 1, +1))
    residuals[idx] = None
    S = subspace_curvature_from_probes(residuals, meta, r0, U, h)
    assert np.isnan(S[0, 0, 1]) and np.isnan(S[0, 1, 0])
    assert not np.isnan(S[0, 0, 0]) and not np.isnan(S[0, 2, 2])  # unaffected entries


def test_restoration_step_quadratic_lowrank_none_is_byte_identical():
    from constellaration_methods.restoration.p3_feasibility_restoration import (
        restoration_step_quadratic,
    )

    r0 = r_quad(np.zeros(3))
    d1, t1, m1 = restoration_step_quadratic(r0, A, Q, 0.05, tol=0.01)
    d2, t2, m2 = restoration_step_quadratic(r0, A, Q, 0.05, tol=0.01, lowrank=None)
    assert np.array_equal(d1, d2) and t1 == t2
    assert m1 == m2 == "diagonal"  # no subspace to fall back from: one rung only


def test_restoration_step_quadratic_lowrank_uses_measured_cross_terms():
    from constellaration_methods.restoration.p3_feasibility_restoration import (
        restoration_step_quadratic,
    )

    # row 0 has a real, strong cross term between coords 0 and 1 that the
    # diagonal-only model cannot see; giving the exact low-rank block must
    # change the proposed step relative to the diagonal-only model
    A3 = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    B3 = np.array([0.2, 0.15, -0.05])
    C3 = np.zeros((3, 3))  # no per-axis diagonal curvature at all
    U = np.eye(2, 3)
    S = np.zeros((3, 2, 2))
    S[0] = [[0.0, 6.0], [6.0, 0.0]]  # row 0: pure cross term between x0, x1
    r0 = B3.copy()
    d_diag, _, m_diag = restoration_step_quadratic(r0, A3, C3, delta=0.3, tol=0.01)
    d_lr, _, m_lr = restoration_step_quadratic(r0, A3, C3, delta=0.3, tol=0.01, lowrank=(U, S))
    assert not np.allclose(d_diag, d_lr)
    assert m_diag == "diagonal"
    assert m_lr == "subspace"  # the solver succeeded on the first rung


def test_restoration_step_quadratic_lowrank_convexifies_a_negative_eigenvalue_block():
    # E19 fix (reviewer round: the 8-attempt SLSQP failure on the real smoke):
    # clipping AFTER summing diagonal + cross makes the per-row model non-
    # convex wherever the combined block has a negative eigenvalue, which is
    # exactly what SLSQP choked on. The fix eigen-clips the row's measured
    # K x K block (ambient diagonal projected into the subspace, plus the
    # measured off-diagonal) BEFORE folding it back in, so the model is
    # provably convex; this test's row 0 block [[0,-6],[-6,0]] has one
    # negative eigenvalue (-6) by construction.
    from constellaration_methods.restoration.p3_feasibility_restoration import (
        restoration_step_quadratic,
    )

    A3 = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    r0 = np.array([0.2, 0.15, -0.05])
    C3 = np.zeros((3, 3))
    U = np.eye(2, 3)
    S = np.zeros((3, 2, 2))
    S[0] = [[0.0, -6.0], [-6.0, 0.0]]  # indefinite cross curvature, row 0
    d, t_pred, model_used = restoration_step_quadratic(
        r0, A3, C3, delta=0.3, tol=0.01, lowrank=(U, S)
    )
    assert model_used == "subspace"
    assert t_pred >= -1.0  # sanity: solver returned a real prediction
    # the row's OWN (convexified) model prediction is never below the
    # linear-only value -- the eigen-clipped block can never contribute
    # negative curvature, by construction
    from constellaration_methods.restoration.p3_feasibility_restoration import (
        quadratic_model_step_terms,
    )

    row0_term = quadratic_model_step_terms(C3, d, lowrank=(U, S))[0]
    assert row0_term >= -1e-9


def test_quadratic_model_step_terms_lowrank_convexifies_indefinite_block_for_random_d():
    # item 1's central invariant: whatever the measured off-diagonal block
    # looks like, the row's quadratic term must be >= 0 for EVERY d, not
    # just the ones a particular test happens to pick.
    from constellaration_methods.restoration.p3_feasibility_restoration import (
        quadratic_model_step_terms,
    )

    rng = np.random.default_rng(20260924)
    n, K = 6, 3
    Qr, _ = np.linalg.qr(rng.standard_normal((n, K)))
    U = Qr[:, :K].T  # K x n orthonormal rows, NOT axis-aligned
    C = np.abs(rng.standard_normal((1, n))) * rng.uniform(1.0, 50.0, size=n)
    S = np.zeros((1, K, K))
    cross = rng.standard_normal((K, K))
    cross = (cross + cross.T) / 2.0
    np.fill_diagonal(cross, 0.0)
    S[0] = cross * 80.0  # scaled up to guarantee an indefinite raw block

    # sanity: the raw (un-convexified) block really is indefinite, so this
    # test actually exercises the fix rather than a case that was fine anyway
    M = (U * C[0]) @ U.T
    raw_eigvals = np.linalg.eigvalsh(M + S[0])
    assert raw_eigvals.min() < 0.0

    for _ in range(500):
        d = rng.standard_normal(n) * rng.uniform(0.01, 5.0)
        term = quadratic_model_step_terms(C, d, lowrank=(U, S))[0]
        assert term >= -1e-9


def test_quadratic_model_step_terms_lowrank_unchanged_outside_the_subspace():
    # the docstring's other half of the guarantee: for a direction entirely
    # orthogonal to U's row space, the measured cross terms must not matter
    # at all -- the row's model is exactly its (already-clipped) diagonal.
    from constellaration_methods.restoration.p3_feasibility_restoration import (
        quadratic_model_step_terms,
    )

    U = np.eye(2, 3)  # spans coordinates 0, 1; coordinate 2 is outside
    C = np.array([[5.0, 3.0, 7.0]])
    S = np.zeros((1, 2, 2))
    S[0] = [[0.0, 100.0], [100.0, 0.0]]  # huge cross term, irrelevant here
    d = np.array([0.0, 0.0, 2.0])  # entirely outside the subspace
    terms_lr = quadratic_model_step_terms(C, d, lowrank=(U, S))
    terms_diag = quadratic_model_step_terms(C, d)
    assert terms_lr[0] == pytest.approx(terms_diag[0])
    assert terms_lr[0] == pytest.approx(0.5 * 7.0 * 4.0)  # = 0.5 * C[0,2] * d[2]**2


def test_restoration_step_quadratic_lowrank_falls_back_through_the_ladder_on_solver_failure(
    monkeypatch,
):
    # item 2: a planted solver failure must fall through subspace ->
    # subspace_half -> diagonal, in order, and record which rung succeeded.
    import scipy.optimize

    from constellaration_methods.restoration.p3_feasibility_restoration import (
        restoration_step_quadratic,
    )

    real_minimize = scipy.optimize.minimize
    calls = {"n": 0}

    def flaky_minimize(*args, **kwargs):
        calls["n"] += 1
        res = real_minimize(*args, **kwargs)
        if calls["n"] < 3:  # fail the first two rungs regardless of their result
            res.success = False
            res.message = "planted failure"
        return res

    monkeypatch.setattr(scipy.optimize, "minimize", flaky_minimize)

    A3 = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    r0 = np.array([0.2, 0.15, -0.05])
    C3 = np.zeros((3, 3))
    U = np.eye(2, 3)
    S = np.zeros((3, 2, 2))
    S[0] = [[0.0, 6.0], [6.0, 0.0]]
    d, t_pred, model_used = restoration_step_quadratic(
        r0, A3, C3, delta=0.3, tol=0.01, lowrank=(U, S)
    )
    assert calls["n"] == 3  # subspace, subspace_half, diagonal -- all three tried
    assert model_used == "diagonal"


def test_restoration_step_quadratic_lowrank_raises_when_every_rung_fails(monkeypatch):
    import scipy.optimize

    from constellaration_methods.restoration.p3_feasibility_restoration import (
        RestorationLPInfeasible,
        restoration_step_quadratic,
    )

    real_minimize = scipy.optimize.minimize
    calls = {"n": 0}

    def always_fails(*args, **kwargs):
        calls["n"] += 1
        res = real_minimize(*args, **kwargs)
        res.success = False
        res.message = "planted failure"
        return res

    monkeypatch.setattr(scipy.optimize, "minimize", always_fails)

    A3 = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    r0 = np.array([0.2, 0.15, -0.05])
    C3 = np.zeros((3, 3))
    U = np.eye(2, 3)
    S = np.zeros((3, 2, 2))
    S[0] = [[0.0, 6.0], [6.0, 0.0]]
    with pytest.raises(RestorationLPInfeasible):
        restoration_step_quadratic(r0, A3, C3, delta=0.3, tol=0.01, lowrank=(U, S))
    assert calls["n"] == 3  # every rung of the ladder was attempted


def test_fake_quadratic_coupling_adds_a_cross_term_to_row_one():
    from constellaration_methods.restoration.p3_feasibility_restoration import (
        _fake_quadratic_residuals,
    )

    coupling = np.array([[0.0, 5.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    x_plain = np.array([0.3, 0.0, 0.0, 0.0, 0.0])
    x_coupled = np.array([0.3, 0.3, 0.0, 0.0, 0.0])
    base = _fake_quadratic_residuals(x_plain)
    with_coupling_off = _fake_quadratic_residuals(x_plain, coupling=coupling)
    assert base[1] == pytest.approx(with_coupling_off[1])  # x1=0 -> no cross term yet
    with_coupling_on = _fake_quadratic_residuals(x_coupled, coupling=coupling)
    no_coupling_on = _fake_quadratic_residuals(x_coupled, coupling=None)
    # the coupling term (0.5 * 2 * 5 * 0.3 * 0.3 = 0.45) shows up only in row 1
    assert with_coupling_on[1] - no_coupling_on[1] == pytest.approx(0.45)
    for i in (0, 2, 3, 4):
        assert with_coupling_on[i] == pytest.approx(no_coupling_on[i])


def test_quadratic_model_step_terms_matches_restoration_step_quadratics_own_model():
    # (rewritten for E19's convexification fix, 2026-09-24: the previous
    # version hardcoded the removed clip-AFTER-sum formula
    # `0.5 * max(Q[0] @ (d*d) + 2*S_01*z0*z1, 0.0)`, which is exactly the
    # non-convex, non-smooth model that made SLSQP fail on the real smoke
    # (Iteration limit reached / Positive directional derivative for
    # linesearch, all 8 attempts). The new model eigen-clips the row's
    # measured block BEFORE folding it back in (see
    # test_quadratic_model_step_terms_lowrank_convexifies_indefinite_block_for_random_d
    # for the general invariant this guarantees); this test now checks
    # restoration_step_quadratic's own model function (via the d it returns)
    # is consistent with quadratic_model_step_terms, plus the two structural
    # guarantees the docstring promises: untouched rows unchanged, and
    # byte-identical to the diagonal-only model outside the subspace.
    from constellaration_methods.restoration.p3_feasibility_restoration import (
        quadratic_model_step_terms,
    )

    U = np.eye(2, 3)
    S = np.zeros((3, 2, 2))
    S[0] = [[0.0, 4.0], [4.0, 0.0]]
    d = np.array([0.1, -0.2, 0.05])
    terms_diag_only = quadratic_model_step_terms(Q, d)
    assert np.allclose(terms_diag_only, 0.5 * Q @ (d * d))
    terms_lowrank = quadratic_model_step_terms(Q, d, lowrank=(U, S))
    # untouched rows (no cross term for rows 1, 2) are unaffected
    assert np.allclose(terms_lowrank[1:], terms_diag_only[1:])
    # row 0 does change relative to the diagonal-only model (the cross term
    # is not simply dropped) and stays non-negative (never predicts a step
    # is better than the linear term alone)
    assert terms_lowrank[0] != pytest.approx(terms_diag_only[0])
    assert terms_lowrank[0] >= -1e-9


def test_quadratic_model_step_terms_lowrank_is_never_negative_for_a_full_rank_subspace():
    # (rewritten for E19's convexification fix, 2026-09-24: the previous
    # version's expected value of exactly 0.0 was a coincidence of the old
    # clip-after-sum formula, not a property of this construction -- see the
    # commit message on this change for the worked numbers.) With U full
    # rank (K == n), there is no "outside the subspace" at all, so the whole
    # row is governed by the eigen-clipped block; a strongly negative raw
    # cross term must still never drive the row's term below zero.
    from constellaration_methods.restoration.p3_feasibility_restoration import (
        quadratic_model_step_terms,
    )

    C = np.zeros((1, 2))  # no diagonal curvature at all
    U = np.eye(2, 2)
    S = np.array([[[0.0, -10.0], [-10.0, 0.0]]])  # strongly negative cross curvature
    d = np.array([1.0, 1.0])
    terms = quadratic_model_step_terms(C, d, lowrank=(U, S))
    assert terms[0] >= -1e-9


# --------------------------------------------------------------------------- #
# fake_front seam honours constraint_overrides (regression ratchet 2026-10-07:
# the seeded NSGA-II override toy recorded the override but the seam ignored
# it). The seam's residuals come from fake RAW measurements through the
# benchmark's own normalisation (problems.overrides.normalised_constraint_
# violations), so a changed threshold changes exactly its own row by exactly
# the benchmark's formula. Host-runnable: no constellaration import.
# --------------------------------------------------------------------------- #
def _seam(x, overrides=None):
    from constellaration_methods.restoration.p3_feasibility_restoration import (
        _fake_front_residuals_and_raw,
    )

    return _fake_front_residuals_and_raw(x, overrides)


def test_fake_front_stock_residuals_unchanged_and_consistent_with_raw():
    from constellaration_methods.problems.overrides import normalised_constraint_violations

    r, raw = _seam([0.0, 0.0, 0.0])
    assert r == pytest.approx([-0.05] * 5, abs=1e-12)
    r_step, raw_step = _seam([0.05, 0.0, 0.0])
    assert r_step == pytest.approx([-0.05, -0.05, -0.05, 0.0, -0.05], abs=1e-12)
    for rr, rw in ((r, raw), (r_step, raw_step)):
        assert rr == normalised_constraint_violations(rw)
        assert rw["vacuum_well"] >= 0.0
    # raw measurements do not depend on the thresholds in force
    assert _seam([0.0, 0.0, 0.0], {"mirror_upper": 0.23})[1] == raw


@pytest.mark.parametrize(
    "name, value, row, expected_of",
    [
        ("iota_lower", 0.26, 0, lambda m: (0.26 - abs(m)) / 0.26),
        ("log10_qi_upper", -3.6, 1, lambda m: (m - (-3.6)) / 3.6),
        ("mirror_upper", 0.23, 2, lambda m: (m - 0.23) / 0.23),
        ("flux_upper", 0.8, 3, lambda m: (m - 0.8) / 0.8),
        ("well_lower", 0.02, 4, lambda m: (0.02 - m) / 0.1),
    ],
)
def test_fake_front_override_changes_exactly_its_own_row(name, value, row, expected_of):
    raw_key = {
        0: "iota_over_nfp",
        1: "log10_qi",
        2: "edge_mirror",
        3: "flux_compression",
        4: "vacuum_well",
    }[row]
    stock, raw = _seam([0.0, 0.0, 0.0])
    over, _ = _seam([0.0, 0.0, 0.0], {name: value})
    assert over[row] == expected_of(raw[raw_key])
    assert over[row] != stock[row]
    for i in range(5):
        if i != row:
            assert over[i] == stock[i]


def test_fake_front_well_override_uses_the_benchmarks_0p1_floor_on_the_target():
    """`well_lower` normalises by max(0.1, lower), like the class: a lower
    bound of 0.2 divides by 0.2, a lower bound of 0.05 still divides by 0.1."""
    _, raw = _seam([0.0, 0.0, 0.0])
    w = raw["vacuum_well"]
    assert _seam([0.0, 0.0, 0.0], {"well_lower": 0.2})[0][4] == (0.2 - w) / 0.2
    assert _seam([0.0, 0.0, 0.0], {"well_lower": 0.05})[0][4] == (0.05 - w) / 0.1


def test_fake_front_unknown_override_name_refused_like_problem_with_overrides():
    with pytest.raises(ValueError, match="mirror_upper"):
        _seam([0.0, 0.0, 0.0], {"mirror_uper": 0.23})
