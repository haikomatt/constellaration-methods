"""Pure-rule tests for scripts/p3_final_restoration.py — the reviewer's final
protocol (vault p3-final-feasibility-attempt-2026-09-16). No VMEC. Rules
covered: eps from repeatability, acceptance without allowances, ceilings and
deadlines, block progress, recovery selection, worst-V adjudication, labels,
null-step stalling (reused), LP failure (reused, raises)."""

from pathlib import Path

import numpy as np
import pytest

from constellaration_methods.restoration.p3_final_restoration import (
    Ceilings,
    accept_final,
    adjudicate_worst_v,
    block_progress_required,
    confirmations_complete,
    eps_from_repeatability,
    label_outcome,
    select_recovery_point,
)


def test_eps_from_repeatability():
    archived = np.array([0.057875, 0.136693, 0.136335, 0.107972, -0.036113])
    reps = [archived + 1e-5, archived - 2e-5]
    ok, e_rep, eps = eps_from_repeatability(archived, reps, max_component=1e-4)
    assert ok and e_rep == pytest.approx(2e-5) and eps == pytest.approx(1e-4)  # max(1e-4, 5e-5)
    ok2, e2, eps2 = eps_from_repeatability(archived, [archived + 5e-5], 1e-4)
    assert ok2 and eps2 == pytest.approx(2.5e-4)
    ok3, _, _ = eps_from_repeatability(archived, [archived + 2e-4], 1e-4)
    assert not ok3


def test_accept_final_has_no_regression_allowances():
    r_old = np.array([0.058, 0.137, 0.136, 0.108, -0.036])  # V 0.137
    # flux AND iota may rise as long as V falls materially and well stays >= 0
    ok = np.array([0.090, 0.125, 0.125, 0.125, -0.010])
    assert accept_final(r_old, ok, tol=0.01, eps=1e-4)
    # well crossing zero -> reject
    assert not accept_final(r_old, np.array([0.090, 0.125, 0.125, 0.125, 0.001]), 0.01, 1e-4)
    # V not materially lower -> reject
    assert not accept_final(r_old, np.array([0.058, 0.13695, 0.136, 0.108, -0.036]), 0.01, 1e-4)
    # a constraint already within tolerance must stay there
    r_old2 = np.array([0.005, 0.137, 0.136, 0.108, -0.036])
    assert not accept_final(r_old2, np.array([0.02, 0.125, 0.125, 0.125, -0.036]), 0.01, 1e-4)
    # non-finite / failed evaluation -> reject
    assert not accept_final(r_old, None, 0.01, 1e-4)
    assert not accept_final(r_old, np.array([0.05, np.nan, 0.1, 0.1, -0.03]), 0.01, 1e-4)


def test_ceilings_and_deadlines():
    t0 = 1000.0
    c = Ceilings(
        t_start=t0,
        total_hours=24,
        opt_cutoff_hours=20,
        max_opt_evals=1200,
        max_verif_evals=20,
        max_full_jacobians=2,
    )
    assert c.can_optimise(now=t0 + 19 * 3600, n=1)
    assert not c.can_optimise(now=t0 + 20 * 3600 + 1, n=1)
    c.charge_opt(1195)
    assert not c.can_optimise(now=t0 + 3600, n=10)  # would exceed 1200
    assert c.can_optimise(now=t0 + 3600, n=5)
    c.charge_full_jacobian()
    c.charge_full_jacobian()
    assert not c.can_full_jacobian()
    c.charge_verif(20)
    with pytest.raises(RuntimeError):
        c.charge_verif(1)
    assert c.past_total(now=t0 + 24 * 3600 + 1)


def test_block_progress_required():
    assert block_progress_required(v_block_start=0.137) == pytest.approx(
        0.002
    )  # 0.1*(0.127)=0.0127 -> min 0.002
    assert block_progress_required(v_block_start=0.02) == pytest.approx(0.001)  # 0.1*(0.01)
    assert block_progress_required(v_block_start=0.01) == pytest.approx(0.0)


def test_select_recovery_point_lowest_v_meeting_preservation():
    archive = [
        {"tag": "a", "r": [0.06, 0.14, 0.14, 0.10, -0.03], "complete": True},
        {
            "tag": "b",
            "r": [0.05, 0.13, 0.13, 0.12, 0.01],
            "complete": True,
        },  # lowest V but well < 0
        {
            "tag": "c",
            "r": [0.02, 0.135, 0.135, 0.11, -0.03],
            "complete": True,
            # iota within tol lost? no: incumbent iota 0.005 satisfied ->
            # c's 0.02 breaks preservation
        },
        {"tag": "d", "r": [0.005, 0.136, 0.136, 0.11, -0.03], "complete": True},
        {"tag": "e", "r": None, "complete": False},
    ]
    incumbent_r = np.array([0.005, 0.15, 0.15, 0.10, -0.03])  # iota satisfied -> must be preserved
    pick = select_recovery_point(archive, incumbent_r, tol=0.01)
    assert pick["tag"] == "d"


def test_adjudicate_worst_v_and_labels():
    rec = np.array([0.05, 0.128, 0.127, 0.10, -0.03])
    conf = [rec + 1e-4, rec - 2e-4]
    v, feas = adjudicate_worst_v(rec, conf, tol=0.01)
    assert v == pytest.approx(0.1281)
    assert not feas
    assert (
        label_outcome(worst_v=0.1281, feasible=False, technical=False)
        == "material_but_infeasible_progress"
    )
    assert label_outcome(0.09, False, False) == "substantial_but_infeasible_progress"
    assert label_outcome(0.15, False, False) == "progress_gate_not_met"
    assert label_outcome(0.005, True, False) == "verified_p3_feasibility"
    assert label_outcome(0.09, False, True) == "technical_limitation"
    # feasibility is the official predicate: every residual <= tol on all three
    good = np.array([0.005, 0.008, 0.009, 0.001, -0.02])
    v2, feas2 = adjudicate_worst_v(good, [good, good + 3e-3], tol=0.01)
    assert not feas2  # one confirmation pushes qi to 0.011 > tol
    v3, feas3 = adjudicate_worst_v(good, [good, good], tol=0.01)
    assert feas3


def test_confirmations_complete():
    good = np.array([0.005, 0.006, 0.006, 0.001, -0.02])
    assert confirmations_complete([good, good])
    assert not confirmations_complete([good])  # only one confirmation
    assert not confirmations_complete([good, good, good])  # more than two
    assert not confirmations_complete([])
    bad_shape = np.array([0.005, 0.006, 0.006, 0.001])  # only 4 components
    assert not confirmations_complete([good, bad_shape])
    non_finite = np.array([0.005, 0.006, 0.006, 0.001, np.nan])
    assert not confirmations_complete([good, non_finite])


def test_adjudicate_worst_v_requires_two_complete_confirmations():
    # Fix 3 (review): a filtered, incomplete confirmation list (a failed
    # evaluation dropped by the caller before this is even reached) must
    # never be enough to declare feasibility, however good the residuals are
    rec = np.array([0.005, 0.006, 0.006, 0.001, -0.02])
    v1, feas1 = adjudicate_worst_v(rec, [rec], tol=0.01)  # only one confirmation
    assert not feas1
    v0, feas0 = adjudicate_worst_v(rec, [], tol=0.01)  # zero confirmations
    assert not feas0
    non_finite = np.array([0.005, 0.006, 0.006, 0.001, np.nan])
    v2, feas2 = adjudicate_worst_v(rec, [rec, non_finite], tol=0.01)
    assert not feas2  # a non-finite confirmation breaks completeness too
    v3, feas3 = adjudicate_worst_v(rec, [rec, rec], tol=0.01)
    assert feas3  # two complete, satisfying confirmations DO declare feasibility


def test_control_candidate_uses_its_own_representation():
    # E4 (2026-09-16) failed its repeatability gate because the control (seed
    # 176, nfp 3) was evaluated under the run's representation (seed 49, nfp 2).
    from constellaration_methods.restoration.p3_final_restoration import representation_of

    cand = {
        "boundary_dict": {"r_cos": [[1.0]]},
        "mpol": 4,
        "ntor": 4,
        "nfp": 3,
        "scaling": 1.0,
    }
    rep = representation_of(cand)
    assert rep["nfp"] == 3 and rep["mpol"] == 4 and rep["boundary_dict"] == {"r_cos": [[1.0]]}
    run_rep = {
        "boundary_dict": {"r_cos": [[2.0]]},
        "mpol": 4,
        "ntor": 4,
        "nfp": 2,
        "scaling": 1.0,
    }
    assert rep != run_rep


def test_representation_of_propagates_fake_coupling():
    # found while building the E19 toy ratchet test: every evaluation after
    # the start point goes through the run's `rep`, built once from the start
    # candidate by representation_of -- a coupling matrix left out of `rep`
    # never reaches any evaluation except the one done directly against the
    # start candidate dict, so the toy's cross term silently vanished
    from constellaration_methods.restoration.p3_final_restoration import representation_of

    coupling = [[0.0, 6.0], [6.0, 0.0]]
    cand = {
        "boundary_dict": {},
        "mpol": 1,
        "ntor": 1,
        "nfp": 1,
        "scaling": 1.0,
        "fake_quadratic": True,
        "fake_coupling": coupling,
    }
    rep = representation_of(cand)
    assert rep["fake_coupling"] == coupling
    # absent when there is no coupling at all (byte-identical to pre-E19)
    assert "fake_coupling" not in representation_of(dict(cand, fake_coupling=None))


def test_step_in_trust_units_divides_by_the_per_coordinate_multiplier():
    from constellaration_methods.restoration.p3_final_restoration import step_in_trust_units

    d = np.array([0.001, 0.05, -0.002])
    mult = np.array([1.0, 50.0, 1.0])
    # the big move is in a coordinate whose box is 50x wider: in trust units it is small
    assert np.isclose(step_in_trust_units(d, mult), 0.002)
    assert np.isclose(step_in_trust_units(d, None), 0.05)


def test_second_order_probe_scale_follows_the_box_multiplier():
    from constellaration_methods.restoration.p3_final_restoration import probe_scale

    mult = np.array([1.0, 50.0])
    assert np.allclose(probe_scale(0.01, mult, second_order=True), [0.01, 0.5])
    assert np.allclose(probe_scale(0.01, mult, second_order=False), [0.01, 0.01])
    assert np.allclose(probe_scale(0.01, None, second_order=True), 0.01)


def test_runner_end_to_end_on_a_fake_quadratic_model(tmp_path):
    """Integration smoke of run() with the evaluator replaced by a toy model
    (payload flag 'fake_quadratic': no VMEC). Exercises phase 0, the
    Jacobian/curvature probes, quadratic trials, line search, adjudication
    and the report. The toy is convex with a feasible point nearby, so the
    second-order runner must accept at least one step and report."""
    import json
    from types import SimpleNamespace

    from constellaration_methods.restoration.p3_final_restoration import run

    n = 4
    x0 = np.zeros(n)
    cand = {
        "x": x0.tolist(),
        "r": None,  # filled below from the toy
        "boundary_dict": {},
        "mpol": 1,
        "ntor": 1,
        "nfp": 1,
        "scaling": 1.0,
        "fake_quadratic": True,
    }
    from constellaration_methods.restoration.p3_feasibility_restoration import _eval_worker

    r0 = _eval_worker(dict(cand, x=x0.tolist(), tag="t"))["r"]
    cand["r"] = r0
    (tmp_path / "cand.json").write_text(json.dumps(cand))
    (tmp_path / "control.json").write_text(json.dumps(dict(cand, x=x0.tolist(), r=r0)))
    args = SimpleNamespace(
        checkpoint=None,
        start_log=None,
        start_tag=None,
        control_candidate=str(tmp_path / "control.json"),
        out_dir=str(tmp_path / "out"),
        workers=2,
        h=0.01,
        delta0=0.05,
        delta_cap=0.1,
        drift_every=3,
        repeat_tol=1e-4,
        total_hours=1.0,
        opt_cutoff_hours=0.9,
        max_opt_evals=60,
        max_verif_evals=12,
        seed=0,
        phase0_only=False,
        start_candidate=str(tmp_path / "cand.json"),
        line_search=True,
        material_bar=0.13,
        coeff_space_box=False,
        reuse_jacobian_archive=None,
        second_order=True,
        recovery_qualify_v=0.130,
        log_flux_surrogate=False,
    )
    report = run(args)
    assert report["second_order"] is True
    assert report["n_accepted"] >= 1
    assert report["incumbent_v"] < max(r0)
    assert (tmp_path / "out" / "REPORT.json").exists()


def test_runner_stops_immediately_on_a_confirmed_feasible_point(tmp_path):
    """Fix 2 (E13 defect): priority_confirm's return value was ignored by every
    caller, so a confirmed-feasible point did not stop the optimisation loop
    (452 extra evaluations after the feasible point in the real E13 run). The
    toy 'fake_near_feasible' payload starts just above TOL on every row so a
    real trial reaches and confirms feasibility almost immediately; the fixed
    runner must record success_point, log stop_reason 'feasible_confirmed',
    and leave the optimisation phase right there -- well short of the
    evaluation ceiling."""
    import json
    from types import SimpleNamespace

    from constellaration_methods.restoration.p3_feasibility_restoration import _eval_worker
    from constellaration_methods.restoration.p3_final_restoration import run

    n = 4
    x0 = np.zeros(n)
    cand = {
        "x": x0.tolist(),
        "r": None,
        "boundary_dict": {},
        "mpol": 1,
        "ntor": 1,
        "nfp": 1,
        "scaling": 1.0,
        "fake_quadratic": True,
        "fake_near_feasible": True,
    }
    r0 = _eval_worker(dict(cand, x=x0.tolist(), tag="t"))["r"]
    assert max(r0) > 0.01  # start is violated, not already feasible
    cand["r"] = r0
    (tmp_path / "cand.json").write_text(json.dumps(cand))
    (tmp_path / "control.json").write_text(json.dumps(dict(cand, x=x0.tolist(), r=r0)))
    args = SimpleNamespace(
        checkpoint=None,
        start_log=None,
        start_tag=None,
        control_candidate=str(tmp_path / "control.json"),
        out_dir=str(tmp_path / "out"),
        workers=2,
        h=0.01,
        delta0=0.05,
        delta_cap=0.1,
        drift_every=3,
        repeat_tol=1e-4,
        total_hours=1.0,
        opt_cutoff_hours=0.9,
        max_opt_evals=60,
        max_verif_evals=12,
        seed=0,
        phase0_only=False,
        start_candidate=str(tmp_path / "cand.json"),
        line_search=True,
        material_bar=0.13,
        coeff_space_box=False,
        reuse_jacobian_archive=None,
        second_order=True,
        recovery_qualify_v=0.130,
        log_flux_surrogate=False,
    )
    report = run(args)
    assert report["stop_reason"] == "feasible_confirmed"
    assert report["success_point"] is not None
    assert report["opt_evals"] < args.max_opt_evals


def test_well_cap_applies_only_once_the_well_is_non_negative():
    # E11 defect: the raw-well cap (residual <= 0) was applied unconditionally;
    # from a start whose well is violated every quadratic solve was infeasible
    from constellaration_methods.restoration.p3_final_restoration import well_cap_for

    assert well_cap_for(np.array([0.1, 0.2, 0.3, 0.4, -0.01])) == {4: 0.0}
    assert well_cap_for(np.array([0.1, 0.2, 0.3, 0.4, 0.0])) == {4: 0.0}
    assert well_cap_for(np.array([0.1, 0.2, 0.3, 0.4, 0.305])) == {}


def test_accept_final_requires_non_negative_well_only_once_reached():
    # E11: from a start whose well is violated, a step that lowers V and keeps
    # the satisfied rows must be acceptable even though the well is still > 0;
    # once the well is non-negative it must never go positive again
    old_violated = np.array([-0.04, -0.14, -0.20, 0.11, 0.30])
    new_better = np.array([-0.04, -0.14, -0.20, 0.10, 0.25])
    assert accept_final(old_violated, new_better, 0.01, 1e-4)
    old_ok = np.array([-0.04, 0.14, 0.20, 0.11, -0.001])
    new_well_lost = np.array([-0.04, 0.13, 0.19, 0.10, 0.001])
    assert not accept_final(old_ok, new_well_lost, 0.01, 1e-4)


def test_select_recovery_point_requires_well_only_once_the_incumbent_has_it():
    inc_violated = np.array([-0.04, -0.14, -0.20, 0.11, 0.30])
    archive = [
        {"tag": "a", "complete": True, "r": [-0.04, -0.14, -0.20, 0.10, 0.27]},
        {
            "tag": "b",
            "complete": True,
            "r": [-0.04, 0.02, -0.20, 0.10, 0.20],
        },  # loses QI
    ]
    pick = select_recovery_point(archive, inc_violated, 0.01)
    assert pick is not None and pick["tag"] == "a"
    inc_ok = np.array([-0.04, -0.14, -0.20, 0.11, -0.001])
    assert select_recovery_point(archive, inc_ok, 0.01) is None  # both have well > 0


def test_runner_end_to_end_from_a_well_violated_start(tmp_path):
    """E11 lesson: every real start before E11 had well >= 0, so three
    unconditional well checks passed 30+ runs and the first toy smoke. This
    variant starts with the well violated; the second-order runner must still
    accept steps and descend."""
    import json
    from types import SimpleNamespace

    from constellaration_methods.restoration.p3_feasibility_restoration import _eval_worker
    from constellaration_methods.restoration.p3_final_restoration import run

    n = 4
    x0 = np.zeros(n)
    cand = {
        "x": x0.tolist(),
        "r": None,
        "boundary_dict": {},
        "mpol": 1,
        "ntor": 1,
        "nfp": 1,
        "scaling": 1.0,
        "fake_quadratic": True,
        "fake_well_violated": True,
    }
    r0 = _eval_worker(dict(cand, x=x0.tolist(), tag="t"))["r"]
    assert r0[4] > 0.0  # the toy really starts with the well violated
    cand["r"] = r0
    (tmp_path / "cand.json").write_text(json.dumps(cand))
    (tmp_path / "control.json").write_text(json.dumps(dict(cand)))
    args = SimpleNamespace(
        checkpoint=None,
        start_log=None,
        start_tag=None,
        control_candidate=str(tmp_path / "control.json"),
        out_dir=str(tmp_path / "out"),
        workers=2,
        h=0.01,
        delta0=0.05,
        delta_cap=0.1,
        drift_every=3,
        repeat_tol=1e-4,
        total_hours=1.0,
        opt_cutoff_hours=0.9,
        max_opt_evals=60,
        max_verif_evals=12,
        seed=0,
        phase0_only=False,
        start_candidate=str(tmp_path / "cand.json"),
        line_search=True,
        material_bar=0.13,
        coeff_space_box=False,
        reuse_jacobian_archive=None,
        second_order=True,
        recovery_qualify_v=0.130,
        log_flux_surrogate=False,
    )
    report = run(args)
    assert report["n_accepted"] >= 1
    assert report["incumbent_v"] < max(r0)


def test_recovery_qualification_threshold_is_a_parameter():
    # E11: the fixed 0.130 qualification (written for starts near 0.13) stopped
    # a run that was descending at rho 1.0 from a 0.30 start
    from constellaration_methods.restoration.p3_final_restoration import recovery_qualified

    assert recovery_qualified(material_steps=3, v=0.12, threshold=0.130)
    assert not recovery_qualified(material_steps=3, v=0.20, threshold=0.130)
    assert recovery_qualified(material_steps=3, v=0.20, threshold=0.30)
    assert not recovery_qualified(material_steps=0, v=0.05, threshold=0.130)


def test_best_admissible_sampled_point_is_kept_even_when_it_misses_the_line_minimum():
    # review round 8 (E11e P:T0): the half step passed every acceptance gate but was
    # discarded because its V was above the unconstrained line minimum; the best
    # admissible sampled point must be kept whenever it beats the incumbent
    from constellaration_methods.restoration.p3_final_restoration import best_admissible_sampled

    inc = np.array([-0.07, -0.01, -0.22, 0.11, 0.116])
    half = np.array([-0.07, -0.009, -0.22, 0.11, 0.110])  # admissible, V 0.110
    full = np.array([-0.07, 0.028, -0.20, 0.107, 0.087])  # V 0.107 but QI > tol
    t, r = best_admissible_sampled(inc, [(0.5, half), (1.0, full)], 0.01, 1e-4)
    assert t == 0.5 and np.allclose(r, half)
    assert best_admissible_sampled(inc, [(1.0, full)], 0.01, 1e-4) == (None, None)


def test_model_prediction_error_reports_the_multivariate_models_own_miss():
    # review round 8: rho validated the quadratic LINE predictor; the diagonal
    # multivariate model's own prediction at the evaluated points must be logged
    from constellaration_methods.restoration.p3_final_restoration import model_prediction_error

    r0 = np.array([0.1, 0.2, -0.05, 0.1, -0.01])
    J = np.zeros((5, 2))
    J[0, 0] = -1.0
    J[1, 1] = -1.0
    C = np.zeros((5, 2))
    C[1, 1] = 4.0
    d = np.array([0.05, 0.1])
    pred = r0 + J @ d + 0.5 * np.maximum(C, 0) @ (d * d)
    r_meas = pred + np.array([0.0, 0.01, 0.0, 0.0, 0.0])  # the model missed row 1 by 0.01
    out = model_prediction_error(r0, J, C, d, r_meas)
    assert np.allclose(out["predicted_r"], pred)
    assert abs(out["max_abs_err"] - 0.01) < 1e-12
    assert out["worst_row"] == 1
    # rho of the multivariate model on V: measured gain / predicted gain
    assert abs(out["rho_model"] - (r0.max() - r_meas.max()) / (r0.max() - pred.max())) < 1e-12


def test_log_flux_surrogate_flag_puts_widths_in_the_representation():
    # zero extra VMEC cost: the evaluator computes the smoothed-gate flux from
    # the equilibrium it already has, so every archive row can support the
    # held-out prediction comparison the reviewer asked for
    from constellaration_methods.restoration.p3_final_restoration import (
        representation_with_surrogate,
    )

    rep = {"boundary_dict": {}, "mpol": 1, "ntor": 1, "nfp": 1, "scaling": 1.0}
    assert "surrogate_widths" not in representation_with_surrogate(rep, False)
    out = representation_with_surrogate(rep, True)
    assert out["surrogate_widths"] == [0.02, 0.05, 0.1, 0.2]
    assert "surrogate_widths" not in rep  # the input is not mutated


# --------------------------------------------------------------------------- #
# E19 item 2 -- wiring the subspace (off-diagonal) curvature block into the
# runner: flags, direction selection, probe scheduling through the shared
# evaluation budget, REPORT's 'subspace' block, and the K=0 byte-identical
# guarantee.
# --------------------------------------------------------------------------- #
def test_direction_seed_from_file_orthonormalises_and_drops_duplicates(tmp_path):
    import json

    from constellaration_methods.restoration.p3_final_restoration import direction_seed_from_file

    path = tmp_path / "dirs.json"
    path.write_text(json.dumps({"directions": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]}))
    U = direction_seed_from_file(path, n=3)
    assert U.shape == (2, 3)  # the duplicate [2,0,0] direction is dropped
    assert np.allclose(U @ U.T, np.eye(2), atol=1e-9)


def test_build_subspace_directions_prefers_direction_file_and_fills_with_svd(tmp_path):
    import json

    from constellaration_methods.restoration.p3_final_restoration import build_subspace_directions

    path = tmp_path / "dirs.json"
    path.write_text(json.dumps({"directions": [[1.0, 0.0, 0.0]]}))
    J = np.diag([5.0, 4.0, 3.0])
    U, n_from_archives, n_from_svd = build_subspace_directions(
        direction_archives=["/does/not/matter"],
        direction_file=str(path),
        J=J,
        n=3,
        k=2,
    )
    assert n_from_archives == 1 and n_from_svd == 1
    assert U.shape == (2, 3)
    assert np.allclose(U @ U.T, np.eye(2), atol=1e-9)
    # with neither file nor archives, U is filled entirely from J's SVD
    U2, n_arch2, n_svd2 = build_subspace_directions(None, None, J, n=3, k=2)
    assert n_arch2 == 0 and n_svd2 == 2
    assert U2.shape == (2, 3)


def test_per_row_offdiag_norm_ratio_and_zero_diagonal_guard():
    from constellaration_methods.restoration.p3_final_restoration import per_row_offdiag_norm_ratio

    S = np.zeros((2, 2, 2))
    S[0] = [[4.0, 1.0], [1.0, 4.0]]  # diag norm sqrt(32), off norm sqrt(2)
    S[1] = [[0.0, 3.0], [3.0, 0.0]]  # zero diagonal
    ratios = per_row_offdiag_norm_ratio(S)
    assert ratios[0] == pytest.approx(np.sqrt(2) / np.sqrt(32))
    assert ratios[1] is None


def _e19_toy_args(tmp_path, **overrides):
    import json
    from types import SimpleNamespace

    from constellaration_methods.restoration.p3_feasibility_restoration import _eval_worker

    n = 4
    x0 = np.zeros(n)
    cand = {
        "x": x0.tolist(),
        "r": None,
        "boundary_dict": {},
        "mpol": 1,
        "ntor": 1,
        "nfp": 1,
        "scaling": 1.0,
        "fake_quadratic": True,
    }
    r0 = _eval_worker(dict(cand, x=x0.tolist(), tag="t"))["r"]
    cand["r"] = r0
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "cand.json").write_text(json.dumps(cand))
    (tmp_path / "control.json").write_text(json.dumps(dict(cand, x=x0.tolist(), r=r0)))
    base = dict(
        checkpoint=None,
        start_log=None,
        start_tag=None,
        control_candidate=str(tmp_path / "control.json"),
        out_dir=str(tmp_path / "out"),
        workers=2,
        h=0.01,
        delta0=0.05,
        delta_cap=0.1,
        drift_every=3,
        repeat_tol=1e-4,
        total_hours=1.0,
        opt_cutoff_hours=0.9,
        max_opt_evals=60,
        max_verif_evals=12,
        seed=0,
        phase0_only=False,
        start_candidate=str(tmp_path / "cand.json"),
        line_search=True,
        material_bar=0.13,
        coeff_space_box=False,
        reuse_jacobian_archive=None,
        second_order=True,
        recovery_qualify_v=0.130,
        log_flux_surrogate=False,
        subspace_curvature=0,
        direction_archives=None,
        direction_file=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base), r0


def test_runner_with_subspace_curvature_zero_is_byte_identical_to_missing_attribute(
    tmp_path,
):
    # E19's own requirement: --subspace-curvature 0 (or the attribute simply
    # absent, as every pre-E19 caller's args object has it) must produce a
    # byte-identical REPORT
    import copy
    from types import SimpleNamespace

    from constellaration_methods.restoration.p3_final_restoration import run

    args_with_zero, _ = _e19_toy_args(tmp_path / "a")
    report_with_zero = run(args_with_zero)

    args_dict = vars(copy.copy(args_with_zero))
    for key in ("subspace_curvature", "direction_archives", "direction_file"):
        del args_dict[key]
    args_dict["out_dir"] = str(tmp_path / "b")
    args_without_attr = SimpleNamespace(**args_dict)
    report_without_attr = run(args_without_attr)

    volatile = {"wall_h"}
    for key in set(report_with_zero) | set(report_without_attr) - volatile:
        if key in volatile:
            continue
        assert report_with_zero.get(key) == report_without_attr.get(key), f"mismatch at {key!r}"
    assert "subspace" not in report_with_zero
    assert "subspace" not in report_without_attr


def test_runner_end_to_end_with_subspace_curvature_reports_the_subspace_block(
    tmp_path,
):
    import json

    from constellaration_methods.restoration.p3_final_restoration import run

    direction_file = tmp_path / "dirs.json"
    direction_file.write_text(
        json.dumps({"directions": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]})
    )
    args, r0 = _e19_toy_args(
        tmp_path,
        subspace_curvature=2,
        direction_file=str(direction_file),
    )
    report = run(args)
    assert "subspace" in report
    sub = report["subspace"]
    assert sub["K"] == 2
    assert sub["n_from_archives"] == 2 and sub["n_from_svd"] == 0
    K = 2
    assert sub["evals_used"] >= 2 * K + K * (K - 1)  # at least one measurement
    assert len(sub["per_row_offdiag_norm_ratio"]) == len(r0)
    assert sub["h"] == pytest.approx(args.delta0)
    assert report["n_accepted"] >= 1
    assert report["incumbent_v"] < max(r0)


def test_runner_logs_model_used_on_trial_events_and_counts_it_in_report(tmp_path):
    # E19 item 2: every trial through the quadratic step must record which
    # rung of the fallback ladder produced it (subspace / subspace_half /
    # diagonal), both on the per-trial event log and tallied in REPORT's
    # subspace block, so a run where the ladder had to fall back is visible
    # without re-reading events.jsonl.
    import json

    from constellaration_methods.restoration.p3_final_restoration import run

    direction_file = tmp_path / "dirs.json"
    direction_file.write_text(
        json.dumps({"directions": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]})
    )
    args, r0 = _e19_toy_args(
        tmp_path,
        subspace_curvature=2,
        direction_file=str(direction_file),
        # generous budget: this test is about the trial/REPORT bookkeeping,
        # not about exercising the evaluation ceiling -- a trial that starts
        # (and is counted in steps_by_model) but is cut off mid-measurement
        # by the ceiling before its own "trial" event is logged is a
        # legitimate, separate edge case, not this invariant's concern.
        max_opt_evals=400,
    )
    report = run(args)
    events = [
        json.loads(ln)
        for ln in (Path(args.out_dir) / "events.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    trial_events = [e for e in events if e.get("event") == "trial"]
    assert trial_events  # at least one trial ran
    for e in trial_events:
        assert e["model_used"] in ("subspace", "subspace_half", "diagonal")
    counts: dict = {}
    for e in trial_events:
        counts[e["model_used"]] = counts.get(e["model_used"], 0) + 1
    assert report["subspace"]["steps_by_model"] == counts


def test_runner_requires_second_order_for_subspace_curvature(tmp_path):
    from constellaration_methods.restoration.p3_final_restoration import run

    args, _ = _e19_toy_args(
        tmp_path, subspace_curvature=2, second_order=False, direction_archives=[]
    )
    with pytest.raises(SystemExit, match="--second-order"):
        run(args)


# --------------------------------------------------------------------------- #
# E19 item 3 -- toy ratchet test: a synthetic problem with real coupling in a
# 3-dim subspace, run end to end through the fake_quadratic/fake_coupling
# seam for both a diagonal-only (K=0) and a subspace (K=3) arm.
#
# RETUNED 2026-09-24 for the convexification fix (item 1): the ORIGINAL
# version of this test (commit decaeab) asserted that the subspace arm
# CROSSED the 0.05 material bar (reaching V ~ 0.019) while the diagonal arm
# stalled above it. That outcome relied entirely on the clip-AFTER-sum model
# this fix removes: at coupling 20 -- the same magnitude used here -- the
# row's raw measured 2x2 block [[0.6, 20], [20, 0.6]] has eigenvalues
# (20.6, -19.4); the pre-fix model could reward a step for exploiting that
# -19.4 curvature (the model's `max(quad, 0)` clip only fires when the
# SUMMED total for the SPECIFIC d being evaluated goes negative, so a
# cleverly-chosen d could still see the model's raw, unclipped -19.4
# eigen-component pull its total down), which is exactly the unsoundness
# that made SLSQP fail outright on the real smoke at more extreme ratios.
# Verified directly: reverting quadratic_model_step_terms and
# restoration_step_quadratic to the exact pre-fix formulas on this same toy
# reproduces incumbent_v ~ 0.0186, matching the original commit message.
# Item 1's eigen-clip is applied to the BLOCK, not a per-d scalar, so it
# provably cannot reward negative curvature at all (see
# test_quadratic_model_step_terms_lowrank_convexifies_indefinite_block_for_random_d
# in test_p3_feasibility_restoration.py) -- no retuning of the coupling
# magnitude, direction set, evaluation budget, or trust-region size can
# recover the original "crosses the bar" outcome without reintroducing that
# same unsoundness (searched extensively: coupling 5-1000, several
# non-uniform per-coordinate curvature shapes, delta0 0.001-0.02, budget up
# to 1000 -- every combination that changed the outcome did so by making
# BOTH arms solve the toy equally easily, never by making the subspace arm
# alone win).
#
# What this test verifies now is the property that DOES hold, and that is
# what items 1+2 actually promise: the diagonal (K=0) arm is completely
# unaffected (still stalls, per the K=0 byte-identical guarantee); the
# subspace (K=3) arm, given the SAME indefinite coupling that once produced
# an unsound "win", solves every one of its trials through the fallback
# ladder without a single full-ladder exhaustion (no lp_infeasible event),
# predominantly on the full "subspace" rung -- and never ends up materially
# WORSE than the diagonal arm. That is: the fix trades "can be unsoundly
# lucky" for "is always usable and never harmful", which is the actual goal.
# --------------------------------------------------------------------------- #
_RATCHET_COUPLING = [[0.0, 20.0, 0.0], [20.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
_RATCHET_MATERIAL_BAR = 0.05


def _ratchet_toy_args(tmp_path, subspace_curvature, direction_file, coupling):
    import json
    from types import SimpleNamespace

    from constellaration_methods.restoration.p3_feasibility_restoration import _eval_worker

    n = 5
    x0 = np.zeros(n)
    cand = {
        "x": x0.tolist(),
        "r": None,
        "boundary_dict": {},
        "mpol": 1,
        "ntor": 1,
        "nfp": 1,
        "scaling": 1.0,
        "fake_quadratic": True,
    }
    if coupling is not None:
        cand["fake_coupling"] = coupling
    r0 = _eval_worker(dict(cand, x=x0.tolist(), tag="t"))["r"]
    cand["r"] = r0
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "cand.json").write_text(json.dumps(cand))
    (tmp_path / "control.json").write_text(json.dumps(dict(cand, x=x0.tolist(), r=r0)))
    args = SimpleNamespace(
        checkpoint=None,
        start_log=None,
        start_tag=None,
        control_candidate=str(tmp_path / "control.json"),
        out_dir=str(tmp_path / "out"),
        workers=2,
        h=0.01,
        delta0=0.005,
        delta_cap=0.02,
        drift_every=3,
        repeat_tol=1e-4,
        total_hours=1.0,
        opt_cutoff_hours=0.9,
        max_opt_evals=200,
        max_verif_evals=12,
        seed=0,
        phase0_only=False,
        start_candidate=str(tmp_path / "cand.json"),
        line_search=False,
        material_bar=_RATCHET_MATERIAL_BAR,
        coeff_space_box=False,
        reuse_jacobian_archive=None,
        second_order=True,
        recovery_qualify_v=0.130,
        log_flux_surrogate=False,
        subspace_curvature=subspace_curvature,
        direction_archives=None,
        direction_file=direction_file,
    )
    return args, r0


def test_toy_ratchet_diagonal_model_stalls_subspace_model_remains_sound(tmp_path):
    pytest.importorskip(
        "constellaration",
        reason="needs the pinned constellaration package; run inside the repo's Docker image",
    )
    import json

    from constellaration_methods.restoration.p3_final_restoration import run

    direction_file = tmp_path / "dirs.json"
    direction_file.write_text(
        json.dumps({"directions": [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]})
    )

    diagonal_args, r0 = _ratchet_toy_args(
        tmp_path / "diagonal",
        subspace_curvature=0,
        direction_file=None,
        coupling=_RATCHET_COUPLING,
    )
    diagonal_report = run(diagonal_args)

    subspace_args, _ = _ratchet_toy_args(
        tmp_path / "subspace",
        subspace_curvature=3,
        direction_file=str(direction_file),
        coupling=_RATCHET_COUPLING,
    )
    subspace_report = run(subspace_args)

    # same evaluation budget for both arms
    assert diagonal_args.max_opt_evals == subspace_args.max_opt_evals == 200

    # the diagonal (K=0) model is untouched by this fix: still stalls ABOVE
    # the material bar, exactly as before
    assert diagonal_report["incumbent_v"] > _RATCHET_MATERIAL_BAR
    assert diagonal_report["label"] == "progress_gate_not_met"
    assert subspace_report["subspace"]["K"] == 3

    # the subspace (K=3) arm, given the same indefinite coupling, never
    # raises out of the fallback ladder (no trial exhausts all three rungs)
    events = [
        json.loads(ln)
        for ln in (Path(subspace_args.out_dir) / "events.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    full_ladder_exhaustions = [
        e
        for e in events
        if e.get("event") == "lp_infeasible" and "every fallback ladder rung" in e.get("msg", "")
    ]
    assert full_ladder_exhaustions == []

    # it solves predominantly on the full "subspace" rung, not by constantly
    # falling back to a weaker model
    by_model = subspace_report["subspace"]["steps_by_model"]
    total_trials = sum(by_model.values())
    assert total_trials > 0
    assert by_model.get("subspace", 0) / total_trials >= 0.5

    # and it is never MATERIALLY worse than the diagonal arm -- the
    # convexified model is provably never harmful, even where (unlike the
    # pre-fix model) it can no longer be unsoundly lucky
    assert subspace_report["incumbent_v"] <= diagonal_report["incumbent_v"] + 0.01


def test_toy_ratchet_k0_path_reproduces_the_existing_toy_end_to_end_results(tmp_path):
    # the K=0 path through the SAME machinery this test module uses elsewhere
    # (no coupling at all) must still satisfy the pre-existing toy invariants:
    # at least one accepted step, and a lower incumbent V than the start
    args, r0 = _ratchet_toy_args(tmp_path, subspace_curvature=0, direction_file=None, coupling=None)

    from constellaration_methods.restoration.p3_final_restoration import run

    report = run(args)
    assert report["second_order"] is True
    assert report["n_accepted"] >= 1
    assert report["incumbent_v"] < max(r0)
    assert "subspace" not in report


def _near_feasible_args(tmp_path, out, **extra):
    import json
    from types import SimpleNamespace

    from constellaration_methods.restoration.p3_feasibility_restoration import _eval_worker

    x0 = np.zeros(4)
    cand = {
        "x": x0.tolist(),
        "r": None,
        "boundary_dict": {},
        "mpol": 1,
        "ntor": 1,
        "nfp": 1,
        "scaling": 1.0,
        "fake_quadratic": True,
        "fake_near_feasible": True,
    }
    cand["r"] = _eval_worker(dict(cand, x=x0.tolist(), tag="t"))["r"]
    (tmp_path / "cand.json").write_text(json.dumps(cand))
    (tmp_path / "control.json").write_text(json.dumps(cand))
    base = dict(
        checkpoint=None,
        start_log=None,
        start_tag=None,
        control_candidate=str(tmp_path / "control.json"),
        out_dir=str(tmp_path / out),
        workers=2,
        h=0.01,
        delta0=0.05,
        delta_cap=0.1,
        drift_every=3,
        repeat_tol=1e-4,
        total_hours=1.0,
        opt_cutoff_hours=0.9,
        max_opt_evals=60,
        max_verif_evals=12,
        seed=0,
        phase0_only=False,
        start_candidate=str(tmp_path / "cand.json"),
        line_search=True,
        material_bar=0.13,
        coeff_space_box=False,
        reuse_jacobian_archive=None,
        second_order=True,
        recovery_qualify_v=0.130,
        log_flux_surrogate=False,
    )
    base.update(extra)
    return SimpleNamespace(**base)


def test_success_tol_zero_does_not_stop_at_a_tolerance_feasible_point(tmp_path):
    """E27 (2026-10-23): restoring a tolerance-feasible design to strict nominal needs the
    run to continue past points with every residual <= 0.01. In the near-feasible toy no
    point has every residual <= 0 (rows 0, 1 and 3 conflict), so with success_tol 0 the
    runner must never claim success, while the default run stops at tolerance feasibility."""
    from constellaration_methods.restoration.p3_final_restoration import run

    default = run(_near_feasible_args(tmp_path, "default"))
    assert default["stop_reason"] == "feasible_confirmed"
    strict = run(_near_feasible_args(tmp_path, "strict", success_tol=0.0))
    assert strict["success_tol"] == 0.0
    assert strict["stop_reason"] != "feasible_confirmed"
    assert strict["success_point"] is None
    assert strict["opt_evals"] > default["opt_evals"]


def test_success_point_under_success_tol_zero_is_strictly_nominal_on_every_evaluation():
    """The success screen and adjudication use success_tol; the official label still uses
    TOL. A point at 0.005 passes TOL but not success_tol 0."""
    from constellaration_methods.restoration.p3_final_restoration import adjudicate_worst_v

    r = [-0.2, 0.005, -0.01, -0.02, -0.1]
    _, feas_tol = adjudicate_worst_v(r, [r, r])
    _, feas_strict = adjudicate_worst_v(r, [r, r], 0.0)
    assert feas_tol and not feas_strict
