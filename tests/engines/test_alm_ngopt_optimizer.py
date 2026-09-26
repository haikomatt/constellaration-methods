"""`claude-code-brief-candidate-logging-fix.md`: tests for
`CandidateTracker`, the fix for the 2026-07-13 overnight run only ever
coefficient-logging one arbitrary strict-best-objective candidate (which
then flipped at canonical fidelity, undermining the verdict). These are
pure unit tests against synthetic records -- no physics, no multiprocessing,
no nevergrad -- deliberately, since `run_alm_ngopt_optimization`'s
concurrent ProcessPoolExecutor loop isn't practically mockable at the
forkserver-worker level (see module docstring elsewhere in this project for
the same "not a unit test, that's what the real run is for" discipline).
"""

import inspect
import json
import random

import numpy as np
import pytest
from constellaration import forward_model, problems

from constellaration_methods.engines.alm_ngopt_optimizer import (
    N_FIELD_PERIODS,
    CandidateTracker,
    _alm_state_from_jsonable,
    _alm_state_to_jsonable,
    _apply_canonical_gate,
    _AsyncConfirmSlot,
    _bank_gate_verdict,
    _canonical_settings_for,
    _convergence_stalled,
    _deepens_margin,
    _forward_model_call_budget_reached,
    _improves_on_incumbent,
    _objective_converged,
    _resolve_needs_qi,
    _resume_tracking_state,
    _rng_state_from_jsonable,
    _rng_state_to_jsonable,
    _search_settings_for,
    _should_confirm_canonically,
    _should_confirm_objective,
    _stall_exceeded,
    _updated_violation_watermark,
    _write_checkpoint_atomic,
    prepare_initial_boundary,
    run_alm_ngopt_optimization,
)


def _record(call_index, objective, violation):
    return {
        "call_index": call_index,
        "outer_iteration": 0,
        "objective": objective,
        "constraint_violation": violation,
        "boundary": {"r_cos": [[0.0]], "z_sin": [[0.0]]},
        "random_seed": 1,
    }


def test_deep_margin_reservoir_stays_within_cap():
    tracker = CandidateTracker(
        deep_margin_threshold=1e-6,
        reservoir_cap=3,
        shortlist_n=0,
        rng=random.Random(42),
    )
    for i in range(50):
        tracker.consider(_record(i, objective=10.0 + i, violation=0.0))

    assert len(tracker._reservoir) == 3
    assert tracker._deep_margin_seen == 50


def test_reservoir_only_admits_deep_margin_calls():
    tracker = CandidateTracker(deep_margin_threshold=1e-6, reservoir_cap=10, shortlist_n=0)
    tracker.consider(_record(1, objective=1.0, violation=0.5))  # not deep-margin
    tracker.consider(_record(2, objective=1.0, violation=1e-7))  # deep-margin

    assert [r["call_index"] for r in tracker._reservoir] == [2]


def test_reservoir_under_cap_keeps_every_deep_margin_call():
    tracker = CandidateTracker(deep_margin_threshold=1e-6, reservoir_cap=500, shortlist_n=0)
    for i in range(10):
        tracker.consider(_record(i, objective=1.0, violation=0.0))

    assert len(tracker._reservoir) == 10


def test_shortlist_keeps_best_n_by_objective_then_violation():
    tracker = CandidateTracker(
        deep_margin_threshold=-1.0,  # nothing qualifies as deep-margin
        reservoir_cap=500,
        shortlist_n=2,
    )
    tracker.consider(_record(1, objective=5.0, violation=0.1))
    tracker.consider(_record(2, objective=1.0, violation=0.1))
    tracker.consider(_record(3, objective=3.0, violation=0.1))

    kept = [r["call_index"] for r in tracker._shortlist]
    assert kept == [2, 3]  # best two objectives (1.0, 3.0), 5.0 dropped


def test_shortlist_tiebreaks_on_violation_when_objectives_match():
    tracker = CandidateTracker(deep_margin_threshold=-1.0, reservoir_cap=500, shortlist_n=1)
    tracker.consider(_record(1, objective=1.0, violation=0.5))
    tracker.consider(_record(2, objective=1.0, violation=0.0))  # same objective, deeper margin

    kept = [r["call_index"] for r in tracker._shortlist]
    assert kept == [2]


def test_shortlist_keeps_best_n_by_objective_descending_for_maximize_problem():
    """QI retarget: `SimpleToBuildQIStellarator.get_objective` maximizes
    (minimize_objective=False) -- the shortlist must keep the LARGEST
    objectives, not the smallest (the default, minimize-objective
    behaviour exercised by the other shortlist tests above)."""
    tracker = CandidateTracker(deep_margin_threshold=-1.0, reservoir_cap=500, shortlist_n=2)
    tracker.consider(_record(1, objective=5.0, violation=0.1), minimize_objective=False)
    tracker.consider(_record(2, objective=1.0, violation=0.1), minimize_objective=False)
    tracker.consider(_record(3, objective=3.0, violation=0.1), minimize_objective=False)

    kept = [r["call_index"] for r in tracker._shortlist]
    assert kept == [1, 3]  # best two objectives (5.0, 3.0), 1.0 dropped


def test_shortlist_tiebreak_on_violation_unaffected_by_maximize_direction():
    """The violation tie-break stays ascending (smaller margin always
    better) regardless of which way the objective is optimized."""
    tracker = CandidateTracker(deep_margin_threshold=-1.0, reservoir_cap=500, shortlist_n=1)
    tracker.consider(_record(1, objective=1.0, violation=0.5), minimize_objective=False)
    tracker.consider(_record(2, objective=1.0, violation=0.0), minimize_objective=False)

    kept = [r["call_index"] for r in tracker._shortlist]
    assert kept == [2]


def test_combined_dedupes_overlap_between_reservoir_and_shortlist():
    tracker = CandidateTracker(deep_margin_threshold=1e-6, reservoir_cap=500, shortlist_n=25)
    # Deep-margin and also the best objective -- qualifies for both.
    tracker.consider(_record(1, objective=1.0, violation=0.0))

    combined = tracker.combined()
    assert len(combined) == 1
    assert combined[0]["call_index"] == 1


def test_this_is_the_reported_flip_scenario_deep_margin_near_twin_gets_saved():
    """Regression for the exact overnight-run failure mode: a strict-best
    objective candidate with a thin margin (0.0041) beat an earlier,
    functionally-identical candidate with a deep margin (0.0) on objective
    alone -- so only the thin-margin one got coefficient-logged and
    canonical-checked. With CandidateTracker, both are retained."""
    tracker = CandidateTracker(deep_margin_threshold=1e-6, reservoir_cap=500, shortlist_n=25)
    tracker.consider(_record(28182, objective=1.321303, violation=0.0041))
    tracker.consider(_record(28239, objective=1.321368, violation=0.0))

    combined = {r["call_index"]: r for r in tracker.combined()}
    assert 28182 in combined  # strict-best-objective point, as before
    assert 28239 in combined  # its deep-margin near-twin, previously lost


def _boundary_dict_at(max_poloidal_mode: int, max_toroidal_mode: int) -> dict:
    """Synthetic stand-in for the source repo's real-boundary fixture
    (`tests/fixtures/nfp3_boundaries.jsonl`, a dataset/leaderboard-derived
    design -- not ported here, since this repo's test fixtures must be
    synthetic, never a real leaderboard boundary). Every test using this
    helper only checks shape/padding/pass-through behaviour of
    `prepare_initial_boundary`, never a specific physical value, so a
    deterministic random, stellarator-symmetric array at the requested mode
    budget is equivalent for their purposes."""
    from constellaration_methods.engines.boundary_generator import (
        enforce_stellarator_symmetry,
    )

    rng = np.random.default_rng(0)
    shape = (max_poloidal_mode + 1, 2 * max_toroidal_mode + 1)
    r_cos = rng.normal(size=shape)
    z_sin = rng.normal(size=shape)
    enforce_stellarator_symmetry(r_cos, z_sin)
    return {"r_cos": r_cos.tolist(), "z_sin": z_sin.tolist()}


def test_prepare_initial_boundary_accepts_boundary_at_the_full_budget():
    """G0 (session-handoff-2026-07-15): warm-starting from a leaderboard
    8x8 boundary with an 8x8 budget must round-trip at that shape."""
    boundary = prepare_initial_boundary(
        _boundary_dict_at(8, 8), max_poloidal_mode=8, max_toroidal_mode=8
    )
    assert np.asarray(boundary.r_cos).shape == (9, 17)


def test_prepare_initial_boundary_pads_a_smaller_boundary_up_to_budget():
    """A 4x4 anchor warm-started into an 8x8 run gets zero-padded up with
    its original coefficients preserved -- the representation-expansion
    move (descend-from-anchor in the widened space)."""
    small = _boundary_dict_at(4, 4)

    boundary = prepare_initial_boundary(small, max_poloidal_mode=8, max_toroidal_mode=8)

    r_cos = np.asarray(boundary.r_cos)
    assert r_cos.shape == (9, 17)
    # rows m=0..4 stay rows 0..4; columns n=-4..4 land at indices 4..12 of
    # the widened n=-8..8 axis
    np.testing.assert_array_equal(r_cos[:5, 4:13], np.asarray(small["r_cos"]))
    padding = r_cos.copy()
    padding[:5, 4:13] = 0.0
    assert not padding.any()


def test_prepare_initial_boundary_refuses_silent_truncation():
    """An 8x8 boundary into a 4x4-budget run must raise, not truncate --
    a silent representational truncation is the exact defect that
    manufactured the fake ~1.6 wall."""
    with pytest.raises(ValueError, match="truncat"):
        prepare_initial_boundary(_boundary_dict_at(8, 8), max_poloidal_mode=4, max_toroidal_mode=4)


def test_write_produces_one_json_line_per_candidate(tmp_path):
    tracker = CandidateTracker(deep_margin_threshold=1e-6, reservoir_cap=500, shortlist_n=25)
    tracker.consider(_record(1, objective=1.0, violation=0.0))
    tracker.consider(_record(2, objective=2.0, violation=0.0))
    out_path = tmp_path / "candidates.ndjson"

    tracker.write(out_path)

    lines = out_path.read_text().splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert {r["call_index"] for r in records} == {1, 2}


# ---------------------------------------------------------------------------
# qi-discovery-phase-plan.md Phase 0: retarget wiring tests. Deterministic
# only (pydantic model construction / attribute access) -- no VMEC, no
# forward_model call, per the plan's "TDD the deterministic wiring" brief.
# ---------------------------------------------------------------------------


def test_run_alm_ngopt_optimization_problem_annotation_accepts_both_problem_types():
    """The plan's Phase 0 ask: loosen the `problem` annotation (previously
    `problems.GeometricalProblem` only) to also accept
    `SimpleToBuildQIStellarator` -- a lightweight regression guard (reads the
    literal annotation string, since `from __future__ import annotations`
    makes it one) so a future edit doesn't silently narrow it back."""
    annotation = inspect.signature(run_alm_ngopt_optimization).parameters["problem"].annotation
    assert "GeometricalProblem" in annotation
    assert "SimpleToBuildQIStellarator" in annotation


def test_needs_qi_true_for_simple_to_build_qi_stellarator():
    assert _resolve_needs_qi(problems.SimpleToBuildQIStellarator()) is True


def test_needs_qi_false_for_geometrical_problem():
    assert _resolve_needs_qi(problems.GeometricalProblem()) is False


def test_needs_qi_defaults_true_for_an_object_with_no_such_attribute():
    assert _resolve_needs_qi(object()) is True


def test_search_settings_activate_qi_for_qi_problem():
    settings = _search_settings_for(problems.SimpleToBuildQIStellarator(), None)
    assert settings.qi_settings is not None


def test_search_settings_skip_qi_for_geometrical_problem():
    settings = _search_settings_for(problems.GeometricalProblem(), None)
    assert settings.qi_settings is None


def test_search_settings_honor_explicit_override_regardless_of_problem():
    explicit = forward_model.ConstellarationSettings(qi_settings=None)
    settings = _search_settings_for(problems.SimpleToBuildQIStellarator(), explicit)
    assert settings is explicit


def test_canonical_settings_use_default_high_fidelity_for_qi_problem():
    settings = _canonical_settings_for(problems.SimpleToBuildQIStellarator(), None)
    assert settings == forward_model.ConstellarationSettings.default_high_fidelity()


def test_canonical_settings_skip_qi_for_geometrical_problem():
    settings = _canonical_settings_for(problems.GeometricalProblem(), None)
    assert settings == forward_model.ConstellarationSettings.default_high_fidelity_skip_qi()


def test_canonical_settings_honor_explicit_override_regardless_of_problem():
    explicit = forward_model.ConstellarationSettings.default_high_fidelity()
    settings = _canonical_settings_for(problems.SimpleToBuildQIStellarator(), explicit)
    assert settings is explicit


def test_improves_on_incumbent_minimize_direction():
    """GeometricalProblem-style: lower objective is better."""
    assert _improves_on_incumbent(1.0, True, 2.0) is True
    assert _improves_on_incumbent(2.0, True, 1.0) is False
    assert _improves_on_incumbent(1.0, True, 1.0) is False


def test_improves_on_incumbent_maximize_direction():
    """SimpleToBuildQIStellarator-style: higher objective is better -- the
    exact case a bare `<` comparison gets backwards."""
    assert _improves_on_incumbent(2.0, False, 1.0) is True
    assert _improves_on_incumbent(1.0, False, 2.0) is False
    assert _improves_on_incumbent(1.0, False, 1.0) is False


# ---------------------------------------------------------------------------
# qi-discovery-phase-plan.md Phase 2a: nfp override + eval-budget cap.
# Deterministic only, per the same "TDD the wiring, not the physics loop"
# discipline as the Phase 0 section above.
# ---------------------------------------------------------------------------


def test_prepare_initial_boundary_defaults_n_field_periods_to_task_template_pin():
    """Every pre-Phase-2a caller omits `n_field_periods` -- behaviour must
    stay the historical nfp=3 pin when it's not passed."""
    boundary = prepare_initial_boundary(
        _boundary_dict_at(4, 4), max_poloidal_mode=4, max_toroidal_mode=4
    )

    assert boundary.n_field_periods == N_FIELD_PERIODS


def test_prepare_initial_boundary_honors_explicit_n_field_periods_override():
    """qi-discovery-phase-plan.md Phase 2a: NAE-seeded QI boundaries are
    generated at nfp=1 (the benchmark's own recipe), not the P1-only nfp=3
    pin -- prepare_initial_boundary must forward an explicit override
    through to the boundary parser rather than silently relabeling the
    seed's real geometry."""
    boundary = prepare_initial_boundary(
        _boundary_dict_at(4, 4),
        max_poloidal_mode=4,
        max_toroidal_mode=4,
        n_field_periods=1,
    )

    assert boundary.n_field_periods == 1
    assert boundary.n_field_periods != N_FIELD_PERIODS


@pytest.mark.parametrize(
    "call_count,in_flight,max_forward_model_calls,expected",
    [
        (0, 0, None, False),  # cap disabled
        (10_000, 500, None, False),  # cap disabled regardless of scale
        (0, 0, 5, False),  # nothing submitted/completed yet
        (4, 0, 5, False),  # 4 completed, budget is 5 -- one more may go out
        (5, 0, 5, True),  # exactly at budget via completed calls alone
        (3, 2, 5, True),  # completed + in-flight already reaches budget
        (3, 1, 5, False),  # completed + in-flight is still under budget
        (0, 5, 5, True),  # budget reached purely by in-flight submissions
    ],
)
def test_forward_model_call_budget_reached(
    call_count, in_flight, max_forward_model_calls, expected
):
    assert (
        _forward_model_call_budget_reached(
            call_count,
            in_flight=in_flight,
            max_forward_model_calls=max_forward_model_calls,
        )
        is expected
    )


# ---------------------------------------------------------------------------
# _stall_exceeded: the fail-don't-hang watchdog predicate
# (session-handoff-2026-07-24-fix-qi-resource-management). Pure, so the
# stall-detection logic is pinned without the real forkserver/physics loop.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "last_progress,now,max_stall_seconds,expected",
    [
        (0.0, 100.0, None, False),  # watchdog disabled -> never trips
        (0.0, 1_000_000.0, None, False),  # disabled regardless of scale
        (0.0, 10.0, 60.0, False),  # 10s < 60s cap, still making progress
        (0.0, 60.0, 60.0, False),  # exactly at the cap is not yet exceeded
        (0.0, 61.0, 60.0, True),  # past the cap -> trips
        (100.0, 100.0, 30.0, False),  # zero elapsed since last progress
    ],
)
def test_stall_exceeded(last_progress, now, max_stall_seconds, expected):
    assert _stall_exceeded(last_progress, now, max_stall_seconds=max_stall_seconds) is expected


# ---------------------------------------------------------------------------
# _convergence_stalled: the smart-cap convergence-plateau predicate
# (constellaration-smart-cap-experiment). Pure, so the retire-a-plateaued-seed
# logic is pinned without the real forkserver/physics loop. `violation_history`
# is the running-best violation at each completed outer iteration (non-increasing).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "violation_history,patience_iters,min_delta,expected",
    [
        # disabled -> never trips, regardless of history
        ([1.0, 0.9, 0.8, 0.7, 0.6], None, 0.0, False),
        ([0.5, 0.5, 0.5, 0.5, 0.5], 0, 0.0, False),
        ([0.5, 0.5, 0.5, 0.5, 0.5], -1, 0.0, False),
        # not enough history yet: need strictly more than patience_iters points
        ([0.5, 0.5, 0.5], 3, 0.0, False),
        ([0.5, 0.5, 0.5, 0.4], 3, 0.0, False),  # window improved, still descending
        # a genuine plateau: the whole trailing window fails to beat the prior
        # best (the drop must PRECEDE the window, not sit inside it)
        ([0.5, 0.4, 0.3, 0.3, 0.3, 0.3], 3, 0.0, True),  # flat window -> trips
        ([0.5, 0.4, 0.3, 0.299, 0.298, 0.297], 3, 0.005, True),  # 0.003 <= 0.005
        # still descending faster than min_delta -> does not trip
        ([0.5, 0.4, 0.3, 0.299, 0.298, 0.297], 3, 0.001, False),  # 0.003 > 0.001
        ([0.5, 0.4, 0.3, 0.2, 0.1], 3, 0.0, False),  # steady descent
        # a drop INSIDE the trailing window still counts as improvement
        ([0.5, 0.4, 0.30, 0.30, 0.30], 3, 0.0, False),  # 0.4->0.3 is within window
        # improvement measured vs best BEFORE the window, not just the last point
        ([0.2, 0.2, 0.2, 0.2, 0.2], 2, 0.0, True),  # long flat -> trips
    ],
)
def test_convergence_stalled(violation_history, patience_iters, min_delta, expected):
    assert (
        _convergence_stalled(
            violation_history,
            patience_iters=patience_iters,
            min_delta=min_delta,
        )
        is expected
    )


# ---------------------------------------------------------------------------
# Full ALM-state checkpoint/resume round-trips (constellaration-smart-cap-
# experiment). The whole point of the fix is that resuming restores the exact
# matured dual state + RNG stream, so these pin the (de)serialization.
# ---------------------------------------------------------------------------


def test_alm_state_round_trip_preserves_all_fields():
    import jax.numpy as jnp
    from constellaration.optimization import augmented_lagrangian as al

    state = al.AugmentedLagrangianState(
        x=jnp.asarray([0.1, -0.2, 0.3, 0.4], dtype=jnp.float32),
        multipliers=jnp.asarray([1.5, -2.5], dtype=jnp.float32),
        penalty_parameters=jnp.asarray([1000.0, 2500.0], dtype=jnp.float32),
        objective=jnp.asarray(3.14, dtype=jnp.float32),
        constraints=jnp.asarray([0.01, -0.02], dtype=jnp.float32),
        bounds=jnp.asarray([0.05, 0.05, 0.05, 0.05], dtype=jnp.float32),
    )
    # survives a JSON encode/decode (the real path writes/reads JSON on disk)
    restored = _alm_state_from_jsonable(json.loads(json.dumps(_alm_state_to_jsonable(state))))
    for field in ("x", "multipliers", "penalty_parameters", "constraints", "bounds"):
        np.testing.assert_allclose(
            np.asarray(getattr(restored, field)),
            np.asarray(getattr(state, field)),
            rtol=1e-6,
        )
    np.testing.assert_allclose(
        float(np.asarray(restored.objective)),
        float(np.asarray(state.objective)),
        rtol=1e-6,
    )


def test_rng_state_round_trip_continues_identical_stream():
    rng = np.random.RandomState(12345)
    rng.randint(0, 2**31 - 1, size=7)  # advance the stream partway
    restored = _rng_state_from_jsonable(json.loads(json.dumps(_rng_state_to_jsonable(rng))))
    # both generators must now emit the SAME next sub-seeds (this is what makes
    # a resumed run continue the per-outer-iteration seed stream faithfully)
    assert [rng.randint(0, 2**31 - 1) for _ in range(5)] == [
        restored.randint(0, 2**31 - 1) for _ in range(5)
    ]


def test_write_checkpoint_atomic_round_trips_and_leaves_no_tmp(tmp_path):
    path = tmp_path / "nested" / "seed_17.json"
    payload = {"schema": "alm_ngopt_checkpoint_v1", "k_completed": 3, "budget": 2280}
    _write_checkpoint_atomic(path, payload)
    assert json.loads(path.read_text()) == payload
    # no leftover .tmp sidecar after a successful atomic replace
    assert not list(path.parent.glob("*.tmp"))


# ---------------------------------------------------------------------------
# Canonical feasibility gate (session-handoff-2026-08-17): the in-loop
# forward model's feasibility verdict diverges from the canonical scorer's
# exactly in the endgame (violation ratio ~1.01 at 0.10+, 3.39x at 0.003 --
# it declared the -3.96 point feasible, canonical says infeasible). A
# search-fidelity `feasible=True` must therefore be CONFIRMED at canonical
# fidelity before it may flip `best_feasible_found` and switch best-tracking
# from violation-minimising to objective-improving. Pure/injectable, per the
# module-docstring discipline: no physics in unit tests.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gate_enabled,feasible,violation,last_confirm_violation,expected",
    [
        # gate off -> never confirm, regardless of the rest
        (False, True, 0.001, float("inf"), False),
        (False, True, 0.0, 0.5, False),
        # infeasible at search fidelity -> nothing to confirm
        (True, False, 0.2, float("inf"), False),
        # first search-feasible claim -> confirm
        (True, True, 0.003, float("inf"), True),
        # strictly better than the last confirmed attempt -> confirm again
        (True, True, 0.002, 0.003, True),
        # equal or worse than the last attempt -> do NOT re-confirm (this is
        # the cost bound: confirmations only on the descent's improvement
        # steps, never per-call -- 67% of the 2026-07-13 run's 15k calls were
        # search-feasible)
        (True, True, 0.003, 0.003, False),
        (True, True, 0.004, 0.003, False),
    ],
)
def test_should_confirm_canonically(
    gate_enabled, feasible, violation, last_confirm_violation, expected
):
    assert (
        _should_confirm_canonically(
            gate_enabled=gate_enabled,
            feasible=feasible,
            violation=violation,
            last_confirm_violation=last_confirm_violation,
        )
        is expected
    )


def test_gate_demotes_a_false_feasible_verdict_to_infeasible():
    """The exact seed-5 failure: search says feasible (violation 0.00296),
    canonical says infeasible (margin 0.01004). Effective verdict must be
    infeasible, and the canonical margin must be surfaced for logging."""
    verdict = _apply_canonical_gate(
        feasible=True,
        violation=0.00296,
        last_confirm_violation=float("inf"),
        gate_enabled=True,
        confirm_fn=lambda: (False, 0.01004),
    )
    assert verdict.effective_feasible is False
    assert verdict.canonical_margin == 0.01004
    # the attempt is recorded so the next equal-or-worse claim skips the
    # ~148s canonical call
    assert verdict.last_confirm_violation == 0.00296


def test_gate_passes_a_confirmed_feasible_verdict_through():
    verdict = _apply_canonical_gate(
        feasible=True,
        violation=0.001,
        last_confirm_violation=float("inf"),
        gate_enabled=True,
        confirm_fn=lambda: (True, 0.0),
    )
    assert verdict.effective_feasible is True
    assert verdict.canonical_margin == 0.0


def test_gate_does_not_call_the_expensive_confirm_when_not_needed():
    calls = []

    def confirm_fn():
        calls.append(1)
        return (True, 0.0)

    # infeasible -> no confirmation
    verdict = _apply_canonical_gate(
        feasible=False,
        violation=0.2,
        last_confirm_violation=float("inf"),
        gate_enabled=True,
        confirm_fn=confirm_fn,
    )
    assert verdict.effective_feasible is False
    assert verdict.canonical_margin is None
    # feasible but not better than the last confirmed attempt -> no call,
    # and the verdict is DEMOTED (not trusted) because the gate is on and
    # the claim was not confirmed
    verdict = _apply_canonical_gate(
        feasible=True,
        violation=0.003,
        last_confirm_violation=0.003,
        gate_enabled=True,
        confirm_fn=confirm_fn,
    )
    assert verdict.effective_feasible is False
    assert calls == []


def test_gate_disabled_trusts_the_search_verdict_unchanged():
    calls = []

    def confirm_fn():
        calls.append(1)
        return (False, 0.5)

    verdict = _apply_canonical_gate(
        feasible=True,
        violation=0.001,
        last_confirm_violation=float("inf"),
        gate_enabled=False,
        confirm_fn=confirm_fn,
    )
    assert verdict.effective_feasible is True
    assert calls == []


# ---------------------------------------------------------------------------
# Gate thrash fix (2026-08-17 afternoon, live-run finding): in a converged
# pocket the search produced a hairline proxy improvement every ~60-90s, each
# strictly beating the watermark and firing a ~50s canonical call -- 93
# confirmations in ~2h15, ~85% duty cycle, all landing at the SAME canonical
# margin (0.0097 +- 0.0001). A confirmation must therefore require a
# MEANINGFUL relative improvement over the watermark, not merely a strict one.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "violation,last_confirm_violation,min_improvement,expected",
    [
        # default 0.0 preserves the original strictly-better behaviour
        (0.0029, 0.0030, 0.0, True),
        (0.0030, 0.0030, 0.0, False),
        # 2% threshold: a hairline improvement no longer fires...
        (0.000813, 0.000814, 0.02, False),
        # ...a >=2% improvement does
        (0.00078, 0.00080, 0.02, True),
        # boundary: exactly at the threshold is NOT enough (must be strictly
        # inside), so a stream of exactly-2% steps cannot thrash either
        (0.00098, 0.00100, 0.02, False),
        # first-ever claim (watermark inf) always confirms
        (0.5, float("inf"), 0.02, True),
    ],
)
def test_confirm_requires_meaningful_improvement_over_watermark(
    violation, last_confirm_violation, min_improvement, expected
):
    assert (
        _should_confirm_canonically(
            gate_enabled=True,
            feasible=True,
            violation=violation,
            last_confirm_violation=last_confirm_violation,
            min_improvement=min_improvement,
        )
        is expected
    )


# ---------------------------------------------------------------------------
# Event-driven gate banking (2026-08-17 afternoon, live-run finding): five+
# confirmed-feasible boundaries existed ONLY in process memory for hours
# because candidates are written at run end and checkpoints per-k -- stopping
# the run meant losing them. Every gate verdict must hit disk the moment it
# exists: append-one-flushed-JSONL-line-per-verdict.
# ---------------------------------------------------------------------------


def test_bank_gate_verdict_appends_flushed_jsonl_lines(tmp_path):
    path = tmp_path / "nested" / "gate_log.jsonl"
    first = {
        "outer_iteration": 43,
        "search_violation": 0.00227,
        "canonical_feasible": True,
        "canonical_margin": 0.00806,
        "boundary": {"r_cos": [[0.1]], "z_sin": [[0.2]]},
    }
    second = {
        "outer_iteration": 43,
        "search_violation": 0.00221,
        "canonical_feasible": False,
        "canonical_margin": 0.01118,
        "boundary": {"r_cos": [[0.3]], "z_sin": [[0.4]]},
    }
    _bank_gate_verdict(path, first)
    # durable immediately after the first append -- this is the whole point
    assert [json.loads(line) for line in path.read_text().splitlines()] == [first]
    _bank_gate_verdict(path, second)
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        first,
        second,
    ]


# ---------------------------------------------------------------------------
# Objective-mode confirmation trigger (qi-family-search-campaign-plan Phase C;
# the B1 watermark-zero lockout): once a run holds a CONFIRMED feasible
# incumbent, a claim with a meaningfully better OBJECTIVE must also be
# confirmable -- B1 drove its violation watermark to 0 and the gate closed,
# freezing the incumbent at the first-confirmed objective forever.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "objective,incumbent_objective,minimize,objective_min_improvement,expected",
    [
        # maximize (SimpleToBuildQIStellarator): meaningfully better -> fire
        (11.5, 11.0, False, 0.02, True),  # +4.5% > 2%
        (11.1, 11.0, False, 0.02, False),  # +0.9% < 2%
        (10.5, 11.0, False, 0.02, False),  # worse never fires
        # minimize direction mirrors
        (10.5, 11.0, True, 0.02, True),
        (10.9, 11.0, True, 0.02, False),
        # no incumbent yet -> objective path never fires (violation path owns
        # the pre-feasible regime)
        (11.5, None, False, 0.02, False),
    ],
)
def test_objective_trigger_fires_on_meaningful_improvement_over_incumbent(
    objective, incumbent_objective, minimize, objective_min_improvement, expected
):
    assert (
        _should_confirm_objective(
            feasible=True,
            gate_enabled=True,
            objective=objective,
            incumbent_objective=incumbent_objective,
            minimize_objective=minimize,
            min_improvement=objective_min_improvement,
            last_confirm_objective=None,
        )
        is expected
    )


def test_objective_trigger_has_its_own_watermark():
    """Symmetric to the violation watermark: repeated hairline objective
    improvements must not thrash the gate. A claim must beat the last
    CONFIRMED-ATTEMPT objective by the margin too."""
    common = dict(
        feasible=True,
        gate_enabled=True,
        incumbent_objective=11.0,
        minimize_objective=False,
        min_improvement=0.02,
    )
    # beats incumbent by 4.5% but not the last attempt (11.4) by 2% -> no
    assert _should_confirm_objective(objective=11.5, last_confirm_objective=11.4, **common) is False
    # beats both -> yes
    assert _should_confirm_objective(objective=11.7, last_confirm_objective=11.4, **common) is True


def test_objective_trigger_never_fires_for_infeasible_or_gate_off():
    for feasible, gate_enabled in [(False, True), (True, False)]:
        assert (
            _should_confirm_objective(
                feasible=feasible,
                gate_enabled=gate_enabled,
                objective=99.0,
                incumbent_objective=11.0,
                minimize_objective=False,
                min_improvement=0.02,
                last_confirm_objective=None,
            )
            is False
        )


def test_violation_watermark_never_rises_from_an_objective_confirmation():
    """min-semantics: an objective-triggered confirmation at HIGHER violation
    than the violation watermark must not raise the watermark -- otherwise
    it reopens the violation-thrash the 2%-rule closed."""
    assert _updated_violation_watermark(0.0005, 0.008) == pytest.approx(0.0005)
    assert _updated_violation_watermark(0.008, 0.0005) == pytest.approx(0.0005)


# ---------------------------------------------------------------------------
# Async confirm slot (Phase C): while one canonical confirmation is in
# flight the cheap search keeps running; new claims HOLD (best-only,
# supersede-stale) rather than stall; never two in flight.
# ---------------------------------------------------------------------------


def _claim(call_index, violation, objective=11.0):
    return {"call_index": call_index, "violation": violation, "objective": objective}


def test_async_slot_submits_when_idle():
    slot = _AsyncConfirmSlot()
    assert slot.offer(_claim(1, 0.008)) == "submit"
    assert slot.in_flight["call_index"] == 1


def test_async_slot_holds_best_pending_while_busy():
    """Second claim while busy: held, not submitted. A better third claim
    supersedes the held one; a worse one is dropped."""
    slot = _AsyncConfirmSlot()
    slot.offer(_claim(1, 0.008))
    assert slot.offer(_claim(2, 0.006)) == "held"
    assert slot.offer(_claim(3, 0.007)) == "dropped"  # worse than held 0.006
    assert slot.offer(_claim(4, 0.004)) == "superseded"  # better than held
    assert slot.pending["call_index"] == 4


def test_async_slot_completion_promotes_the_pending_claim():
    slot = _AsyncConfirmSlot()
    slot.offer(_claim(1, 0.008))
    slot.offer(_claim(2, 0.004))
    finished = slot.complete()
    assert finished["call_index"] == 1
    # pending is promoted to next-in-flight by the caller asking for it
    nxt = slot.take_pending()
    assert nxt["call_index"] == 2
    assert slot.in_flight is None or slot.in_flight["call_index"] == 2


def test_async_slot_ranks_held_claims_by_violation_then_objective():
    """Pending replacement prefers lower violation; ties break on better
    objective (maximize)."""
    slot = _AsyncConfirmSlot(minimize_objective=False)
    slot.offer(_claim(1, 0.008))
    slot.offer(_claim(2, 0.005, objective=11.0))
    assert slot.offer(_claim(3, 0.005, objective=11.4)) == "superseded"
    assert slot.pending["call_index"] == 3


def test_async_slot_never_two_in_flight():
    slot = _AsyncConfirmSlot()
    slot.offer(_claim(1, 0.008))
    for i in range(2, 6):
        assert slot.offer(_claim(i, 0.008 / i)) in ("held", "superseded", "dropped")
    assert slot.in_flight["call_index"] == 1


# ---------------------------------------------------------------------------
# Near-feasible candidate retention (session-handoff-2026-08-17): the k=38
# true best (violation 0.008620) was LOST because per-k history only started
# later and the candidates log only admitted feasible calls -- it was 0
# bytes after a 46-iteration descent. Infeasible-but-close points must be
# retained too, best-N by violation.
# ---------------------------------------------------------------------------


def test_near_feasible_infeasible_point_below_threshold_is_retained():
    tracker = CandidateTracker(near_feasible_threshold=0.05, near_shortlist_n=5)
    tracker.consider(_record(1, objective=1.0, violation=0.008620), feasible=False)
    assert [r["call_index"] for r in tracker.near_feasible()] == [1]
    # and it must NOT leak into the feasible stream
    assert tracker.combined() == []


def test_near_feasible_point_above_threshold_is_dropped():
    tracker = CandidateTracker(near_feasible_threshold=0.05, near_shortlist_n=5)
    tracker.consider(_record(1, objective=1.0, violation=0.2), feasible=False)
    assert tracker.near_feasible() == []


def test_near_feasible_shortlist_keeps_best_n_by_violation():
    tracker = CandidateTracker(near_feasible_threshold=0.05, near_shortlist_n=2)
    tracker.consider(_record(1, objective=1.0, violation=0.03), feasible=False)
    tracker.consider(_record(2, objective=1.0, violation=0.01), feasible=False)
    tracker.consider(_record(3, objective=1.0, violation=0.02), feasible=False)
    assert [r["call_index"] for r in tracker.near_feasible()] == [2, 3]


def test_feasible_records_are_unaffected_by_the_near_feasible_stream():
    """Default `feasible=True` keeps every existing call site's behaviour."""
    tracker = CandidateTracker(near_feasible_threshold=0.05, near_shortlist_n=5)
    tracker.consider(_record(1, objective=1.0, violation=0.0))
    assert [r["call_index"] for r in tracker.combined()] == [1]
    assert tracker.near_feasible() == []


def test_write_emits_near_feasible_records_to_a_separate_file(tmp_path):
    tracker = CandidateTracker(near_feasible_threshold=0.05, near_shortlist_n=5)
    tracker.consider(_record(1, objective=1.0, violation=0.0))
    tracker.consider(_record(2, objective=1.0, violation=0.009), feasible=False)
    feasible_path = tmp_path / "candidates.jsonl"
    near_path = tmp_path / "near_feasible.jsonl"
    tracker.write(feasible_path, near_feasible_path=near_path)
    feasible_rows = [json.loads(line) for line in feasible_path.read_text().splitlines()]
    near_rows = [json.loads(line) for line in near_path.read_text().splitlines()]
    assert [r["call_index"] for r in feasible_rows] == [1]
    assert [r["call_index"] for r in near_rows] == [2]


# ---------------------------------------------------------------------------
# Resume best-tracking policy (session-handoff-2026-08-17): the seed-5 k=43
# checkpoint records `best_feasible_found=True` -- a FALSE claim made by the
# unreliable in-loop fidelity. A gated re-descent must be able to demote that
# claim WITHOUT the full `resume_reset_best_tracking` reset (which also
# throws away the matured 12,680 budget and the convergence history, the
# very things the resume exists to keep).
# ---------------------------------------------------------------------------


def _ckpt(
    best_violation=0.0096,
    best_objective=7.08,
    best_feasible_found=True,
    budget=12680,
    violation_history=(0.5, 0.1, 0.0096),
):
    return {
        "best_violation": best_violation,
        "best_objective": best_objective,
        "best_feasible_found": best_feasible_found,
        "budget": budget,
        "violation_history": list(violation_history),
    }


def test_resume_tracking_default_restores_the_checkpoint_verbatim():
    state = _resume_tracking_state(
        _ckpt(), reset_best_tracking=False, clear_feasible_incumbent=False
    )
    assert state.best_violation == 0.0096
    assert state.best_objective == 7.08
    assert state.best_feasible_found is True
    assert state.budget == 12680
    assert state.violation_history == [0.5, 0.1, 0.0096]


def test_resume_tracking_full_reset_clears_everything():
    """`resume_reset_best_tracking` semantics, unchanged: fidelity-change
    resume re-establishes best, budget, AND history from scratch."""
    state = _resume_tracking_state(
        _ckpt(), reset_best_tracking=True, clear_feasible_incumbent=False
    )
    assert state.best_violation == float("inf")
    assert state.best_objective == float("inf")
    assert state.best_feasible_found is False
    assert state.budget is None
    assert state.violation_history == []


def test_resume_tracking_clear_feasible_incumbent_demotes_but_keeps_the_descent():
    state = _resume_tracking_state(
        _ckpt(), reset_best_tracking=False, clear_feasible_incumbent=True
    )
    # the false feasible claim is demoted...
    assert state.best_feasible_found is False
    assert state.best_objective == float("inf")
    # ...but the descent state the resume exists to keep is KEPT
    assert state.best_violation == 0.0096
    assert state.budget == 12680
    assert state.violation_history == [0.5, 0.1, 0.0096]


def test_resume_tracking_rejects_both_flags_at_once():
    """The two policies contradict each other on budget/history -- a caller
    passing both has a bug; fail loud, not quiet precedence."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        _resume_tracking_state(_ckpt(), reset_best_tracking=True, clear_feasible_incumbent=True)


# ---------------------------------------------------------------------------
# qi-canonical-polish-plan Phase 1: the fidelity_switch resume policy -- the
# CORRECT version of what `--search-fidelity high` should have been. The
# polish-5b failure decomposed into two pieces mapped to two policies:
# best-tracking DOES need a full reset at a fidelity change (old-fidelity
# violations don't compare -- keeping the proxy's 0.0018 baseline would
# freeze best_x against canonical violations ~4x larger), but the nevergrad
# BUDGET does not: resetting it restarted the inner search at 1,760 evals in
# an 80-D box against a 12,680-eval incumbent, which is why polish-5b could
# only produce 0.15-0.09 iterates against its own 0.010 checkpoint.
# ---------------------------------------------------------------------------


def test_resume_tracking_fidelity_switch_resets_best_but_keeps_budget():
    state = _resume_tracking_state(_ckpt(), fidelity_switch=True)
    # best-tracking fully re-established (values from the OLD fidelity do
    # not compare at the new one)...
    assert state.best_violation == float("inf")
    assert state.best_objective == float("inf")
    assert state.best_feasible_found is False
    # ...history cleared (the false-stall bug fix, same reasoning)...
    assert state.violation_history == []
    # ...but the matured inner-search budget is KEPT -- the polish-5b fix
    assert state.budget == 12680


def test_resume_tracking_fidelity_switch_is_exclusive_with_the_other_policies():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _resume_tracking_state(_ckpt(), reset_best_tracking=True, fidelity_switch=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _resume_tracking_state(_ckpt(), clear_feasible_incumbent=True, fidelity_switch=True)


def test_resume_tracking_clear_stall_history_keeps_best_and_budget():
    """Phase C climb resume: after a premature objective-stall (patience
    fired before the duals matured), the resume must keep the REAL confirmed
    incumbent and the matured budget but clear the stall histories -- a
    verbatim restore re-imports the flat history and re-stalls on the first
    iteration."""
    state = _resume_tracking_state(_ckpt(), clear_stall_history=True)
    assert state.best_violation == 0.0096
    assert state.best_feasible_found is True
    assert state.budget == 12680
    assert state.violation_history == []


def test_resume_tracking_clear_stall_history_exclusive_with_other_policies():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _resume_tracking_state(_ckpt(), clear_stall_history=True, fidelity_switch=True)


def test_resume_tracking_flags_default_off_preserves_verbatim_restore():
    """All three policies default off -> the checkpoint restores verbatim
    (guards the existing callers as the signature grows)."""
    state = _resume_tracking_state(_ckpt())
    assert state.best_violation == 0.0096
    assert state.best_feasible_found is True
    assert state.budget == 12680


# ---------------------------------------------------------------------------
# qi-canonical-polish-plan Phase 1: margin incumbent. The objective chase
# (feasible + better objective wins) can wander to thin-margin winners --
# run 1's incumbent went 0.00806 -> objective-better points at ~0.0097. The
# deepest-margin FEASIBLE point must be tracked alongside best_x so the
# robustness candidate is never lost to the objective race.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "feasible,violation,best_margin,expected",
    [
        (True, 0.005, 0.008, True),  # feasible + deeper -> new margin incumbent
        (True, 0.008, 0.008, False),  # equal is not deeper
        (True, 0.009, 0.008, False),  # shallower
        (False, 0.001, 0.008, False),  # infeasible never deepens, however small
        (True, 0.005, float("inf"), True),  # first feasible point seeds it
    ],
)
def test_deepens_margin(feasible, violation, best_margin, expected):
    assert (
        _deepens_margin(feasible=feasible, violation=violation, best_margin=best_margin) is expected
    )


# ---------------------------------------------------------------------------
# qi-canonical-polish-plan Phase 1: objective-stall mode. The polish run's
# convergence is judged on the OBJECTIVE among feasible incumbents (violation
# flatlines at ~0 the moment the run is feasible, so the existing
# violation-history stall would fire immediately/never meaningfully).
# Direction-aware: SimpleToBuildQIStellarator MAXIMIZES its objective.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "objective_history,patience_iters,min_delta,minimize,expected",
    [
        # disabled -> never trips
        ([7.0, 7.0, 7.0, 7.0], None, 0.0, False, False),
        # not enough history yet
        ([7.0, 7.0, 7.0], 3, 0.0, False, False),
        # maximize: flat window after a rise -> trips
        ([6.8, 6.9, 7.0, 7.0, 7.0, 7.0], 3, 0.0, False, True),
        # maximize: still climbing faster than min_delta -> does not trip
        ([6.8, 6.9, 7.0, 7.05, 7.1, 7.15], 3, 0.01, False, False),
        # maximize: sub-min_delta creep -> trips
        ([6.8, 6.9, 7.0, 7.001, 7.002, 7.003], 3, 0.01, False, True),
        # minimize direction: falling objective is improvement
        ([7.2, 7.1, 7.0, 6.9, 6.8, 6.7], 3, 0.0, True, False),
        ([7.2, 7.1, 7.0, 7.0, 7.0, 7.0], 3, 0.0, True, True),
    ],
)
def test_objective_converged(objective_history, patience_iters, min_delta, minimize, expected):
    assert (
        _objective_converged(
            objective_history,
            patience_iters=patience_iters,
            min_delta=min_delta,
            minimize_objective=minimize,
        )
        is expected
    )


def test_candidate_tracker_write_is_atomic(tmp_path):
    """qi-canonical-polish-plan Phase 1: with periodic mid-run flushes, a
    crash mid-write must never corrupt the candidates file -- write goes
    through a tmp sidecar + atomic replace, leaving no sidecar behind."""
    tracker = CandidateTracker(near_feasible_threshold=0.05, near_shortlist_n=5)
    tracker.consider(_record(1, objective=1.0, violation=0.0))
    tracker.consider(_record(2, objective=1.0, violation=0.01), feasible=False)
    feasible_path = tmp_path / "candidates.jsonl"
    near_path = tmp_path / "near.jsonl"
    tracker.write(feasible_path, near_feasible_path=near_path)
    assert [json.loads(line) for line in feasible_path.read_text().splitlines()]
    assert [json.loads(line) for line in near_path.read_text().splitlines()]
    assert not list(tmp_path.glob("*.tmp"))
