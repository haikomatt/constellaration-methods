"""qi-discovery-phase-plan.md Phase 1: TDD for the proxy-vs-canonical audit
instrumentation (`src/qi_audit_instrument.py`).

Two capture paths, per the plan:

1. DESC proxy path (`optimize_boundary_omnigenity_desc` /
   `_optimize_equilibrium_for_omnigenity_desc`) discards the optimizer's
   `OptimizeResult` via `(equilibrium,), _ = optimizer.optimize(...)` -- see
   that private function's body in the installed `constellaration` package.
   `_capture_desc_optimize_result`/`run_desc_omnigenity_with_capture` recover
   it via the same "monkeypatch the exact discarding seam, call the
   library's own public unmodified entry point" idiom
   `alm_ngopt_optimizer.py` already uses for its
   `pytree.register_pydantic_data` patch.
2. ALM-NGOpt canonical path already writes a per-call trajectory ndjson
   (`run_alm_ngopt_optimization`'s `trajectory_log_path`) -- captured for
   completeness/symmetry via `summarize_alm_ngopt_trajectory`, which reuses
   `trajectory_logger.read_trajectory` rather than re-implementing an ndjson
   reader.

All tests here are deterministic: no VMEC, no real DESC equilibrium solve.
The DESC-path functions are exercised either against a monkeypatched
`desc.optimize.Optimizer.optimize` (the real class, real capture-context
code, fake solver body) or via injected `optimize_fn`/`forward_model_fn`
callables standing in for the expensive real ones. A single real-seam probe
(not part of this suite) lives in `scripts/qi_audit_instrument_probe.py` --
see its module docstring and the Phase 1 report for the evidence it produced.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip(
    "constellaration",
    reason="needs the pinned constellaration/vmecpp stack; run inside the repo's Docker image",
)
pytest.importorskip("desc", reason="needs the desc package; run inside the repo's Docker image")
from constellaration import forward_model
from desc import optimize as desc_optimize

from constellaration_methods.evaluation import qi_audit_instrument as instrument

pytestmark = pytest.mark.container


def _fixture_boundary():
    """A synthetic (never leaderboard-derived), minimal, valid
    `SurfaceRZFourier`, used here only as an inert pass-through value (these
    tests never call real DESC/VMEC on it)."""
    from constellaration.geometry import surface_rz_fourier

    r_cos = np.zeros((5, 9))
    z_sin = np.zeros((5, 9))
    r_cos[0, 4] = 1.0  # major radius
    return surface_rz_fourier.SurfaceRZFourier(
        r_cos=r_cos, z_sin=z_sin, n_field_periods=3, is_stellarator_symmetric=True
    )


def _fake_metrics(qi_value):
    return forward_model.ConstellarationMetrics(
        aspect_ratio=8.0,
        aspect_ratio_over_edge_rotational_transform=16.0,
        max_elongation=4.0,
        axis_rotational_transform_over_n_field_periods=0.4,
        edge_rotational_transform_over_n_field_periods=0.5,
        axis_magnetic_mirror_ratio=0.1,
        edge_magnetic_mirror_ratio=0.15,
        average_triangularity=-0.2,
        vacuum_well=0.01,
        minimum_normalized_magnetic_gradient_scale_length=3.0,
        qi=qi_value,
    )


def _fake_objective_values(term_key="Omnigenity error: ", f0_mean=1.0e-3, f_mean=2.0e-5):
    """Shaped exactly like the real `OptimizeResult["Objective values"]`
    dict DESC's `Optimizer.optimize` returns for a non-scalar, non-linear
    objective term (the "weighted mean" `print_value` branch) -- a dict of
    term-name -> list-of-one-dict-per-instance, each with `f0_*`/`f_*` stat
    keys. Confirmed directly against
    `desc.objectives._omnigenity.Omnigenity._print_value_fmt` and
    `desc.objectives.objective_funs.ObjectiveFunction.print_value` in
    `localhost/constellaration-methods:test`."""
    return {
        term_key: [
            {
                "f0_max": f0_mean * 3,
                "f_max": f_mean * 3,
                "f0_min": f0_mean * 0.1,
                "f_min": f_mean * 0.1,
                "f0_mean": f0_mean,
                "f_mean": f_mean,
            }
        ]
    }


# ---------------------------------------------------------------------------
# 1. _capture_desc_optimize_result
# ---------------------------------------------------------------------------


def test_capture_desc_optimize_result_captures_the_discarded_result(monkeypatch):
    fake_result = {"Objective values": _fake_objective_values()}

    def fake_optimize(self, *args, **kwargs):
        return (["thing"], fake_result)

    monkeypatch.setattr(desc_optimize.Optimizer, "optimize", fake_optimize)

    with instrument._capture_desc_optimize_result() as captured:
        things, result = desc_optimize.Optimizer.optimize(object(), things=None, objective=None)

    assert things == ["thing"]
    assert result is fake_result
    assert captured["results"] == [fake_result]


def test_capture_desc_optimize_result_captures_every_call_in_order(monkeypatch):
    """The real DESC path calls `optimizer.optimize` up to three times per
    invocation (see module docstring's real-seam-probe finding) -- the
    capture context must keep all of them, not just the last."""
    results_by_call = [{"Objective values": {"a": i}} for i in range(3)]
    call_index = {"n": 0}

    def fake_optimize(self, *args, **kwargs):
        result = results_by_call[call_index["n"]]
        call_index["n"] += 1
        return (["thing"], result)

    monkeypatch.setattr(desc_optimize.Optimizer, "optimize", fake_optimize)

    with instrument._capture_desc_optimize_result() as captured:
        for _ in range(3):
            desc_optimize.Optimizer.optimize(object())

    assert captured["results"] == results_by_call


def test_capture_desc_optimize_result_restores_original_after(monkeypatch):
    def fake_optimize(self, *args, **kwargs):
        return (["thing"], {"Objective values": {}})

    monkeypatch.setattr(desc_optimize.Optimizer, "optimize", fake_optimize)

    with instrument._capture_desc_optimize_result():
        pass

    assert desc_optimize.Optimizer.optimize is fake_optimize


def test_capture_desc_optimize_result_restores_original_even_on_error(monkeypatch):
    def fake_optimize(self, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(desc_optimize.Optimizer, "optimize", fake_optimize)

    with pytest.raises(RuntimeError, match="boom"):
        with instrument._capture_desc_optimize_result():
            desc_optimize.Optimizer.optimize(object())

    assert desc_optimize.Optimizer.optimize is fake_optimize


# ---------------------------------------------------------------------------
# 2. run_desc_omnigenity_with_capture
# ---------------------------------------------------------------------------


def test_run_desc_omnigenity_with_capture_returns_boundary_and_results_list(
    monkeypatch,
):
    fake_result = {"Objective values": _fake_objective_values()}
    sentinel_boundary = object()

    def fake_optimize(self, *args, **kwargs):
        return (["thing"], fake_result)

    monkeypatch.setattr(desc_optimize.Optimizer, "optimize", fake_optimize)

    def fake_optimize_boundary_fn(targets, settings):
        # Simulates optimize_boundary_omnigenity_desc: it calls
        # optimizer.optimize(...) internally, discards the result, and
        # returns just a boundary.
        desc_optimize.Optimizer.optimize(object(), things=None, objective=None)
        return sentinel_boundary

    boundary, results = instrument.run_desc_omnigenity_with_capture(
        targets="fake-targets",
        settings="fake-settings",
        optimize_boundary_fn=fake_optimize_boundary_fn,
    )

    assert boundary is sentinel_boundary
    assert results == [fake_result]


def test_run_desc_omnigenity_with_capture_returns_all_three_real_seam_calls(
    monkeypatch,
):
    """Regression for the real-seam probe finding: the real
    `_optimize_equilibrium_for_omnigenity_desc` calls `optimizer.optimize`
    THREE times (force-balance solve, omnigenity optimize, force-balance
    re-solve) via `solve_desc_equilibrium`/`Equilibrium.solve` internally
    calling `Optimizer.optimize` too -- confirmed by
    `scripts/qi_audit_instrument_probe.py`'s first run capturing the
    re-solve's `"Force error: "` result instead of `"Omnigenity error: "`
    when only the last call was kept. This fakes that exact three-call
    shape and asserts all three survive, in order."""
    force_balance_result = {"Objective values": {"Force error: ": [{"f": 1.0}]}}
    omnigenity_result = {"Objective values": _fake_objective_values()}
    force_balance_resolve_result = {"Objective values": {"Force error: ": [{"f": 0.1}]}}
    call_results = [
        force_balance_result,
        omnigenity_result,
        force_balance_resolve_result,
    ]

    def counting_optimize(self, *args, **kwargs):
        result = call_results[counting_optimize.calls]
        counting_optimize.calls += 1
        return (["thing"], result)

    counting_optimize.calls = 0
    monkeypatch.setattr(desc_optimize.Optimizer, "optimize", counting_optimize)

    def fake_optimize_boundary_fn(targets, settings):
        # solve_desc_equilibrium (force-balance)
        desc_optimize.Optimizer.optimize(object())
        # the omnigenity-objective optimize
        desc_optimize.Optimizer.optimize(object())
        # solve_desc_equilibrium again (force-balance re-solve)
        desc_optimize.Optimizer.optimize(object())
        return object()

    _boundary, results = instrument.run_desc_omnigenity_with_capture(
        targets="fake-targets",
        settings="fake-settings",
        optimize_boundary_fn=fake_optimize_boundary_fn,
    )

    assert results == call_results


def test_run_desc_omnigenity_with_capture_raises_if_optimize_never_called():
    def fake_optimize_boundary_fn(targets, settings):
        return object()  # never touches Optimizer.optimize

    with pytest.raises(RuntimeError, match="optimize.*not called|never called"):
        instrument.run_desc_omnigenity_with_capture(
            targets="fake-targets",
            settings="fake-settings",
            optimize_boundary_fn=fake_optimize_boundary_fn,
        )


# ---------------------------------------------------------------------------
# 2b. select_optimize_result_with_term
# ---------------------------------------------------------------------------


def test_select_optimize_result_with_term_picks_the_matching_one_among_several():
    force_balance_result = {"Objective values": {"Force error: ": [{"f": 1.0}]}}
    omnigenity_result = {"Objective values": _fake_objective_values()}
    results = [force_balance_result, omnigenity_result, force_balance_result]

    selected = instrument.select_optimize_result_with_term(results)

    assert selected is omnigenity_result


def test_select_optimize_result_with_term_raises_when_none_match():
    results = [{"Objective values": {"Force error: ": [{"f": 1.0}]}}]

    with pytest.raises(RuntimeError, match="none of the"):
        instrument.select_optimize_result_with_term(results)


def test_select_optimize_result_with_term_raises_when_more_than_one_matches():
    duplicate = {"Objective values": _fake_objective_values()}
    results = [duplicate, duplicate]

    with pytest.raises(RuntimeError, match="ambiguous"):
        instrument.select_optimize_result_with_term(results)


# ---------------------------------------------------------------------------
# 3. extract_proxy_objective_values
# ---------------------------------------------------------------------------


def test_extract_proxy_objective_values_returns_before_after_mean():
    values = _fake_objective_values(f0_mean=1.0e-3, f_mean=2.0e-5)

    before, after = instrument.extract_proxy_objective_values(values)

    assert before == pytest.approx(1.0e-3)
    assert after == pytest.approx(2.0e-5)


def test_extract_proxy_objective_values_supports_alternate_stat():
    values = _fake_objective_values(f0_mean=1.0e-3, f_mean=2.0e-5)

    before, after = instrument.extract_proxy_objective_values(values, stat="max")

    assert before == pytest.approx(3.0e-3)
    assert after == pytest.approx(6.0e-5)


def test_extract_proxy_objective_values_raises_clear_error_for_missing_term_key():
    values = _fake_objective_values()

    with pytest.raises(KeyError, match="Omnigenity"):
        instrument.extract_proxy_objective_values(values, term_key="Not a real term")


def test_extract_proxy_objective_values_raises_clear_error_for_missing_stat():
    values = _fake_objective_values()

    with pytest.raises(KeyError, match="not_a_real_stat"):
        instrument.extract_proxy_objective_values(values, stat="not_a_real_stat")


def test_extract_proxy_objective_values_raises_for_empty_entry_list():
    with pytest.raises(ValueError, match="empty"):
        instrument.extract_proxy_objective_values({"Omnigenity error: ": []})


# ---------------------------------------------------------------------------
# 4. recompute_canonical_qi
# ---------------------------------------------------------------------------


def test_recompute_canonical_qi_defaults_to_default_high_fidelity():
    captured_settings = {}

    def fake_forward_model_fn(boundary, settings):
        captured_settings["settings"] = settings
        return _fake_metrics(qi_value=1e-5), None

    instrument.recompute_canonical_qi(_fixture_boundary(), forward_model_fn=fake_forward_model_fn)

    assert (
        captured_settings["settings"]
        == forward_model.ConstellarationSettings.default_high_fidelity()
    )


def test_recompute_canonical_qi_returns_qi_and_log10_qi():
    def fake_forward_model_fn(boundary, settings):
        return _fake_metrics(qi_value=1e-5), None

    result = instrument.recompute_canonical_qi(
        _fixture_boundary(), forward_model_fn=fake_forward_model_fn
    )

    assert result["qi"] == pytest.approx(1e-5)
    assert result["log10_qi"] == pytest.approx(-5.0)
    assert result["canonical_non_convergent"] is False


def test_recompute_canonical_qi_handles_none_qi_gracefully():
    def fake_forward_model_fn(boundary, settings):
        return _fake_metrics(qi_value=None), None

    result = instrument.recompute_canonical_qi(
        _fixture_boundary(), forward_model_fn=fake_forward_model_fn
    )

    assert result["qi"] is None
    assert result["log10_qi"] is None


def test_recompute_canonical_qi_honors_explicit_settings_override():
    explicit = forward_model.ConstellarationSettings(qi_settings=None)
    captured_settings = {}

    def fake_forward_model_fn(boundary, settings):
        captured_settings["settings"] = settings
        return _fake_metrics(qi_value=1e-5), None

    instrument.recompute_canonical_qi(
        _fixture_boundary(), forward_model_fn=fake_forward_model_fn, settings=explicit
    )

    assert captured_settings["settings"] is explicit


def test_recompute_canonical_qi_marks_non_convergent_on_exception():
    def failing_forward_model_fn(boundary, settings):
        raise RuntimeError("VMEC++ did not converge")

    result = instrument.recompute_canonical_qi(
        _fixture_boundary(), forward_model_fn=failing_forward_model_fn
    )

    assert result["qi"] is None
    assert result["log10_qi"] is None
    assert result["canonical_non_convergent"] is True


# ---------------------------------------------------------------------------
# 5. record_desc_proxy_vs_canonical (end-to-end against fixtures)
# ---------------------------------------------------------------------------


def test_record_desc_proxy_vs_canonical_end_to_end_with_fixtures():
    fixture_boundary = _fixture_boundary()
    fake_result = {"Objective values": _fake_objective_values(f0_mean=1.0e-3, f_mean=2.0e-5)}

    def fake_optimize_fn(targets, settings):
        return fixture_boundary, [fake_result]

    def fake_forward_model_fn(boundary, settings):
        assert boundary is fixture_boundary
        return _fake_metrics(qi_value=1.0e-4), None

    record = instrument.record_desc_proxy_vs_canonical(
        targets="fake-targets",
        desc_settings="fake-settings",
        optimize_fn=fake_optimize_fn,
        forward_model_fn=fake_forward_model_fn,
        label="probe-1",
    )

    assert record["label"] == "probe-1"
    assert record["proxy_omnigenity_before"] == pytest.approx(1.0e-3)
    assert record["proxy_omnigenity_after"] == pytest.approx(2.0e-5)
    assert record["canonical_qi"] == pytest.approx(1.0e-4)
    assert record["canonical_log10_qi"] == pytest.approx(-4.0)
    assert record["canonical_non_convergent"] is False
    # delta/ratio between the in-loop proxy's final value and canonical QI --
    # the actual audit finding this instrument exists to characterize.
    assert record["proxy_vs_canonical_delta"] == pytest.approx(1.0e-4 - 2.0e-5)
    assert record["proxy_vs_canonical_ratio"] == pytest.approx(1.0e-4 / 2.0e-5)
    assert set(record["boundary"]) == {"r_cos", "z_sin"}
    np.testing.assert_array_equal(
        np.asarray(record["boundary"]["r_cos"]), np.asarray(fixture_boundary.r_cos)
    )


def test_record_desc_proxy_vs_canonical_handles_non_convergent_canonical():
    fixture_boundary = _fixture_boundary()
    fake_result = {"Objective values": _fake_objective_values()}

    def fake_optimize_fn(targets, settings):
        return fixture_boundary, [fake_result]

    def failing_forward_model_fn(boundary, settings):
        raise RuntimeError("VMEC++ did not converge")

    record = instrument.record_desc_proxy_vs_canonical(
        targets="fake-targets",
        desc_settings="fake-settings",
        optimize_fn=fake_optimize_fn,
        forward_model_fn=failing_forward_model_fn,
    )

    assert record["canonical_non_convergent"] is True
    assert record["canonical_qi"] is None
    assert record["proxy_vs_canonical_delta"] is None
    assert record["proxy_vs_canonical_ratio"] is None
    # the proxy side is independent of canonical convergence -- still captured.
    assert record["proxy_omnigenity_after"] == pytest.approx(2.0e-5)


def test_record_desc_proxy_vs_canonical_honors_custom_term_key_and_stat():
    fixture_boundary = _fixture_boundary()
    fake_result = {
        "Objective values": _fake_objective_values(
            term_key="Quasi-symmetry error: ", f0_mean=5.0, f_mean=1.0
        )
    }

    def fake_optimize_fn(targets, settings):
        return fixture_boundary, [fake_result]

    def fake_forward_model_fn(boundary, settings):
        return _fake_metrics(qi_value=2.0), None

    record = instrument.record_desc_proxy_vs_canonical(
        targets="fake-targets",
        desc_settings="fake-settings",
        optimize_fn=fake_optimize_fn,
        forward_model_fn=fake_forward_model_fn,
        proxy_term_key="Quasi-symmetry error: ",
        proxy_stat="max",
    )

    assert record["proxy_omnigenity_before"] == pytest.approx(15.0)
    assert record["proxy_omnigenity_after"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 6. summarize_alm_ngopt_trajectory (ALM-NGOpt symmetry capture)
# ---------------------------------------------------------------------------


def _traj_line(call, objective, feasible, is_new_best, converged=True):
    return {
        "outer_iteration": 0,
        "call": call,
        "elapsed_s": call * 0.1,
        "converged": converged,
        "objective": objective,
        "constraint_violation": 0.0 if feasible else 1.0,
        "feasible": feasible,
        "is_new_best": is_new_best,
    }


def test_summarize_alm_ngopt_trajectory_computes_counts_and_best_record(tmp_path):
    path = tmp_path / "trajectory.ndjson"
    lines = [
        _traj_line(1, objective=1.0, feasible=False, is_new_best=False),
        _traj_line(2, objective=2.0, feasible=True, is_new_best=True),
        _traj_line(3, objective=3.0, feasible=True, is_new_best=True),
        _traj_line(4, objective=1.5, feasible=True, is_new_best=False),
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    summary = instrument.summarize_alm_ngopt_trajectory(path)

    assert summary["n_calls"] == 4
    assert summary["n_converged"] == 4
    assert summary["n_feasible"] == 3
    assert summary["best_record"]["call"] == 3
    assert summary["proxy_gap"] is None
    assert "canonically" in summary["proxy_gap_note"]
    assert len(summary["trajectory"]) == 4


def test_summarize_alm_ngopt_trajectory_best_record_is_none_when_never_feasible(
    tmp_path,
):
    path = tmp_path / "trajectory.ndjson"
    lines = [
        _traj_line(1, objective=None, feasible=False, is_new_best=False, converged=False),
        _traj_line(2, objective=None, feasible=False, is_new_best=False, converged=False),
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    summary = instrument.summarize_alm_ngopt_trajectory(path)

    assert summary["n_feasible"] == 0
    assert summary["best_record"] is None
