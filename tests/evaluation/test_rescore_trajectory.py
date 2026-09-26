"""High-fidelity re-score and feasibility-flip detection.

Deterministic tests against a mocked forward_model (flip scenarios are easy
to construct precisely this way). Needs `constellaration` installed (runs
inside the repo's own Docker image, same convention as the rest of this
package) because it builds real `SurfaceRZFourier` objects.
"""

from unittest.mock import patch

import numpy as np
import pytest
from constellaration import problems

from constellaration_methods.engines.trajectory_logger import TrajectoryLogger, read_trajectory
from constellaration_methods.evaluation.rescore_trajectory import (
    rescore_step,
    rescore_trajectory_file,
)

# The fixed P1 pin this module's `_boundary_from_dict` assumes (see its
# module docstring) -- mirrored here rather than imported from the
# (out-of-scope) LLM task-prompt template these constants originated in.
_MAX_POLOIDAL_MODE = 4
_MAX_TOROIDAL_MODE = 4


def _synthetic_boundary_dict(max_poloidal_mode=4, max_toroidal_mode=4):
    """A synthetic (never leaderboard-derived) boundary dict of the given
    mode budget: zero everywhere except a unit major radius (m=0, n=0),
    matching the shape `rescore_step` expects from a logged trajectory
    step."""
    m_rows = max_poloidal_mode + 1
    n_cols = 2 * max_toroidal_mode + 1
    r_cos = np.zeros((m_rows, n_cols))
    z_sin = np.zeros((m_rows, n_cols))
    r_cos[0, max_toroidal_mode] = 1.0
    return {"r_cos": r_cos.tolist(), "z_sin": z_sin.tolist()}


class _FakeMetrics:
    """Real feasibility/objective values are derived from these fields by
    GeometricalProblem's actual logic -- no mocking of problem methods,
    since pydantic models reject arbitrary attribute assignment via
    patch.object anyway."""

    def __init__(
        self,
        max_elongation,
        aspect_ratio,
        average_triangularity,
        edge_rotational_transform_over_n_field_periods,
    ):
        self.max_elongation = max_elongation
        self.aspect_ratio = aspect_ratio
        self.average_triangularity = average_triangularity
        self.edge_rotational_transform_over_n_field_periods = (
            edge_rotational_transform_over_n_field_periods
        )


# aspect_ratio<=4.0, average_triangularity<=-0.5, |edge_rotational_transform|>=0.3
_FEASIBLE_METRICS = _FakeMetrics(
    max_elongation=3.0,
    aspect_ratio=2.0,
    average_triangularity=-1.0,
    edge_rotational_transform_over_n_field_periods=0.5,
)
_INFEASIBLE_METRICS = _FakeMetrics(
    max_elongation=3.0,
    aspect_ratio=10.0,
    average_triangularity=-1.0,
    edge_rotational_transform_over_n_field_periods=0.5,
)

pytestmark = pytest.mark.container


def test_rescore_step_with_no_boundary_is_untouched():
    problem = problems.GeometricalProblem()
    settings = object()  # never used; forward_model isn't called
    record = {"step": 1, "boundary": None, "canonical_objective_value": None}

    result = rescore_step(record, problem, settings)

    assert result["canonical_objective_value"] is None


@patch("constellaration_methods.evaluation.rescore_trajectory.forward_model.forward_model")
def test_rescore_step_fills_canonical_fields_on_success(mock_forward_model):
    problem = problems.GeometricalProblem()
    mock_forward_model.return_value = (_FEASIBLE_METRICS, None)
    record = {
        "step": 1,
        "boundary": _synthetic_boundary_dict(_MAX_POLOIDAL_MODE, _MAX_TOROIDAL_MODE),
        "feasible": False,
    }
    result = rescore_step(record, problem, object())

    assert result["canonical_non_convergent"] is False
    assert result["canonical_objective_value"] == 3.0
    assert result["canonical_feasible"] is True
    assert result["canonical_feasibility_margin"] == 0.0


@patch("constellaration_methods.evaluation.rescore_trajectory.forward_model.forward_model")
def test_rescore_step_scores_boundary_larger_than_the_pinned_truncation(mock_forward_model):
    """An 8x8 candidate record (leaderboard-shaped) must not be silently
    skipped here -- the record's own shape is the mode-budget authority."""
    problem = problems.GeometricalProblem()
    mock_forward_model.return_value = (_FEASIBLE_METRICS, None)
    record = {
        "step": 1,
        "boundary": _synthetic_boundary_dict(8, 8),
        "feasible": True,
    }

    result = rescore_step(record, problem, object())

    assert result["canonical_non_convergent"] is False
    assert result["canonical_feasible"] is True


@patch("constellaration_methods.evaluation.rescore_trajectory.forward_model.forward_model")
def test_rescore_step_records_canonical_non_convergence(mock_forward_model):
    problem = problems.GeometricalProblem()
    mock_forward_model.side_effect = RuntimeError("VMEC++ did not converge")
    record = {
        "step": 1,
        "boundary": _synthetic_boundary_dict(_MAX_POLOIDAL_MODE, _MAX_TOROIDAL_MODE),
        "feasible": False,
    }

    result = rescore_step(record, problem, object())

    assert result["canonical_non_convergent"] is True
    assert result["canonical_feasible"] is None
    assert "VMEC++ did not converge" in result["canonical_error"]


@patch("constellaration_methods.evaluation.rescore_trajectory.forward_model.forward_model")
def test_rescore_trajectory_file_detects_flip_to_infeasible(mock_forward_model, tmp_path):
    log_path = tmp_path / "run.ndjson"
    with TrajectoryLogger(log_path, {}) as logger:
        logger.log_step(
            {
                "step": 1,
                "source": "llm",
                "boundary": _synthetic_boundary_dict(),
                "feasible": True,  # agent-visible said feasible
            }
        )

    problem = problems.GeometricalProblem()
    mock_forward_model.return_value = (_INFEASIBLE_METRICS, None)
    summary = rescore_trajectory_file(log_path, tmp_path / "rescored.ndjson", problem=problem)

    assert summary["flips_to_infeasible"] == 1
    assert summary["flips_to_feasible"] == 0
    assert summary["n_steps_rescored"] == 1


@patch("constellaration_methods.evaluation.rescore_trajectory.forward_model.forward_model")
def test_rescore_trajectory_file_detects_flip_to_feasible(mock_forward_model, tmp_path):
    log_path = tmp_path / "run.ndjson"
    with TrajectoryLogger(log_path, {}) as logger:
        logger.log_step(
            {
                "step": 1,
                "source": "llm",
                "boundary": _synthetic_boundary_dict(),
                "feasible": False,  # agent-visible said infeasible
            }
        )

    problem = problems.GeometricalProblem()
    mock_forward_model.return_value = (_FEASIBLE_METRICS, None)
    summary = rescore_trajectory_file(log_path, tmp_path / "rescored.ndjson", problem=problem)

    assert summary["flips_to_feasible"] == 1
    assert summary["flips_to_infeasible"] == 0


@patch("constellaration_methods.evaluation.rescore_trajectory.forward_model.forward_model")
def test_rescore_trajectory_file_output_is_readable_and_preserves_agent_visible_fields(
    mock_forward_model, tmp_path
):
    log_path = tmp_path / "run.ndjson"
    with TrajectoryLogger(log_path, {"model": "test-model"}) as logger:
        logger.log_step(
            {
                "step": 1,
                "source": "llm",
                "boundary": _synthetic_boundary_dict(),
                "objective_value": 5.0,
                "feasible": False,
            }
        )

    problem = problems.GeometricalProblem()
    mock_forward_model.return_value = (_FEASIBLE_METRICS, None)
    out_path = tmp_path / "rescored.ndjson"
    rescore_trajectory_file(log_path, out_path, problem=problem)

    lines = read_trajectory(out_path)
    assert lines[0]["rescored_at"] == "default_high_fidelity"
    step_line = lines[1]
    # agent-visible fields untouched
    assert step_line["objective_value"] == 5.0
    assert step_line["feasible"] is False
    # canonical fields filled in
    assert step_line["canonical_objective_value"] == 3.0
    assert step_line["canonical_feasible"] is True


@patch("constellaration_methods.evaluation.rescore_trajectory.forward_model.forward_model")
def test_no_flips_when_agent_visible_and_canonical_agree(mock_forward_model, tmp_path):
    log_path = tmp_path / "run.ndjson"
    with TrajectoryLogger(log_path, {}) as logger:
        logger.log_step(
            {
                "step": 1,
                "source": "seed",
                "boundary": _synthetic_boundary_dict(),
                "feasible": False,
            }
        )

    problem = problems.GeometricalProblem()
    mock_forward_model.return_value = (_INFEASIBLE_METRICS, None)
    summary = rescore_trajectory_file(log_path, tmp_path / "rescored.ndjson", problem=problem)

    assert summary["flips_to_infeasible"] == 0
    assert summary["flips_to_feasible"] == 0


@patch("constellaration_methods.evaluation.rescore_trajectory.forward_model.forward_model")
def test_convergence_flip_default_fail_high_succeed(mock_forward_model, tmp_path):
    """A candidate that failed to converge at default fidelity (agent-visible
    objective_value is None) converged fine at high fidelity. This isn't a
    feasibility flip -- there's no agent-visible feasibility to compare
    against -- so it must be tracked separately, not silently dropped or
    crash a summary that assumes both objectives are always present
    together."""
    log_path = tmp_path / "run.ndjson"
    with TrajectoryLogger(log_path, {}) as logger:
        logger.log_step(
            {
                "step": 1,
                "source": "llm",
                "boundary": _synthetic_boundary_dict(),
                "objective_value": None,
                "feasible": None,
                "non_convergent": True,
            }
        )

    problem = problems.GeometricalProblem()
    mock_forward_model.return_value = (_FEASIBLE_METRICS, None)
    summary = rescore_trajectory_file(log_path, tmp_path / "rescored.ndjson", problem=problem)

    assert summary["convergence_flips_to_convergent"] == 1
    assert summary["convergence_flips_to_non_convergent"] == 0
    assert summary["flips_to_infeasible"] == 0
    assert summary["flips_to_feasible"] == 0


@patch("constellaration_methods.evaluation.rescore_trajectory.forward_model.forward_model")
def test_convergence_flip_default_succeed_high_fail(mock_forward_model, tmp_path):
    """Mirror case: converged at default fidelity, failed at high fidelity."""
    log_path = tmp_path / "run.ndjson"
    with TrajectoryLogger(log_path, {}) as logger:
        logger.log_step(
            {
                "step": 1,
                "source": "llm",
                "boundary": _synthetic_boundary_dict(),
                "objective_value": 5.0,
                "feasible": False,
                "non_convergent": False,
            }
        )

    problem = problems.GeometricalProblem()
    mock_forward_model.side_effect = RuntimeError("VMEC++ did not converge")
    summary = rescore_trajectory_file(log_path, tmp_path / "rescored.ndjson", problem=problem)

    assert summary["convergence_flips_to_convergent"] == 0
    assert summary["convergence_flips_to_non_convergent"] == 1
    assert summary["flips_to_infeasible"] == 0
    assert summary["flips_to_feasible"] == 0
