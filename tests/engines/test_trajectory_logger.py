"""Build step 4: NDJSON trajectory logger. Pure logic, no physics stack."""

from constellaration_methods.engines.trajectory_logger import (
    TrajectoryLogger,
    read_trajectory,
)


def test_first_line_is_run_metadata(tmp_path):
    log_path = tmp_path / "run.ndjson"
    with TrajectoryLogger(log_path, {"model": "test-model", "buffer_k": 5}):
        pass

    lines = read_trajectory(log_path)
    assert lines[0]["type"] == "run_metadata"
    assert lines[0]["model"] == "test-model"
    assert lines[0]["buffer_k"] == 5


def test_step_lines_default_canonical_fields_to_none(tmp_path):
    log_path = tmp_path / "run.ndjson"
    with TrajectoryLogger(log_path, {}) as logger:
        logger.log_step({"step": 1, "objective_value": 3.0})

    lines = read_trajectory(log_path)
    step_line = lines[1]
    assert step_line["type"] == "step"
    assert step_line["objective_value"] == 3.0
    assert step_line["canonical_objective_value"] is None
    assert step_line["canonical_feasible"] is None
    assert step_line["canonical_feasibility_margin"] is None
    assert step_line["canonical_non_convergent"] is None


def test_step_lines_preserve_provided_values_over_none_default(tmp_path):
    log_path = tmp_path / "run.ndjson"
    with TrajectoryLogger(log_path, {}) as logger:
        logger.log_step({"step": 1, "canonical_objective_value": 2.5})

    lines = read_trajectory(log_path)
    assert lines[1]["canonical_objective_value"] == 2.5


def test_multiple_steps_are_appended_in_order(tmp_path):
    log_path = tmp_path / "run.ndjson"
    with TrajectoryLogger(log_path, {}) as logger:
        logger.log_step({"step": 1})
        logger.log_step({"step": 2})
        logger.log_step({"step": 3})

    lines = read_trajectory(log_path)
    steps = [line["step"] for line in lines if line["type"] == "step"]
    assert steps == [1, 2, 3]


def test_log_is_flushed_incrementally(tmp_path):
    """Each log_step call flushes to disk immediately, so a crash mid-run
    doesn't lose already-completed steps."""
    log_path = tmp_path / "run.ndjson"
    logger = TrajectoryLogger(log_path, {})
    logger.log_step({"step": 1})

    lines_before_close = read_trajectory(log_path)
    logger.close()

    assert len(lines_before_close) == 2  # run_metadata + step 1
