"""Re-score every logged candidate at `default_high_fidelity` and write
canonical scores back into the trajectory.

This is the mandatory validation pass: agent-visible scores (what an
optimizer saw, at cheap default fidelity) and canonical scores (what the
physics actually says, at high fidelity) are kept in separate fields by
`engines.trajectory_logger`'s data model -- this module is what fills the
canonical_* fields in, and reports feasibility flips so it's known (not
assumed) whether a default-fidelity debugging shortcut changed the
trajectory's actual content, not just its scores.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import numpy as np
from constellaration import forward_model, problems
from constellaration.geometry import surface_rz_fourier

from constellaration_methods.engines.trajectory_logger import TrajectoryLogger, read_trajectory

# P1's fixed search-space pin (field-period count and symmetry), inlined here
# because this module only needs the boundary-construction primitive, not
# the rest of the (out-of-scope) LLM task-prompt template it originated in.
_N_FIELD_PERIODS = 3
_IS_STELLARATOR_SYMMETRIC = True


def _boundary_from_dict(
    boundary_dict: dict[str, Any],
    max_poloidal_mode: int,
    max_toroidal_mode: int,
) -> surface_rz_fourier.SurfaceRZFourier:
    """Parses a logged `boundary` dict into a validated SurfaceRZFourier,
    enforcing the given mode truncation. Raises pydantic.ValidationError
    (shape/type) or ValueError (wrong shape for the given truncation) on
    malformed input -- callers should treat both as a non-convergent step,
    not a crash.

    Every trajectory this repo re-scores was logged at the fixed P1 pin
    (n_field_periods=3, stellarator-symmetric) -- see
    `docs/methods.md` -- so those are not read from `boundary_dict`.
    """
    r_cos = np.asarray(boundary_dict["r_cos"], dtype=float)
    z_sin = np.asarray(boundary_dict["z_sin"], dtype=float)

    expected_shape = (max_poloidal_mode + 1, 2 * max_toroidal_mode + 1)
    if r_cos.shape != expected_shape or z_sin.shape != expected_shape:
        raise ValueError(
            f"Expected r_cos/z_sin shape {expected_shape} for "
            f"max_poloidal_mode={max_poloidal_mode}/max_toroidal_mode="
            f"{max_toroidal_mode}, got r_cos={r_cos.shape} z_sin={z_sin.shape}"
        )

    return surface_rz_fourier.SurfaceRZFourier(
        r_cos=r_cos,
        z_sin=z_sin,
        n_field_periods=_N_FIELD_PERIODS,
        is_stellarator_symmetric=_IS_STELLARATOR_SYMMETRIC,
    )


def rescore_step(
    step_record: dict[str, Any],
    problem: problems.GeometricalProblem,
    forward_model_settings: forward_model.ConstellarationSettings,
) -> dict[str, Any]:
    """Returns a copy of `step_record` with canonical_* fields filled in.
    Leaves them at their None default if there's no boundary to re-score
    (e.g. a step whose candidate failed to parse in the first place)."""
    record = dict(step_record)
    boundary_dict = record.get("boundary")
    if boundary_dict is None:
        return record

    try:
        # The record's own array shape is the mode-budget authority -- an
        # 8x8 run's candidate records are (9, 17), not the pinned 4x4 shape.
        r_cos = np.asarray(boundary_dict["r_cos"], dtype=float)
        boundary = _boundary_from_dict(
            boundary_dict,
            max_poloidal_mode=r_cos.shape[0] - 1,
            max_toroidal_mode=(r_cos.shape[1] - 1) // 2,
        )
    except (ValueError, IndexError):
        # Shouldn't happen for a boundary that was already accepted once,
        # but don't let a re-score pass crash a whole trajectory over it.
        return record

    try:
        metrics, _ = forward_model.forward_model(boundary, settings=forward_model_settings)
    except Exception:
        record["canonical_non_convergent"] = True
        record["canonical_objective_value"] = None
        record["canonical_feasible"] = None
        record["canonical_feasibility_margin"] = None
        # Keep the actual exception text, not just a boolean, so a flip to
        # non-convergent is diagnosable without re-running the solve.
        record["canonical_error"] = traceback.format_exc()
        return record

    objective_value, _minimize = problem.get_objective(metrics)
    record["canonical_non_convergent"] = False
    record["canonical_objective_value"] = objective_value
    record["canonical_feasible"] = problem.is_feasible(metrics)
    record["canonical_feasibility_margin"] = problem.compute_feasibility(metrics)
    record["canonical_error"] = None
    return record


def rescore_trajectory_file(
    log_path: Path,
    out_path: Path,
    problem: problems.GeometricalProblem | None = None,
) -> dict[str, Any]:
    """Re-scores every step in `log_path` at `default_high_fidelity` and
    writes the result to `out_path`. Returns a summary dict: counts of
    feasibility flips, objective agreement, and canonical non-convergence.
    """
    problem = problem or problems.GeometricalProblem()
    high_fidelity_settings = forward_model.ConstellarationSettings.default_high_fidelity()

    lines = read_trajectory(log_path)
    run_metadata = lines[0]
    step_lines = [line for line in lines if line["type"] == "step"]

    rescored_metadata = dict(run_metadata)
    rescored_metadata["rescored_at"] = "default_high_fidelity"
    rescored_metadata.pop("type", None)

    flips_to_infeasible = 0
    flips_to_feasible = 0
    n_rescored = 0
    n_canonical_non_convergent = 0
    # Convergence can itself flip with fidelity -- a candidate right at the
    # edge of VMEC's convergence basin can fail at one fidelity and succeed
    # at the other, in either direction. Distinct from feasibility flips
    # (which only apply to candidates that converged at both fidelities);
    # tracked separately so a caller comparing "best objective" doesn't
    # silently trip over a step with an objective at one fidelity and None
    # at the other.
    convergence_flips_to_convergent = 0
    convergence_flips_to_non_convergent = 0

    with TrajectoryLogger(out_path, rescored_metadata) as logger:
        for step_line in step_lines:
            rescored = rescore_step(step_line, problem, high_fidelity_settings)
            logger.log_step(rescored)

            if rescored.get("boundary") is None:
                continue
            n_rescored += 1

            agent_non_convergent = bool(rescored.get("non_convergent"))
            canonical_non_convergent = bool(rescored.get("canonical_non_convergent"))
            if canonical_non_convergent:
                n_canonical_non_convergent += 1
            if agent_non_convergent and not canonical_non_convergent:
                convergence_flips_to_convergent += 1
            elif not agent_non_convergent and canonical_non_convergent:
                convergence_flips_to_non_convergent += 1

            if agent_non_convergent or canonical_non_convergent:
                continue  # feasibility isn't comparable without both objectives

            agent_feasible = rescored.get("feasible")
            canonical_feasible = rescored.get("canonical_feasible")
            if agent_feasible is True and canonical_feasible is False:
                flips_to_infeasible += 1
            elif agent_feasible is False and canonical_feasible is True:
                flips_to_feasible += 1

    return {
        "n_steps_rescored": n_rescored,
        "convergence_flips_to_convergent": convergence_flips_to_convergent,
        "convergence_flips_to_non_convergent": convergence_flips_to_non_convergent,
        "n_canonical_non_convergent": n_canonical_non_convergent,
        "flips_to_infeasible": flips_to_infeasible,
        "flips_to_feasible": flips_to_feasible,
        "out_path": str(out_path),
    }
