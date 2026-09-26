"""Tests for constellaration_methods.evaluation.score_mhd_stable.

This module has no VMEC/constellaration-independent test in the source
campaign (the missing test the curated-repo plan calls out). It covers only
the pure-logic parts that don't need the physics stack: CLI argument
parsing, and report assembly (`evaluate_and_record`) driven by a fake
evaluator standing in for `problems.MHDStableQIStellarator`.
"""

from __future__ import annotations

from constellaration_methods.evaluation.score_mhd_stable import (
    build_arg_parser,
    evaluate_and_record,
)


class _FakeEvaluation:
    """Stands in for constellaration.problems.ProblemEvaluation: exposes the
    three fields evaluate_and_record reads (score, objectives, feasibility)."""

    def __init__(self, score, length_obj, aspect_obj, feasibility):
        self.score = score
        self.objectives = [(length_obj, aspect_obj)]
        self.feasibility = [feasibility]


class _FakeProblem:
    """Stands in for problems.MHDStableQIStellarator: evaluate([boundary])
    returns a canned _FakeEvaluation for a batch of one, exactly the call
    shape score_mhd_stable.py relies on."""

    def __init__(self, evaluation: _FakeEvaluation):
        self._evaluation = evaluation
        self.calls: list[list] = []

    def evaluate(self, boundaries):
        self.calls.append(boundaries)
        return self._evaluation


def test_build_arg_parser_defaults_have_no_other_call_indices():
    args = build_arg_parser().parse_args([])
    assert args.boundary is None
    assert args.other_call_index == []


def test_build_arg_parser_parses_boundary_files_and_call_indices():
    args = build_arg_parser().parse_args(
        [
            "--boundary",
            "a.json",
            "b.json",
            "--other-call-index",
            "68164",
            "69761",
        ]
    )
    assert args.boundary == ["a.json", "b.json"]
    assert args.other_call_index == [68164, 69761]


def test_evaluate_and_record_feasible_design_records_score_and_feasible_true():
    evaluation = _FakeEvaluation(
        score=0.5818, length_obj=[11.636], aspect_obj=[10.2], feasibility=0.0072
    )
    problem = _FakeProblem(evaluation)
    results: list[dict] = []

    rec = evaluate_and_record(
        problem, tol=0.01, results=results, label="submitted", boundary=object(), extra={"id": "x"}
    )

    assert rec["score"] == 0.5818
    assert rec["min_normalized_magnetic_gradient_scale_length"] == 11.636
    assert rec["aspect_ratio"] == 10.2
    assert rec["feasibility_violation"] == 0.0072
    assert rec["feasible"] is True
    assert results == [rec]
    assert len(problem.calls) == 1
    assert len(problem.calls[0]) == 1  # batch of one, per the module's own convention


def test_evaluate_and_record_violation_above_tolerance_is_infeasible():
    evaluation = _FakeEvaluation(score=0.0, length_obj=[5.0], aspect_obj=[9.0], feasibility=0.05)
    problem = _FakeProblem(evaluation)
    results: list[dict] = []

    rec = evaluate_and_record(
        problem, tol=0.01, results=results, label="other", boundary=object(), extra={"id": "y"}
    )

    assert rec["feasible"] is False


def test_evaluate_and_record_exception_is_captured_not_raised():
    class _RaisingProblem:
        def evaluate(self, boundaries):
            raise RuntimeError("VMEC non-convergent")

    results: list[dict] = []
    rec = evaluate_and_record(
        _RaisingProblem(),
        tol=0.01,
        results=results,
        label="other",
        boundary=object(),
        extra={"id": "z"},
    )

    assert "error" in rec
    assert "RuntimeError" in rec["error"]
    assert results == [rec]


def test_evaluate_and_record_preserves_extra_fields():
    evaluation = _FakeEvaluation(score=1.0, length_obj=[1.0], aspect_obj=[1.0], feasibility=0.0)
    problem = _FakeProblem(evaluation)
    results: list[dict] = []

    rec = evaluate_and_record(
        problem,
        tol=0.01,
        results=results,
        label="submitted",
        boundary=object(),
        extra={"id": "best_feasible_simple_to_build", "call_index": 30548},
    )

    assert rec["id"] == "best_feasible_simple_to_build"
    assert rec["call_index"] == 30548
    assert rec["cohort"] == "submitted"
