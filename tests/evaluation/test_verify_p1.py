"""Tests for constellaration_methods.evaluation.verify_p1 -- the single
accurate P1 packaging script (replaces the source campaign's
verify_p1_final.py / verify_p1_overnight.py / package_own_geometrical.py).

Pure-logic tests driven by a synthetic candidates.ndjson and a fake
evaluate_fn -- no VMEC, no real SurfaceRZFourier construction.
"""

from __future__ import annotations

import json

from constellaration_methods.evaluation.verify_p1 import (
    DEFAULT_BOARD_TOP,
    boundary_dict_from_record,
    build_arg_parser,
    build_provenance,
    find_best_feasible,
    read_records,
    verify_saved_file,
)


def _synthetic_boundary(seed: int = 0) -> dict:
    """A tiny, deterministic, never-leaderboard-derived boundary payload."""
    return {
        "r_cos": [[1.0 + 0.01 * seed, 0.0], [0.05, 0.0]],
        "z_sin": [[0.0, 0.0], [0.05, 0.0]],
    }


def _write_ndjson(path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_read_records_parses_one_json_object_per_line(tmp_path):
    path = tmp_path / "candidates.ndjson"
    _write_ndjson(path, [{"a": 1}, {"a": 2}])

    records = read_records(path)

    assert records == [{"a": 1}, {"a": 2}]


def test_read_records_skips_blank_lines(tmp_path):
    path = tmp_path / "candidates.ndjson"
    path.write_text('{"a": 1}\n\n{"a": 2}\n')

    assert read_records(path) == [{"a": 1}, {"a": 2}]


def test_boundary_dict_from_record_defaults_n_field_periods_to_three():
    record = {"boundary": _synthetic_boundary()}

    boundary_dict = boundary_dict_from_record(record)

    assert boundary_dict["n_field_periods"] == 3
    assert boundary_dict["is_stellarator_symmetric"] is True
    assert boundary_dict["r_sin"] is None
    assert boundary_dict["z_cos"] is None


def test_boundary_dict_from_record_honors_explicit_n_field_periods():
    record = {"boundary": _synthetic_boundary(), "n_field_periods": 5}

    boundary_dict = boundary_dict_from_record(record)

    assert boundary_dict["n_field_periods"] == 5


def test_boundary_dict_from_record_supports_alternate_field_name():
    record = {"optimized_boundary": _synthetic_boundary(), "n_field_periods": 3}

    boundary_dict = boundary_dict_from_record(record, boundary_field="optimized_boundary")

    assert boundary_dict["r_cos"] == _synthetic_boundary()["r_cos"]


def test_find_best_feasible_walks_objective_ascending_and_stops_at_first_hit():
    records = [
        {"objective": 3.0, "boundary": _synthetic_boundary(3)},
        {"objective": 1.0, "boundary": _synthetic_boundary(1)},  # best objective
        {"objective": 2.0, "boundary": _synthetic_boundary(2)},
    ]
    calls = []

    def fake_evaluate(boundary_dict):
        calls.append(boundary_dict)
        # only the best-objective (seed=1) candidate is feasible
        if boundary_dict["r_cos"][0][0] == 1.01:
            return (0.95, 1.2, 0.0)
        return (0.0, 5.0, 0.5)

    found = find_best_feasible(records, fake_evaluate)

    assert found is not None
    rank, boundary_dict, evaluation = found
    assert rank == 0  # first candidate tried after sorting by objective ascending
    assert evaluation["score"] == 0.95
    assert len(calls) == 1  # stopped at the first feasible hit


def test_find_best_feasible_skips_records_without_objective_or_boundary():
    records = [
        {"objective": None, "boundary": _synthetic_boundary()},
        {"objective": 1.0},  # no boundary field
        {"objective": 2.0, "boundary": _synthetic_boundary()},
    ]

    found = find_best_feasible(records, lambda b: (0.9, 1.0, 0.0))

    assert found is not None
    rank, _boundary_dict, _evaluation = found
    assert rank == 0  # the only valid record, at whatever its sorted position is


def test_find_best_feasible_returns_none_when_nothing_feasible():
    records = [{"objective": 1.0, "boundary": _synthetic_boundary()}]

    found = find_best_feasible(records, lambda b: (0.0, 5.0, 0.5))

    assert found is None


def test_find_best_feasible_respects_max_candidates():
    records = [{"objective": float(i), "boundary": _synthetic_boundary(i)} for i in range(20)]
    calls = []

    def fake_evaluate(boundary_dict):
        calls.append(boundary_dict)
        return (0.0, 5.0, 0.5)  # never feasible

    find_best_feasible(records, fake_evaluate, max_candidates=5)

    assert len(calls) == 5


def test_verify_saved_file_reloads_from_disk_not_memory(tmp_path):
    path = tmp_path / "best_own_geometrical.json"
    path.write_text(json.dumps(_synthetic_boundary(7)))

    seen = {}

    def fake_evaluate(boundary_dict):
        seen["boundary"] = boundary_dict
        return (0.97, 1.1, 0.0)

    score, objective, feasibility = verify_saved_file(path, fake_evaluate)

    assert score == 0.97
    assert objective == 1.1
    assert feasibility == 0.0
    assert seen["boundary"] == _synthetic_boundary(7)


def test_build_provenance_matches_the_flat_schema_of_the_real_file():
    provenance = build_provenance(
        score=0.9752223935436287,
        objective=1.2229984581073414,
        feasibility=0.0,
        lineage="own -- rotating-ellipse analytic seed, random_seed 20260907/20260908, "
        "NO leader boundary, 4h+8h+10h ALM/NGOpt descent, resumed via checkpoint",
        mode_budget="8x8 NFP=3",
    )

    # Exactly the keys docs/results/submission/PROVENANCE_own_geometrical.json has --
    # no nested "source"/"verified_on_saved_file"/"run_reported_canonical" blocks.
    assert set(provenance) == {
        "problem",
        "score",
        "objective",
        "feasibility",
        "board_top_at_submission",
        "beats_board",
        "lineage",
        "mode_budget",
    }
    assert provenance["problem"] == "GeometricalProblem"
    assert provenance["board_top_at_submission"] == DEFAULT_BOARD_TOP
    assert provenance["beats_board"] is True


def test_build_provenance_beats_board_false_when_score_below_board_top():
    provenance = build_provenance(
        score=0.5, objective=5.0, feasibility=0.0, lineage="own", board_top=0.97
    )

    assert provenance["beats_board"] is False


def test_build_arg_parser_requires_records_and_lineage():
    args = build_arg_parser().parse_args(
        ["--records", "results/p1-own-lineage/result.ndjson", "--lineage", "own -- test"]
    )

    assert args.records == "results/p1-own-lineage/result.ndjson"
    assert args.boundary_field == "boundary"
    assert args.board_top == DEFAULT_BOARD_TOP
