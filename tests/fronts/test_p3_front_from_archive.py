"""Pure-rule tests for scripts/p3_front_from_archive.py — E15 (vault
p3-reopen-three-experiments-2026-09-16, "confirm and export the front
already in E13's archive"). No VMEC for these; hypervolume() calls the
library's own primitive (`constellaration.problems._hypervolume`), so those
two tests need the pinned image (run inside the container, not on host)."""

import json

import pytest

from constellaration_methods.fronts.p3_front_from_archive import (
    best_v_of_rows,
    hypervolume,
    nondominated,
    retained,
    strict_nominal,
    tolerance_feasible_rows,
    v_curve_from_rows,
    worse_confirmation,
)


# --------------------------------------------------------------------------- #
# tolerance_feasible_rows
# --------------------------------------------------------------------------- #
def test_tolerance_feasible_rows_filters_and_dedupes():
    rows = [
        {"tag": "a", "x": [0.1, 0.2], "r": [0.001, -0.002, 0.003, 0.004, 0.005]},
        # duplicate x (same to 6 dp) as "a" -> dropped
        {"tag": "b", "x": [0.1000000001, 0.2], "r": [0.0, 0.0, 0.0, 0.0, 0.0]},
        # max residual 0.02 > tol -> dropped
        {"tag": "c", "x": [0.9, 0.9], "r": [0.02, 0.0, 0.0, 0.0, 0.0]},
        # non-finite -> dropped
        {"tag": "d", "x": [0.3, 0.3], "r": [float("nan"), 0.0, 0.0, 0.0, 0.0]},
        # missing r entirely (a non-convergent archive row) -> dropped
        {"tag": "e", "x": [0.4, 0.4]},
        # exactly at tol -> kept
        {"tag": "f", "x": [0.5, 0.5], "r": [0.01, 0.01, 0.01, 0.01, 0.01]},
    ]
    kept = tolerance_feasible_rows(rows, tol=0.01)
    assert [r["tag"] for r in kept] == ["a", "f"]


def test_tolerance_feasible_rows_default_tol_is_benchmark_predicate():
    rows = [{"tag": "z", "x": [1.0], "r": [0.0099, -0.5, -0.5, -0.5, -0.5]}]
    assert [r["tag"] for r in tolerance_feasible_rows(rows)] == ["z"]


# --------------------------------------------------------------------------- #
# nondominated
# --------------------------------------------------------------------------- #
def test_nondominated_keeps_the_tradeoff_frontier():
    # aspect_ratio: minimise: min_len: maximise
    a = {"tag": "A", "aspect_ratio": 10.53, "min_len": 10.44}
    b = {"tag": "B", "aspect_ratio": 10.61, "min_len": 10.56}
    c = {"tag": "C", "aspect_ratio": 10.64, "min_len": 10.76}
    # dominated by B (aspect higher, min_len lower) and by C likewise
    d = {"tag": "D", "aspect_ratio": 10.70, "min_len": 10.50}
    out = nondominated([a, b, c, d])
    assert {p["tag"] for p in out} == {"A", "B", "C"}


def test_nondominated_ties_both_survive():
    p = {"tag": "x", "aspect_ratio": 5.0, "min_len": 5.0}
    q = {"tag": "y", "aspect_ratio": 5.0, "min_len": 5.0}
    out = nondominated([p, q])
    assert {r["tag"] for r in out} == {"x", "y"}


def test_nondominated_single_point_is_its_own_front():
    p = {"tag": "only", "aspect_ratio": 8.0, "min_len": 3.0}
    assert nondominated([p]) == [p]


def test_nondominated_strict_in_one_no_worse_in_both():
    # e beats f in aspect only, equal min_len -> f dominated
    e = {"tag": "e", "aspect_ratio": 9.0, "min_len": 4.0}
    f = {"tag": "f", "aspect_ratio": 9.5, "min_len": 4.0}
    out = nondominated([e, f])
    assert {p["tag"] for p in out} == {"e"}


# --------------------------------------------------------------------------- #
# hypervolume (needs the pinned image: constellaration.problems._hypervolume)
# --------------------------------------------------------------------------- #
def test_hypervolume_single_point_hand_computable():
    pytest.importorskip(
        "constellaration",
        reason="needs the pinned constellaration package; run inside the repo's Docker image",
    )
    # HV of one feasible point vs reference (1.0, 20.0):
    # box = (ref_x - (-min_len)) * (ref_y - aspect) = (1 + min_len) * (20 - aspect)
    p = {"aspect_ratio": 10.5, "min_len": 11.24}
    expected = (1.0 + 11.24) * (20.0 - 10.5)
    assert hypervolume([p], ref=(1.0, 20.0)) == pytest.approx(expected, rel=1e-9)


def test_hypervolume_two_point_hand_computable():
    pytest.importorskip(
        "constellaration",
        reason="needs the pinned constellaration package; run inside the repo's Docker image",
    )
    # Two points, neither dominating the other. Each contributes a box
    # [-min_len, 1] x [aspect, 20]; the union's volume is box1 + box2 minus
    # their rectangular intersection (inclusion-exclusion) -- computable by
    # hand from the two boxes alone, no library needed to derive the
    # expected value (only to confirm it).
    p1 = {"aspect_ratio": 10.5, "min_len": 11.24}
    p2 = {"aspect_ratio": 9.0, "min_len": 10.0}
    ref = (1.0, 20.0)

    def box(p):
        return (ref[0] + p["min_len"]) * (ref[1] - p["aspect_ratio"])

    x_lo = max(-p1["min_len"], -p2["min_len"])
    x_hi = min(ref[0], ref[0])
    y_lo = max(p1["aspect_ratio"], p2["aspect_ratio"])
    y_hi = min(ref[1], ref[1])
    inter = max(0.0, x_hi - x_lo) * max(0.0, y_hi - y_lo)
    expected = box(p1) + box(p2) - inter

    assert hypervolume([p1, p2], ref=ref) == pytest.approx(expected, rel=1e-9)
    assert hypervolume([p1, p2], ref=ref) == pytest.approx(132.78, rel=1e-6)


# --------------------------------------------------------------------------- #
# retained
# --------------------------------------------------------------------------- #
ARCHIVED = [-0.0209, -0.0065, -0.0887, -0.0020, 0.00007]


def test_retained_true_when_both_confirmations_match():
    confs = [list(ARCHIVED), list(ARCHIVED)]
    assert retained(ARCHIVED, confs, tol=0.01, atol=1e-4) is True


def test_retained_false_with_only_one_confirmation():
    assert retained(ARCHIVED, [list(ARCHIVED)], tol=0.01, atol=1e-4) is False


def test_retained_false_with_three_confirmations():
    confs = [list(ARCHIVED)] * 3
    assert retained(ARCHIVED, confs, tol=0.01, atol=1e-4) is False


def test_retained_false_when_a_confirmation_drifts_past_atol():
    drifted = list(ARCHIVED)
    drifted[0] += 3e-4  # > atol 1e-4
    confs = [list(ARCHIVED), drifted]
    assert retained(ARCHIVED, confs, tol=0.01, atol=1e-4) is False


def test_retained_true_when_drift_is_within_atol():
    nudged = list(ARCHIVED)
    nudged[0] += 5e-5  # < atol 1e-4
    confs = [list(ARCHIVED), nudged]
    assert retained(ARCHIVED, confs, tol=0.01, atol=1e-4) is True


def test_retained_false_when_a_confirmation_exceeds_tol():
    over_tol = [0.02, -0.5, -0.5, -0.5, -0.5]
    confs = [list(ARCHIVED), over_tol]
    assert retained(ARCHIVED, confs, tol=0.01, atol=1e-4) is False


def test_retained_false_on_incomplete_or_nonfinite_confirmation():
    confs_short = [list(ARCHIVED), list(ARCHIVED)[:4]]
    assert retained(ARCHIVED, confs_short, tol=0.01, atol=1e-4) is False
    nan_conf = list(ARCHIVED)
    nan_conf[2] = float("nan")
    confs_nan = [list(ARCHIVED), nan_conf]
    assert retained(ARCHIVED, confs_nan, tol=0.01, atol=1e-4) is False


def test_retained_false_on_missing_confirmation():
    confs = [list(ARCHIVED), None]
    assert retained(ARCHIVED, confs, tol=0.01, atol=1e-4) is False


# --------------------------------------------------------------------------- #
# strict_nominal
# --------------------------------------------------------------------------- #
def test_strict_nominal_all_five_strictly_negative():
    assert strict_nominal([-0.02, -0.01, -0.05, -0.002, -0.0003]) is True


def test_strict_nominal_false_when_one_residual_nonnegative():
    # P:T3-like: raw well is a hair positive -> not strictly nominal
    assert strict_nominal([-0.02, -0.01, -0.05, -0.002, 0.0000254]) is False


def test_strict_nominal_false_on_exact_zero():
    assert strict_nominal([-0.02, -0.01, -0.05, -0.002, 0.0]) is False


def test_strict_nominal_ignores_nan_by_failing_closed():
    assert strict_nominal([-0.02, -0.01, float("nan"), -0.002, -0.0003]) is False


# --- confirmation cap as a CLI parameter (E17 adjudication, 2026-09-19) -------
# E15 locked a 30-evaluation cap in the module constant; E17's registration
# specifies two confirmations per member without that cap, so the cap becomes
# a flag whose DEFAULT preserves E15 behaviour byte-for-byte.


def test_confirmation_cap_default_is_the_e15_constant():
    from constellaration_methods.fronts.p3_front_from_archive import (
        MAX_CANONICAL_EVALS,
        build_parser,
    )

    args = build_parser().parse_args(["results/x", "--out-dir", "o"])
    assert args.max_confirmations == MAX_CANONICAL_EVALS == 30


def test_confirmation_cap_flag_overrides_default():
    from constellaration_methods.fronts.p3_front_from_archive import build_parser

    args = build_parser().parse_args(["results/x", "--out-dir", "o", "--max-confirmations", "50"])
    assert args.max_confirmations == 50


def test_check_confirmation_cap_raises_only_when_exceeded():
    from constellaration_methods.fronts.p3_front_from_archive import check_confirmation_cap

    check_confirmation_cap(n_members=15, cap=30)
    with pytest.raises(RuntimeError, match="32 confirmations for 16 members"):
        check_confirmation_cap(n_members=16, cap=30)
    check_confirmation_cap(n_members=25, cap=50)


# --------------------------------------------------------------------------- #
# E18 item 3 -- best_v_of_rows / v_curve_from_rows (V-vs-evaluations curve;
# no VMEC, pure over already-archived residuals)
# --------------------------------------------------------------------------- #
def test_best_v_of_rows_is_the_min_max_residual_over_convergent_rows():
    rows = [
        {"tag": "a", "r": [0.05, -0.1, -0.1, -0.1, -0.1]},  # V=0.05
        {"tag": "b", "r": [0.02, 0.01, -0.1, -0.1, -0.1]},  # V=0.02 (best)
        {"tag": "c", "non_convergent": True},  # no r -> ignored
        {"tag": "d", "r": [float("nan"), 0.0, 0.0, 0.0, 0.0]},  # non-finite -> ignored
    ]
    assert best_v_of_rows(rows) == pytest.approx(0.02)


def test_best_v_of_rows_is_none_when_nothing_convergent():
    rows = [{"tag": "a", "non_convergent": True}, {"tag": "b"}]
    assert best_v_of_rows(rows) is None


def test_v_curve_from_rows_hand_computed():
    # V per row: 0.10, 0.08, 0.12, 0.03, 0.05, 0.20 -> running best:
    # 0.10, 0.08, 0.08, 0.03, 0.03, 0.03
    vs = [0.10, 0.08, 0.12, 0.03, 0.05, 0.20]
    rows = [{"tag": f"r{i}", "r": [v, -1, -1, -1, -1]} for i, v in enumerate(vs)]
    curve = v_curve_from_rows(rows, every=2)
    assert curve == [
        [2, pytest.approx(0.08)],
        [4, pytest.approx(0.03)],
        [6, pytest.approx(0.03)],
    ]


def test_v_curve_from_rows_drops_trailing_partial_block():
    rows = [{"tag": f"r{i}", "r": [0.1, -1, -1, -1, -1]} for i in range(5)]
    curve = v_curve_from_rows(rows, every=2)
    assert [e for e, _v in curve] == [2, 4]  # the 5th row starts an unfinished block


def test_v_curve_from_rows_none_before_any_convergent_row():
    rows = [{"tag": "a", "non_convergent": True}, {"tag": "b", "non_convergent": True}]
    curve = v_curve_from_rows(rows, every=2)
    assert curve == [[2, None]]


# --------------------------------------------------------------------------- #
# E18 item 1 -- --constraint-override on p3_front_from_archive.py (parses via
# p3_feasibility_restoration.parse_constraint_overrides, no VMEC needed here)
# --------------------------------------------------------------------------- #
def test_build_parser_accepts_repeated_constraint_override_flags():
    from constellaration_methods.fronts.p3_front_from_archive import build_parser

    args = build_parser().parse_args(
        [
            "results/x",
            "--out-dir",
            "o",
            "--constraint-override",
            "mirror_upper=0.23",
            "--constraint-override",
            "well_lower=0.01",
        ]
    )
    assert args.constraint_override == ["mirror_upper=0.23", "well_lower=0.01"]


def test_build_parser_constraint_override_defaults_to_none():
    from constellaration_methods.fronts.p3_front_from_archive import build_parser

    args = build_parser().parse_args(["results/x", "--out-dir", "o"])
    assert args.constraint_override is None


def test_run_records_empty_constraint_overrides_in_report_without_confirm(tmp_path):
    from constellaration_methods.fronts.p3_front_from_archive import build_parser, run

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "archive.jsonl").write_text(
        json.dumps(
            {
                "tag": "t0",
                "x": [0.0],
                "r": [-0.05, -0.05, -0.05, -0.05, -0.05],
                "raw": {"aspect_ratio": 9.0, "min_len": 9.0},
            }
        )
        + "\n"
    )
    (archive_dir / "start_serialised.json").write_text(
        json.dumps({"boundary_dict": {}, "mpol": 1, "ntor": 1, "nfp": 1})
    )
    out_dir = tmp_path / "out"
    args = build_parser().parse_args([str(archive_dir), "--out-dir", str(out_dir)])
    report = run(args)
    assert report["constraint_overrides"] == {}


def test_run_records_given_constraint_overrides_in_report_without_confirm(tmp_path):
    from constellaration_methods.fronts.p3_front_from_archive import build_parser, run

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "archive.jsonl").write_text(
        json.dumps(
            {
                "tag": "t0",
                "x": [0.0],
                "r": [-0.05, -0.05, -0.05, -0.05, -0.05],
                "raw": {"aspect_ratio": 9.0, "min_len": 9.0},
            }
        )
        + "\n"
    )
    (archive_dir / "start_serialised.json").write_text(
        json.dumps({"boundary_dict": {}, "mpol": 1, "ntor": 1, "nfp": 1})
    )
    out_dir = tmp_path / "out"
    args = build_parser().parse_args(
        [
            str(archive_dir),
            "--out-dir",
            str(out_dir),
            "--constraint-override",
            "mirror_upper=0.23",
        ]
    )
    report = run(args)
    assert report["constraint_overrides"] == {"mirror_upper": 0.23}


def test_run_rejects_unknown_constraint_override_name(tmp_path):
    from constellaration_methods.fronts.p3_front_from_archive import build_parser, run

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "archive.jsonl").write_text("")
    (archive_dir / "start_serialised.json").write_text(
        json.dumps({"boundary_dict": {}, "mpol": 1, "ntor": 1, "nfp": 1})
    )
    args = build_parser().parse_args(
        [
            str(archive_dir),
            "--out-dir",
            str(tmp_path / "out"),
            "--constraint-override",
            "not_a_real_bound=1.0",
        ]
    )
    with pytest.raises(ValueError, match="unknown constraint override"):
        run(args)


# --------------------------------------------------------------------------- #
# confirmed-field convention: worst-case, one evaluation for every field
# --------------------------------------------------------------------------- #
def test_worse_confirmation_picks_the_larger_max_residual_with_its_raw():
    c_mild = [-0.05, -0.05, -0.05, -0.05, 0.001]
    c_worse = [-0.05, -0.05, -0.05, -0.05, 0.002]
    raw_mild = {"aspect_ratio": 9.0, "min_len": 9.0}
    raw_worse = {"aspect_ratio": 9.1, "min_len": 8.9}
    assert worse_confirmation([c_mild, c_worse], [raw_mild, raw_worse]) == (
        c_worse,
        raw_worse,
    )
    assert worse_confirmation([c_worse, c_mild], [raw_worse, raw_mild]) == (
        c_worse,
        raw_worse,
    )


def test_worse_confirmation_tie_keeps_the_first():
    c = [-0.05, -0.05, -0.05, -0.05, 0.001]
    raw1 = {"aspect_ratio": 9.0, "min_len": 9.0}
    raw2 = {"aspect_ratio": 9.1, "min_len": 8.9}
    assert worse_confirmation([c, list(c)], [raw1, raw2]) == (c, raw1)


def _write_single_member_archive(archive_dir, r, raw):
    archive_dir.mkdir()
    (archive_dir / "archive.jsonl").write_text(
        json.dumps({"tag": "t0", "x": [0.0], "r": r, "raw": raw}) + "\n"
    )
    (archive_dir / "start_serialised.json").write_text(
        json.dumps({"boundary_dict": {}, "mpol": 1, "ntor": 1, "nfp": 1})
    )


@pytest.mark.parametrize("worse_slot", [0, 1])
def test_run_confirmed_fields_all_come_from_the_worse_confirmation(
    tmp_path, monkeypatch, worse_slot
):
    """Regression (review of E15/E16/E17 exports): confirmed_V and
    strict_nominal used the worse confirmation while confirmed_aspect_ratio
    and confirmed_min_len averaged the two, so the hypervolume was computed
    from metrics no single evaluation produced. The two confirmations here
    differ in aspect/min_len by 3e-4 (>> 1e-5), so a mean is observable."""
    from constellaration_methods.fronts.p3_front_from_archive import build_parser, run
    from constellaration_methods.restoration import p3_final_restoration, p3_promote_candidate

    archived_r = [-0.05, -0.05, -0.05, -0.05, 0.0010]
    archived_raw = {"aspect_ratio": 9.0, "min_len": 9.0}
    _write_single_member_archive(tmp_path / "archive", archived_r, archived_raw)

    mild = ([-0.05, -0.05, -0.05, -0.05, 0.00102], {"aspect_ratio": 9.0, "min_len": 9.0})
    worse = (
        [-0.05, -0.05, -0.05, -0.05, 0.00108],
        {"aspect_ratio": 9.0003, "min_len": 8.9997},
    )
    pair = [mild, worse] if worse_slot == 1 else [worse, mild]

    def fake_fresh_evaluate(rep, xs, tags, workers, log_path, budget):
        assert len(xs) == 2 and len(tags) == 2
        return [c for c, _ in pair], [{"raw": raw} for _, raw in pair]

    monkeypatch.setattr(p3_final_restoration, "_fresh_evaluate", fake_fresh_evaluate)
    monkeypatch.setattr(p3_promote_candidate, "boundary_dict_of_candidate", lambda c: {})
    monkeypatch.setattr(p3_promote_candidate, "reflect_toroidally", lambda bd: {})

    args = build_parser().parse_args(
        [str(tmp_path / "archive"), "--out-dir", str(tmp_path / "out"), "--confirm"]
    )
    report = run(args)

    (m,) = report["members"]
    assert m["retained"] is True
    assert m["confirmed_r"] == worse[0]
    assert m["confirmed_V"] == 0.00108
    assert m["strict_nominal"] is False
    # the observable part: no averaging, the worse evaluation's own metrics
    assert m["confirmed_aspect_ratio"] == 9.0003
    assert m["confirmed_min_len"] == 8.9997
    expected_hv = hypervolume([{"aspect_ratio": 9.0003, "min_len": 8.9997}], ref=(1.0, 20.0))
    assert m["solo_hypervolume"] == expected_hv
    assert report["confirmed_hypervolume"] == expected_hv
