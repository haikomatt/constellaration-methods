"""Tests for scripts/b3_nsga2_mhd.py -- E17 (seeding an NSGA-II population
from an existing confirmed front rather than from scratch). Pure-rule tests
need no VMEC/pymoo; the toy end-to-end run additionally needs `pymoo`
(the ask/tell loop) and `constellaration` (hypervolume/nondominated, via
p3_front_from_archive) and so needs the pinned image, same as
test_p3_front_builder.py's own toy end-to-end test.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from constellaration_methods.fronts.b3_nsga2_mhd import (
    INIT_SCALE_DEFAULT,
    POP_SIZE_SEEDED_DEFAULT,
    SEEDED_MAX_EVALS_DEFAULT,
    SEEDED_TOTAL_HOURS_DEFAULT,
    apply_smoke_ceilings_seeded,
    build_arg_parser,
    init_scale_box,
    load_seed_candidate,
    objectives_and_constraints_from_raws,
    old_mask_from_anchor_xs,
    polynomial_mutate,
    resolve_eval_fidelity,
    resolve_out_dir_seeded,
    run_seeded_from_anchors,
    seed_population_from_anchors,
    seeded_stop_reason,
    validate_seeded_args,
)
from constellaration_methods.restoration.p3_promote_candidate import (
    coefficient_space_multiplier,
)


# --------------------------------------------------------------------------- #
# polynomial_mutate
# --------------------------------------------------------------------------- #
def test_polynomial_mutate_stays_within_bounds():
    rng = np.random.RandomState(0)
    x = np.array([0.0, 0.0, 0.0])
    lo = x - 0.02
    hi = x + 0.02
    for _ in range(200):
        out = polynomial_mutate(x, lo, hi, rng)
        assert np.all(out >= lo - 1e-12)
        assert np.all(out <= hi + 1e-12)


def test_polynomial_mutate_is_deterministic_given_a_seeded_rng():
    x = np.array([0.1, -0.2])
    lo, hi = x - 0.05, x + 0.05
    out1 = polynomial_mutate(x, lo, hi, np.random.RandomState(7))
    out2 = polynomial_mutate(x, lo, hi, np.random.RandomState(7))
    assert np.allclose(out1, out2)


def test_polynomial_mutate_can_move_away_from_the_parent():
    # over many draws, at least some samples must differ meaningfully from x
    # (not a no-op operator)
    rng = np.random.RandomState(1)
    x = np.array([0.0])
    lo, hi = x - 0.02, x + 0.02
    draws = [polynomial_mutate(x, lo, hi, rng)[0] for _ in range(50)]
    assert max(abs(d) for d in draws) > 0.001


# --------------------------------------------------------------------------- #
# old_mask_from_anchor_xs
# --------------------------------------------------------------------------- #
def test_old_mask_from_anchor_xs_flags_columns_any_anchor_occupies():
    anchor_xs = np.array(
        [
            [0.1, 0.0, 0.0],
            [0.0, 0.0, 0.2],
        ]
    )
    mask = old_mask_from_anchor_xs(anchor_xs)
    assert mask.tolist() == [True, False, True]


def test_old_mask_from_anchor_xs_all_false_when_every_anchor_is_zero():
    anchor_xs = np.zeros((1, 3))
    mask = old_mask_from_anchor_xs(anchor_xs)
    assert mask.tolist() == [False, False, False]


# --------------------------------------------------------------------------- #
# init_scale_box
# --------------------------------------------------------------------------- #
def test_init_scale_box_is_isotropic_when_every_mode_is_old():
    scale = np.array([1.0, 2.0, 5.0])
    old_mask = np.array([True, True, True])
    box = init_scale_box(scale, old_mask, 0.02)
    assert np.allclose(box, 0.02)


def test_init_scale_box_widens_new_modes_via_coefficient_space_multiplier():
    scale = np.array([1.0, 1.0, 0.01])
    old_mask = np.array([True, True, False])
    box = init_scale_box(scale, old_mask, 0.02)
    expected = 0.02 * coefficient_space_multiplier(scale, old_mask)
    assert np.allclose(box, expected)
    # the new (small-scale) mode must get a WIDER box than the old ones
    assert box[2] > box[0]


def test_init_scale_box_falls_back_to_isotropic_with_no_old_modes():
    # degenerate case: every anchor is exactly zero everywhere (e.g. a
    # single all-zero toy anchor) -- no reference scale to widen against,
    # so every coordinate gets plain init_scale rather than NaN from an
    # empty median.
    scale = np.array([1.0, 3.0])
    old_mask = np.array([False, False])
    box = init_scale_box(scale, old_mask, 0.02)
    assert np.allclose(box, [0.02, 0.02])
    assert np.all(np.isfinite(box))


# --------------------------------------------------------------------------- #
# seed_population_from_anchors
# --------------------------------------------------------------------------- #
def _anchors_and_box():
    anchor_xs = np.array([[0.1, 0.2, 0.0], [0.0, -0.1, 0.3]])
    box = np.array([0.02, 0.02, 0.02])
    xl, xu = anchor_xs.min(axis=0) - 0.5, anchor_xs.max(axis=0) + 0.5
    return anchor_xs, box, xl, xu


def test_seed_population_from_anchors_keeps_anchors_verbatim_in_generation_zero():
    anchor_xs, box, xl, xu = _anchors_and_box()
    pop = seed_population_from_anchors(
        anchor_xs, pop_size=6, box=box, xl=xl, xu=xu, rng=np.random.RandomState(0)
    )
    assert np.allclose(pop[0], anchor_xs[0])
    assert np.allclose(pop[1], anchor_xs[1])


def test_seed_population_from_anchors_count_equals_pop_size():
    anchor_xs, box, xl, xu = _anchors_and_box()
    for pop_size in (2, 6, 48):
        pop = seed_population_from_anchors(
            anchor_xs,
            pop_size=pop_size,
            box=box,
            xl=xl,
            xu=xu,
            rng=np.random.RandomState(0),
        )
        assert pop.shape == (pop_size, anchor_xs.shape[1])


def test_seed_population_from_anchors_mutated_members_within_the_box():
    anchor_xs, box, xl, xu = _anchors_and_box()
    pop = seed_population_from_anchors(
        anchor_xs, pop_size=48, box=box, xl=xl, xu=xu, rng=np.random.RandomState(3)
    )
    for member in pop[len(anchor_xs) :]:
        # every mutated member must lie within SOME anchor's own +/- box
        within_some_anchor = any(
            np.all(member >= a - box - 1e-9) and np.all(member <= a + box + 1e-9) for a in anchor_xs
        )
        assert within_some_anchor
        assert np.all(member >= xl - 1e-9) and np.all(member <= xu + 1e-9)


def test_seed_population_from_anchors_rejects_pop_size_below_anchor_count():
    anchor_xs, box, xl, xu = _anchors_and_box()
    with pytest.raises(ValueError):
        seed_population_from_anchors(
            anchor_xs, pop_size=1, box=box, xl=xl, xu=xu, rng=np.random.RandomState(0)
        )


# --------------------------------------------------------------------------- #
# objectives_and_constraints_from_raws
# --------------------------------------------------------------------------- #
def test_objectives_and_constraints_from_raws_builds_F_and_G():
    raws = [
        {
            "tag": "a",
            "r": [0.001, -0.1, -0.2, -0.3, -0.4],
            "raw": {"aspect_ratio": 9.5, "min_len": 11.0},
        }
    ]
    F, G = objectives_and_constraints_from_raws(raws)
    assert F.shape == (1, 2)
    assert G.shape == (1, 5)
    assert F[0, 0] == pytest.approx(-11.0)
    assert F[0, 1] == pytest.approx(9.5)
    assert G[0].tolist() == pytest.approx([0.001, -0.1, -0.2, -0.3, -0.4])


def test_objectives_and_constraints_from_raws_penalises_non_convergent():
    from constellaration_methods.fronts.b3_nsga2_mhd import (
        NON_CONVERGENT_CONSTRAINT_PENALTY,
        NON_CONVERGENT_OBJECTIVE_PENALTY,
    )

    raws = [{"tag": "a", "non_convergent": True}]
    F, G = objectives_and_constraints_from_raws(raws)
    assert np.all(F == NON_CONVERGENT_OBJECTIVE_PENALTY)
    assert np.all(G == NON_CONVERGENT_CONSTRAINT_PENALTY)
    assert np.all(np.isfinite(F)) and np.all(np.isfinite(G))


# --------------------------------------------------------------------------- #
# fidelity default / arg validation
# --------------------------------------------------------------------------- #
def test_resolve_eval_fidelity_defaults_to_canonical_when_anchors_given():
    args = build_arg_parser().parse_args(["--anchors", "a.json", "--representation", "r.json"])
    assert resolve_eval_fidelity(args) == "canonical"


def test_resolve_eval_fidelity_defaults_to_cheap_without_anchors():
    args = build_arg_parser().parse_args([])
    assert resolve_eval_fidelity(args) == "cheap"


def test_resolve_eval_fidelity_defaults_to_canonical_when_seed_candidate_given():
    args = build_arg_parser().parse_args(["--seed-candidate", "c.json"])
    assert resolve_eval_fidelity(args) == "canonical"


def test_resolve_eval_fidelity_respects_an_explicit_choice():
    args = build_arg_parser().parse_args(
        [
            "--anchors",
            "a.json",
            "--representation",
            "r.json",
            "--eval-fidelity",
            "cheap",
        ]
    )
    assert resolve_eval_fidelity(args) == "cheap"


def test_validate_seeded_args_requires_anchors_and_representation_together():
    args = build_arg_parser().parse_args(["--anchors", "a.json"])
    with pytest.raises(SystemExit):
        validate_seeded_args(args)


def test_validate_seeded_args_accepts_anchors_with_representation():
    validate_seeded_args(
        build_arg_parser().parse_args(["--anchors", "a.json", "--representation", "r.json"])
    )


# E18 item 2 changed this function's contract: it used to accept "neither
# anchors nor representation given" (there was no second seeding source, so
# main()'s own dispatch gate made that case unreachable in practice, and the
# old test pinned it anyway). Now --seed-candidate is a second seeding
# source and exactly one is required, so "neither given" is refused too --
# the old "accepts ... or neither" assertion no longer describes what this
# function should do and is replaced by the two tests below.
def test_validate_seeded_args_rejects_neither_anchors_nor_seed_candidate():
    with pytest.raises(SystemExit):
        validate_seeded_args(build_arg_parser().parse_args([]))


def test_validate_seeded_args_rejects_both_anchors_and_seed_candidate():
    args = build_arg_parser().parse_args(
        [
            "--anchors",
            "a.json",
            "--representation",
            "r.json",
            "--seed-candidate",
            "c.json",
        ]
    )
    with pytest.raises(SystemExit):
        validate_seeded_args(args)


def test_validate_seeded_args_accepts_seed_candidate_alone():
    validate_seeded_args(build_arg_parser().parse_args(["--seed-candidate", "c.json"]))


# --------------------------------------------------------------------------- #
# ceilings / defaults
# --------------------------------------------------------------------------- #
def test_default_seeded_ceilings_match_the_spec():
    args = build_arg_parser().parse_args(["--anchors", "a.json", "--representation", "r.json"])
    assert args.pop_size == POP_SIZE_SEEDED_DEFAULT == 48
    assert args.init_scale == pytest.approx(INIT_SCALE_DEFAULT) == pytest.approx(0.02)
    assert args.total_hours == pytest.approx(SEEDED_TOTAL_HOURS_DEFAULT) == pytest.approx(24.0)
    assert args.max_evals == SEEDED_MAX_EVALS_DEFAULT == 3000


def test_smoke_forces_seeded_ceilings_even_if_others_given():
    args = build_arg_parser().parse_args(
        [
            "--anchors",
            "a.json",
            "--representation",
            "r.json",
            "--pop-size",
            "48",
            "--total-hours",
            "24",
            "--smoke",
        ]
    )
    apply_smoke_ceilings_seeded(args)
    assert args.pop_size == 4
    assert args.max_generations == 1
    assert args.max_evals == 20
    assert args.total_hours == pytest.approx(0.05)


def test_resolve_out_dir_seeded_defaults_to_the_smoke_path():
    args = build_arg_parser().parse_args(
        ["--anchors", "a.json", "--representation", "r.json", "--smoke"]
    )
    assert resolve_out_dir_seeded(args) == "results/b3-nsga2-e17-smoke"


def test_resolve_out_dir_seeded_required_without_smoke():
    args = build_arg_parser().parse_args(["--anchors", "a.json", "--representation", "r.json"])
    with pytest.raises(SystemExit):
        resolve_out_dir_seeded(args)


def test_seeded_stop_reason_wall_cap():
    assert seeded_stop_reason(3601, 1.0, 0, None, 0, 3000, 48) == "wall_cap"


def test_seeded_stop_reason_max_generations():
    assert seeded_stop_reason(0.0, 24.0, 5, 5, 0, 3000, 48) == "max_generations"


def test_seeded_stop_reason_max_evals():
    assert seeded_stop_reason(0.0, 24.0, 0, None, 2980, 3000, 48) == "max_evals"


def test_seeded_stop_reason_none_when_nothing_binds():
    assert seeded_stop_reason(0.0, 24.0, 0, None, 0, 3000, 48) is None


# --------------------------------------------------------------------------- #
# archive/report schema round-trip with p3_front_from_archive
# --------------------------------------------------------------------------- #
def test_archive_rows_from_canonical_evaluator_are_front_from_archive_compatible(
    tmp_path,
):
    """A toy canonical evaluation batch (payload flag 'fake_front', no VMEC)
    writes rows p3_front_from_archive's own pure readers accept unchanged:
    tolerance_feasible_rows/nondominated see the same 'r'/'x'/'raw' shape
    E15/E16 already produce. (hypervolume() itself needs `constellaration`
    and so needs the pinned image; this test exercises everything up to
    that call on host.)"""
    from concurrent.futures import ProcessPoolExecutor

    from constellaration_methods.fronts.p3_front_from_archive import (
        nondominated,
        tolerance_feasible_rows,
    )
    from constellaration_methods.restoration.p3_feasibility_restoration import (
        Budget,
        CanonicalEvaluator,
    )

    representation = {
        "boundary_dict": {},
        "mpol": 1,
        "ntor": 1,
        "nfp": 1,
        "scaling": 1.0,
        "fake_front": True,
    }
    archive_path = tmp_path / "archive.jsonl"
    budget = Budget(cap=10)
    with ProcessPoolExecutor(max_workers=2) as pool:
        evaluator = CanonicalEvaluator(representation, pool, archive_path, budget)
        xs = [np.array([0.0, 0.0, 0.0]), np.array([0.05, 0.0, 0.0])]
        rs, raws = evaluator.evaluate(xs, ["t0", "t1"])

    assert all(r is not None for r in rs)
    rows = [__import__("json").loads(line) for line in archive_path.read_text().splitlines()]
    assert len(rows) == 2
    feas = tolerance_feasible_rows(rows, tol=0.01)
    points = [dict(row["raw"], tag=row["tag"]) for row in feas]
    front = nondominated(points)
    assert len(front) >= 1
    for p in front:
        assert "aspect_ratio" in p and "min_len" in p


# --------------------------------------------------------------------------- #
# toy end-to-end (needs pymoo for the ask/tell loop AND constellaration for
# hypervolume/nondominated via p3_front_from_archive -- pinned image only,
# same limitation as test_p3_front_builder.py's own toy end-to-end test)
# --------------------------------------------------------------------------- #
def _toy_anchor_and_representation():
    representation = {
        "boundary_dict": {},
        "mpol": 1,
        "ntor": 1,
        "nfp": 1,
        "scaling": 1.0,
        "fake_front": True,
    }
    anchor = {
        "tag": "anchor0",
        "r": [-0.05, -0.05, -0.05, -0.05, -0.05],
        "aspect": 10.0,
        "min_len": 10.0,
        "x": [0.0, 0.0, 0.0],
    }
    return anchor, representation


def test_run_seeded_from_anchors_rejects_pop_size_below_anchor_count_names_both(
    tmp_path,
):
    """Bug 2: the refusal is correct (seeding needs one population slot per
    anchor) but the message must name both the anchor count and the minimum
    --pop-size, not just repeat one number."""
    anchor, representation = _toy_anchor_and_representation()
    anchor2 = dict(anchor, tag="anchor1")
    args = build_arg_parser().parse_args(
        [
            "--anchors",
            "a.json",
            "--representation",
            "r.json",
            "--pop-size",
            "1",
        ]
    )
    with pytest.raises(SystemExit) as excinfo:
        run_seeded_from_anchors([anchor, anchor2], representation, tmp_path / "out", args)
    message = str(excinfo.value)
    assert "1" in message
    assert "2" in message
    assert "anchor" in message and "minimum" in message.lower()


def test_runner_smoke_toy_end_to_end_one_generation_pop_four(tmp_path):
    """--smoke's own shape (pop 4, one generation), driven with the fake_front
    seam so it needs no VMEC -- matches p3_front_builder's own smoke test."""
    pytest.importorskip(
        "constellaration",
        reason="needs the pinned constellaration package; run inside the repo's Docker image",
    )
    anchor, representation = _toy_anchor_and_representation()
    args = build_arg_parser().parse_args(
        ["--anchors", "a.json", "--representation", "r.json", "--smoke"]
    )
    apply_smoke_ceilings_seeded(args)
    args.eval_fidelity = resolve_eval_fidelity(args)
    assert (args.pop_size, args.max_generations) == (4, 1)
    report = run_seeded_from_anchors([anchor], representation, tmp_path / "out", args)
    assert report["generations"] == 1
    assert report["pop_size"] == 4
    assert report["stop_reason"] == "max_generations"
    assert (tmp_path / "out" / "REPORT.json").exists()


def test_runner_toy_end_to_end_two_generations_hold_or_increase_hypervolume(tmp_path):
    pytest.importorskip(
        "constellaration",
        reason="needs the pinned constellaration package; run inside the repo's Docker image",
    )
    anchor, representation = _toy_anchor_and_representation()
    args = SimpleNamespace(
        smoke=False,
        seed=0,
        pop_size=6,
        init_scale=0.02,
        max_generations=2,
        max_evals=600,
        total_hours=1.0,
        eval_fidelity="canonical",
        workers=2,
    )
    report = run_seeded_from_anchors([anchor], representation, tmp_path / "out", args)
    assert report["generations"] == 2
    assert report["stop_reason"] == "max_generations"
    assert report["final_hypervolume"] >= report["initial_hypervolume"]
    assert (tmp_path / "out" / "REPORT.json").exists()
    assert (tmp_path / "out" / "archive.jsonl").exists()
    assert (tmp_path / "out" / "start_serialised.json").exists()


def test_runner_writes_report_on_keyboard_interrupt(tmp_path, monkeypatch):
    """A KeyboardInterrupt raised mid-loop (e.g. Ctrl+C during a canonical
    evaluation batch) must still leave a REPORT.json behind. The evaluator
    itself is stubbed out, but `run_seeded_from_anchors` still constructs a
    real pymoo NSGA2 and calls the real `hypervolume()` before the loop, so
    this test needs the pinned image too (pymoo + constellaration), same as
    the toy end-to-end test above."""
    pytest.importorskip(
        "constellaration",
        reason="needs the pinned constellaration package; run inside the repo's Docker image",
    )
    import constellaration_methods.fronts.b3_nsga2_mhd as b3_nsga2_mhd

    representation = {
        "boundary_dict": {},
        "mpol": 1,
        "ntor": 1,
        "nfp": 1,
        "scaling": 1.0,
        "fake_front": True,
    }
    anchor = {
        "tag": "anchor0",
        "r": [-0.05, -0.05, -0.05, -0.05, -0.05],
        "aspect": 10.0,
        "min_len": 10.0,
        "x": [0.0, 0.0, 0.0],
    }
    args = SimpleNamespace(
        smoke=False,
        seed=0,
        pop_size=6,
        init_scale=0.02,
        max_generations=5,
        max_evals=600,
        total_hours=1.0,
        eval_fidelity="canonical",
        workers=2,
    )

    def _raise(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        b3_nsga2_mhd,
        "_make_seeded_evaluator",
        lambda *a, **k: (_raise, __import__("contextlib").nullcontext()),
    )
    report = run_seeded_from_anchors([anchor], representation, tmp_path / "out", args)
    assert report["stop_reason"] == "keyboard_interrupt"
    assert (tmp_path / "out" / "REPORT.json").exists()


# --------------------------------------------------------------------------- #
# E18 item 1 -- --constraint-override on b3_nsga2_mhd.py (seeded mode)
# --------------------------------------------------------------------------- #
def test_build_arg_parser_has_constraint_override_repeatable():
    args = build_arg_parser().parse_args(
        [
            "--anchors",
            "a.json",
            "--representation",
            "r.json",
            "--constraint-override",
            "mirror_upper=0.23",
            "--constraint-override",
            "well_lower=0.01",
        ]
    )
    assert args.constraint_override == ["mirror_upper=0.23", "well_lower=0.01"]


def test_build_arg_parser_constraint_override_defaults_to_none():
    args = build_arg_parser().parse_args(["--anchors", "a.json", "--representation", "r.json"])
    assert args.constraint_override is None


def test_runner_records_empty_constraint_overrides_in_report_by_default(tmp_path):
    pytest.importorskip(
        "constellaration",
        reason="needs the pinned constellaration package; run inside the repo's Docker image",
    )
    anchor, representation = _toy_anchor_and_representation()
    args = SimpleNamespace(
        smoke=False,
        seed=0,
        pop_size=6,
        init_scale=0.02,
        max_generations=1,
        max_evals=600,
        total_hours=1.0,
        eval_fidelity="canonical",
        workers=2,
        constraint_override=None,
    )
    report = run_seeded_from_anchors([anchor], representation, tmp_path / "out", args)
    assert report["constraint_overrides"] == {}


def test_runner_records_given_constraint_overrides_in_report(tmp_path):
    pytest.importorskip(
        "constellaration",
        reason="needs the pinned constellaration package; run inside the repo's Docker image",
    )
    anchor, representation = _toy_anchor_and_representation()
    args = SimpleNamespace(
        smoke=False,
        seed=0,
        pop_size=6,
        init_scale=0.02,
        max_generations=1,
        max_evals=600,
        total_hours=1.0,
        eval_fidelity="canonical",
        workers=2,
        constraint_override=["mirror_upper=0.23"],
    )
    report = run_seeded_from_anchors([anchor], representation, tmp_path / "out", args)
    assert report["constraint_overrides"] == {"mirror_upper": 0.23}


# --------------------------------------------------------------------------- #
# E18 item 2 -- --seed-candidate (single-seed population, as an alternative
# to --anchors)
# --------------------------------------------------------------------------- #
def test_build_arg_parser_has_seed_candidate_checkpoint_and_coeff_space_box():
    args = build_arg_parser().parse_args(
        ["--seed-candidate", "c.json", "--checkpoint", "ck.json", "--coeff-space-box"]
    )
    assert args.seed_candidate == "c.json"
    assert args.checkpoint == "ck.json"
    assert args.coeff_space_box is True


def test_build_arg_parser_seed_candidate_defaults_to_none_and_box_to_false():
    args = build_arg_parser().parse_args(["--anchors", "a.json", "--representation", "r.json"])
    assert args.seed_candidate is None
    assert args.checkpoint is None
    assert args.coeff_space_box is False


def test_single_seed_population_is_seed_population_from_anchors_with_one_anchor():
    """The single-seed population --seed-candidate needs IS
    seed_population_from_anchors given a length-1 anchor array: the seed
    verbatim at index 0, pop_size total, every mutant within the box -- no
    separate population-construction function was needed (E18 item 2)."""
    rng = np.random.RandomState(0)
    x0 = np.array([0.1, -0.2, 0.3])
    box = np.array([0.02, 0.02, 0.02])
    xl, xu = x0 - 1.0, x0 + 1.0
    pop = seed_population_from_anchors(np.array([x0]), pop_size=10, box=box, xl=xl, xu=xu, rng=rng)
    assert pop.shape == (10, 3)
    assert np.allclose(pop[0], x0)
    for row in pop[1:]:
        assert np.all(np.abs(row - x0) <= box + 1e-9)


def _toy_candidate(tmp_path, **overrides):
    cand = dict(
        x=[0.1, 0.2, 0.3],
        r=[-0.1, -0.1, -0.1, -0.1, -0.1],
        raw={"aspect_ratio": 9.0, "min_len": 9.5},
        boundary_dict={"r_cos": [[1.0]]},
        mpol=8,
        ntor=8,
        nfp=3,
        scaling=1.0,
        source={"path": "src.json", "mpol": 4, "ntor": 4},
    )
    cand.update(overrides)
    path = tmp_path / "cand.json"
    path.write_text(json.dumps(cand))
    return path, cand


def test_load_seed_candidate_uses_the_candidates_own_representation(tmp_path):
    path, cand = _toy_candidate(tmp_path)
    anchors, representation, old_mask = load_seed_candidate(str(path), None, False)
    assert len(anchors) == 1
    assert anchors[0]["x"] == cand["x"]
    assert anchors[0]["r"] == cand["r"]
    assert anchors[0]["aspect"] == cand["raw"]["aspect_ratio"]
    assert anchors[0]["min_len"] == cand["raw"]["min_len"]
    assert representation["boundary_dict"] == cand["boundary_dict"]
    assert representation["mpol"] == cand["mpol"]
    assert old_mask is None


def test_load_seed_candidate_requires_checkpoint_without_a_boundary_dict(tmp_path):
    cand = dict(
        x=[0.1],
        r=[-0.1, -0.1, -0.1, -0.1, -0.1],
        raw={"aspect_ratio": 9.0, "min_len": 9.5},
    )
    path = tmp_path / "cand.json"
    path.write_text(json.dumps(cand))
    with pytest.raises(SystemExit):
        load_seed_candidate(str(path), None, False)


def test_load_seed_candidate_falls_back_to_checkpoint_representation(tmp_path):
    cand = dict(
        x=[0.1],
        r=[-0.1, -0.1, -0.1, -0.1, -0.1],
        raw={"aspect_ratio": 9.0, "min_len": 9.5},
    )
    cand_path = tmp_path / "cand.json"
    cand_path.write_text(json.dumps(cand))
    checkpoint = dict(
        boundary_dict={"r_cos": [[2.0]]},
        max_poloidal_mode=8,
        max_toroidal_mode=8,
        n_field_periods=3,
        infinity_norm_spectrum_scaling=1.0,
    )
    ck_path = tmp_path / "checkpoint.json"
    ck_path.write_text(json.dumps(checkpoint))
    anchors, representation, old_mask = load_seed_candidate(str(cand_path), str(ck_path), False)
    assert representation["boundary_dict"] == checkpoint["boundary_dict"]
    assert representation["mpol"] == 8
    assert old_mask is None


def test_load_seed_candidate_coeff_space_box_needs_a_source_budget(tmp_path):
    path, _cand = _toy_candidate(tmp_path, source=None)
    with pytest.raises(SystemExit):
        load_seed_candidate(str(path), None, True)


def test_load_seed_candidate_coeff_space_box_calls_old_mode_mask_with_source_budget(
    monkeypatch, tmp_path
):
    import constellaration_methods.fronts.b3_nsga2_mhd as b3_nsga2_mhd

    path, cand = _toy_candidate(tmp_path)
    captured = {}

    def fake_old_mode_mask(c, mpol_old, ntor_old):
        captured["cand"] = c
        captured["mpol_old"] = mpol_old
        captured["ntor_old"] = ntor_old
        return np.array([True, False, True]), np.array([1.0, 2.0, 3.0])

    monkeypatch.setattr(b3_nsga2_mhd, "old_mode_mask", fake_old_mode_mask)
    anchors, representation, old_mask = b3_nsga2_mhd.load_seed_candidate(str(path), None, True)
    assert captured["mpol_old"] == 4
    assert captured["ntor_old"] == 4
    assert old_mask.tolist() == [True, False, True]


def test_run_seeded_dispatches_seed_candidate_to_load_seed_candidate(monkeypatch, tmp_path):
    import constellaration_methods.fronts.b3_nsga2_mhd as b3_nsga2_mhd

    path, cand = _toy_candidate(tmp_path)
    args = build_arg_parser().parse_args(
        ["--seed-candidate", str(path), "--out-dir", str(tmp_path / "out")]
    )
    captured = {}

    def fake_run_seeded_from_anchors(anchors, representation, out_dir, args, old_mask=None):
        captured["anchors"] = anchors
        captured["representation"] = representation
        captured["old_mask"] = old_mask
        return {"stub": True}

    monkeypatch.setattr(b3_nsga2_mhd, "run_seeded_from_anchors", fake_run_seeded_from_anchors)
    result = b3_nsga2_mhd.run_seeded(args)
    assert result == {"stub": True}
    assert captured["anchors"][0]["x"] == cand["x"]
    assert captured["old_mask"] is None


# --------------------------------------------------------------------------- #
# E18 item 3 -- V-vs-evaluations curve in events.jsonl / REPORT.json
# --------------------------------------------------------------------------- #
def test_runner_generation_events_carry_best_v_so_far_and_n_feasible_so_far(tmp_path):
    pytest.importorskip(
        "constellaration",
        reason="needs the pinned constellaration package; run inside the repo's Docker image",
    )
    anchor, representation = _toy_anchor_and_representation()
    args = SimpleNamespace(
        smoke=False,
        seed=0,
        pop_size=6,
        init_scale=0.02,
        max_generations=2,
        max_evals=600,
        total_hours=1.0,
        eval_fidelity="canonical",
        workers=2,
        constraint_override=None,
    )
    run_seeded_from_anchors([anchor], representation, tmp_path / "out", args)
    events = [
        json.loads(ln)
        for ln in (tmp_path / "out" / "events.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    gens = [e for e in events if e["event"] == "generation"]
    assert len(gens) == 2
    for e in gens:
        assert "best_v_so_far" in e
        assert "n_feasible_so_far" in e
        assert isinstance(e["n_feasible_so_far"], int)


def test_runner_report_carries_v_curve_sampled_at_every_configured_count(monkeypatch, tmp_path):
    pytest.importorskip(
        "constellaration",
        reason="needs the pinned constellaration package; run inside the repo's Docker image",
    )
    import constellaration_methods.fronts.b3_nsga2_mhd as b3_nsga2_mhd

    monkeypatch.setattr(b3_nsga2_mhd, "V_CURVE_EVERY", 4)
    anchor, representation = _toy_anchor_and_representation()
    args = SimpleNamespace(
        smoke=False,
        seed=0,
        pop_size=6,
        init_scale=0.02,
        max_generations=2,
        max_evals=600,
        total_hours=1.0,
        eval_fidelity="canonical",
        workers=2,
        constraint_override=None,
    )
    report = b3_nsga2_mhd.run_seeded_from_anchors([anchor], representation, tmp_path / "out", args)
    assert "v_curve" in report
    # 2 generations of pop_size 6 = 12 archive rows -> 3 blocks of 4
    assert [e for e, _v in report["v_curve"]] == [4, 8, 12]


# --------------------------------------------------------------------------- #
# E18 item 4 -- combined toy end-to-end: a single synthetic seed candidate
# (the --seed-candidate shape: one anchor, not a front), a constraint
# override, two generations -- REPORT.json must carry both the overrides
# and the curve. Exercises the same run_seeded_from_anchors entry point the
# other toy end-to-end tests above use, mirroring their own convention
# (they call it directly rather than through the CLI/main()).
# --------------------------------------------------------------------------- #
def test_toy_end_to_end_single_seed_with_constraint_override_two_generations(monkeypatch, tmp_path):
    pytest.importorskip(
        "constellaration",
        reason="needs the pinned constellaration package; run inside the repo's Docker image",
    )
    import constellaration_methods.fronts.b3_nsga2_mhd as b3_nsga2_mhd

    monkeypatch.setattr(b3_nsga2_mhd, "V_CURVE_EVERY", 4)
    representation = {
        "boundary_dict": {},
        "mpol": 1,
        "ntor": 1,
        "nfp": 1,
        "scaling": 1.0,
        "fake_front": True,
    }
    seed_candidate_anchor = {
        "tag": "seed:synthetic_candidate.json",
        "r": [-0.05, -0.05, -0.05, -0.05, -0.05],
        "aspect": 10.0,
        "min_len": 10.0,
        "x": [0.0, 0.0, 0.0],
    }
    args = SimpleNamespace(
        smoke=False,
        seed=0,
        pop_size=6,
        init_scale=0.02,
        max_generations=2,
        max_evals=600,
        total_hours=1.0,
        eval_fidelity="canonical",
        workers=2,
        constraint_override=["mirror_upper=0.23"],
    )
    report = b3_nsga2_mhd.run_seeded_from_anchors(
        [seed_candidate_anchor], representation, tmp_path / "out", args
    )
    assert report["generations"] == 2
    assert report["n_anchors"] == 1
    assert report["constraint_overrides"] == {"mirror_upper": 0.23}
    assert [e for e, _v in report["v_curve"]] == [4, 8, 12]
    events = [
        json.loads(ln)
        for ln in (tmp_path / "out" / "events.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    gens = [e for e in events if e["event"] == "generation"]
    assert len(gens) == 2
    for e in gens:
        assert "best_v_so_far" in e and "n_feasible_so_far" in e
    archive_rows = [
        json.loads(ln)
        for ln in (tmp_path / "out" / "archive.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    assert len(archive_rows) == 12
    for row in archive_rows:
        assert "r" in row and len(row["r"]) == 5


def _seed_row(out_dir: Path) -> dict:
    """The G0:0 archive row: generation 0's first member is the anchor
    verbatim (`seed_population_from_anchors`), so it is the toy's seed at
    x = 0 whichever thresholds were in force."""
    rows = [
        json.loads(ln) for ln in (out_dir / "archive.jsonl").read_text().splitlines() if ln.strip()
    ]
    (row,) = [r for r in rows if r["tag"] == "G0:0"]
    return row


def test_toy_constraint_override_changes_the_seed_row_mirror_residual(tmp_path):
    """Regression ratchet (2026-10-07). The override toy above proved only
    that `--constraint-override mirror_upper=0.23` was RECORDED in
    REPORT.json: the fake_front seam ignored `constraint_overrides`, so no
    evaluation changed. The seed row's mirror residual must differ between a
    stock run and an overridden run by exactly the benchmark's own
    normalised-violation formula applied to the seam's raw mirror value m,
    (m - 0.23) / 0.23 - (m - 0.25) / 0.25, and m itself (a measurement, not
    a threshold) must be the same in both runs."""
    pytest.importorskip(
        "constellaration",
        reason="needs the pinned constellaration package; run inside the repo's Docker image",
    )
    import constellaration_methods.fronts.b3_nsga2_mhd as b3_nsga2_mhd

    anchor, representation = _toy_anchor_and_representation()
    runs = {}
    for name, overrides in (("stock", []), ("override", ["mirror_upper=0.23"])):
        args = SimpleNamespace(
            smoke=False,
            seed=0,
            pop_size=6,
            init_scale=0.02,
            max_generations=1,
            max_evals=600,
            total_hours=1.0,
            eval_fidelity="canonical",
            workers=2,
            constraint_override=overrides,
        )
        b3_nsga2_mhd.run_seeded_from_anchors([anchor], representation, tmp_path / name, args)
        runs[name] = _seed_row(tmp_path / name)
    stock, over = runs["stock"], runs["override"]
    assert stock["x"] == [0.0, 0.0, 0.0] and over["x"] == [0.0, 0.0, 0.0]
    assert over["r"][2] != stock["r"][2], (
        "the override must change the evaluation, not just the report"
    )
    m = stock["raw"]["edge_mirror"]
    assert over["raw"]["edge_mirror"] == m
    assert over["r"][2] - stock["r"][2] == (m - 0.23) / 0.23 - (m - 0.25) / 0.25
    # every other row is untouched by a mirror-only override
    for i in (0, 1, 3, 4):
        assert over["r"][i] == stock["r"][i]
