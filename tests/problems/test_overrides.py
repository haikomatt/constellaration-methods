"""Tests for constellaration_methods.problems.overrides. Moved out of
test_p3_feasibility_restoration.py (the E18 item 1 block) because the
override mechanism was promoted to its own module -- see
src/constellaration_methods/problems/overrides.py's docstring.

Two of these need the pinned constellaration image
(`constellaration.forward_model.ConstellarationMetrics`); they run inside the
repo's own Docker image, same convention as the rest of this package.
"""

from __future__ import annotations

import pytest

from constellaration_methods.problems.overrides import (
    CONSTRAINT_OVERRIDE_NAMES,
    parse_constraint_overrides,
    problem_with_overrides,
)


def _toy_metrics():
    """A fixed, arbitrary-but-finite ConstellarationMetrics: every field the
    benchmark schema requires, values chosen with no special relationship to
    any bound so a changed threshold visibly changes exactly its own row."""
    from constellaration import forward_model

    return forward_model.ConstellarationMetrics(
        aspect_ratio=8.0,
        aspect_ratio_over_edge_rotational_transform=10.0,
        max_elongation=3.0,
        axis_rotational_transform_over_n_field_periods=0.3,
        edge_rotational_transform_over_n_field_periods=0.30,
        axis_magnetic_mirror_ratio=0.1,
        edge_magnetic_mirror_ratio=0.22,
        average_triangularity=-0.3,
        vacuum_well=0.01,
        minimum_normalized_magnetic_gradient_scale_length=5.0,
        qi=1e-4,
        flux_compression_in_regions_of_bad_curvature=0.5,
    )


def test_problem_with_overrides_empty_dict_matches_stock_class():
    """Empty overrides -> byte-identical residuals to the stock class."""
    pytest.importorskip(
        "constellaration",
        reason="needs the pinned constellaration package; run inside the repo's Docker image",
    )
    from constellaration import problems

    metrics = _toy_metrics()
    stock = problems.MHDStableQIStellarator()._normalized_constraint_violations(metrics)
    overridden = problem_with_overrides({})._normalized_constraint_violations(metrics)
    assert list(overridden) == list(stock)
    none_case = problem_with_overrides(None)._normalized_constraint_violations(metrics)
    assert list(none_case) == list(stock)


def test_problem_with_overrides_mirror_upper_changes_only_the_mirror_row():
    pytest.importorskip(
        "constellaration",
        reason="needs the pinned constellaration package; run inside the repo's Docker image",
    )
    metrics = _toy_metrics()
    stock = problem_with_overrides({})._normalized_constraint_violations(metrics)
    overridden = problem_with_overrides({"mirror_upper": 0.23})._normalized_constraint_violations(
        metrics
    )
    expected_mirror = (metrics.edge_magnetic_mirror_ratio - 0.23) / 0.23
    assert overridden[2] == pytest.approx(expected_mirror)
    for i in (0, 1, 3, 4):
        assert overridden[i] == pytest.approx(stock[i])


def test_problem_with_overrides_unknown_name_refused():
    pytest.importorskip(
        "constellaration",
        reason="needs the pinned constellaration package; run inside the repo's Docker image",
    )
    with pytest.raises(ValueError, match="unknown constraint override"):
        problem_with_overrides({"not_a_real_bound": 1.0})


def test_problem_with_overrides_unknown_name_lists_valid_names():
    pytest.importorskip(
        "constellaration",
        reason="needs the pinned constellaration package; run inside the repo's Docker image",
    )
    with pytest.raises(ValueError) as excinfo:
        problem_with_overrides({"nope": 1.0})
    for name in CONSTRAINT_OVERRIDE_NAMES:
        assert name in str(excinfo.value)


def test_parse_constraint_overrides_empty_and_none():
    assert parse_constraint_overrides(None) == {}
    assert parse_constraint_overrides([]) == {}


def test_parse_constraint_overrides_parses_repeated_flags():
    out = parse_constraint_overrides(["mirror_upper=0.23", "well_lower=0.01"])
    assert out == {"mirror_upper": 0.23, "well_lower": 0.01}


def test_parse_constraint_overrides_rejects_malformed_pair():
    with pytest.raises(ValueError, match="NAME=VALUE"):
        parse_constraint_overrides(["mirror_upper_0.23"])


def test_parse_constraint_overrides_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown constraint override"):
        parse_constraint_overrides(["not_a_real_bound=1.0"])
