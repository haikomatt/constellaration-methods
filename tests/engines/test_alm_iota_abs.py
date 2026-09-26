"""The pinned library's search residual for the iota constraint
(`augmented_lagrangian_runner.objective_constraints`) is `lower_bound - iota`
with no absolute value in the P2 and P3 branches, while BOTH problems'
canonical `_normalized_constraint_violations` use `lower_bound - |iota|`.
Negative-iota seeds are therefore descended the long way round (budget
ladder seed 176, 2026-09-13, lost a 2 h rung to it). Our engine wraps the
library call and re-expresses the iota residual with |iota| for those two
problems, leaving everything else untouched.
"""

import numpy as np
import pytest

problems = pytest.importorskip("constellaration.problems")
from constellaration_methods.engines.alm_ngopt_optimizer import (  # noqa: E402
    iota_abs_constraints,
)


class _M:
    def __init__(self, iota):
        self.edge_rotational_transform_over_n_field_periods = iota


def test_p3_negative_iota_residual_uses_abs():
    p = problems.MHDStableQIStellarator()
    cons = np.array([-5.0, 0.25 - (-0.107), 0.3, 0.4, -0.1, -0.7])
    out = iota_abs_constraints(p, _M(-0.107), cons)
    assert out[1] == pytest.approx(0.25 - 0.107)
    assert np.allclose(np.delete(out, 1), np.delete(cons, 1))


def test_p2_negative_iota_residual_uses_abs():
    p = problems.SimpleToBuildQIStellarator()
    cons = np.array([-1.0, 0.25 - (-0.2), 0.1, 0.1, 0.1])
    assert iota_abs_constraints(p, _M(-0.2), cons)[1] == pytest.approx(0.25 - 0.2)


def test_positive_iota_unchanged():
    p = problems.MHDStableQIStellarator()
    cons = np.array([-5.0, 0.25 - 0.19, 0.3, 0.4, -0.1, -0.7])
    assert np.allclose(iota_abs_constraints(p, _M(0.19), cons), cons)


def test_none_metrics_and_other_problems_unchanged():
    cons = np.array([1.0, 2.0, 3.0])
    assert iota_abs_constraints(problems.MHDStableQIStellarator(), None, cons) is cons
    assert iota_abs_constraints(problems.GeometricalProblem(), _M(-0.3), cons) is cons
