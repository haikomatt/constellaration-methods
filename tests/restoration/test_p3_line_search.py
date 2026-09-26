"""E2's second-order piece (vault p3-reopen-three-experiments-2026-09-16):
a 3-point quadratic line model per residual along the LP direction, and the
choice of t* minimising predicted V subject to predicted well <= 0."""

import numpy as np
import pytest

from constellaration_methods.restoration.p3_final_restoration import (
    best_t_on_line,
    quadratic_line_model,
)


def test_quadratic_line_model_recovers_true_quadratics():
    a = np.array([2.0, -1.0, 0.5, 0.0, 1.0])
    b = np.array([-1.0, -0.5, -0.2, -0.3, 0.1])
    c = np.array([0.12, 0.12, 0.12, 0.10, -0.03])
    r = lambda t: a * t * t + b * t + c  # noqa: E731
    A, B, C = quadratic_line_model(r(0.0), r(0.5), r(1.0))
    assert np.allclose(A, a) and np.allclose(B, b) and np.allclose(C, c)


def test_best_t_minimises_predicted_v_and_keeps_well_nonpositive():
    # residual 0 curves back up (a>0), residual 4 (well) crosses zero beyond t=1.2
    A = np.array([0.5, 0.0, 0.0, 0.0, 0.0])
    B = np.array([-0.4, -0.05, -0.05, -0.05, 0.05])
    C = np.array([0.12, 0.12, 0.12, 0.10, -0.06])
    t, v_pred = best_t_on_line(A, B, C, t_grid=np.linspace(0.1, 1.5, 15))
    # V(t) = max(rows); row 0 minimised at t=0.4 (0.04); others fall with t;
    # well = -0.06 + 0.05 t <= 0 for t <= 1.2 -> t* is where max is smallest inside [0.1, 1.2]
    assert 0.1 <= t <= 1.2
    assert v_pred == pytest.approx(max(A * t * t + B * t + C))
    # if well would go positive for every t, return None
    C2 = C.copy()
    C2[4] = 0.01
    assert best_t_on_line(A, B, C2, np.linspace(0.1, 1.5, 15))[0] is None


def test_best_t_requires_well_nonpositive_only_once_reached():
    # E11 run 2: from a start whose well is violated, every grid point had a
    # positive predicted well and the line search returned None every trial
    from constellaration_methods.restoration.p3_final_restoration import best_t_on_line

    # a line along which well falls 0.30 -> 0.20 but stays positive, V falls
    A = np.zeros(5)
    B = np.array([0.0, 0.0, 0.0, -0.01, -0.10])
    C = np.array([-0.04, -0.14, -0.20, 0.11, 0.30])
    assert best_t_on_line(A, B, C, [0.5, 1.0])[0] is None  # well reached: keep old rule
    t, v = best_t_on_line(A, B, C, [0.5, 1.0], require_well_nonpositive=False)
    assert t == 1.0 and abs(v - 0.20) < 1e-12


def test_best_t_respects_every_preservation_row_not_only_the_well():
    # review round 8: a line along which QI (row 1, satisfied at -0.009) crosses its
    # tolerance at t >= 1 must not be selected there even though V keeps falling
    from constellaration_methods.restoration.p3_final_restoration import best_t_on_line

    A = np.zeros(5)
    B = np.array([0.0, 0.03, 0.0, -0.02, -0.02])
    C = np.array([-0.05, -0.009, -0.2, 0.12, 0.12])
    # the well is violated here (0.12), so its own <= 0 requirement is off
    t, v = best_t_on_line(A, B, C, [0.5, 1.0, 1.5], require_well_nonpositive=False, r0=C, tol=0.01)
    assert t == 0.5  # at t=1 QI would be +0.021 > tol; at 0.5 it is +0.006
    t2, _ = best_t_on_line(
        A, B, C, [0.5, 1.0, 1.5], require_well_nonpositive=False
    )  # legacy call: no preservation rows checked
    assert t2 == 1.5
