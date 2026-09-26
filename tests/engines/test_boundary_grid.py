"""Phase 0: pure-numpy tests for the shared boundary-grid helpers -- pad
correctness, over-budget truncation refusal, known-fixture effective-mode
energies, and rel_dist reproducing a known value."""

import numpy as np
import pytest

from constellaration_methods.engines.boundary_grid import (
    effective_modes,
    pad_to_common_grid,
    relative_seed_distance,
)


def test_pad_to_common_grid_preserves_values_and_centers_n0_column():
    # 2x3 grid (mpol=1, ntor=1): columns are n=-1,0,1
    r_cos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    z_sin = np.zeros_like(r_cos)

    rp, zp = pad_to_common_grid(r_cos, z_sin, target_mpol=3, target_ntor=2)

    assert rp.shape == (4, 5)
    # n=0 column of the 3-wide input is index 1; the 5-wide output's n=0
    # column is index 2, so col_off = 1 -- input columns 0,1,2 land at 1,2,3.
    np.testing.assert_array_equal(rp[:2, 1:4], r_cos)
    # everything outside the original block is zero
    rp_masked = rp.copy()
    rp_masked[:2, 1:4] = 0
    assert np.all(rp_masked == 0)
    assert np.all(zp == 0)


def test_pad_to_common_grid_noop_on_exact_shape():
    r_cos = np.arange(9 * 17, dtype=float).reshape(9, 17)
    z_sin = -r_cos
    rp, zp = pad_to_common_grid(r_cos, z_sin, target_mpol=8, target_ntor=8)
    np.testing.assert_array_equal(rp, r_cos)
    np.testing.assert_array_equal(zp, z_sin)


def test_pad_to_common_grid_refuses_over_budget_input():
    r_cos = np.zeros((10, 5))
    z_sin = np.zeros((10, 5))
    with pytest.raises(ValueError):
        pad_to_common_grid(r_cos, z_sin, target_mpol=8, target_ntor=8)

    r_cos = np.zeros((5, 21))
    z_sin = np.zeros((5, 21))
    with pytest.raises(ValueError):
        pad_to_common_grid(r_cos, z_sin, target_mpol=8, target_ntor=8)


def test_effective_modes_known_fixture():
    # 5x5 grid (mpol=4, ntor=2): only m=0,n=0 and m=3,n=2 nonzero.
    r_cos = np.zeros((5, 5))
    z_sin = np.zeros((5, 5))
    r_cos[0, 2] = 1.0  # m=0, n=0 (col 2 is n=0 in a 5-wide grid)
    r_cos[3, 4] = 1.0  # m=3, n=2 (col 4 is n=2)

    eff_m, eff_n, tail_frac = effective_modes(r_cos, z_sin, tol=1e-8, m_cut=4, n_cut=4)
    assert eff_m == 3
    assert eff_n == 2
    # both nonzero coefficients sit within the m<=4,|n|<=4 box -> no tail energy
    assert tail_frac == 0.0

    eff_m, eff_n, tail_frac = effective_modes(r_cos, z_sin, tol=1e-8, m_cut=2, n_cut=1)
    # now m=3,n=2 falls in the tail; the m=0,n=0 coefficient still doesn't ->
    # tail energy is half of the total (one of the two unit-magnitude coeffs)
    assert tail_frac == pytest.approx(0.5)


def test_effective_modes_all_zero_returns_zero_tail_frac():
    r_cos = np.zeros((5, 5))
    z_sin = np.zeros((5, 5))
    eff_m, eff_n, tail_frac = effective_modes(r_cos, z_sin, tol=1e-8)
    assert (eff_m, eff_n, tail_frac) == (0, 0, 0.0)


def test_relative_seed_distance_known_value():
    seed = np.array([3.0, 4.0])  # norm = 5
    x = np.array([3.0, 4.0])
    assert relative_seed_distance(x, seed) == pytest.approx(0.0)

    x = np.array([0.0, 0.0])
    # dist = 5, seed norm = 5 -> rel_dist = 1.0
    assert relative_seed_distance(x, seed) == pytest.approx(1.0)

    x = np.array([6.0, 8.0])
    # dist = ||[3,4]|| = 5, seed norm = 5 -> rel_dist = 1.0
    assert relative_seed_distance(x, seed) == pytest.approx(1.0)
