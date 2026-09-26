"""Tests for constellaration_methods.analysis.boundary_symmetry. Moved out
of test_qi_family.py (window_4x4 / symmetry_variants / align_to only --
the family-campaign seed-construction tests in that file are out of scope
for this repository, see boundary_symmetry.py's module docstring).
"""

import numpy as np
import pytest

from constellaration_methods.analysis.boundary_symmetry import (
    align_to,
    symmetry_variants,
    window_4x4,
)


def _boundary(r4, z4, pad_to=None):
    """A boundary dict whose 4x4 window equals (r4, z4); optionally embedded
    in a larger (pad_to x pad_to) mode budget like the 8x15 submissions."""
    if pad_to is None:
        return {"r_cos": r4.tolist(), "z_sin": z4.tolist(), "n_field_periods": 3}
    m_rows = pad_to + 1
    n_cols = 2 * pad_to + 1
    r = np.zeros((m_rows, n_cols))
    z = np.zeros((m_rows, n_cols))
    r[:5, pad_to - 4 : pad_to + 5] = r4
    z[:5, pad_to - 4 : pad_to + 5] = z4
    return {"r_cos": r.tolist(), "z_sin": z.tolist(), "n_field_periods": 3}


def _sample(seed=0):
    rng = np.random.default_rng(seed)
    r4 = rng.normal(scale=0.05, size=(5, 9))
    z4 = rng.normal(scale=0.05, size=(5, 9))
    r4[0, 4] = 1.0  # major radius
    z4[0, 4] = 0.0
    return r4, z4


def test_window_4x4_extracts_the_low_mode_block_from_a_larger_budget():
    r4, z4 = _sample()
    big = _boundary(r4, z4, pad_to=7)  # 8x15, like the top submissions
    wr, wz = window_4x4(big)
    np.testing.assert_allclose(wr, r4)
    np.testing.assert_allclose(wz, z4)


def test_symmetry_variants_include_identity_and_preserve_shape():
    r4, z4 = _sample()
    variants = symmetry_variants(r4, z4)
    assert len(variants) == 8
    assert any(np.allclose(r, r4) and np.allclose(z, z4) for r, z, _ in variants)


def test_align_recovers_a_symmetry_transformed_copy():
    """A design transformed by any of the discrete symmetry operations must
    align back onto the original at ~zero distance -- otherwise family
    distances are contaminated by gauge, not physics."""
    r4, z4 = _sample()
    for r_var, z_var, _tag in symmetry_variants(r4, z4):
        d, _r, _z, _tag2 = align_to(r4, z4, r_var, z_var)
        assert d == pytest.approx(0.0, abs=1e-12)
