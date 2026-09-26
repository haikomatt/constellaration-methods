"""E6: promote a 4x4 candidate to 8x8 by embedding its boundary with the new
modes at zero. The shape is unchanged, so the canonical residuals must be
identical (verified live by the repeatability gate)."""

import numpy as np

from constellaration_methods.restoration.p3_promote_candidate import pad_fourier


def test_pad_fourier_embeds_and_zero_fills():
    # 4x4 arrays are (mpol+1) x (2*ntor+1) = 5 x 9, centred on n=0 at column ntor
    r4 = np.arange(45, dtype=float).reshape(5, 9)
    r8 = pad_fourier(r4, mpol_from=4, ntor_from=4, mpol_to=8, ntor_to=8)
    assert r8.shape == (9, 17)
    # old (m, n) entries land at the same (m, n): column offset = ntor_to - ntor_from
    assert np.allclose(r8[:5, 4:13], r4)
    assert (
        np.allclose(r8[5:, :], 0.0)
        and np.allclose(r8[:5, :4], 0.0)
        and np.allclose(r8[:5, 13:], 0.0)
    )


def test_coefficient_space_multiplier_gives_new_modes_the_old_modes_reach():
    from constellaration_methods.restoration.p3_promote_candidate import (
        coefficient_space_multiplier,
    )

    scale = np.array([0.05, 0.04, 0.06, 0.001, 0.0009])  # 3 old, 2 new
    old = np.array([True, True, True, False, False])
    mult = coefficient_space_multiplier(scale, old)
    s_ref = np.median(scale[old])
    # post-Green change (documented): the pre-registration's control condition
    # is that old modes keep EXACTLY E6's box, so mult is 1 there, not s_ref/scale
    assert np.allclose(mult[old], 1.0)
    assert np.allclose(mult[~old], s_ref / scale[~old])
    assert mult[3] > 40 and mult[4] > 40  # new modes get the median old coefficient reach


def test_new_mode_energy_fraction_is_measured_in_coefficient_units():
    from constellaration_methods.restoration.p3_promote_candidate import new_mode_energy_fraction

    scale = np.array([0.1, 0.1, 0.001, 0.001])
    old = np.array([True, True, False, False])
    # equal scaled steps: the new modes move 100x less in coefficient units
    d = np.array([1.0, 1.0, 1.0, 1.0])
    f = new_mode_energy_fraction(d, scale, old)
    assert 0 < f < 1e-3
    # a step only in new modes has fraction 1; a zero step is reported as 0
    assert new_mode_energy_fraction(np.array([0, 0, 1.0, 0]), scale, old) == 1.0
    assert new_mode_energy_fraction(np.zeros(4), scale, old) == 0.0


def test_old_mode_mask_matches_the_source_budget_coordinate_count():
    # needs the engine (jax): the old-mode coordinates of an 8x8 promoted
    # boundary are exactly the coordinates of the same boundary at 4x4
    pytest = __import__("pytest")
    pytest.importorskip("jax")
    from constellaration_methods.engines.alm_ngopt_optimizer import masked_coefficients_and_scale
    from constellaration_methods.restoration.p3_promote_candidate import old_mode_mask

    mpol, ntor, nfp = 8, 8, 3
    r_cos = np.zeros((mpol + 1, 2 * ntor + 1))
    z_sin = np.zeros_like(r_cos)
    r_cos[0, ntor] = 1.0
    r_cos[1, ntor] = 0.15
    z_sin[1, ntor] = 0.15
    r_cos[1, ntor + 1] = 0.02
    z_sin[1, ntor + 1] = 0.02
    bd = {
        "r_cos": r_cos.tolist(),
        "z_sin": z_sin.tolist(),
        "r_sin": np.zeros_like(r_cos).tolist(),
        "z_cos": np.zeros_like(z_sin).tolist(),
        "n_field_periods": nfp,
        "is_stellarator_symmetric": True,
    }
    cand = {"boundary_dict": bd, "mpol": mpol, "ntor": ntor, "nfp": nfp, "scaling": 1.0}
    old, scale = old_mode_mask(cand, 4, 4)
    # the engine refuses an 8x8-shaped array under a 4x4 budget, so compare
    # against the same shape written at 4x4 (all its content is within 4x4)
    bd44 = dict(bd)
    for k in ("r_cos", "z_sin", "r_sin", "z_cos"):
        bd44[k] = np.asarray(bd[k])[:5, ntor - 4 : ntor + 5].tolist()
    _, _, coeffs44, _, _ = masked_coefficients_and_scale(
        bd44, max_poloidal_mode=4, max_toroidal_mode=4, n_field_periods=nfp
    )
    assert old.dtype == bool and len(old) == len(scale)
    assert int(old.sum()) == len(np.asarray(coeffs44))
    assert 0 < int(old.sum()) < len(old)


def test_candidate_from_boundary_dict_round_trips_through_the_engine():
    # E11: start a restoration from a raw boundary file (no prior run): x must
    # be the engine's scaled coordinates and reproduce the boundary exactly
    pytest = __import__("pytest")
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from constellaration_methods.engines.alm_ngopt_optimizer import masked_coefficients_and_scale
    from constellaration_methods.restoration.p3_promote_candidate import (
        candidate_from_boundary_dict,
    )

    mpol, ntor, nfp = 3, 3, 3
    r_cos = np.zeros((mpol + 1, 2 * ntor + 1))
    z_sin = np.zeros_like(r_cos)
    r_cos[0, ntor] = 1.0
    r_cos[1, ntor] = 0.15
    z_sin[1, ntor] = 0.15
    r_cos[2, ntor + 1] = 0.01
    bd = {
        "r_cos": r_cos.tolist(),
        "z_sin": z_sin.tolist(),
        "r_sin": np.zeros_like(r_cos).tolist(),
        "z_cos": np.zeros_like(z_sin).tolist(),
        "n_field_periods": nfp,
        "is_stellarator_symmetric": True,
    }
    cand = candidate_from_boundary_dict(bd, r=[0.1, 0.2, 0.3, 0.4, -0.5], raw={"a": 1.0})
    assert cand["mpol"] == mpol and cand["ntor"] == ntor and cand["nfp"] == nfp
    _, _, _, unravel, scale = masked_coefficients_and_scale(
        bd, max_poloidal_mode=mpol, max_toroidal_mode=ntor, n_field_periods=nfp
    )
    back = unravel(jnp.asarray(cand["x"]) * jnp.asarray(scale))
    assert np.allclose(np.asarray(back.r_cos), r_cos) and np.allclose(np.asarray(back.z_sin), z_sin)
    assert cand["r"] == [0.1, 0.2, 0.3, 0.4, -0.5] and cand["scaling"] == 1.0


def test_boundary_dict_of_candidate_round_trips_with_candidate_from_boundary():
    # close-out: export a chain best (x in scaled coordinates) as a leaderboard-format
    # boundary JSON; re-importing it must give the same x
    pytest = __import__("pytest")
    pytest.importorskip("jax")
    from constellaration_methods.restoration.p3_promote_candidate import (
        boundary_dict_of_candidate,
        candidate_from_boundary_dict,
    )

    mpol, ntor, nfp = 3, 3, 3
    r_cos = np.zeros((mpol + 1, 2 * ntor + 1))
    z_sin = np.zeros_like(r_cos)
    r_cos[0, ntor] = 1.0
    r_cos[1, ntor] = 0.15
    z_sin[1, ntor] = 0.15
    r_cos[2, ntor + 1] = 0.01
    bd = {
        "r_cos": r_cos.tolist(),
        "z_sin": z_sin.tolist(),
        "r_sin": np.zeros_like(r_cos).tolist(),
        "z_cos": np.zeros_like(z_sin).tolist(),
        "n_field_periods": nfp,
        "is_stellarator_symmetric": True,
    }
    cand = candidate_from_boundary_dict(bd, r=[0] * 5)
    x = np.asarray(cand["x"])
    x[3] += 0.01  # move one scaled coordinate
    cand["x"] = x.tolist()
    out = boundary_dict_of_candidate(cand)
    assert set(out) >= {
        "r_cos",
        "z_sin",
        "r_sin",
        "z_cos",
        "n_field_periods",
        "is_stellarator_symmetric",
    }
    back = candidate_from_boundary_dict(out, r=[0] * 5)
    assert np.allclose(back["x"], cand["x"], atol=1e-12)
    assert not np.allclose(np.asarray(out["r_cos"]), r_cos)  # the move is visible in the boundary


def test_reflect_toroidally_is_an_involution_and_keeps_the_symmetry_convention():
    # close-out: phi -> -phi maps a stellarator-symmetric boundary to its
    # mirror image (iota changes sign, every other benchmark metric is
    # unchanged, verified in results/p3-close-out); m=0, n<0 must stay zero
    from constellaration_methods.restoration.p3_promote_candidate import reflect_toroidally

    rng = np.random.default_rng(0)
    mpol, ntor = 2, 2
    r_cos = rng.normal(size=(mpol + 1, 2 * ntor + 1))
    z_sin = rng.normal(size=(mpol + 1, 2 * ntor + 1))
    r_cos[0, :ntor] = 0.0
    z_sin[0, :ntor] = 0.0  # the convention: m=0 has no n<0 entries
    bd = {
        "r_cos": r_cos.tolist(),
        "z_sin": z_sin.tolist(),
        "r_sin": None,
        "z_cos": None,
        "n_field_periods": 3,
        "is_stellarator_symmetric": True,
    }
    out = reflect_toroidally(bd)
    R, Z = np.asarray(out["r_cos"]), np.asarray(out["z_sin"])
    assert np.all(R[0, :ntor] == 0.0) and np.all(Z[0, :ntor] == 0.0)
    assert np.allclose(R[1:], r_cos[1:, ::-1]) and np.allclose(Z[1:], z_sin[1:, ::-1])
    assert np.allclose(Z[0], -z_sin[0]) and np.allclose(R[0], r_cos[0])
    back = reflect_toroidally(out)
    assert np.allclose(back["r_cos"], r_cos) and np.allclose(back["z_sin"], z_sin)
