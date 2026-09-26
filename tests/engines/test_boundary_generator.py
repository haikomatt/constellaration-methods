"""G1b Phase 2: pure-numpy tests for `boundary_generator.py` -- decay-fit
sign, fixed-seed determinism, low-order dampening (measured on the actual
`perturb_anchor` output, not just by construction), and round-tripping every
generated candidate through a validated `SurfaceRZFourier` parse at 8x8. No
VMEC calls anywhere in this file."""

import numpy as np
import pytest

pytest.importorskip(
    "constellaration",
    reason="needs the pinned constellaration/vmecpp stack; run inside the repo's Docker image",
)
from constellaration.geometry import surface_rz_fourier

from constellaration_methods.engines.boundary_generator import (
    enforce_stellarator_symmetry,
    fit_covariance_prior,
    fit_spectral_prior,
    generate_batch,
    generate_covariance_batch,
    perturb_along_covariance,
    perturb_anchor,
    sample_from_prior,
)


def candidate_dict_to_boundary(
    boundary_dict: dict, max_poloidal_mode: int, max_toroidal_mode: int
) -> surface_rz_fourier.SurfaceRZFourier:
    """Local stand-in for the excluded `task_template.candidate_dict_to_boundary`
    (that module also carries LLM prompt-templating out of this repo's
    scope). This test only needs to confirm every generated candidate
    round-trips into a valid `SurfaceRZFourier` at the given mode budget, so
    it builds one directly here instead of importing the excluded module."""
    r_cos = np.asarray(boundary_dict["r_cos"], dtype=float)
    z_sin = np.asarray(boundary_dict["z_sin"], dtype=float)
    expected_shape = (max_poloidal_mode + 1, 2 * max_toroidal_mode + 1)
    if r_cos.shape != expected_shape or z_sin.shape != expected_shape:
        raise ValueError(
            f"expected shape {expected_shape} for max_poloidal_mode="
            f"{max_poloidal_mode}/max_toroidal_mode={max_toroidal_mode}, "
            f"got r_cos={r_cos.shape} z_sin={z_sin.shape}"
        )
    return surface_rz_fourier.SurfaceRZFourier(
        r_cos=r_cos, z_sin=z_sin, n_field_periods=3, is_stellarator_symmetric=True
    )


def _synthetic_decaying_pool(n_boundaries: int = 6, seed: int = 0) -> list[dict]:
    """Boundaries whose coefficient magnitude decays with (m+|n|) -- a clean
    signal for the decay-fit slope test."""
    rng = np.random.default_rng(seed)
    mpol, ntor = 8, 8
    shape = (mpol + 1, 2 * ntor + 1)
    m = np.arange(shape[0])[:, None] * np.ones((1, shape[1]))
    n = np.abs(np.arange(-ntor, ntor + 1))[None, :] * np.ones((shape[0], 1))
    index_sum = m + n

    pool = []
    for _ in range(n_boundaries):
        base = np.exp(-0.6 * index_sum)
        r_cos = base * (1.0 + 0.05 * rng.normal(size=shape))
        z_sin = base * (1.0 + 0.05 * rng.normal(size=shape))
        pool.append({"r_cos": r_cos, "z_sin": z_sin})
    return pool


def _synthetic_flat_pool(n_boundaries: int = 8, seed: int = 0) -> list[dict]:
    """Boundaries with roughly uniform coefficient variance across modes (no
    spectral decay) -- isolates the low-order dampening effect from the
    decay-fit floor's own (m+|n|)-dependent scale, which the decaying-pool
    fixture above would otherwise confound it with."""
    rng = np.random.default_rng(seed)
    shape = (9, 17)
    return [
        {
            "r_cos": rng.normal(scale=0.05, size=shape).tolist(),
            "z_sin": rng.normal(scale=0.05, size=shape).tolist(),
        }
        for _ in range(n_boundaries)
    ]


def _synthetic_anchor(mpol: int = 4, ntor: int = 4, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    shape = (mpol + 1, 2 * ntor + 1)
    return {
        "r_cos": rng.normal(scale=0.1, size=shape).tolist(),
        "z_sin": rng.normal(scale=0.1, size=shape).tolist(),
        "n_field_periods": 3,
    }


def test_fitted_decay_slope_negative_on_synthetic_decaying_boundaries():
    pool = _synthetic_decaying_pool()
    prior = fit_spectral_prior(pool)
    assert prior.decay_slope < 0


def test_fit_spectral_prior_floors_zero_variance_modes():
    # Every boundary in the pool is identical -> empirical std is exactly
    # zero everywhere, but the decay-fit floor must still leave the prior
    # perturbable in modes where the pool magnitude is nonzero.
    single = _synthetic_decaying_pool(n_boundaries=1)[0]
    pool = [single, single, single]
    prior = fit_spectral_prior(pool)
    assert np.all(prior.r_cos_std > 0)
    assert np.all(prior.z_sin_std > 0)


def test_generate_batch_is_deterministic_for_a_fixed_seed():
    pool = _synthetic_decaying_pool()
    prior = fit_spectral_prior(pool)
    anchors = [_synthetic_anchor(seed=1), _synthetic_anchor(seed=2)]

    batch_a = generate_batch(anchors, prior, n_per_anchor=5, seed=42)
    batch_b = generate_batch(anchors, prior, n_per_anchor=5, seed=42)

    assert batch_a["candidates"] == batch_b["candidates"]


def test_generate_batch_differs_for_a_different_seed():
    pool = _synthetic_decaying_pool()
    prior = fit_spectral_prior(pool)
    anchors = [_synthetic_anchor(seed=1)]

    batch_a = generate_batch(anchors, prior, n_per_anchor=5, seed=1)
    batch_b = generate_batch(anchors, prior, n_per_anchor=5, seed=2)

    assert batch_a["candidates"] != batch_b["candidates"]


def test_generate_batch_respects_perturb_prior_fraction_split():
    pool = _synthetic_decaying_pool()
    prior = fit_spectral_prior(pool)
    anchors = [_synthetic_anchor(seed=1), _synthetic_anchor(seed=2)]

    batch = generate_batch(
        anchors,
        prior,
        n_per_anchor=10,
        fractions={"perturb": 0.8, "prior": 0.2},
        seed=7,
    )
    ops = [c["provenance"]["op"] for c in batch["candidates"]]
    n_perturb = ops.count("perturb")
    n_prior = ops.count("prior")

    assert n_perturb == 20  # 2 anchors x 10 n_per_anchor
    assert n_prior == round(20 * 0.2 / 0.8)


def test_generate_batch_tags_provenance_on_every_candidate():
    pool = _synthetic_decaying_pool()
    prior = fit_spectral_prior(pool)
    anchors = [_synthetic_anchor(seed=1)]
    batch = generate_batch(anchors, prior, n_per_anchor=3, seed=3)

    for c in batch["candidates"]:
        prov = c["provenance"]
        assert prov["op"] in ("perturb", "prior")
        assert "seed" in prov
        if prov["op"] == "perturb":
            assert prov["anchor_id"] is not None
        else:
            assert prov["anchor_id"] is None


def test_low_order_dampening_holds_in_expectation_over_n_draws():
    pool = _synthetic_flat_pool()
    prior = fit_spectral_prior(pool)
    anchor = _synthetic_anchor(seed=1)

    from boundary_grid import pad_to_common_grid

    r0, z0 = pad_to_common_grid(np.asarray(anchor["r_cos"]), np.asarray(anchor["z_sin"]), 8, 8)

    shape = r0.shape
    m = np.arange(shape[0])[:, None] * np.ones((1, shape[1]))
    n = np.abs(np.arange(-8, 9))[None, :] * np.ones((shape[0], 1))
    lo_mask = (m <= 4) & (n <= 4)
    tail_mask = ~lo_mask

    n_draws = 50
    lo_abs_diffs = []
    tail_abs_diffs = []
    for seed in range(n_draws):
        r, z = perturb_anchor(anchor, prior, tail_scale=1.0, low_order_scale=0.1, seed=seed)
        r_diff = np.abs(r - r0)
        z_diff = np.abs(z - z0)
        lo_abs_diffs.append(r_diff[lo_mask].mean() + z_diff[lo_mask].mean())
        tail_abs_diffs.append(r_diff[tail_mask].mean() + z_diff[tail_mask].mean())

    mean_lo = float(np.mean(lo_abs_diffs))
    mean_tail = float(np.mean(tail_abs_diffs))
    assert mean_lo < mean_tail


def test_sample_from_prior_returns_grid_shaped_arrays():
    pool = _synthetic_decaying_pool()
    prior = fit_spectral_prior(pool)
    r, z = sample_from_prior(prior, seed=0)
    assert r.shape == (9, 17)
    assert z.shape == (9, 17)


def test_generated_candidates_round_trip_through_candidate_dict_to_boundary():
    pool = _synthetic_decaying_pool()
    prior = fit_spectral_prior(pool)
    anchors = [
        _synthetic_anchor(mpol=4, ntor=4, seed=1),
        _synthetic_anchor(mpol=8, ntor=8, seed=9),
    ]
    batch = generate_batch(anchors, prior, n_per_anchor=4, seed=11)

    for c in batch["candidates"]:
        # Should not raise for any generated candidate at the 8x8 budget.
        candidate_dict_to_boundary(c["boundary"], max_poloidal_mode=8, max_toroidal_mode=8)


def _synthetic_correlated_pool(n_boundaries: int = 12, seed: int = 5) -> list[dict]:
    """A pool where two specific low-order modes co-vary strongly across
    boundaries (a shared latent factor) while everything else is small
    independent noise -- isolates whether `fit_covariance_prior` actually
    captures joint low-order variation, as opposed to `SpectralPrior`'s
    per-mode-independent view of the same pool."""
    rng = np.random.default_rng(seed)
    shape = (9, 17)
    pool = []
    for _ in range(n_boundaries):
        base = np.full(shape, 0.5)
        factor = rng.normal(scale=0.2)
        # ntor=8 -> column index 16 is n=8; couple (m=1, n=0) and (m=2, n=0)
        # (column index 8) together via the same latent factor.
        base[1, 8] += factor
        base[2, 8] += factor
        base += rng.normal(scale=0.01, size=shape)
        pool.append({"r_cos": base.tolist(), "z_sin": (base * 0.3).tolist()})
    return pool


def test_fit_covariance_prior_rank_capped_at_n_minus_1():
    pool = _synthetic_decaying_pool(n_boundaries=6)
    prior = fit_covariance_prior(pool)
    assert prior.components.shape[0] == 5  # n - 1
    assert prior.variances.shape[0] == 5
    assert prior.components.shape[1] == 9 * 17 * 2


def test_fit_covariance_prior_requires_at_least_two_boundaries():
    with pytest.raises(ValueError):
        fit_covariance_prior(_synthetic_decaying_pool(n_boundaries=1))


def test_fit_covariance_prior_n_components_caps_rank():
    pool = _synthetic_decaying_pool(n_boundaries=8)
    prior = fit_covariance_prior(pool, n_components=3)
    assert prior.components.shape[0] == 3
    assert prior.variances.shape[0] == 3


def test_perturb_along_covariance_is_deterministic_for_a_fixed_seed():
    pool = _synthetic_correlated_pool()
    prior = fit_covariance_prior(pool)
    anchor = _synthetic_anchor(seed=1)

    r_a, z_a = perturb_along_covariance(anchor, prior, scale=1.0, seed=42)
    r_b, z_b = perturb_along_covariance(anchor, prior, scale=1.0, seed=42)

    assert np.array_equal(r_a, r_b)
    assert np.array_equal(z_a, z_b)


def test_perturb_along_covariance_zero_components_returns_anchor_unchanged():
    pool = _synthetic_correlated_pool()
    prior = fit_covariance_prior(pool)
    anchor = _synthetic_anchor(seed=1)

    from boundary_grid import pad_to_common_grid

    r0, z0 = pad_to_common_grid(np.asarray(anchor["r_cos"]), np.asarray(anchor["z_sin"]), 8, 8)
    r, z = perturb_along_covariance(anchor, prior, scale=1.0, n_components=0, seed=1)
    enforce_stellarator_symmetry(r0, z0)
    assert np.allclose(r, r0)
    assert np.allclose(z, z0)


def test_perturb_along_covariance_does_not_damp_the_low_order_block():
    """Contrast with `perturb_anchor`'s intentional low-order damping: on a
    pool where low-order modes genuinely co-vary, covariance-based
    perturbation should move the low-order block by *at least* as much as
    the tail, not systematically less -- unlike `perturb_anchor` on the same
    kind of pool (see `test_low_order_dampening_holds_in_expectation_over_n_draws`)."""
    pool = _synthetic_correlated_pool()
    prior = fit_covariance_prior(pool)
    anchor = _synthetic_anchor(seed=1)

    from boundary_grid import pad_to_common_grid

    r0, z0 = pad_to_common_grid(np.asarray(anchor["r_cos"]), np.asarray(anchor["z_sin"]), 8, 8)
    shape = r0.shape
    m = np.arange(shape[0])[:, None] * np.ones((1, shape[1]))
    n = np.abs(np.arange(-8, 9))[None, :] * np.ones((shape[0], 1))
    lo_mask = (m <= 4) & (n <= 4)
    tail_mask = ~lo_mask

    n_draws = 50
    lo_abs_diffs = []
    tail_abs_diffs = []
    for seed in range(n_draws):
        r, z = perturb_along_covariance(anchor, prior, scale=1.0, seed=seed)
        r_diff = np.abs(r - r0)
        z_diff = np.abs(z - z0)
        lo_abs_diffs.append(r_diff[lo_mask].mean() + z_diff[lo_mask].mean())
        tail_abs_diffs.append(r_diff[tail_mask].mean() + z_diff[tail_mask].mean())

    mean_lo = float(np.mean(lo_abs_diffs))
    mean_tail = float(np.mean(tail_abs_diffs))
    assert mean_lo >= mean_tail


def test_generate_covariance_batch_is_deterministic_for_a_fixed_seed():
    pool = _synthetic_correlated_pool()
    prior = fit_covariance_prior(pool)
    anchors = [_synthetic_anchor(seed=1), _synthetic_anchor(seed=2)]

    batch_a = generate_covariance_batch(anchors, prior, n_per_anchor=4, seed=42)
    batch_b = generate_covariance_batch(anchors, prior, n_per_anchor=4, seed=42)

    assert batch_a["candidates"] == batch_b["candidates"]


def test_generate_covariance_batch_tags_provenance_and_count():
    pool = _synthetic_correlated_pool()
    prior = fit_covariance_prior(pool)
    anchors = [_synthetic_anchor(seed=1), _synthetic_anchor(seed=2)]

    batch = generate_covariance_batch(anchors, prior, n_per_anchor=3, seed=7)
    assert len(batch["candidates"]) == 6  # 2 anchors x 3 n_per_anchor
    for c in batch["candidates"]:
        assert c["provenance"]["op"] == "perturb_covariance"
        assert c["provenance"]["anchor_id"] in (0, 1)


def test_covariance_generated_candidates_round_trip_through_candidate_dict_to_boundary():
    pool = _synthetic_correlated_pool()
    prior = fit_covariance_prior(pool)
    anchors = [
        _synthetic_anchor(mpol=4, ntor=4, seed=1),
        _synthetic_anchor(mpol=8, ntor=8, seed=9),
    ]
    batch = generate_covariance_batch(anchors, prior, n_per_anchor=4, seed=11)

    for c in batch["candidates"]:
        candidate_dict_to_boundary(c["boundary"], max_poloidal_mode=8, max_toroidal_mode=8)
