"""H2 (qi-discovery-stuck-analysis-and-directions.md): TDD for the pure
near-axis search in `src/near_axis_search.py`.

Host-testable: the module imports only numpy + scipy; the physics (pyQIC build
-> observables) is the injected `observe_fn`, faked here. Same split as
`tests/test_nae_generator.py`.
"""

from __future__ import annotations

import numpy as np

from constellaration_methods.engines import near_axis_search as nas

# ---------------------------------------------------------------------------
# config_arrays -- the param -> near-axis mapping (must match generate's from_x)
# ---------------------------------------------------------------------------


def test_config_arrays_matches_generate_from_x():
    nfp = 2
    v = np.array([-0.3, 0.4, -0.05, 0.02, 0.7, 0.18])
    arr = nas.config_arrays(v, nfp)
    r_cos = arr["r_cos"]
    z_sin = arr["z_sin"]
    # r_cos_0 fixed at major_radius, r_cos_1 free, r_cos_2/3 fixed-by-construction
    assert r_cos[0] == 1.0
    assert r_cos[1] == -0.3
    assert r_cos[2] == -1.0 / (1 + 4 * nfp**2)
    assert np.isclose(r_cos[3], -(-0.3) * (nfp**2 + 1) / (9 * nfp**2 + 1))
    assert z_sin[0] == 0.0
    assert list(z_sin[1:]) == [0.4, -0.05, 0.02]
    assert list(arr["B0_cos"]) == [1.0, 0.18]
    assert arr["d_over_curvature"] == 0.7
    assert arr["n_field_periods"] == 2


def test_config_arrays_r_cos_3_tracks_r_cos_1_sign():
    # r_cos_1 negative -> r_cos_3 positive (the Goodman-et-al coupling)
    arr = nas.config_arrays(np.array([-0.5, 0.3, 0.0, 0.0, 1.0, 0.1]), 1)
    assert arr["r_cos"][3] > 0


# ---------------------------------------------------------------------------
# is_viable
# ---------------------------------------------------------------------------


def test_viable_requires_finite_iota_and_real_qi():
    assert nas.is_viable({"iota": -0.9, "qi_residual": 1e-6}, 2) is True  # 0.45 >= 0.25


def test_not_viable_when_iota_proxy_below_bound():
    # |iota|=0.4 at nfp=2 -> proxy 0.2 < 0.25
    assert nas.is_viable({"iota": 0.4, "qi_residual": 1e-6}, 2) is False


def test_not_viable_when_qi_residual_too_large():
    assert nas.is_viable({"iota": -0.9, "qi_residual": 5.0}, 1) is False


def test_not_viable_on_missing_or_nonfinite():
    assert nas.is_viable({"iota": np.nan, "qi_residual": 1e-6}, 1) is False
    assert nas.is_viable({"qi_residual": 1e-6}, 1) is False
    assert nas.is_viable({"iota": -0.9, "qi_residual": float("inf")}, 1) is False


def test_viable_counts_iota_by_magnitude():
    # negative iota is feasible (problem takes |iota|)
    assert nas.is_viable({"iota": -0.6, "qi_residual": 1e-6}, 1) is True


# ---------------------------------------------------------------------------
# analytic_objective
# ---------------------------------------------------------------------------


def test_objective_infinite_for_failed_observation():
    assert nas.analytic_objective({"qi_residual": float("inf")}, 1) == float("inf")


def test_objective_rewards_low_qi_and_penalises_low_iota():
    # A viable config (finite iota, low elong, low qi) beats one with tiny iota.
    good = nas.analytic_objective({"iota": -0.9, "max_elongation": 3.0, "qi_residual": 1e-6}, 1)
    bad = nas.analytic_objective({"iota": -0.02, "max_elongation": 3.0, "qi_residual": 1e-6}, 1)
    assert good < bad
    assert good < 1e-3  # no active barriers -> just the tiny qi residual


def test_objective_penalises_high_elongation():
    lo = nas.analytic_objective({"iota": -0.9, "max_elongation": 4.0, "qi_residual": 1e-6}, 1)
    hi = nas.analytic_objective({"iota": -0.9, "max_elongation": 12.0, "qi_residual": 1e-6}, 1)
    assert hi > lo


# ---------------------------------------------------------------------------
# dedup + select_diverse
# ---------------------------------------------------------------------------


def _cfg(label, nfp, r1, z1, z2=0.0, z3=0.0, d=1.0, mirror=0.1):
    return nas.NearAxisConfigVector(
        label=label,
        n_field_periods=nfp,
        r_cos_1=r1,
        z_sin_1=z1,
        z_sin_2=z2,
        z_sin_3=z3,
        d_over_curvature=d,
        mirror=mirror,
    )


def test_dedup_collapses_near_identical_same_nfp():
    space = nas.NearAxisSpace()
    a = _cfg("a", 2, -0.3, 0.4)
    b = _cfg("b", 2, -0.3001, 0.4001)  # essentially a
    c = _cfg("c", 2, -0.9, 0.9)  # far
    kept = nas.dedup([a, b, c], space, tol=0.02)
    assert [k.label for k in kept] == ["a", "c"]


def test_dedup_keeps_same_vector_different_nfp():
    space = nas.NearAxisSpace()
    a = _cfg("a", 1, -0.3, 0.4)
    b = _cfg("b", 2, -0.3, 0.4)  # same vector, different nfp -> distinct config
    kept = nas.dedup([a, b], space, tol=0.02)
    assert len(kept) == 2


def test_select_diverse_returns_all_when_k_exceeds():
    space = nas.NearAxisSpace()
    cfgs = [_cfg("a", 1, -0.3, 0.4), _cfg("b", 2, -0.9, 0.9)]
    assert len(nas.select_diverse(cfgs, space, 5)) == 2


def test_select_diverse_prefers_spread():
    space = nas.NearAxisSpace()
    # 3 clustered + 1 far; choosing 2 should take one cluster member + the far one
    cfgs = [
        _cfg("c0", 2, -0.30, 0.40),
        _cfg("c1", 2, -0.31, 0.41),
        _cfg("c2", 2, -0.29, 0.39),
        _cfg("far", 2, -1.1, 1.1),
    ]
    chosen = nas.select_diverse(cfgs, space, 2)
    labels = {c.label for c in chosen}
    assert "c0" in labels  # first is the seed
    assert "far" in labels  # max-min distance picks the far one second


# ---------------------------------------------------------------------------
# multi_start_search -- the driver, with an injected analytic observe_fn
# ---------------------------------------------------------------------------


def _quadratic_observe(target=None):
    """A fake observe_fn whose analytic optimum is a fixed, viable config: a
    single quadratic bowl in the FULL 6-vector (so every search dimension is
    constrained, the way the real qi_residual is), minimised at `target`. No
    physics -- exercises the driver's search/collect/dedup logic."""
    if target is None:
        target = np.array([-0.3, 0.5, 0.0, 0.0, 1.0, 0.1])
    target = np.asarray(target, dtype=float)

    def observe(vector, nfp):
        dist2 = float(np.sum((np.asarray(vector, dtype=float) - target) ** 2))
        return {
            "iota": -0.9,  # viable magnitude at any nfp<=3
            "max_elongation": 3.0,
            "qi_residual": 1e-6 + dist2,  # minimised at the target
        }

    return observe


def test_latin_hypercube_starts_shape_and_bounds():
    space = nas.NearAxisSpace()
    rng = np.random.RandomState(0)
    starts = nas.latin_hypercube_starts(space, 12, rng)
    assert starts.shape == (12, len(nas.CONTINUOUS))
    bounds = space.bounds()
    for j, (lo, hi) in enumerate(bounds):
        assert (starts[:, j] >= lo).all()
        assert (starts[:, j] <= hi).all()


def test_latin_hypercube_starts_deterministic_from_rng_state():
    space = nas.NearAxisSpace()
    a = nas.latin_hypercube_starts(space, 10, np.random.RandomState(42))
    b = nas.latin_hypercube_starts(space, 10, np.random.RandomState(42))
    assert np.allclose(a, b)


def test_multi_start_finds_viable_configs():
    space = nas.NearAxisSpace(n_field_periods=(1, 2))
    res = nas.multi_start_search(_quadratic_observe(), space, n_starts_per_nfp=8, random_seed=0)
    assert res.n_starts == 16  # 8 per nfp x 2 nfp
    assert len(res.viable) >= 1
    # best config's qi should be near the bowl minimum
    best = res.viable[0]
    assert res.objective_by_label[best.label] < 1e-2


def test_multi_start_deterministic_from_seed():
    space = nas.NearAxisSpace(n_field_periods=(2,))
    a = nas.multi_start_search(_quadratic_observe(), space, n_starts_per_nfp=6, random_seed=7)
    b = nas.multi_start_search(_quadratic_observe(), space, n_starts_per_nfp=6, random_seed=7)
    assert [c.vector.tolist() for c in a.viable] == [c.vector.tolist() for c in b.viable]


def test_multi_start_dedups_common_basin():
    # All starts fall into one basin -> after dedup, few distinct viable configs.
    space = nas.NearAxisSpace(n_field_periods=(2,))
    res = nas.multi_start_search(_quadratic_observe(), space, n_starts_per_nfp=20, random_seed=1)
    assert res.n_viable_raw >= len(res.viable)  # dedup never grows the pool
    assert len(res.viable) <= 3  # one basin -> collapses hard


def test_multi_start_survives_failed_observations():
    def flaky(vector, nfp):
        if float(vector[0]) < -0.6:
            return {"qi_residual": float("inf")}  # "solve failed" region
        return {"iota": -0.9, "max_elongation": 3.0, "qi_residual": 1e-6}

    space = nas.NearAxisSpace(n_field_periods=(1,))
    res = nas.multi_start_search(flaky, space, n_starts_per_nfp=10, random_seed=3)
    # driver must not crash and should still collect viable configs
    assert len(res.viable) >= 1


# ---------------------------------------------------------------------------
# p3_wide_space -- E20 optional widened box; the default box must stay
# byte-identical (regression ratchet: `NearAxisSpace()`'s pre-registered
# bounds are the H2 registration itself, never touched by a later widening)
# ---------------------------------------------------------------------------


def test_default_space_bounds_unchanged_by_the_p3_widening():
    space = nas.NearAxisSpace()
    assert space.r_cos_1 == (-1.2, 0.0)
    assert space.z_sin_1 == (0.0, 1.2)
    assert space.z_sin_2 == (-0.35, 0.35)
    assert space.z_sin_3 == (-0.25, 0.25)
    assert space.d_over_curvature == (0.1, 2.0)
    assert space.mirror == (0.05, 0.20)
    assert space.n_field_periods == (1, 2, 3, 4)
    assert space.major_radius == 1.0


def test_p3_wide_space_only_widens_r_cos_1_upper_bound():
    default = nas.NearAxisSpace()
    wide = nas.p3_wide_space()
    assert wide.r_cos_1 == (-1.2, 0.4)
    # every other continuous bound is inherited unchanged from the default
    assert wide.z_sin_1 == default.z_sin_1
    assert wide.z_sin_2 == default.z_sin_2
    assert wide.z_sin_3 == default.z_sin_3
    assert wide.d_over_curvature == default.d_over_curvature
    assert wide.mirror == default.mirror
    assert wide.major_radius == default.major_radius


def test_p3_wide_space_restricts_nfp_to_3_and_4():
    assert nas.p3_wide_space().n_field_periods == (3, 4)


def test_p3_wide_space_bounds_is_a_superset_of_default_on_r_cos_1():
    default_lo, default_hi = nas.NearAxisSpace().r_cos_1
    wide_lo, wide_hi = nas.p3_wide_space().r_cos_1
    assert wide_lo == default_lo  # lower edge not widened
    assert wide_hi > default_hi  # upper edge is the widening
