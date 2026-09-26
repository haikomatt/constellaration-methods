"""qi-discovery-loop-plan-v0.1.md Phase A: TDD for `src/nae_generator.py`, the
NAE-recipe generator for the QI discovery loop.

Pure and deterministic -- no physics, no `constellaration` import. This suite
pins the recipe-sampling logic (Latin-hypercube coverage over the NAE recipe
space, reproducibility from a seed, range/choice respect) and the recipe-space
spread metric. The physics validation (does sampling yield convergent, spread,
QI-relevant seeds whose 8x8 descend beats the 4x4 baseline?) is the gated
in-container step in `scripts/nae_generator_gate.py`, pre-registered separately
-- NOT a unit test.
"""

from __future__ import annotations

import pytest

from constellaration_methods.engines import nae_generator as ng


def _space(**overrides) -> ng.NAERecipeSpace:
    return ng.NAERecipeSpace(**overrides)


# ---------------------------------------------------------------------------
# sample_recipes: draw N recipes from the recipe space
# ---------------------------------------------------------------------------


def test_sample_recipes_returns_n_recipes_within_the_space_ranges():
    space = _space()
    recipes = ng.sample_recipes(space, 24, random_seed=0)

    assert len(recipes) == 24
    for r in recipes:
        assert space.aspect_ratio[0] <= r.aspect_ratio <= space.aspect_ratio[1]
        assert space.max_elongation[0] <= r.max_elongation <= space.max_elongation[1]
        assert (
            space.rotational_transform[0] <= r.rotational_transform <= space.rotational_transform[1]
        )
        assert space.mirror_ratio[0] <= r.mirror_ratio <= space.mirror_ratio[1]
        assert r.n_field_periods in space.n_field_periods
        # seeds stay at the coarse resolution where NAE converges
        assert r.max_poloidal_mode == space.max_poloidal_mode
        assert r.max_toroidal_mode == space.max_toroidal_mode


def test_sample_recipes_labels_are_unique_and_ordered():
    recipes = ng.sample_recipes(_space(), 5, random_seed=1, label_prefix="gen")
    labels = [r.label for r in recipes]
    assert labels == ["gen_000", "gen_001", "gen_002", "gen_003", "gen_004"]
    assert len(set(labels)) == 5


def test_sample_recipes_is_reproducible_from_the_seed():
    a = ng.sample_recipes(_space(), 16, random_seed=42)
    b = ng.sample_recipes(_space(), 16, random_seed=42)
    assert a == b


def test_sample_recipes_differs_across_seeds():
    a = ng.sample_recipes(_space(), 16, random_seed=1)
    b = ng.sample_recipes(_space(), 16, random_seed=2)
    assert a != b


def test_sample_recipes_rejects_non_positive_n():
    with pytest.raises(ValueError, match="n must be >= 1"):
        ng.sample_recipes(_space(), 0, random_seed=0)


def test_sample_recipes_respects_multiple_nfp_choices():
    space = _space(n_field_periods=(1, 2, 3))
    recipes = ng.sample_recipes(space, 60, random_seed=7)
    used = {r.n_field_periods for r in recipes}
    assert used <= {1, 2, 3}
    assert len(used) >= 2  # 60 draws over 3 choices should hit at least two


def test_latin_hypercube_sampling_covers_each_dimension():
    # LHS guarantees one sample per 1/n stratum per dim, so with n draws the
    # spanned range of every continuous dim is near the full [lo, hi] -- this
    # is the coverage the geometrical generator lacked (it clustered near a
    # seed). Coverage of a uniform-random sampler would be looser.
    space = _space()
    recipes = ng.sample_recipes(space, 20, random_seed=3)
    spread = ng.recipe_spread(recipes, space)
    for dim, coverage in spread["per_dim_coverage"].items():
        assert coverage > 0.8, f"{dim} coverage {coverage} too low for LHS"


# ---------------------------------------------------------------------------
# recipe_spread: recipe-space diversity metric
# ---------------------------------------------------------------------------


def test_recipe_spread_reports_coverage_and_pairwise_distances():
    space = _space()
    recipes = ng.sample_recipes(space, 12, random_seed=5)
    spread = ng.recipe_spread(recipes, space)

    assert spread["n"] == 12
    assert set(spread["per_dim_coverage"]) == {
        "aspect_ratio",
        "max_elongation",
        "rotational_transform",
        "mirror_ratio",
    }
    assert 0.0 < spread["min_pairwise_distance"] <= spread["mean_pairwise_distance"]


def test_recipe_spread_of_a_single_recipe_is_zero():
    space = _space()
    recipes = ng.sample_recipes(space, 1, random_seed=0)
    spread = ng.recipe_spread(recipes, space)
    assert spread["n"] == 1
    assert spread["min_pairwise_distance"] == 0.0
    assert spread["mean_pairwise_distance"] == 0.0


def test_recipe_spread_normalizes_by_the_space_ranges():
    # two recipes at opposite corners of every continuous dim -> normalized
    # distance = sqrt(4) = 2.0 exactly, independent of the raw units.
    space = _space()
    lo = ng.NAERecipe(
        label="lo",
        aspect_ratio=space.aspect_ratio[0],
        max_elongation=space.max_elongation[0],
        rotational_transform=space.rotational_transform[0],
        mirror_ratio=space.mirror_ratio[0],
        n_field_periods=1,
        max_poloidal_mode=space.max_poloidal_mode,
        max_toroidal_mode=space.max_toroidal_mode,
    )
    hi = ng.NAERecipe(
        label="hi",
        aspect_ratio=space.aspect_ratio[1],
        max_elongation=space.max_elongation[1],
        rotational_transform=space.rotational_transform[1],
        mirror_ratio=space.mirror_ratio[1],
        n_field_periods=1,
        max_poloidal_mode=space.max_poloidal_mode,
        max_toroidal_mode=space.max_toroidal_mode,
    )
    spread = ng.recipe_spread([lo, hi], space)
    assert spread["min_pairwise_distance"] == pytest.approx(2.0)
    for coverage in spread["per_dim_coverage"].values():
        assert coverage == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# recipe_as_naeseed_kwargs: the bridge to qi_discovery.NAESeed (no import here)
# ---------------------------------------------------------------------------


def test_recipe_as_naeseed_kwargs_has_exactly_the_naeseed_fields():
    recipes = ng.sample_recipes(_space(), 1, random_seed=0)
    kwargs = ng.recipe_as_naeseed_kwargs(recipes[0])
    assert set(kwargs) == {
        "label",
        "aspect_ratio",
        "max_elongation",
        "rotational_transform",
        "mirror_ratio",
        "n_field_periods",
        "max_poloidal_mode",
        "max_toroidal_mode",
    }
    assert kwargs["label"] == recipes[0].label
    assert kwargs["aspect_ratio"] == recipes[0].aspect_ratio
