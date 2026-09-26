"""qi-discovery-loop-plan-v0.1.md Phase A: the NAE-recipe generator for the QI
discovery loop.

The geometrical discovery loop's structured-Fourier generator capped at
davidkh-adjacency (novel-space-discovery-strategy, G3 memo, 2026-07-19): it
perturbs Fourier coefficients off a seed-clustered pool, so its output
diversity is bounded by the diversity of that pool. This generator instead
samples the *NAE recipe space* -- aspect_ratio / max_elongation /
rotational_transform / mirror_ratio / n_field_periods -- so its diversity comes
from genuinely distinct physical configurations: each recipe produces a QI seed
that is physical-by-construction via near-axis expansion. NAE is "the wrong
prior for the geometric problem, the natural coordinate system for QI" (strategy
doc); this is that lever.

Each drawn `NAERecipe` mirrors `qi_discovery.NAESeed`'s fields exactly, at the
*coarse* seed resolution (default max_poloidal_mode=2, max_toroidal_mode=3). The
NAE Phase-0 spike found NAE generation fails VMEC at the 8x8 grid, so the seed
stays low-mode and the discovery loop's *descend* phase expands it into 8x8 by
zero-padding + canonical-scored optimization. The QI feasibility constraints
(max_elongation <= 5, edge_mirror <= 0.2) inform the default ranges: elongation
and mirror straddle their constraint boundaries so descend has feasible-side
room to move into.

This module is PURE (numpy only, no `constellaration` import) so the sampler is
host-testable and fast. The physics validation -- does a sampled batch yield
convergent, spread, QI-relevant seeds whose 8x8 descend beats the 4x4 baseline?
-- is the gated in-container step (`scripts/nae_generator_gate.py`), with its
numeric bars pre-registered separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The continuous recipe dimensions, in a fixed order (the Latin-hypercube
# columns map to these by index).
_CONTINUOUS = (
    "aspect_ratio",
    "max_elongation",
    "rotational_transform",
    "mirror_ratio",
)


@dataclass(frozen=True)
class NAERecipeSpace:
    """The bounds the generator samples over. Continuous dims are `(lo, hi)`
    ranges; `n_field_periods` is a discrete choice set. `max_poloidal_mode` /
    `max_toroidal_mode` are the fixed *seed* resolution (coarse, where NAE
    converges -- see module docstring), NOT the descend/search budget.

    Defaults sit in the validated convergent QI regime (the benchmark's own
    `SimpleToBuildQIStellarator` recipe: aspect ~7-10, elongation ~7, iota
    ~0.5, mirror ~0.2, nfp=1) but widened for diversity, with elongation
    (feasibility bound 5) and mirror (bound 0.2) straddling their constraint
    boundaries so descend has feasible-side room. These defaults are part of
    the Phase A pre-registration -- change them deliberately, not silently."""

    aspect_ratio: tuple[float, float] = (7.0, 10.0)
    max_elongation: tuple[float, float] = (4.0, 8.0)
    rotational_transform: tuple[float, float] = (0.3, 0.7)
    mirror_ratio: tuple[float, float] = (0.1, 0.25)
    n_field_periods: tuple[int, ...] = (1,)
    max_poloidal_mode: int = 2
    max_toroidal_mode: int = 3


@dataclass(frozen=True)
class NAERecipe:
    """One sampled recipe. Fields mirror `qi_discovery.NAESeed` exactly so the
    gate/loop can build a seed via `qi_discovery.NAESeed(**recipe_as_naeseed_kwargs(r))`."""

    label: str
    aspect_ratio: float
    max_elongation: float
    rotational_transform: float
    mirror_ratio: float
    n_field_periods: int
    max_poloidal_mode: int
    max_toroidal_mode: int


def _latin_hypercube(n: int, d: int, rng: np.random.RandomState) -> np.ndarray:
    """`n x d` Latin-hypercube sample in [0, 1): each column places exactly one
    point in each of the `n` equal strata, then permutes independently. Gives
    near-full coverage of every dimension with few samples -- the property the
    davidkh-clustered geometrical generator lacked. Pure w.r.t. `rng` so it is
    deterministic from the seed."""
    result = np.empty((n, d))
    lower_edges = np.arange(n) / n
    for j in range(d):
        points = lower_edges + rng.random_sample(n) / n  # one per stratum
        rng.shuffle(points)
        result[:, j] = points
    return result


def sample_recipes(
    space: NAERecipeSpace,
    n: int,
    *,
    random_seed: int | None = None,
    label_prefix: str = "nae_gen",
) -> list[NAERecipe]:
    """Draw `n` NAE recipes from `space` by Latin-hypercube over the continuous
    dims plus a discrete `n_field_periods` choice. Deterministic from
    `random_seed` (`None` uses OS entropy). Labels are `f"{label_prefix}_{i:03d}"`."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    rng = np.random.RandomState(random_seed)
    unit = _latin_hypercube(n, len(_CONTINUOUS), rng)
    recipes: list[NAERecipe] = []
    for i in range(n):
        values: dict[str, float] = {}
        for j, name in enumerate(_CONTINUOUS):
            lo, hi = getattr(space, name)
            values[name] = float(lo + unit[i, j] * (hi - lo))
        nfp = int(rng.choice(np.asarray(space.n_field_periods)))
        recipes.append(
            NAERecipe(
                label=f"{label_prefix}_{i:03d}",
                n_field_periods=nfp,
                max_poloidal_mode=space.max_poloidal_mode,
                max_toroidal_mode=space.max_toroidal_mode,
                **values,
            )
        )
    return recipes


def _normalized_coords(recipes: list[NAERecipe], space: NAERecipeSpace) -> np.ndarray:
    """`len(recipes) x len(_CONTINUOUS)` array with each continuous dim scaled
    to [0, 1] by its space range (a zero-width range maps to 0), so spread is
    unit-independent."""
    coords = np.empty((len(recipes), len(_CONTINUOUS)))
    for i, r in enumerate(recipes):
        for j, name in enumerate(_CONTINUOUS):
            lo, hi = getattr(space, name)
            coords[i, j] = (getattr(r, name) - lo) / (hi - lo) if hi > lo else 0.0
    return coords


def recipe_spread(recipes: list[NAERecipe], space: NAERecipeSpace) -> dict[str, object]:
    """Recipe-space diversity of a batch: per-dimension coverage (fraction of
    the [0,1]-normalized range spanned) and normalized pairwise distances
    (min/mean Euclidean over the continuous dims). This measures diversity in
    the *recipe* space; the gate additionally measures diversity in physical
    boundary space (rel_dist) on the convergent-feasible subset."""
    n = len(recipes)
    coords = _normalized_coords(recipes, space)
    if n > 1:
        spans = coords.max(axis=0) - coords.min(axis=0)
        dists = [
            float(np.linalg.norm(coords[a] - coords[b])) for a in range(n) for b in range(a + 1, n)
        ]
        min_pairwise = min(dists)
        mean_pairwise = float(np.mean(dists))
    else:
        spans = np.zeros(len(_CONTINUOUS))
        min_pairwise = 0.0
        mean_pairwise = 0.0
    return {
        "n": n,
        "per_dim_coverage": {name: float(spans[j]) for j, name in enumerate(_CONTINUOUS)},
        "min_pairwise_distance": min_pairwise,
        "mean_pairwise_distance": mean_pairwise,
    }


def recipe_as_naeseed_kwargs(recipe: NAERecipe) -> dict[str, object]:
    """The kwargs to build a `qi_discovery.NAESeed` from a recipe. Kept here
    (rather than importing `qi_discovery`, which pulls in `constellaration`) so
    this module stays pure and host-testable; the gate/loop, which already runs
    in-container, does `qi_discovery.NAESeed(**recipe_as_naeseed_kwargs(r))`."""
    return {
        "label": recipe.label,
        "aspect_ratio": recipe.aspect_ratio,
        "max_elongation": recipe.max_elongation,
        "rotational_transform": recipe.rotational_transform,
        "mirror_ratio": recipe.mirror_ratio,
        "n_field_periods": recipe.n_field_periods,
        "max_poloidal_mode": recipe.max_poloidal_mode,
        "max_toroidal_mode": recipe.max_toroidal_mode,
    }
