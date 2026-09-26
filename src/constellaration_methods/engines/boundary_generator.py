"""G1b Phase 2: structured 8x8 boundary generator -- the surrogate-screened
discovery loop's proposal mechanism (Phase 3 consumes `generate_batch`'s
output as its screen-phase input).

Per `task_template.py`'s module docstring, anything that builds/handles
boundaries must take mode budget as an explicit parameter -- everything here
is pinned to the (8, 8) common grid (`boundary_grid.pad_to_common_grid`'s own
default), never assuming the LLM-prompt path's 4x4 pin.

`fit_spectral_prior` fits a per-mode empirical std over a "known-good" 8x8
pool (leaderboard boundaries + G0 canonical_feasible confirmations), plus a
parametric log-linear decay `log|c| ~ a + slope*(m+|n|)` (slope negative for
a decaying spectrum) used as a floor wherever the empirical std collapses to
~0 -- some high-order modes have zero variance across the pool simply because
G0's own descent barely moved them, not because they are physically
unperturbable, so a hard floor keeps those modes explorable.

`perturb_anchor` is the primary generation op: pad a confirmed anchor
(currently 4x4-confirmed own designs; 8x8 archive elites feed in from Phase
3) up to (9, 17) and add prior-scaled Gaussian noise, damped in the
m<=4,|n|<=4 block so the feasibility-defining low-order content is preserved
while the tail is perturbed freely.

`sample_from_prior` draws fresh boundaries with no anchor at all (a small
exploration fraction of every batch).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from constellaration_methods.engines.boundary_grid import pad_to_common_grid

TARGET_MPOL = 8
TARGET_NTOR = 8
_GRID_SHAPE = (TARGET_MPOL + 1, 2 * TARGET_NTOR + 1)

DEFAULT_LOW_ORDER_M_CUT = 4
DEFAULT_LOW_ORDER_N_CUT = 4

# Floor applied to the decay-fit magnitude before treating it as a std floor
# -- the fit is on raw coefficient magnitude across the pool, which runs
# systematically larger than a per-mode std would; without damping, the floor
# would swamp the empirical std almost everywhere instead of only filling in
# the near-zero modes it's meant for.
_DECAY_FLOOR_SCALE = 0.25
_LOG_EPS = 1e-12


@dataclasses.dataclass(frozen=True)
class SpectralPrior:
    """Per-mode Gaussian prior over the (9, 17) common grid, fit from a pool
    of known-good boundaries."""

    target_mpol: int
    target_ntor: int
    r_cos_mean: np.ndarray
    z_sin_mean: np.ndarray
    r_cos_std: np.ndarray
    z_sin_std: np.ndarray
    decay_intercept: float
    decay_slope: float
    n_boundaries: int


def _mode_index_sum(shape: tuple[int, int]) -> np.ndarray:
    """(m + |n|) for every cell of a (mpol+1, 2*ntor+1) grid."""
    mpol1, ncols = shape
    ntor = (ncols - 1) // 2
    m = np.arange(mpol1)[:, None] * np.ones((1, ncols))
    n = np.abs(np.arange(-ntor, ntor + 1))[None, :] * np.ones((mpol1, 1))
    return m + n


def _low_order_mask(shape: tuple[int, int], m_cut: int, n_cut: int) -> np.ndarray:
    mpol1, ncols = shape
    ntor = (ncols - 1) // 2
    m = np.arange(mpol1)[:, None] * np.ones((1, ncols))
    n = np.abs(np.arange(-ntor, ntor + 1))[None, :] * np.ones((mpol1, 1))
    return (m <= m_cut) & (n <= n_cut)


def enforce_stellarator_symmetry(r_cos: np.ndarray, z_sin: np.ndarray) -> None:
    """Zeroes the entries `SurfaceRZFourier` requires to be exactly 0.0 for a
    stellarator-symmetric surface (`r_cos` for m=0,n<0; `z_sin` for
    m=0,n<=0) -- in place. Gaussian perturbation of every mode would
    otherwise put nonzero noise into these fixed-zero slots and every
    generated candidate would fail to parse."""
    ntor = (r_cos.shape[1] - 1) // 2
    r_cos[0, :ntor] = 0.0
    z_sin[0, : ntor + 1] = 0.0


def _fit_decay(magnitudes: np.ndarray, index_sum: np.ndarray) -> tuple[float, float]:
    """Fits log|c| ~ intercept + slope*(m+|n|) by ordinary least squares over
    every (mode, boundary) pair in the pool with magnitude above `_LOG_EPS`
    (exact zeros are pad artifacts, not signal). Returns (intercept, slope);
    slope is negative for a decaying spectrum."""
    mask = magnitudes > _LOG_EPS
    if mask.sum() < 2:
        # Degenerate pool (e.g. an all-zero synthetic fixture): no decay
        # signal available, floor contributes nothing.
        return -np.inf, 0.0
    x = index_sum[mask]
    y = np.log(magnitudes[mask])
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), float(slope)


def fit_spectral_prior(
    boundaries: list[dict[str, Any]],
    target_mpol: int = TARGET_MPOL,
    target_ntor: int = TARGET_NTOR,
) -> SpectralPrior:
    """Fits a `SpectralPrior` from a pool of known-good boundaries (each a
    dict with `r_cos`/`z_sin` keys, any shape within the target grid budget).
    """
    if not boundaries:
        raise ValueError("fit_spectral_prior requires at least one boundary")

    r_stack = []
    z_stack = []
    for b in boundaries:
        r = np.asarray(b["r_cos"], dtype=float)
        z = np.asarray(b["z_sin"], dtype=float)
        rp, zp = pad_to_common_grid(r, z, target_mpol, target_ntor)
        r_stack.append(rp)
        z_stack.append(zp)
    r_stack = np.stack(r_stack)
    z_stack = np.stack(z_stack)

    r_mean = r_stack.mean(axis=0)
    z_mean = z_stack.mean(axis=0)
    r_std = r_stack.std(axis=0)
    z_std = z_stack.std(axis=0)

    shape = (target_mpol + 1, 2 * target_ntor + 1)
    index_sum = _mode_index_sum(shape)
    mag_stack = np.sqrt(r_stack**2 + z_stack**2)
    index_sum_tiled = np.broadcast_to(index_sum, mag_stack.shape)
    intercept, slope = _fit_decay(mag_stack.ravel(), index_sum_tiled.ravel())

    floor = _DECAY_FLOOR_SCALE * np.exp(intercept + slope * index_sum)
    floor = np.where(np.isfinite(floor), floor, 0.0)
    r_std = np.maximum(r_std, floor)
    z_std = np.maximum(z_std, floor)

    return SpectralPrior(
        target_mpol=target_mpol,
        target_ntor=target_ntor,
        r_cos_mean=r_mean,
        z_sin_mean=z_mean,
        r_cos_std=r_std,
        z_sin_std=z_std,
        decay_intercept=intercept,
        decay_slope=slope,
        n_boundaries=len(boundaries),
    )


def perturb_anchor(
    anchor: dict[str, Any],
    prior: SpectralPrior,
    tail_scale: float,
    low_order_scale: float = 0.1,
    low_order_m_cut: int = DEFAULT_LOW_ORDER_M_CUT,
    low_order_n_cut: int = DEFAULT_LOW_ORDER_N_CUT,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pads `anchor` (dict with `r_cos`/`z_sin`, any shape within the prior's
    target grid) up to (target_mpol+1, 2*target_ntor+1) and adds Gaussian
    noise scaled by the prior's per-mode std, damped by `low_order_scale` in
    the m<=low_order_m_cut,|n|<=low_order_n_cut block (preserves
    feasibility-defining low-order content; the tail is perturbed at full
    `tail_scale`). Returns the perturbed (r_cos, z_sin) padded arrays.
    """
    r = np.asarray(anchor["r_cos"], dtype=float)
    z = np.asarray(anchor["z_sin"], dtype=float)
    rp, zp = pad_to_common_grid(r, z, prior.target_mpol, prior.target_ntor)

    shape = rp.shape
    lo_mask = _low_order_mask(shape, low_order_m_cut, low_order_n_cut)
    damping = np.where(lo_mask, low_order_scale, 1.0)

    rng = np.random.default_rng(seed)
    r_noise = rng.normal(size=shape) * prior.r_cos_std * tail_scale * damping
    z_noise = rng.normal(size=shape) * prior.z_sin_std * tail_scale * damping

    r_out, z_out = rp + r_noise, zp + z_noise
    enforce_stellarator_symmetry(r_out, z_out)
    return r_out, z_out


def sample_from_prior(
    prior: SpectralPrior,
    scale: float = 1.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fresh draws from the fitted prior (mean +/- std*scale per mode), no
    anchor -- a minor exploration fraction of every batch."""
    rng = np.random.default_rng(seed)
    r = prior.r_cos_mean + rng.normal(size=prior.r_cos_mean.shape) * prior.r_cos_std * scale
    z = prior.z_sin_mean + rng.normal(size=prior.z_sin_mean.shape) * prior.z_sin_std * scale
    enforce_stellarator_symmetry(r, z)
    return r, z


@dataclasses.dataclass(frozen=True)
class CovariancePrior:
    """A PCA-derived alternative to `SpectralPrior`: instead of a per-mode
    independent Gaussian (which `perturb_anchor` must damp on the low-order
    block to avoid breaking feasibility, freezing exactly the coefficients
    that dominate `relative_seed_distance` -- see the G3 memo,
    novel-space-discovery-strategy.md 2026-07-19), this fits the *joint*
    covariance of coefficients across a pool of known-good boundaries and
    perturbs along its principal directions. The hypothesis: real feasible
    boundaries vary along correlated combinations of modes (move several
    low-order coefficients together in a way that stays near the feasible
    manifold), not independently -- so sampling along those directions can
    move the low-order block substantially (real novelty, by the same
    metric G3 measures) without the naive damping `perturb_anchor` needs.

    `components` rows are unit-norm principal directions over the flattened
    (r_cos then z_sin, each on the common (target_mpol+1, 2*target_ntor+1)
    grid) coefficient vector; `variances` is the corresponding per-component
    variance (descending). Rank is capped at `n_boundaries - 1` (SVD of a
    mean-centered pool), so a small pool yields few usable directions --
    this is a real limitation of the method, not a bug (see module-level
    caveat in the generator-upgrade scoping)."""

    target_mpol: int
    target_ntor: int
    grid_shape: tuple[int, int]
    mean: np.ndarray
    components: np.ndarray
    variances: np.ndarray
    n_boundaries: int


def fit_covariance_prior(
    boundaries: list[dict[str, Any]],
    target_mpol: int = TARGET_MPOL,
    target_ntor: int = TARGET_NTOR,
    n_components: int | None = None,
) -> CovariancePrior:
    """Fits a `CovariancePrior` via SVD of the mean-centered, flattened
    coefficient pool. `n_components` caps the number of principal
    directions kept (default: all available, i.e. `n_boundaries - 1`)."""
    if len(boundaries) < 2:
        raise ValueError("fit_covariance_prior requires at least 2 boundaries")

    shape = (target_mpol + 1, 2 * target_ntor + 1)
    flats = []
    for b in boundaries:
        r = np.asarray(b["r_cos"], dtype=float)
        z = np.asarray(b["z_sin"], dtype=float)
        rp, zp = pad_to_common_grid(r, z, target_mpol, target_ntor)
        flats.append(np.concatenate([rp.ravel(), zp.ravel()]))
    x = np.stack(flats)

    mean = x.mean(axis=0)
    x_centered = x - mean
    n = x.shape[0]

    u, s, vt = np.linalg.svd(x_centered, full_matrices=False)
    variances = (s**2) / (n - 1)

    max_rank = min(n - 1, vt.shape[0])
    k = min(n_components, max_rank) if n_components is not None else max_rank

    return CovariancePrior(
        target_mpol=target_mpol,
        target_ntor=target_ntor,
        grid_shape=shape,
        mean=mean,
        components=vt[:k],
        variances=variances[:k],
        n_boundaries=n,
    )


def perturb_along_covariance(
    anchor: dict[str, Any],
    prior: CovariancePrior,
    scale: float = 1.0,
    n_components: int | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pads `anchor` up to the prior's common grid and perturbs it by moving
    along the prior's principal directions -- `delta = sum_k z_k *
    sqrt(variance_k) * component_k * scale`, `z_k ~ N(0, 1)` -- rather than
    `perturb_anchor`'s per-mode independent, low-order-damped noise. No
    damping is applied here: the low-order block is free to move, which is
    the entire point (correlated variance, not damping, is what's meant to
    keep the result plausible). Returns the perturbed (r_cos, z_sin) padded
    arrays, stellarator-symmetry enforced."""
    r = np.asarray(anchor["r_cos"], dtype=float)
    z = np.asarray(anchor["z_sin"], dtype=float)
    rp, zp = pad_to_common_grid(r, z, prior.target_mpol, prior.target_ntor)
    anchor_flat = np.concatenate([rp.ravel(), zp.ravel()])

    k = len(prior.variances) if n_components is None else min(n_components, len(prior.variances))
    rng = np.random.default_rng(seed)
    if k > 0:
        z_scores = rng.normal(size=k)
        delta = (z_scores * np.sqrt(prior.variances[:k]) * scale) @ prior.components[:k]
    else:
        delta = np.zeros_like(anchor_flat)

    out_flat = anchor_flat + delta
    n_r = rp.size
    r_out = out_flat[:n_r].reshape(prior.grid_shape)
    z_out = out_flat[n_r:].reshape(prior.grid_shape)
    enforce_stellarator_symmetry(r_out, z_out)
    return r_out, z_out


def generate_covariance_batch(
    anchors: list[dict[str, Any]],
    prior: CovariancePrior,
    n_per_anchor: int,
    scale: float = 1.0,
    n_components: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """`generate_batch`'s covariance-prior analogue: `n_per_anchor`
    `perturb_along_covariance` draws per anchor, no prior-only-sample
    fraction (the covariance prior's mean is just the known-good pool's
    centroid, not a meaningfully distinct exploration mode the way
    `sample_from_prior` is for `SpectralPrior`)."""
    if not anchors:
        raise ValueError("generate_covariance_batch requires at least one anchor")

    rng = np.random.default_rng(seed)
    candidates: list[dict[str, Any]] = []
    call_index = 0

    for anchor_id, anchor in enumerate(anchors):
        n_field_periods = int(anchor.get("n_field_periods", 3))
        for _ in range(n_per_anchor):
            sub_seed = int(rng.integers(0, 2**31 - 1))
            r, z = perturb_along_covariance(
                anchor,
                prior,
                scale=scale,
                n_components=n_components,
                seed=sub_seed,
            )
            candidates.append(
                {
                    "call_index": call_index,
                    "boundary": {"r_cos": r.tolist(), "z_sin": z.tolist()},
                    "max_poloidal_mode": prior.target_mpol,
                    "max_toroidal_mode": prior.target_ntor,
                    "n_field_periods": n_field_periods,
                    "provenance": {
                        "op": "perturb_covariance",
                        "anchor_id": anchor_id,
                        "seed": sub_seed,
                    },
                }
            )
            call_index += 1

    return {"candidates": candidates, "seed": seed}


def _candidate_row(
    call_index: int,
    r: np.ndarray,
    z: np.ndarray,
    n_field_periods: int,
    prior: SpectralPrior,
    op: str,
    anchor_id: int | None,
    seed: int,
) -> dict[str, Any]:
    return {
        "call_index": call_index,
        "boundary": {"r_cos": r.tolist(), "z_sin": z.tolist()},
        "max_poloidal_mode": prior.target_mpol,
        "max_toroidal_mode": prior.target_ntor,
        "n_field_periods": n_field_periods,
        "provenance": {"op": op, "anchor_id": anchor_id, "seed": seed},
    }


def generate_batch(
    anchors: list[dict[str, Any]],
    prior: SpectralPrior,
    n_per_anchor: int,
    fractions: dict[str, float] | None = None,
    seed: int = 0,
    tail_scale: float = 1.0,
    low_order_scale: float = 0.1,
) -> dict[str, Any]:
    """Generates a batch of candidates: `n_per_anchor` perturbations per
    anchor (op="perturb"), plus a `sample_from_prior` fraction (op="prior")
    sized so the batch's overall perturb/prior split matches `fractions`
    (default 80/20). Every candidate is tagged with provenance (op, anchor
    id, seed). Deterministic for a fixed `seed`.
    """
    fractions = fractions or {"perturb": 0.8, "prior": 0.2}
    if not anchors:
        raise ValueError("generate_batch requires at least one anchor")

    rng = np.random.default_rng(seed)
    candidates: list[dict[str, Any]] = []
    call_index = 0

    for anchor_id, anchor in enumerate(anchors):
        n_field_periods = int(anchor.get("n_field_periods", 3))
        for _ in range(n_per_anchor):
            sub_seed = int(rng.integers(0, 2**31 - 1))
            r, z = perturb_anchor(
                anchor,
                prior,
                tail_scale=tail_scale,
                low_order_scale=low_order_scale,
                seed=sub_seed,
            )
            candidates.append(
                _candidate_row(
                    call_index,
                    r,
                    z,
                    n_field_periods,
                    prior,
                    "perturb",
                    anchor_id,
                    sub_seed,
                )
            )
            call_index += 1

    n_perturb = len(candidates)
    perturb_frac = fractions.get("perturb", 0.8)
    prior_frac = fractions.get("prior", 0.2)
    n_prior = round(n_perturb * prior_frac / perturb_frac) if perturb_frac > 0 else 0

    default_nfp = int(anchors[0].get("n_field_periods", 3))
    for _ in range(n_prior):
        sub_seed = int(rng.integers(0, 2**31 - 1))
        r, z = sample_from_prior(prior, scale=tail_scale, seed=sub_seed)
        candidates.append(
            _candidate_row(call_index, r, z, default_nfp, prior, "prior", None, sub_seed)
        )
        call_index += 1

    return {
        "candidates": candidates,
        "seed": seed,
        "n_anchors": len(anchors),
        "n_per_anchor": n_per_anchor,
        "fractions": fractions,
        "tail_scale": tail_scale,
        "low_order_scale": low_order_scale,
    }
