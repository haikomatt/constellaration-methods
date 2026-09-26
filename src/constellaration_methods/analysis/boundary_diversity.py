"""P3 design-space diversity probe (D1) -- core, pure-numpy building
blocks (vault `p3-design-diversity-probe-2026-09-23.md`, Method section).
Factored out of `scripts/p3_design_diversity_probe.py` so every algorithm
here is host-testable without the pinned image: no VMEC, no jax. The only
jax-dependent step (rebuilding `results/p3-e17-nsga2/archive.jsonl` rows
into boundaries via `p3_promote_candidate.boundary_dict_of_candidate`)
lives in the CLI script's loader, not here.

Reuses (extend-before-create):
- `qi_family.symmetry_variants` / `align_to` for the discrete
  stellarator-symmetry group (theta shift by pi, phi shift by pi/nfp,
  toroidal reversal n -> -n). Both were generalised this session from the
  fixed 4x4 (5x9) window to any (mpol+1, 2*ntor+1) grid -- confirmed
  byte-identical to the old fixed-size version on 5x9 input (existing
  `tests/test_qi_family.py` is unchanged and green) -- so this probe's
  larger common grid (up to 7x7 / 8x15, the davidkh-lineage budget) can
  use them unchanged. In this repository these two functions live in
  `constellaration_methods.analysis.boundary_symmetry`, promoted there as
  the single source of truth (see that module's own docstring).
- `boundary_grid.pad_to_common_grid` for zero-padding boundaries of
  differing native mode budgets onto one comparison grid, and
  `boundary_grid.effective_modes` for the "energy fraction with m > 4 or
  |n| > 4" tail proxy (registration step 6, proxy 2) -- that function
  already computes exactly this fraction.

Conventions this module fixes that the registration leaves implicit
(flagged for review, not silently assumed):
- **Coefficient-space relative L2 is symmetrised.** The registration's
  "repo's `relative_seed_distance` convention" is one-sided (built for a
  fixed seed vs a candidate); here every pair is two peers, so
  `coefficient_distance` reports the mean of `relative_seed_distance`
  computed each way (A relative to B's best-aligned variant, and B
  relative to A's).
- **Minor radius** is the mean of the R and Z half-widths of the phi=0
  cross-section on the R00-normalised boundary:
  `a_minor = mean((max_R - min_R) / 2, (max_Z - min_Z) / 2)` over theta at
  phi=0. It is computed once per boundary, not per pair or per symmetry
  variant: every element of the discrete symmetry group is a
  rotation/reflection of the same physical surface (theta/phi shifts move
  the parametrisation's origin; toroidal reversal mirrors it), so minor
  radius -- a rotation/reflection-invariant scalar -- is identical across
  the whole orbit. A pair's normalising scale for physical distance is
  the mean of the two boundaries' own minor radii.
- **The (theta, phi) grid spans one field period**: theta in [0, 2*pi),
  phi in [0, 2*pi / nfp) -- matching `qi_family_fingerprint.py`'s own
  Boozer-map phi range. Curvature is a local differential-geometric
  quantity, so restricting phi to one field period does not change the
  analytic answer (checked against a circular torus below, whose
  curvature is constant along the minor circle irrespective of which
  field period is sampled).
- **Max principal curvature is reported in units of 1/minor_radius.**
  Every length in this module is already carried in R00-normalised units
  (R00 = 1), so the raw curvature (in 1/R00 units) is multiplied by the
  minor radius to convert to 1/minor_radius units. A circular torus' max
  principal curvature in these units is exactly 1.0 for any aspect ratio
  -- `test_boundary_diversity.py` checks this against the closed form.
"""

from __future__ import annotations

import numpy as np

from constellaration_methods.analysis.boundary_symmetry import align_to, symmetry_variants
from constellaration_methods.engines.boundary_grid import (
    pad_to_common_grid,
    relative_seed_distance,
)

# --------------------------------------------------------------------------- #
# canonicalisation: common grid + R00 normalisation
# --------------------------------------------------------------------------- #


def pad_and_normalize(
    r_cos: np.ndarray, z_sin: np.ndarray, target_mpol: int, target_ntor: int
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-pad onto the (target_mpol, target_ntor) common grid
    (`boundary_grid.pad_to_common_grid`), then scale out R00 (the m=0,n=0
    r_cos coefficient, the major radius) so absolute size is not a
    difference -- registration step 1. Raises via `pad_to_common_grid` if
    the input already exceeds the target grid (no silent truncation)."""
    r_pad, z_pad = pad_to_common_grid(r_cos, z_sin, target_mpol, target_ntor)
    n_max = (r_pad.shape[1] - 1) // 2
    r00 = float(r_pad[0, n_max])
    if r00 == 0.0:
        raise ValueError("pad_and_normalize: R00 (major radius coefficient) is zero")
    return r_pad / r00, z_pad / r00


# --------------------------------------------------------------------------- #
# distances
# --------------------------------------------------------------------------- #


def coefficient_distance(
    r_a: np.ndarray, z_a: np.ndarray, r_b: np.ndarray, z_b: np.ndarray
) -> float:
    """Symmetry-group-minimised relative L2 in coefficient space
    (registration step 2a): the mean of `relative_seed_distance` computed
    each way. Both inputs must already be on the same padded,
    R00-normalised grid."""
    _, r_b_aligned, z_b_aligned, _ = align_to(r_a, z_a, r_b, z_b)
    _, r_a_aligned, z_a_aligned, _ = align_to(r_b, z_b, r_a, z_a)
    vec_a = np.concatenate([r_a.ravel(), z_a.ravel()])
    vec_b = np.concatenate([r_b.ravel(), z_b.ravel()])
    vec_b_aligned = np.concatenate([r_b_aligned.ravel(), z_b_aligned.ravel()])
    vec_a_aligned = np.concatenate([r_a_aligned.ravel(), z_a_aligned.ravel()])
    fwd = relative_seed_distance(vec_b_aligned, vec_a)
    rev = relative_seed_distance(vec_a_aligned, vec_b)
    return 0.5 * (fwd + rev)


def minor_radius(r_cos: np.ndarray, z_sin: np.ndarray, nfp: int) -> float:
    """Mean of the R and Z half-widths of the phi=0 cross-section, on the
    R00-normalised boundary -- see module docstring. Gauge-invariant
    across the symmetry group, so this is computed once per boundary."""
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    r_curve, z_curve = _cross_section(r_cos, z_sin, nfp, theta, phi=0.0)
    half_width_r = (r_curve.max() - r_curve.min()) / 2.0
    half_width_z = (z_curve.max() - z_curve.min()) / 2.0
    return float((half_width_r + half_width_z) / 2.0)


def _cross_section(
    r_cos: np.ndarray, z_sin: np.ndarray, nfp: int, theta: np.ndarray, phi: float
) -> tuple[np.ndarray, np.ndarray]:
    """R(theta), Z(theta) at a fixed toroidal angle `phi`. ang[m, n, k] =
    m*theta_k - nfp*n*phi."""
    mpol1, ncols = r_cos.shape
    ntor = (ncols - 1) // 2
    m = np.arange(mpol1)[:, None, None]  # (mpol1, 1, 1)
    n = np.arange(-ntor, ntor + 1)[None, :, None]  # (1, ncols, 1)
    ang = m * theta[None, None, :] - nfp * n * phi  # (mpol1, ncols, n_theta)
    r_curve = np.tensordot(r_cos, np.cos(ang), axes=([0, 1], [0, 1]))
    z_curve = np.tensordot(z_sin, np.sin(ang), axes=([0, 1], [0, 1]))
    return r_curve, z_curve


# --------------------------------------------------------------------------- #
# physical-space grid: surface evaluation + derivatives
# --------------------------------------------------------------------------- #


def make_grid(nfp: int, n_theta: int = 64, n_phi: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """One field period: theta in [0, 2*pi), phi in [0, 2*pi/nfp)."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi / nfp, n_phi, endpoint=False)
    return theta, phi


_BASIS_CACHE: dict[tuple[int, int, int, int, int], tuple] = {}


def _angle_tensor(mpol1: int, ntor: int, nfp: int, theta: np.ndarray, phi: np.ndarray):
    """ang[m, n_col, i, j] = m*theta_i - nfp*n*phi_j, plus the (m, n)
    index arrays broadcast-ready for derivative weighting.

    Cached on (mpol1, ntor, nfp, len(theta), len(phi)): the CLI's pairwise
    work (near-equivalent search, clustering, the 10,000-pair reference
    distribution) calls this many times with the SAME common-grid shape
    and the SAME `theta`/`phi` arrays (always built by `make_grid`), so
    recomputing the O(mpol*ntor*grid) trig tensor from scratch every call
    would dominate the run time for no reason -- the angle grid does not
    depend on which boundary is being evaluated."""
    key = (mpol1, ntor, nfp, theta.shape[0], phi.shape[0])
    cached = _BASIS_CACHE.get(key)
    if cached is not None:
        return cached
    m = np.arange(mpol1)
    n = np.arange(-ntor, ntor + 1)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    ang = (
        m[:, None, None, None] * th[None, None, :, :]
        - nfp * n[None, :, None, None] * ph[None, None, :, :]
    )
    result = (ang, m, n)
    _BASIS_CACHE[key] = result
    return result


def evaluate_surface(
    r_cos: np.ndarray, z_sin: np.ndarray, nfp: int, theta: np.ndarray, phi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """R, Z on the (theta, phi) grid, direct Fourier sum (mode budgets are
    small -- <= 7x7 -- so a direct tensordot is simpler than an FFT and
    fast enough for single-boundary/on-demand use)."""
    mpol1, ncols = r_cos.shape
    ntor = (ncols - 1) // 2
    ang, _m, _n = _angle_tensor(mpol1, ntor, nfp, theta, phi)
    cos_a = np.cos(ang)
    sin_a = np.sin(ang)
    r = np.tensordot(r_cos, cos_a, axes=([0, 1], [0, 1]))
    z = np.tensordot(z_sin, sin_a, axes=([0, 1], [0, 1]))
    return r, z


def evaluate_surface_with_derivatives(
    r_cos: np.ndarray, z_sin: np.ndarray, nfp: int, theta: np.ndarray, phi: np.ndarray
) -> dict[str, np.ndarray]:
    """R, Z and their first/second theta/phi derivatives on the grid --
    everything `principal_curvature_max` needs. ang = m*theta - nfp*n*phi,
    so d(ang)/dtheta = m and d(ang)/dphi = -nfp*n; the derivative fields
    below follow directly by the chain rule."""
    mpol1, ncols = r_cos.shape
    ntor = (ncols - 1) // 2
    ang, m, n = _angle_tensor(mpol1, ntor, nfp, theta, phi)
    cos_a = np.cos(ang)
    sin_a = np.sin(ang)
    m_col = m[:, None]
    n_row = n[None, :]
    nn = nfp * n_row

    def tdot(weight, basis):
        return np.tensordot(weight, basis, axes=([0, 1], [0, 1]))

    r = tdot(r_cos, cos_a)
    r_th = tdot(r_cos * (-m_col), sin_a)
    r_ph = tdot(r_cos * nn, sin_a)
    r_thth = tdot(r_cos * (-(m_col**2)), cos_a)
    r_phph = tdot(r_cos * (-(nn**2)), cos_a)
    r_thph = tdot(r_cos * (m_col * nn), cos_a)

    z = tdot(z_sin, sin_a)
    z_th = tdot(z_sin * m_col, cos_a)
    z_ph = tdot(z_sin * (-nn), cos_a)
    z_thth = tdot(z_sin * (-(m_col**2)), sin_a)
    z_phph = tdot(z_sin * (-(nn**2)), sin_a)
    z_thph = tdot(z_sin * (m_col * nn), sin_a)

    return dict(
        r=r,
        z=z,
        r_th=r_th,
        r_ph=r_ph,
        r_thth=r_thth,
        r_phph=r_phph,
        r_thph=r_thph,
        z_th=z_th,
        z_ph=z_ph,
        z_thth=z_thth,
        z_phph=z_phph,
        z_thph=z_thph,
        phi=phi,
    )


# --------------------------------------------------------------------------- #
# physical distance (symmetry-group-minimised) and curvature proxy
# --------------------------------------------------------------------------- #


def _evaluate_surface_batch(
    r_stack: np.ndarray,
    z_stack: np.ndarray,
    nfp: int,
    theta: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """R, Z for a BATCH of k coefficient arrays in one tensordot call each
    (`r_stack`/`z_stack` shape (k, mpol1, ncols)) -- this is the
    "vectorise the symmetry-group minimisation" the registration's
    performance note asks for: `physical_distance` calls this once for
    B's 8-element group orbit instead of once per variant, which cuts the
    per-pair Python/BLAS-dispatch overhead by ~8x (measured on real atlas
    data: near-equivalent-pair search alone was 18k+ candidate pairs on a
    2000-boundary slice, so that overhead is what actually gated the
    ~2h/28k-boundary budget, not raw FLOPs)."""
    k, mpol1, ncols = r_stack.shape
    ntor = (ncols - 1) // 2
    ang, _m, _n = _angle_tensor(mpol1, ntor, nfp, theta, phi)
    cos_a = np.cos(ang)
    sin_a = np.sin(ang)
    r_grid = np.tensordot(r_stack, cos_a, axes=([1, 2], [0, 1]))  # (k, n_theta, n_phi)
    z_grid = np.tensordot(z_stack, sin_a, axes=([1, 2], [0, 1]))
    return r_grid, z_grid


def physical_distance(
    r_a: np.ndarray,
    z_a: np.ndarray,
    r_b: np.ndarray,
    z_b: np.ndarray,
    nfp: int,
    theta: np.ndarray,
    phi: np.ndarray,
    minor_radius_a: float,
    minor_radius_b: float,
) -> float:
    """RMS separation of two surfaces over the (theta, phi) grid, in units
    of the pair's mean minor radius, minimised over B's symmetry-group
    orbit (registration step 2b + the "minimise over the group" note) --
    both boundaries must already be on the same padded, R00-normalised
    grid and have the same nfp. B's 8-element orbit is evaluated as one
    batch (see `_evaluate_surface_batch`)."""
    r_a_grid, z_a_grid = evaluate_surface(r_a, z_a, nfp, theta, phi)
    variants = symmetry_variants(r_b, z_b)
    r_stack = np.stack([v[0] for v in variants])
    z_stack = np.stack([v[1] for v in variants])
    r_b_grid, z_b_grid = _evaluate_surface_batch(r_stack, z_stack, nfp, theta, phi)
    rms = np.sqrt(
        np.mean(
            (r_a_grid[None, :, :] - r_b_grid) ** 2 + (z_a_grid[None, :, :] - z_b_grid) ** 2,
            axis=(1, 2),
        )
    )
    best = float(np.min(rms))
    scale = 0.5 * (minor_radius_a + minor_radius_b)
    if scale <= 0.0:
        raise ValueError("physical_distance: non-positive minor-radius scale")
    return best / scale


def principal_curvature_max(fields: dict[str, np.ndarray], a_minor: float) -> float:
    """Max |principal curvature| over the grid, in units of 1/minor_radius
    (registration step 6, proxy 1). The surface is embedded as
    S(theta, phi) = (R cos(phi), R sin(phi), Z); first/second fundamental
    forms follow from the chain rule on that embedding, and the principal
    curvatures are the eigenvalues of the shape operator, i.e. the roots
    of (EG - F^2) k^2 - (EN - 2FM + GL) k + (LN - M^2) = 0."""
    r = fields["r"]
    r_th, r_ph = fields["r_th"], fields["r_ph"]
    r_thth, r_phph, r_thph = fields["r_thth"], fields["r_phph"], fields["r_thph"]
    z_th, z_ph = fields["z_th"], fields["z_ph"]
    z_thth, z_phph, z_thph = fields["z_thth"], fields["z_phph"], fields["z_thph"]

    # phi is a grid axis broadcast against the theta axis already carried
    # by r/z's shape (n_theta, n_phi); `evaluate_surface_with_derivatives`
    # stashes the phi vector in the fields dict for this.
    phi = fields["phi"]
    cos_v = np.cos(phi)[None, :]
    sin_v = np.sin(phi)[None, :]

    s_u = np.stack([r_th * cos_v, r_th * sin_v, z_th], axis=-1)
    s_v = np.stack([r_ph * cos_v - r * sin_v, r_ph * sin_v + r * cos_v, z_ph], axis=-1)
    s_uu = np.stack([r_thth * cos_v, r_thth * sin_v, z_thth], axis=-1)
    s_uv = np.stack(
        [r_thph * cos_v - r_th * sin_v, r_thph * sin_v + r_th * cos_v, z_thph],
        axis=-1,
    )
    s_vv = np.stack(
        [
            r_phph * cos_v - 2 * r_ph * sin_v - r * cos_v,
            r_phph * sin_v + 2 * r_ph * cos_v - r * sin_v,
            z_phph,
        ],
        axis=-1,
    )

    e = np.sum(s_u * s_u, axis=-1)
    f = np.sum(s_u * s_v, axis=-1)
    g = np.sum(s_v * s_v, axis=-1)

    normal = np.cross(s_u, s_v)
    normal_len = np.linalg.norm(normal, axis=-1)
    normal_len = np.where(normal_len == 0.0, 1.0, normal_len)
    unit_normal = normal / normal_len[..., None]

    ell = np.sum(s_uu * unit_normal, axis=-1)
    m_ = np.sum(s_uv * unit_normal, axis=-1)
    n_ = np.sum(s_vv * unit_normal, axis=-1)

    a = e * g - f * f
    a = np.where(a == 0.0, 1e-300, a)
    b = e * n_ - 2.0 * f * m_ + g * ell
    c = ell * n_ - m_ * m_
    disc = np.clip(b * b - 4.0 * a * c, 0.0, None)
    sqrt_disc = np.sqrt(disc)
    k1 = (b + sqrt_disc) / (2.0 * a)
    k2 = (b - sqrt_disc) / (2.0 * a)

    kappa_max_raw = float(np.max(np.maximum(np.abs(k1), np.abs(k2))))
    return kappa_max_raw * a_minor


# --------------------------------------------------------------------------- #
# objective-space bucketing, near-equivalent pairing, reference distribution
# --------------------------------------------------------------------------- #


def bucket_by_cell(
    x: np.ndarray,
    y: np.ndarray,
    feasible: np.ndarray,
    group: np.ndarray,
    cell: float,
    max_per_cell: int | None = None,
) -> dict[tuple, list[int]]:
    """Feasible-only indices grouped into a (group, floor(x/cell),
    floor(y/cell)) grid -- the shared bucketing behind both the
    near-equivalent search (cell = objective_tol, with a 3x3 neighbour
    sweep, see `near_equivalent_pairs`) and per-cell clustering (cell =
    the registration's 0.1 grid, no neighbour sweep -- clusters are
    reported per cell, not merged across cell edges).

    If `max_per_cell` is set, a cell exceeding it is deterministically
    subsampled: every ceil(n / max_per_cell)-th member in input order is
    kept. This is the registration's "otherwise subsample the atlas
    deterministically" fallback -- callers should record which cells were
    subsampled (compare `len(buckets[key])` against a pre-subsample
    count) in the report."""
    buckets: dict[tuple, list[int]] = {}
    for i in range(len(x)):
        if not feasible[i]:
            continue
        key = (int(group[i]), int(np.floor(x[i] / cell)), int(np.floor(y[i] / cell)))
        buckets.setdefault(key, []).append(i)
    if max_per_cell is not None:
        for key, idxs in list(buckets.items()):
            if len(idxs) <= max_per_cell:
                continue
            stride = -(-len(idxs) // max_per_cell)  # ceil division
            buckets[key] = idxs[::stride]
    return buckets


def near_equivalent_pairs(
    x: np.ndarray,
    y: np.ndarray,
    feasible: np.ndarray,
    group: np.ndarray,
    tol: float,
    max_per_cell: int | None = None,
) -> list[tuple[int, int]]:
    """Index pairs (i < j), both feasible, same `group` (nfp), with
    |x_i - x_j| <= tol AND |y_i - y_j| <= tol (registration step 3's
    |d aspect| <= tol and |d min_len| <= tol) -- an L-infinity ball of
    radius `tol` in objective space. Found via a spatial hash with cell
    size = tol and a 3x3 neighbour sweep (any pair within an L-infinity
    ball of radius `tol` is guaranteed to land in the same or an adjacent
    cell), so the search cost is per-cell, not all-pairs, per the
    registration's performance note."""
    buckets = bucket_by_cell(x, y, feasible, group, tol, max_per_cell=max_per_cell)
    pairs: set[tuple[int, int]] = set()
    for (grp, cx, cy), idxs in buckets.items():
        neighborhood: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighborhood.extend(buckets.get((grp, cx + dx, cy + dy), []))
        candidates = sorted(set(neighborhood))
        for a in idxs:
            for b in candidates:
                if b <= a:
                    continue
                if abs(x[a] - x[b]) <= tol and abs(y[a] - y[b]) <= tol:
                    pairs.add((a, b))
    return sorted(pairs)


def sample_reference_pairs(
    feasible: np.ndarray,
    group: np.ndarray,
    rng: np.random.Generator,
    n_pairs: int,
) -> list[tuple[int, int]]:
    """`n_pairs` random (i, j) index pairs, both feasible, same `group`
    (nfp) -- registration step 4's reference distribution. Groups are
    sampled in proportion to their own pair count (n*(n-1)), so a large
    group (e.g. phanerozoic at nfp=3) does not get diluted by a tiny one
    contributing an equal share of draws."""
    by_group: dict[int, list[int]] = {}
    for i, (f, g) in enumerate(zip(feasible, group)):
        if f:
            by_group.setdefault(int(g), []).append(i)
    groups = [g for g, idxs in by_group.items() if len(idxs) >= 2]
    if not groups:
        return []
    weights = np.array([len(by_group[g]) * (len(by_group[g]) - 1) for g in groups], dtype=float)
    weights /= weights.sum()
    pairs = []
    for _ in range(n_pairs):
        g = groups[rng.choice(len(groups), p=weights)]
        idxs = by_group[g]
        pick = rng.choice(len(idxs), size=2, replace=False)
        pairs.append((idxs[pick[0]], idxs[pick[1]]))
    return pairs


# --------------------------------------------------------------------------- #
# single-linkage clustering (union-find) and the pre-registered reading
# --------------------------------------------------------------------------- #


class _UnionFind:
    def __init__(self, items: list[int]) -> None:
        self._parent = {i: i for i in items}

    def find(self, i: int) -> int:
        root = i
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[i] != root:
            self._parent[i], i = root, self._parent[i]
        return root

    def union(self, i: int, j: int) -> None:
        root_i, root_j = self.find(i), self.find(j)
        if root_i != root_j:
            self._parent[root_i] = root_j


def single_linkage_clusters(indices: list[int], distance_fn, threshold: float) -> list[list[int]]:
    """Single-linkage partition of `indices` at `threshold`: union two
    members whenever `distance_fn(i, j) <= threshold`, return the
    surviving connected components as member-index lists (registration
    step 5's per-cell clustering at d_10; also the basis for reporting
    each cluster's curvature/tail-fraction spread, step 6)."""
    uf = _UnionFind(indices)
    for a in range(len(indices)):
        for b in range(a + 1, len(indices)):
            i, j = indices[a], indices[b]
            if distance_fn(i, j) <= threshold:
                uf.union(i, j)
    groups: dict[int, list[int]] = {}
    for i in indices:
        groups.setdefault(uf.find(i), []).append(i)
    return list(groups.values())


def single_linkage_cluster_count(indices: list[int], distance_fn, threshold: float) -> int:
    """Number of single-linkage clusters among `indices` at `threshold` --
    see `single_linkage_clusters`."""
    return len(single_linkage_clusters(indices, distance_fn, threshold))


def classify_reading(
    fraction: float,
    n_pairs: int,
    *,
    degenerate_threshold: float = 0.20,
    rigid_threshold: float = 0.05,
    min_pairs: int = 100,
) -> str:
    """The pre-registered reading (registration "Pre-registered readings"):
    Degenerate if >= `degenerate_threshold` of near-equivalent feasible
    pairs are at physical distance >= d_rand AND there are >= `min_pairs`
    pairs; Rigid if < `rigid_threshold` under the same pair-count
    condition; Inconclusive otherwise (including always below
    `min_pairs`, and always for a cross-lineage split, which the caller
    never feeds through this classifier)."""
    if n_pairs < min_pairs:
        return "Inconclusive"
    if fraction >= degenerate_threshold:
        return "Degenerate"
    if fraction < rigid_threshold:
        return "Rigid"
    return "Inconclusive"
