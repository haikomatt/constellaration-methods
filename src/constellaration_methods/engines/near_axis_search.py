"""H2 (qi-discovery-stuck-analysis-and-directions.md): search the near-axis
parameter space, not the boundary.

H1 diagnosed the root cause of the QI stall: `near_axis_configuration.generate`
runs a SINGLE-start Nelder-Mead/L-BFGS solve over the near-axis parameters and
returns it unchecked, so from the torsion-pinned corner it lands off the
QI-viable manifold 18 times in 21. The fix is not new physics -- it is proper
MULTI-start search over the very same near-axis parameters. This module is that
search.

The near-axis parameter vector mirrors `generate`'s own `from_x` exactly (so we
stay on the QI-by-construction manifold), at `max_toroidal_mode=3`:

    r_cos = [ major_radius,                                   # fixed
              r_cos_1,                                        # FREE
              -major_radius / (1 + 4 nfp^2),                  # fixed-by-construction
              -r_cos_1 (nfp^2+1)/(9 nfp^2+1) ]                # fixed (Goodman et al.)
    z_sin = [ 0, z_sin_1, z_sin_2, z_sin_3 ]                  # z_sin_1..3 FREE
    B0_cos = [ 1, mirror ]                                    # mirror FREE
    d_over_curvature                                          # FREE
    n_field_periods                                          # FREE (discrete)

so the continuous search vector is (r_cos_1, z_sin_1, z_sin_2, z_sin_3,
d_over_curvature, mirror) with n_field_periods a discrete outer choice -- the
6 + 1 knobs H2 pre-registered. `r_cos_2`/`r_cos_3` stay fixed-by-construction
(they hold the axis on the QI manifold; freeing them is an explicit H2 widen,
not a default).

This module is PURE (numpy + scipy only, no `constellaration`/`qic` import) so
the driver is host-testable: the physics -- building a pyQIC `Qic` from a vector
and reading its (iota, max_elongation, qi_residual) -- is an INJECTED callable
(`observe_fn`), defaulting to the real thing only in the in-container runner
`scripts/near_axis_search_run.py`, exactly the seam pattern
`qi_discovery.build_seed_boundary` uses for `generate_nae`. Same split as
`src/nae_generator.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import optimize

# The continuous search dimensions, in the fixed order the vector uses.
CONTINUOUS = (
    "r_cos_1",
    "z_sin_1",
    "z_sin_2",
    "z_sin_3",
    "d_over_curvature",
    "mirror",
)

# The QI feasibility bound this search targets on the near-axis quantities
# (SimpleToBuildQIStellarator: |edge iota|/nfp >= 0.25, max elongation <= 5).
# The near-axis iota-proxy is |iota|/nfp; the QI residual bar separates a
# converged first-order QI construction (~1e-9) from a failed solve (~8.7).
MIN_EDGE_IOTA_OVER_NFP = 0.25
MAX_ELONGATION = 5.0
MAX_QI_RESIDUAL = 1.0e-2

# `generate`'s own barrier weight; keeps the objective's penalty terms on the
# same scale as the physics package's near-axis solve.
_BARRIER_WEIGHT = 1.0e2


@dataclass(frozen=True)
class NearAxisSpace:
    """Box bounds for the continuous knobs + the discrete field-period choices.

    Defaults are anchored on the QI-viable configs H1 actually found (grid
    `results/nae-h1/grid.json`): the viable seeds had r_cos_1 in [-0.28, 0],
    z_sin_1 in [0.11, 0.80], so the box brackets them with margin. `mirror`
    (B0_cos[1]) is capped at 0.20 (the edge-mirror bound is 0.2; the on-axis B0
    is a proxy, so ALM refines the true edge value downstream) and floored at
    0.05. These bounds are the H2 pre-registration -- change them deliberately.

    The sign convention follows `generate`: r_cos_1 <= 0 and z_sin_1 >= 0 give a
    finite iota (the problem takes |iota|, so the mirror-image basin is
    equivalent and need not be searched).
    """

    r_cos_1: tuple[float, float] = (-1.2, 0.0)
    z_sin_1: tuple[float, float] = (0.0, 1.2)
    z_sin_2: tuple[float, float] = (-0.35, 0.35)
    z_sin_3: tuple[float, float] = (-0.25, 0.25)
    d_over_curvature: tuple[float, float] = (0.1, 2.0)
    mirror: tuple[float, float] = (0.05, 0.20)
    n_field_periods: tuple[int, ...] = (1, 2, 3, 4)
    major_radius: float = 1.0

    def bounds(self) -> list[tuple[float, float]]:
        return [getattr(self, name) for name in CONTINUOUS]


def p3_wide_space() -> NearAxisSpace:
    """Widened box for the P3 construction-search stage: a NEW optional
    space, not a change to `NearAxisSpace()`'s own defaults -- the default
    space above stays exactly as pre-registered; this is an opt-in
    (`--space p3-wide`) widening of exactly one bound.

    Evidence (E20 diagnostic + Phase 0 introspection sweep,
    `results/p3-e20-diag/`): a 2,000-sample unpolished Latin-hypercube draw
    of the DEFAULT box (`space_default.jsonl`) already reaches
    `|iota|/nfp >= 0.25` in the raw sample (max 1.73 at nfp=3), but every
    such sample has a catastrophic QI residual (minimum ~427 against the
    `MAX_QI_RESIDUAL=1e-2` bound) -- high iota and a converged QI solve
    never co-occur inside the box as sampled. A one-at-a-time sensitivity
    sweep anchored at the smoke run's best (QI-converged, qi_residual=
    0.0035) config, `results/p3-e20-diag/_sensitivity_sweep_report.json`,
    swept each of the 6 continuous knobs 50% past its own bound on both
    sides holding the rest fixed. Five of the six knobs show iota rising
    only by hitting a near-discontinuous QI cliff (qi_residual jumping by
    2-6 orders of magnitude within one grid step) well inside their current
    bounds -- not a bound-exclusion story. `r_cos_1` is the exception: the
    box forces `r_cos_1 <= 0` (upper bound pinned at exactly 0.0, per this
    module's own sign-convention docstring above, on the assumption that
    `r_cos_1 > 0` is just the `r_cos_1 <= 0` / `z_sin_1 >= 0` mirror-image
    basin and "need not be searched"). Holding `z_sin_1` fixed at the
    (positive) anchor value while sweeping `r_cos_1` positive is NOT that
    mirror pair (the mirror pair also flips `z_sin_1`'s sign) -- it is a
    genuinely unsampled configuration family, and it is the one place the
    sweep found a SMOOTH climb: `r_cos_1` from 0.0 to 0.4 takes
    `iota_over_nfp` from 0.037 to 0.39 (crossing the P3 0.25 bound at
    `r_cos_1~0.2`) while `qi_residual` rises gradually (0.003 -> 0.49 -> 1.8
    -> 9.7 -> 60), never jumping by more than ~1 order of magnitude per
    step -- the kind of gradient a bounded Nelder-Mead polish can actually
    climb, unlike the near-vertical QI cliffs on the other 5 knobs. So the
    minimal widening is exactly this one bound: `r_cos_1`'s upper edge
    0.0 -> 0.4 (the point past which the sweep's qi_residual starts
    climbing as steeply as the other knobs' cliffs). Every other bound,
    including `r_cos_1`'s lower edge, is unchanged from the default box.
    """
    return NearAxisSpace(
        r_cos_1=(-1.2, 0.4),
        n_field_periods=(3, 4),
    )


@dataclass(frozen=True)
class NearAxisConfigVector:
    """One near-axis configuration in search coordinates: the continuous vector
    plus its field-period count, with the near-axis arrays it maps to. `label`
    lets a batch be threaded through the VMEC screen / seed files by name."""

    label: str
    n_field_periods: int
    r_cos_1: float
    z_sin_1: float
    z_sin_2: float
    z_sin_3: float
    d_over_curvature: float
    mirror: float

    @property
    def vector(self) -> np.ndarray:
        return np.array(
            [
                self.r_cos_1,
                self.z_sin_1,
                self.z_sin_2,
                self.z_sin_3,
                self.d_over_curvature,
                self.mirror,
            ]
        )


def config_arrays(
    vector: np.ndarray, n_field_periods: int, *, major_radius: float = 1.0
) -> dict[str, Any]:
    """Map a continuous search vector + nfp to the near-axis arrays, replicating
    `near_axis_configuration.generate`'s `from_x` mapping exactly (r_cos_2/3
    fixed-by-construction). Returns a plain dict so this stays pure -- the runner
    feeds it to `NearAxisConfiguration(**arrays)`."""
    r_cos_1, z_sin_1, z_sin_2, z_sin_3, d_over_curvature, mirror = (float(v) for v in vector)
    nfp = int(n_field_periods)
    r_cos = np.array(
        [
            major_radius,
            r_cos_1,
            -major_radius / (1 + 4 * nfp**2),
            -r_cos_1 * (nfp**2 + 1) / (9 * nfp**2 + 1),
        ]
    )
    z_sin = np.array([0.0, z_sin_1, z_sin_2, z_sin_3])
    return {
        "n_field_periods": nfp,
        "r_cos": r_cos,
        "z_sin": z_sin,
        "B0_cos": np.array([1.0, mirror]),
        "d_over_curvature": d_over_curvature,
    }


def is_viable(observables: dict[str, float], n_field_periods: int) -> bool:
    """QI-viable = finite edge-iota proxy (|iota|/nfp >= 0.25) AND a real
    first-order QI residual (<= MAX_QI_RESIDUAL). The necessary near-axis
    condition H1 showed the old seeds lacked; VMEC-convergence is the separate,
    not-implied second gate (checked downstream)."""
    iota = observables.get("iota")
    qi_residual = observables.get("qi_residual")
    if iota is None or qi_residual is None:
        return False
    if not (np.isfinite(iota) and np.isfinite(qi_residual)):
        return False
    edge_iota_proxy = abs(float(iota)) / n_field_periods if n_field_periods else abs(float(iota))
    return edge_iota_proxy >= MIN_EDGE_IOTA_OVER_NFP and float(qi_residual) <= (MAX_QI_RESIDUAL)


def analytic_objective(observables: dict[str, float], n_field_periods: int) -> float:
    """The scalar the multi-start solve minimises, mirroring `generate`'s own
    objective: first-order QI residual + weighted normalized barriers pulling
    the iota-proxy up to 0.25 and elongation down to 5. A failed/exception
    observation (`qi_residual=inf`) yields `inf`, so the optimiser is repelled
    from it. Mirror is bounded by the search box, not penalised here."""
    qi_residual = observables.get("qi_residual", float("inf"))
    iota = observables.get("iota", 0.0)
    max_elongation = observables.get("max_elongation", float("inf"))
    if not np.isfinite(qi_residual):
        return float("inf")
    edge_iota_proxy = abs(float(iota)) / n_field_periods if n_field_periods else abs(float(iota))
    iota_barrier = max(0.0, (MIN_EDGE_IOTA_OVER_NFP - edge_iota_proxy)) / (MIN_EDGE_IOTA_OVER_NFP)
    elong_barrier = max(0.0, (float(max_elongation) - MAX_ELONGATION)) / MAX_ELONGATION
    return (
        float(qi_residual) + _BARRIER_WEIGHT * iota_barrier**2 + _BARRIER_WEIGHT * elong_barrier**2
    )


def latin_hypercube_starts(
    space: NearAxisSpace, n_starts: int, rng: np.random.RandomState
) -> np.ndarray:
    """`n_starts` Latin-hypercube points over the box `space.bounds()`: one
    stratum per dimension, decorrelated across dimensions by an independent
    shuffle. Factored out of `multi_start_search`'s own start generation so
    other search drivers over the same near-axis vector (e.g. a differently
    objectived construction search) can reuse the identical start
    distribution rather than re-deriving it."""
    bounds = space.bounds()
    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])
    d = len(CONTINUOUS)
    starts = np.empty((n_starts, d))
    edges = np.arange(n_starts) / n_starts
    for j in range(d):
        pts = edges + rng.random_sample(n_starts) / n_starts
        rng.shuffle(pts)
        starts[:, j] = lows[j] + pts * (highs[j] - lows[j])
    return starts


def _normalized(vector: np.ndarray, space: NearAxisSpace) -> np.ndarray:
    """Vector scaled to [0,1] per dimension by the box, so distances are
    unit-independent (used for dedup + diversity selection)."""
    lows = np.array([getattr(space, n)[0] for n in CONTINUOUS])
    highs = np.array([getattr(space, n)[1] for n in CONTINUOUS])
    spans = np.where(highs > lows, highs - lows, 1.0)
    return (np.asarray(vector, dtype=float) - lows) / spans


def dedup(
    configs: list[NearAxisConfigVector], space: NearAxisSpace, *, tol: float = 0.02
) -> list[NearAxisConfigVector]:
    """Drop near-duplicate configs (same nfp AND normalized-vector distance <
    `tol`), keeping the first of each cluster. Multi-start Nelder-Mead from many
    inits lands repeatedly in the same basin; this keeps the pool's spread
    honest."""
    kept: list[NearAxisConfigVector] = []
    kept_norm: list[np.ndarray] = []
    for cfg in configs:
        v = _normalized(cfg.vector, space)
        if any(
            cfg.n_field_periods == k.n_field_periods and float(np.linalg.norm(v - kn)) < tol
            for k, kn in zip(kept, kept_norm)
        ):
            continue
        kept.append(cfg)
        kept_norm.append(v)
    return kept


def select_diverse(
    configs: list[NearAxisConfigVector], space: NearAxisSpace, k: int
) -> list[NearAxisConfigVector]:
    """Greedy max-min-distance subset of size <= k over the normalized vectors,
    so a bounded VMEC screen spends its budget on spread rather than a cluster.
    Deterministic: starts from `configs[0]` (the caller passes them best-first),
    then repeatedly adds the config farthest from the chosen set."""
    if k <= 0 or not configs:
        return []
    if len(configs) <= k:
        return list(configs)
    norms = [_normalized(c.vector, space) for c in configs]
    chosen_idx = [0]
    while len(chosen_idx) < k:
        best_i, best_d = None, -1.0
        for i in range(len(configs)):
            if i in chosen_idx:
                continue
            d = min(float(np.linalg.norm(norms[i] - norms[j])) for j in chosen_idx)
            if d > best_d:
                best_i, best_d = i, d
        assert best_i is not None
        chosen_idx.append(best_i)
    return [configs[i] for i in chosen_idx]


@dataclass
class SearchResult:
    viable: list[NearAxisConfigVector] = field(default_factory=list)
    n_starts: int = 0
    n_solves_ok: int = 0
    n_viable_raw: int = 0  # viable optima before dedup
    objective_by_label: dict[str, float] = field(default_factory=dict)


def multi_start_search(
    observe_fn: Callable[[np.ndarray, int], dict[str, float]],
    space: NearAxisSpace,
    *,
    n_starts_per_nfp: int = 60,
    random_seed: int | None = None,
    label_prefix: str = "nas",
    dedup_tol: float = 0.02,
    method: str = "Nelder-Mead",
) -> SearchResult:
    """The H2 search: for each nfp, draw `n_starts_per_nfp` Latin-hypercube
    starts in the box and polish each toward `analytic_objective` with a local
    solver, then keep the viable, deduplicated optima best-first. This is
    `generate`'s single-start solve done as a proper multi-start global search
    over the same parameters -- the direct fix for the failure H1 diagnosed.

    `observe_fn(vector, nfp) -> {iota, max_elongation, qi_residual}` is the
    injected physics (a pyQIC build); a raising/failed observation should come
    back as `{"qi_residual": inf}` so the objective repels the solver. Pure and
    deterministic given `random_seed`, so it is unit-tested with a fake
    `observe_fn` and no physics.
    """
    rng = np.random.RandomState(random_seed)
    bounds = space.bounds()
    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])

    viable: list[NearAxisConfigVector] = []
    objective_by_label: dict[str, float] = {}
    n_starts = 0
    n_solves_ok = 0

    for nfp in space.n_field_periods:
        starts = latin_hypercube_starts(space, n_starts_per_nfp, rng)

        for s in range(n_starts_per_nfp):
            n_starts += 1

            def _fun(x: np.ndarray, _nfp: int = nfp) -> float:
                return analytic_objective(observe_fn(x, _nfp), _nfp)

            try:
                res = optimize.minimize(_fun, starts[s], method=method, bounds=bounds)
            except Exception:  # noqa: BLE001 -- a solver blow-up is just a dead start
                continue
            x = np.clip(res.x, lows, highs)
            obs = observe_fn(x, nfp)
            n_solves_ok += 1
            if not is_viable(obs, nfp):
                continue
            label = f"{label_prefix}_nfp{nfp}_{s:03d}"
            viable.append(
                NearAxisConfigVector(
                    label=label,
                    n_field_periods=nfp,
                    r_cos_1=float(x[0]),
                    z_sin_1=float(x[1]),
                    z_sin_2=float(x[2]),
                    z_sin_3=float(x[3]),
                    d_over_curvature=float(x[4]),
                    mirror=float(x[5]),
                )
            )
            objective_by_label[label] = float(analytic_objective(obs, nfp))

    n_viable_raw = len(viable)
    viable.sort(key=lambda c: objective_by_label[c.label])
    viable = dedup(viable, space, tol=dedup_tol)
    return SearchResult(
        viable=viable,
        n_starts=n_starts,
        n_solves_ok=n_solves_ok,
        n_viable_raw=n_viable_raw,
        objective_by_label=objective_by_label,
    )
