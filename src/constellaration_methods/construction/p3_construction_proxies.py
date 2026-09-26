"""E20 (p3-construction-search-plan-2026-09-25.md, Mechanism step 1): P3 proxy
objective over the near-axis configuration, for the construction-search stage
(cheap, CPU-light -- no VMEC/forward_model call per candidate).

Reuses (extend-before-create): `engines.near_axis_search`'s `NearAxisSpace`,
`CONTINUOUS`, `config_arrays` and `MAX_QI_RESIDUAL` for the near-axis
configuration vector, its box bounds, and the QI-residual scale already
pre-registered by H2. This module adds the P3-specific proxy rows and
scalarisation on top; it does not touch `near_axis_search.py` (that engine's
own QI-only `analytic_objective`/`is_viable` stay as they are, used elsewhere).

Phase 0 evidence (one podman introspection run, `localhost/private-
constellaration:phase1`, cores 14-15, 2026-09-25): built one
`NearAxisConfiguration` via `near_axis_search.config_arrays` (nfp=3) and
inspected the `qic.qic.Qic` object `near_axis_configuration_to_pyqic` returns,
plus `constellaration.geometry.surface_rz_fourier`. Findings that fix the
proxy formulas below:

- `Qic.iota` -- float, the near-axis analytic rotational transform. Used as
  `iota_over_nfp = abs(iota) / n_field_periods`, the same proxy H2's own
  `near_axis_search.is_viable` uses.
- `Qic.min_geo_qi_consistency(order=1)` -- float, the first-order QI
  omnigenity residual (0 = perfectly QI, ~8.7 = a failed/non-QI solve). Same
  call `near_axis_search_run.make_observe_fn` already uses.
- `Qic.B0_well_depth` -- float, exposed on every `Qic` instance regardless of
  `order` (confirmed via `dir(st)` + value inspection: for a config built with
  mirror=0.12, `B0_well_depth == 0.12` and `Qic.B0` (on-axis |B| along
  varphi) had min=0.8801/max=1.12, matching `(max-min)/(max+min) == 0.12` to
  4 decimals). This is pyQIC's own mirror-ratio quantity, computed from the
  on-axis field -- not a VMEC edge mirror ratio, but it is what the
  registration means by "if pyQIC exposes it": it does, and it is used here.
- **Magnetic well: NOT AVAILABLE.** `inspect.signature(NearAxisConfiguration)`
  has no `order` field, and `near_axis_configuration_to_pyqic`'s source
  hardcodes `order="r1"` when building the `Qic` -- the near-axis engine this
  module reuses never exposes second order (`r2`), so no
  `d2_volume_d_psi2`-style second-order-well quantity is reachable through it.
  `dir(qic.Qic)` (class-level) has no `well`/`d2`/`psi`/`volume` attribute
  either. The registration's fallback ("use the survey's well estimate")
  needs a full `forward_model.forward_model()` (VMEC) call per candidate,
  which costs orders of magnitude more than the ~1.4s near-axis oracle this
  search is built around (5,000 candidates would blow the 3h budget by a
  wide margin) -- so per the registration's own final fallback, the well row
  is DROPPED from the proxy scalarisation. `evaluate_proxies` returns
  `well_available=False` and no `"well"` key in `violations`; the stage-2
  canonical evaluation (which the driver script reuses from
  `evaluation.score_mhd_stable`) reports vacuum_well/well residual properly on
  the eventual boundaries, so the check is not skipped end-to-end -- only
  removed from the cheap stage-1 proxy.
- Aspect ratio: `constellaration.geometry.surface_rz_fourier` has no
  `evaluate_aspect_ratio`, only `evaluate_minor_radius(surface, n_theta,
  n_phi) -> float` (pure geometry on the exported `SurfaceRZFourier`, no VMEC).
  The proxy is `major_radius / evaluate_minor_radius(boundary)`, where
  `boundary = near_axis_configuration_to_plasma_boundary(cfg,
  NearAxisToPlasmaBoundarySettings(minor_radius=<fixed nominal radius>))` --
  the same boundary-export call `near_axis_search_run.screen_config` already
  makes. `major_radius` is the near-axis space's fixed `r_cos[0]` (1.0 by
  `NearAxisSpace` convention).

Scalarisation: each row is a NORMALISED VIOLATION, 0.0 when the row's bound is
satisfied and positive otherwise, on the same shape as P3's own `V` (the
`MHDStableQIStellarator` leaderboard problem's
`_normalized_constraint_violations().max()`, see `evaluation.score_mhd_
stable`). `proxy_v` is the max of whichever violations are available (well
excluded, as above). The `qi` row has no hard P3 bound (it is "minimised", not
constrained) -- it is normalised against `near_axis_search.MAX_QI_RESIDUAL`
(1e-2), the same scale H2 already uses to call a first-order QI solve
"viable", so `qi` violation 0 means "as good as a converged QI solve", not
"exactly zero residual".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from constellaration_methods.analysis.boundary_diversity import (
    make_grid,
    minor_radius,
    pad_and_normalize,
    physical_distance,
)
from constellaration_methods.engines.near_axis_search import MAX_QI_RESIDUAL

# Same recipe as the worked example (`results/p3-close-out/seed_dist_tmp.py`):
# pad to 8x8, evaluate distance on a 32x32 (theta, phi) grid, same-nfp only.
D1_TARGET_MPOL = 8
D1_TARGET_NTOR = 8
D1_N_THETA = 32
D1_N_PHI = 32

MIN_IOTA_OVER_NFP = 0.25
MAX_MIRROR = 0.25
ASPECT_LOW = 8.0
ASPECT_HIGH = 12.0
ALLOWED_NFP = (3, 4)


def iota_violation(iota_over_nfp: float) -> float:
    """Row 'iota': |iota|/nfp >= 0.25. 0 if satisfied, else the shortfall
    normalised by the bound itself."""
    return max(0.0, (MIN_IOTA_OVER_NFP - iota_over_nfp) / MIN_IOTA_OVER_NFP)


def mirror_violation(mirror: float) -> float:
    """Row 'mirror': edge mirror ratio proxy <= 0.25. 0 if satisfied, else the
    excess normalised by the bound."""
    return max(0.0, (mirror - MAX_MIRROR) / MAX_MIRROR)


def qi_violation(qi_residual: float) -> float:
    """Row 'qi': minimised, not a hard bound. Normalised against
    `MAX_QI_RESIDUAL` (1e-2, H2's own QI-viable threshold) so 0 means
    'as good as a converged first-order QI solve'. Non-finite (a failed
    near-axis/pyQIC build) is the worst possible violation."""
    if (
        qi_residual is None
        or qi_residual != qi_residual
        or qi_residual
        in (
            float("inf"),
            float("-inf"),
        )
    ):
        return float("inf")
    return max(0.0, float(qi_residual) / MAX_QI_RESIDUAL)


def aspect_violation(aspect_ratio: float) -> float:
    """Row 'aspect': aspect ratio in [8, 12]. 0 inside the band, else the
    distance to the nearest bound normalised by that bound."""
    if aspect_ratio < ASPECT_LOW:
        return (ASPECT_LOW - aspect_ratio) / ASPECT_LOW
    if aspect_ratio > ASPECT_HIGH:
        return (aspect_ratio - ASPECT_HIGH) / ASPECT_HIGH
    return 0.0


def nfp_violation(n_field_periods: int) -> float:
    """Row 'nfp': n_field_periods in {3, 4}. 0 if allowed, else 1.0 (this
    search's own space is restricted to {3, 4}, so this row exists for
    completeness/testability, not because the search itself can violate it)."""
    return 0.0 if int(n_field_periods) in ALLOWED_NFP else 1.0


@dataclass(frozen=True)
class ProxyRow:
    """One evaluated near-axis configuration's P3 proxies. `well_available`
    is always False (see module docstring); kept explicit so callers/tests
    can assert the row was consciously dropped, not silently missing."""

    iota_over_nfp: float
    mirror: float
    qi_residual: float
    aspect_ratio: float
    n_field_periods: int
    violations: dict[str, float]
    proxy_v: float
    well_available: bool = False


def evaluate_proxies(
    *,
    iota_over_nfp: float,
    mirror: float,
    qi_residual: float,
    aspect_ratio: float,
    n_field_periods: int,
) -> ProxyRow:
    """Compute all five available proxy violations and scalarise as their
    max (the same shape as P3's own V). The well row is never included --
    see module docstring."""
    violations = {
        "iota": iota_violation(iota_over_nfp),
        "mirror": mirror_violation(mirror),
        "qi": qi_violation(qi_residual),
        "aspect": aspect_violation(aspect_ratio),
        "nfp": nfp_violation(n_field_periods),
    }
    proxy_v = max(violations.values())
    return ProxyRow(
        iota_over_nfp=iota_over_nfp,
        mirror=mirror,
        qi_residual=qi_residual,
        aspect_ratio=aspect_ratio,
        n_field_periods=n_field_periods,
        violations=violations,
        proxy_v=proxy_v,
        well_available=False,
    )


# --------------------------------------------------------------------------- #
# stage-2 labelling (registration Bars section)
# --------------------------------------------------------------------------- #

NULL_V = 0.30
MATERIAL_V = 0.20
SUBSTANTIAL_V = 0.10


def stage2_label(worst_of_three_v: float, *, feasible: bool) -> str:
    """Registration Bars, as four mutually exclusive tiers on the best
    confirmed (worst-of-three) V: goal = feasible; substantial = V<=0.10;
    material = V<=0.20; null = anything weaker (including the registration's
    own worked example -- a V<=0.30 construction whose iota/mirror/well are
    all satisfied at the seed still does not reach material's 0.20, so it is
    still labelled null; `iota`/`mirror`/`well` satisfaction at the seed is
    reported alongside the label in REPORT.json, not folded into it -- the
    registration names only these four tiers, all defined by V)."""
    if feasible:
        return "goal"
    if worst_of_three_v <= SUBSTANTIAL_V:
        return "substantial"
    if worst_of_three_v <= MATERIAL_V:
        return "material"
    return "null"


# --------------------------------------------------------------------------- #
# bank row provenance
# --------------------------------------------------------------------------- #


@dataclass
class BankRow:
    """One evaluated configuration, as written to bank.jsonl."""

    label: str
    n_field_periods: int
    vector: list[float]
    proxies: dict[str, Any]
    proxy_v: float
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n_field_periods": self.n_field_periods,
            "vector": list(self.vector),
            "proxies": self.proxies,
            "proxy_v": self.proxy_v,
            "provenance": self.provenance,
        }

    @staticmethod
    def from_json_dict(d: dict[str, Any]) -> BankRow:
        return BankRow(
            label=d["label"],
            n_field_periods=d["n_field_periods"],
            vector=list(d["vector"]),
            proxies=d["proxies"],
            proxy_v=d["proxy_v"],
            provenance=d.get("provenance", {}),
        )


# --------------------------------------------------------------------------- #
# D1 placement distance (reuses `analysis.boundary_diversity`; same recipe as
# `results/p3-close-out/seed_dist_tmp.py`, the worked example)
# --------------------------------------------------------------------------- #


def nearest_family_distance(
    boundary: dict[str, Any],
    family: list[dict[str, Any]],
    *,
    target_mpol: int = D1_TARGET_MPOL,
    target_ntor: int = D1_TARGET_NTOR,
    n_theta: int = D1_N_THETA,
    n_phi: int = D1_N_PHI,
) -> float | None:
    """D1 distance from `boundary` to the nearest same-nfp member of
    `family`. Each boundary dict needs `r_cos`, `z_sin`, `n_field_periods`.
    Returns None (report "n/a") when no family member shares nfp -- distance
    across nfp is not meaningful (`physical_distance` requires equal nfp)."""
    nfp = int(boundary["n_field_periods"])
    same_nfp = [f for f in family if int(f["n_field_periods"]) == nfp]
    if not same_nfp:
        return None
    theta, phi = make_grid(nfp, n_theta, n_phi)
    r_a, z_a = pad_and_normalize(
        np.asarray(boundary["r_cos"], dtype=float),
        np.asarray(boundary["z_sin"], dtype=float),
        target_mpol,
        target_ntor,
    )
    mr_a = minor_radius(r_a, z_a, nfp)
    best: float | None = None
    for f in same_nfp:
        r_b, z_b = pad_and_normalize(
            np.asarray(f["r_cos"], dtype=float),
            np.asarray(f["z_sin"], dtype=float),
            target_mpol,
            target_ntor,
        )
        mr_b = minor_radius(r_b, z_b, nfp)
        d = physical_distance(r_a, z_a, r_b, z_b, nfp, theta, phi, mr_a, mr_b)
        if best is None or d < best:
            best = d
    return best
