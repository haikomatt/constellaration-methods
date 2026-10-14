"""Problem-definition overrides for `constellaration.problems.MHDStableQIStellarator`
(P3): a constraint threshold changes when a design's definition changes (the
benchmark's own scenario for a re-optimisation request), without forking the
upstream class by hand for each new bound.

Promoted out of the restoration runner (it originated as an implementation
detail of `p3_feasibility_restoration`) because both the front builders
(`fronts.b3_nsga2_mhd`, `fronts.p3_front_from_archive`) and the restoration
runners need the same override mechanism and the same `--constraint-override
NAME=VALUE` CLI convention.
"""

from __future__ import annotations

import numpy as np

# Short override name -> the private pydantic attribute it replaces on
# `constellaration.problems.MHDStableQIStellarator`.
CONSTRAINT_OVERRIDE_NAMES: dict[str, str] = {
    "iota_lower": "_edge_rotational_transform_over_n_field_periods_lower_bound",
    "log10_qi_upper": "_log10_qi_upper_bound",
    "mirror_upper": "_edge_magnetic_mirror_ratio_upper_bound",
    "flux_upper": "_flux_compression_in_regions_of_bad_curvature_upper_bound",
    "well_lower": "_vacuum_well_lower_bound",
}

# The stock class's five thresholds (its pydantic private defaults), keyed by
# the short names above. The restoration runner's `fake_front` test seam
# normalises its fake raw measurements against these so it needs no
# `constellaration` import on the host; the pinned-image guard test pins each
# value to the real class.
STOCK_CONSTRAINT_THRESHOLDS: dict[str, float] = {
    "iota_lower": 0.25,
    "log10_qi_upper": -3.5,
    "mirror_upper": 0.25,
    "flux_upper": 0.9,
    "well_lower": 0.0,
}


def _refuse_unknown_override_names(names) -> None:
    """Shared validation for every override entry point (`parse_constraint_
    overrides`, `problem_with_overrides`, `normalised_constraint_violations`):
    an unknown short name is refused with every valid name listed."""
    unknown = sorted(set(names) - set(CONSTRAINT_OVERRIDE_NAMES))
    if unknown:
        raise ValueError(
            f"unknown constraint override name(s) {unknown}; valid names are "
            f"{sorted(CONSTRAINT_OVERRIDE_NAMES)}"
        )


def problem_with_overrides(overrides: dict[str, float] | None = None):
    """A `constellaration.problems.MHDStableQIStellarator` instance with the
    given constraint thresholds replaced. `overrides` maps the short names in
    `CONSTRAINT_OVERRIDE_NAMES` to their new values; an empty/None dict
    returns the stock class, byte-identical behaviour. Unknown names are
    refused with every valid name listed.

    Mechanism: the class's five thresholds are pydantic PRIVATE attributes
    (`_edge_magnetic_mirror_ratio_upper_bound = 0.25`, etc: private, not
    fields, since a submitted design never supplies them). Verified directly
    against the pinned image: a subclass that redeclares one of those names
    in its class namespace overrides pydantic's own recorded default for
    that attribute (`__private_attributes__`) -- a plain value in the
    namespace is enough, no `pydantic.PrivateAttr(default=...)` wrapper
    needed. Built with `type()` rather than a `class` statement so the
    override set can be arbitrary.
    """
    overrides = overrides or {}
    from constellaration import problems

    if not overrides:
        return problems.MHDStableQIStellarator()
    _refuse_unknown_override_names(overrides)
    namespace = {CONSTRAINT_OVERRIDE_NAMES[name]: float(value) for name, value in overrides.items()}
    subclass = type(
        "_OverriddenMHDStableQIStellarator",
        (problems.MHDStableQIStellarator,),
        namespace,
    )
    return subclass()


def parse_constraint_overrides(pairs: list[str] | None) -> dict[str, float]:
    """`--constraint-override NAME=VALUE` (repeatable) -> {name: value}.
    None/empty -> {}. Refuses a malformed pair (no '=') or an unknown name
    eagerly -- the same validation `problem_with_overrides` applies -- so a
    typo fails at parse time, not mid-run.
    """
    if not pairs:
        return {}
    out: dict[str, float] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--constraint-override must be NAME=VALUE, got {pair!r}")
        name, _, value = pair.partition("=")
        out[name] = float(value)
    _refuse_unknown_override_names(out)
    return out


def _thresholds_with_overrides(overrides: dict[str, float] | None) -> dict[str, float]:
    """`STOCK_CONSTRAINT_THRESHOLDS` with `overrides` applied; refuses unknown
    names exactly as `problem_with_overrides` does."""
    overrides = overrides or {}
    _refuse_unknown_override_names(overrides)
    return dict(STOCK_CONSTRAINT_THRESHOLDS, **{k: float(v) for k, v in overrides.items()})


def normalised_constraint_violations(
    raw: dict[str, float], overrides: dict[str, float] | None = None
) -> list[float]:
    """The benchmark's own normalised constraint violations from RAW
    measurements (the restoration runner's raw keys: iota_over_nfp, log10_qi,
    edge_mirror, flux_compression, vacuum_well) against the stock thresholds
    with `overrides` applied -- the same expressions, in the same order, as
    `MHDStableQIStellarator._normalized_constraint_violations`, so the result
    is bit-identical to `problem_with_overrides(overrides)` on a metrics
    object carrying the same values (pinned-image guard test). Lets the
    `fake_front` test seam stand in for the class without a metrics object.
    """
    t = _thresholds_with_overrides(overrides)
    violations = np.array(
        [
            t["iota_lower"] - np.abs(raw["iota_over_nfp"]),
            raw["log10_qi"] - t["log10_qi_upper"],
            raw["edge_mirror"] - t["mirror_upper"],
            raw["flux_compression"] - t["flux_upper"],
            t["well_lower"] - raw["vacuum_well"],
        ]
    )
    targets = np.array(
        [
            t["iota_lower"],
            t["log10_qi_upper"],
            t["mirror_upper"],
            t["flux_upper"],
            np.maximum(1e-1, t["well_lower"]),
        ]
    )
    return [float(v) for v in violations / np.abs(targets)]
