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

# Short override name -> the private pydantic attribute it replaces on
# `constellaration.problems.MHDStableQIStellarator`.
CONSTRAINT_OVERRIDE_NAMES: dict[str, str] = {
    "iota_lower": "_edge_rotational_transform_over_n_field_periods_lower_bound",
    "log10_qi_upper": "_log10_qi_upper_bound",
    "mirror_upper": "_edge_magnetic_mirror_ratio_upper_bound",
    "flux_upper": "_flux_compression_in_regions_of_bad_curvature_upper_bound",
    "well_lower": "_vacuum_well_lower_bound",
}


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
    unknown = sorted(set(overrides) - set(CONSTRAINT_OVERRIDE_NAMES))
    if unknown:
        raise ValueError(
            f"unknown constraint override name(s) {unknown}; valid names are "
            f"{sorted(CONSTRAINT_OVERRIDE_NAMES)}"
        )
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
    unknown = sorted(set(out) - set(CONSTRAINT_OVERRIDE_NAMES))
    if unknown:
        raise ValueError(
            f"unknown constraint override name(s) {unknown}; valid names are "
            f"{sorted(CONSTRAINT_OVERRIDE_NAMES)}"
        )
    return out
