"""qi-discovery-phase-plan.md Phase 1: reusable instrumentation for the
proxy-vs-canonical audit finding, so any optimization run against
`SimpleToBuildQIStellarator` can harvest it as a byproduct.

Two capture paths, matching the plan's two items:

1. **DESC proxy path** (`constellaration.data_generation.desc_optimization
   .optimize_boundary_omnigenity_desc`, and its inner
   `_optimize_equilibrium_for_omnigenity_desc`) refines a boundary against
   the omnigenity-residual PROXY, entirely in DESC's own Boozer-space
   representation -- never against VMEC++. The inner function currently
   discards the optimizer's `OptimizeResult` via
   `(equilibrium,), _ = optimizer.optimize(...)` -- that result's
   `"Objective values"` key holds before/after values per objective term,
   keyed by each term's `_print_value_fmt` string (confirmed directly
   against the installed `desc` package in
   `localhost/private-image:phase1`:
   `desc.objectives._omnigenity.Omnigenity._print_value_fmt ==
   "Omnigenity error: "`, and for a non-scalar/non-linear objective like
   this one, `ObjectiveFunction.print_value`'s "weighted mean" branch
   populates `f0_mean`/`f_mean` (also `_max`/`_min`) before/after each
   term's list entry).

   `run_desc_omnigenity_with_capture` recovers this discarded value the same
   way `alm_ngopt_optimizer.py` already recovers a different library's
   discarded return value (see that module's `pytree.register_pydantic_data`
   patch docstring): monkeypatch the exact discarding seam
   (`desc.optimize.Optimizer.optimize`) for the duration of one call, invoke
   the library's own public, unmodified entry point
   (`optimize_boundary_omnigenity_desc`), capture the side channel, restore
   the patch immediately after. This duplicates none of
   `_optimize_equilibrium_for_omnigenity_desc`'s physics-relevant logic --
   it only taps a return value the library itself throws away.

   IMPORTANT (found by this Phase's own real-seam probe, not assumed):
   `_optimize_equilibrium_for_omnigenity_desc` calls `optimizer.optimize`
   up to THREE times per invocation, not once -- `solve_desc_equilibrium`
   (called both before and after the omnigenity optimize, to force-balance
   -solve the initial guess and then re-solve the optimized equilibrium)
   is itself implemented as `Equilibrium.solve()`, which internally builds
   its own force-balance objective and calls `Optimizer(...).optimize(...)`.
   So the capture seam fires 3 times per call in the order [force-balance
   solve, the omnigenity optimize we actually want, force-balance re-solve],
   and naively keeping only the *last* captured result gets the re-solve's
   force-balance `"Objective values"` (`"Force error: "`,
   `"R boundary error: "`, ...), not the omnigenity one -- confirmed by
   `scripts/qi_audit_instrument_probe.py`'s first run. `run_desc_omnigenity_
   with_capture` therefore captures ALL calls in order, and
   `select_optimize_result_with_term` picks out the one whose
   `"Objective values"` actually contains the requested proxy term -- robust
   to however many solve/re-solve calls wrap the real one, rather than
   assuming a fixed position.

   `extract_proxy_objective_values` then pulls the before/after pair for a
   given term out of that selected result, and `recompute_canonical_qi`
   independently recomputes CANONICAL QI on the same output boundary via
   `forward_model.forward_model` at `default_high_fidelity()` (QI/Boozer via
   VMEC++, unconditionally -- the plan's own convention for a QI problem,
   matching `alm_ngopt_optimizer._canonical_settings_for`'s QI-problem
   branch). `record_desc_proxy_vs_canonical` is the single clean entry point
   that ties these three together per design: run DESC with capture,
   extract the proxy value the optimizer trusted, recompute canonical QI,
   and store both plus their delta/ratio for later characterization of
   where they diverge.

2. **ALM-NGOpt canonical-scored path** already scores every candidate via
   canonical `forward_model` (see `alm_ngopt_optimizer.record()`'s use of
   `problem.get_objective`/`compute_feasibility`/`is_feasible`), so its
   "proxy gap" is minimal by construction -- it has no separate proxy
   channel to compare against. Its `trajectory_log_path` already writes one
   JSON line per evaluation with exactly this ground-truth trajectory.
   `summarize_alm_ngopt_trajectory` captures it for completeness/symmetry
   with the DESC-path record above, reusing `trajectory_logger
   .read_trajectory` (a generic ndjson-of-dicts reader) rather than
   re-implementing file parsing.

Concurrency note: `run_desc_omnigenity_with_capture`'s monkeypatch is
process-global for the duration of one call and is restored immediately
after (even on exception) -- safe for sequential per-design calls within one
process (the expected Phase 2 usage: "a clean function the Phase 2 run will
call per design"), but not reentrant against concurrent calls from multiple
threads in the same process. Multiple processes (e.g. one per design, via
multiprocessing) are unaffected by each other, since the patch is
per-process state.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from constellaration import forward_model
from constellaration.data_generation import desc_optimization
from desc import optimize as desc_optimize

from constellaration_methods.engines import trajectory_logger

_DEFAULT_PROXY_TERM_KEY = "Omnigenity error: "
_DEFAULT_PROXY_STAT = "mean"


@contextlib.contextmanager
def _capture_desc_optimize_result():
    """Monkeypatches `desc.optimize.Optimizer.optimize` for the duration of
    the `with` block so every `(things, result)` return value during it is
    appended (in call order) to the yielded dict's `"results"` list, then
    restores the original method (even if the wrapped code raises). See
    module docstring for why this seam (and why ALL calls are captured, not
    just the last one) rather than a reimplementation of
    `_optimize_equilibrium_for_omnigenity_desc`'s body."""
    captured: dict[str, Any] = {"results": []}
    original_optimize = desc_optimize.Optimizer.optimize

    def _patched_optimize(self, *args, **kwargs):
        things, result = original_optimize(self, *args, **kwargs)
        captured["results"].append(result)
        return things, result

    desc_optimize.Optimizer.optimize = _patched_optimize
    try:
        yield captured
    finally:
        desc_optimize.Optimizer.optimize = original_optimize


def run_desc_omnigenity_with_capture(
    targets: Any,
    settings: Any,
    *,
    optimize_boundary_fn: Callable[
        [Any, Any], Any
    ] = desc_optimization.optimize_boundary_omnigenity_desc,
) -> tuple[Any, list[dict[str, Any]]]:
    """Runs `optimize_boundary_fn` (default: the library's own
    `optimize_boundary_omnigenity_desc`, unmodified) under
    `_capture_desc_optimize_result`, and returns `(boundary, results)` --
    the boundary `optimize_boundary_fn` would normally return on its own,
    plus the list of raw DESC `OptimizeResult`s it otherwise discards
    internally, in call order. The real path calls `optimizer.optimize`
    THREE times per invocation (initial force-balance solve, the omnigenity
    optimize, force-balance re-solve -- see module docstring); use
    `select_optimize_result_with_term` to pick out the one that actually
    scored the proxy term you want, rather than assuming a position.

    `optimize_boundary_fn` is injectable for testing: a fake standing in for
    the real DESC call, expected (like the real one) to invoke
    `desc.optimize.Optimizer.optimize` internally at least once. Raises
    `RuntimeError` if it never does -- that means the capture seam didn't
    fire, which would otherwise silently produce a boundary with no matching
    proxy record.
    """
    with _capture_desc_optimize_result() as captured:
        boundary = optimize_boundary_fn(targets, settings)

    if not captured["results"]:
        raise RuntimeError(
            "run_desc_omnigenity_with_capture: optimize_boundary_fn returned "
            "without ever calling desc.optimize.Optimizer.optimize -- the "
            "capture seam was not called, so no proxy OptimizeResult is "
            "available. This means optimize_boundary_fn does not match the "
            "real optimize_boundary_omnigenity_desc's call sequence."
        )
    return boundary, captured["results"]


def select_optimize_result_with_term(
    results: list[dict[str, Any]],
    term_key: str = _DEFAULT_PROXY_TERM_KEY,
) -> dict[str, Any]:
    """Picks the one `OptimizeResult` (out of possibly several captured by
    `run_desc_omnigenity_with_capture`) whose `"Objective values"` actually
    contains `term_key`. `_optimize_equilibrium_for_omnigenity_desc` calls
    `optimizer.optimize` up to three times per invocation -- the initial
    force-balance `solve_desc_equilibrium` call, the omnigenity-objective
    optimize this audit targets, and a force-balance re-solve afterward --
    and only the middle one scores the omnigenity/aspect/elongation/iota
    terms; the force-balance calls score `"Force error: "`/
    `"R boundary error: "`/etc instead. Selecting by content is robust to
    however many solve/re-solve calls wrap the real one (confirmed to be
    exactly this shape by `scripts/qi_audit_instrument_probe.py`'s real-seam
    probe -- see Phase 1 report), rather than assuming a fixed call index.

    Raises `RuntimeError` if zero or more than one captured result contains
    `term_key` (the latter would mean the selection is ambiguous and needs a
    human to look at it, not a silent pick).
    """
    matches = [r for r in results if term_key in r.get("Objective values", {})]
    if not matches:
        raise RuntimeError(
            f"none of the {len(results)} captured optimizer.optimize() calls "
            f"had {term_key!r} in their 'Objective values' -- present key "
            f"sets were {[sorted(r.get('Objective values', {})) for r in results]}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"{len(matches)} of the {len(results)} captured optimizer.optimize() "
            f"calls had {term_key!r} in their 'Objective values' -- ambiguous "
            "which is the intended one"
        )
    return matches[0]


def extract_proxy_objective_values(
    objective_values: dict[str, Any],
    term_key: str = _DEFAULT_PROXY_TERM_KEY,
    stat: str = _DEFAULT_PROXY_STAT,
) -> tuple[float, float]:
    """Pulls the before/after `(f0, f)` pair for `term_key` out of a DESC
    `OptimizeResult["Objective values"]` dict (or an equivalently-shaped
    fixture). `stat` selects which reduction of the per-node residual to
    read: `"mean"`/`"max"`/`"min"` map to that objective's `f0_<stat>`/
    `f_<stat>` keys (the "weighted mean" `print_value` branch DESC's
    `Omnigenity` objective uses, being non-scalar/non-linear); `"f"` reads
    the plain `f0`/`f` keys instead (the scalar-objective branch), for reuse
    against other objective terms if needed.

    Raises `KeyError` with the available keys listed if `term_key` or the
    requested stat isn't present -- most likely cause is an installed `desc`
    version renaming the objective's `_print_value_fmt` or changing
    `print_value`'s output shape; see
    `desc.objectives._omnigenity.Omnigenity._print_value_fmt`.
    """
    if term_key not in objective_values:
        raise KeyError(
            f"{term_key!r} not present in DESC 'Objective values' -- present "
            f"keys were {sorted(objective_values)}. Check "
            "desc.objectives._omnigenity.Omnigenity._print_value_fmt against "
            "the installed desc version."
        )
    entries = objective_values[term_key]
    if not entries:
        raise ValueError(f"{term_key!r} present but its entry list is empty")
    entry = entries[0]

    before_key = "f0" if stat == "f" else f"f0_{stat}"
    after_key = "f" if stat == "f" else f"f_{stat}"
    if before_key not in entry or after_key not in entry:
        raise KeyError(
            f"stat={stat!r} not available for term {term_key!r} -- entry keys were {sorted(entry)}"
        )
    return float(entry[before_key]), float(entry[after_key])


def recompute_canonical_qi(
    boundary: Any,
    *,
    forward_model_fn: Callable[..., Any] = forward_model.forward_model,
    settings: forward_model.ConstellarationSettings | None = None,
) -> dict[str, Any]:
    """Recomputes CANONICAL QI on `boundary` via `forward_model_fn`
    (default: the real `forward_model.forward_model`) at
    `default_high_fidelity()` (QI/Boozer via VMEC++, unconditionally -- this
    audit's whole point is comparing against the real thing, not an optional
    extra) unless `settings` overrides it.

    Mirrors `alm_ngopt_optimizer.py`'s own canonical-confirm convention: a
    high-fidelity forward-model call may legitimately fail to converge, and
    that's a recorded outcome (`canonical_non_convergent=True`, `qi`/
    `log10_qi` both `None`), not a crash.
    """
    resolved_settings = (
        settings
        if settings is not None
        else forward_model.ConstellarationSettings.default_high_fidelity()
    )
    try:
        metrics, _ = forward_model_fn(boundary, settings=resolved_settings)
    except Exception:
        return {"qi": None, "log10_qi": None, "canonical_non_convergent": True}

    qi_value = metrics.qi
    log10_qi = float(np.log10(qi_value)) if qi_value is not None and qi_value > 0 else None
    return {"qi": qi_value, "log10_qi": log10_qi, "canonical_non_convergent": False}


def record_desc_proxy_vs_canonical(
    targets: Any,
    desc_settings: Any,
    *,
    optimize_fn: Callable[
        [Any, Any], tuple[Any, list[dict[str, Any]]]
    ] = run_desc_omnigenity_with_capture,
    forward_model_fn: Callable[..., Any] = forward_model.forward_model,
    canonical_settings: forward_model.ConstellarationSettings | None = None,
    proxy_term_key: str = _DEFAULT_PROXY_TERM_KEY,
    proxy_stat: str = _DEFAULT_PROXY_STAT,
    label: str | None = None,
) -> dict[str, Any]:
    """The DESC-proxy-path clean entry point Phase 2 calls per design: runs
    the DESC omnigenity optimization with capture, extracts the in-loop
    proxy value the optimizer trusted, recomputes canonical QI on the output
    boundary, and returns a single record with both plus their delta/ratio
    for later characterization of where the proxy diverges from canonical
    QI -- the audit finding this instrumentation exists to harvest.

    `optimize_fn` must return `(boundary, results)` where `results` is a
    list of one or more raw DESC `OptimizeResult`s (see
    `run_desc_omnigenity_with_capture`) -- the one actually scoring
    `proxy_term_key` is picked out via `select_optimize_result_with_term`.
    `optimize_fn`/`forward_model_fn` are injectable for testing (defaults
    are the real, expensive calls). `delta`/`ratio` are `None` whenever
    canonical QI didn't converge, since neither is meaningful against a
    missing value.
    """
    boundary, results = optimize_fn(targets, desc_settings)
    optimize_result = select_optimize_result_with_term(results, term_key=proxy_term_key)
    objective_values = optimize_result["Objective values"]
    proxy_before, proxy_after = extract_proxy_objective_values(
        objective_values, term_key=proxy_term_key, stat=proxy_stat
    )

    canonical = recompute_canonical_qi(
        boundary, forward_model_fn=forward_model_fn, settings=canonical_settings
    )
    canonical_qi = canonical["qi"]

    delta = None if canonical_qi is None else canonical_qi - proxy_after
    ratio = None if canonical_qi is None or not proxy_after else canonical_qi / proxy_after

    return {
        "label": label,
        "proxy_term_key": proxy_term_key,
        "proxy_stat": proxy_stat,
        "proxy_omnigenity_before": proxy_before,
        "proxy_omnigenity_after": proxy_after,
        "canonical_qi": canonical_qi,
        "canonical_log10_qi": canonical["log10_qi"],
        "canonical_non_convergent": canonical["canonical_non_convergent"],
        "proxy_vs_canonical_delta": delta,
        "proxy_vs_canonical_ratio": ratio,
        "boundary": {
            "r_cos": np.asarray(boundary.r_cos).tolist(),
            "z_sin": np.asarray(boundary.z_sin).tolist(),
        },
    }


def summarize_alm_ngopt_trajectory(trajectory_log_path: Path | str) -> dict[str, Any]:
    """ALM-NGOpt canonical-scored-path symmetry capture (plan item 2): reads
    back the per-call trajectory `run_alm_ngopt_optimization` already writes
    via its `trajectory_log_path` (one JSON line per forward-model call --
    see `alm_ngopt_optimizer.record()`), via `trajectory_logger
    .read_trajectory`, and summarizes it.

    This path scores every candidate canonically (VMEC++ `forward_model`
    directly, not a differentiable proxy), so it has no separate in-loop
    proxy value to compare against -- `proxy_gap` is `None` by construction,
    with `proxy_gap_note` explaining why, so a record reader doesn't mistake
    the absence for a missing measurement.

    `best_record` is the last trajectory line with `is_new_best=True` (the
    ALM-NGOpt run's own direction-aware
    `_improves_on_incumbent`/`record()` already decided this -- no need to
    re-derive minimize/maximize direction here), or `None` if no evaluation
    was ever feasible.
    """
    records = trajectory_logger.read_trajectory(trajectory_log_path)
    n_calls = len(records)
    n_converged = sum(1 for r in records if r.get("converged"))
    n_feasible = sum(1 for r in records if r.get("feasible"))
    best_candidates = [r for r in records if r.get("is_new_best")]
    best_record = best_candidates[-1] if best_candidates else None

    return {
        "n_calls": n_calls,
        "n_converged": n_converged,
        "n_feasible": n_feasible,
        "best_record": best_record,
        "proxy_gap": None,
        "proxy_gap_note": (
            "not applicable -- this path scores every candidate canonically "
            "(forward_model/VMEC++) directly, with no separate differentiable "
            "proxy in the loop; captured for completeness/symmetry with the "
            "DESC proxy-path record per qi-discovery-phase-plan.md Phase 1 "
            "item 2."
        ),
        "trajectory": records,
    }
