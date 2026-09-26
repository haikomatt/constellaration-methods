"""Step 0 pivot plan (`step0-pivot-alm-ngopt-plan.md`), Step 2: standalone
ALM-NGOpt wrapper, no LLM involved. Same purpose as `ground_truth_optimizer.py`
(answer whether P1 feasibility is reachable, independent of the LLM's search
quality) but using the paper's own baseline method instead of scipy, after
arXiv 2506.19583 Table 4 confirmed scipy-trust-constr and scipy-COBYQA both
fail to reach feasibility on this exact problem -- matching this project's
own scipy fleet's null result -- while ALM-NGOpt (augmented Lagrangian
wrapping Nevergrad's NGOpt) does.

Reimplements `constellaration.optimization.augmented_lagrangian_runner.run()`'s
loop body rather than calling it directly, for two independent reasons found
reading it for Step 1 of the pivot plan:

1. `run()` has the exact missing-return-value bug `scipy_minimize_runner.run()`
   had: it loops over outer iterations, prints via `_logging()`, and falls off
   the end with no `return`. Every `AugmentedLagrangianState` it builds --
   the optimized `x`, the whole objective/constraint trajectory -- is local
   and discarded.
2. `AugmentedLagrangianMethodSettings.penalty_parameters_initial` is a dead
   field: `run()` hardcodes `state.penalty_parameters = jnp.ones_like(constraints)`
   regardless of what's configured there, so the paper's rho_i^0=10 (Appendix
   A.5.1) would silently never take effect if `run()` were called as-is with
   that field set. Applied directly to the initial state here instead.

Hyperparameter names follow Algorithm 1 of arXiv 2506.19583 (rho_i^0, rho_max,
delta_0, gamma, delta_min, tau, sigma) -- confirmed against the appendix PDF
text directly (Algorithm 1 and its surrounding equations), not assumed from
symbol conventions:
    tau       -> AugmentedLagrangianSettings.constraint_violation_tolerance_reduction_factor
    sigma     -> AugmentedLagrangianSettings.penalty_parameters_increase_factor
    gamma     -> AugmentedLagrangianSettings.bounds_reduction_factor
    delta_min -> AugmentedLagrangianSettings.bounds_min
    rho_max   -> AugmentedLagrangianSettings.penalty_parameters_max
    delta_0   -> AugmentedLagrangianMethodSettings.bounds_initial
    rho_i^0   -> `penalty_parameters_initial` arg below (NOT the dead settings
                 field -- applied directly to the initial AugmentedLagrangianState).
Defaults below are the paper's own geometric-problem values (Appendix A.5.1):
rho_i^0=10, rho_max=1e9, delta_0=0.5, gamma=0.9, delta_min=0.05, tau=0.8,
sigma=5, NGOpt subproblem budget min(20000, 1500 + k*260).

Same fidelity discipline as `ground_truth_optimizer.py`: search runs at
`forward_model_settings` (default `ConstellarationSettings()`, low fidelity --
matching the paper's own stated "we run VMEC++ at low fidelity" during search,
Appendix A.5.1), and the final candidate is always additionally confirmed at
`high_fidelity_settings` (default `ConstellarationSettings.default_high_fidelity()`)
before anything is called feasible -- non-negotiable per this project's
fidelity-flip finding (39% of default-fidelity-converged points fail to
converge at high fidelity), and per this pivot's added obligation to check
whether the paper's own reported ALM-NGOpt result would survive the same
discipline. (Table 4's own footnote convention -- entries marked `*` are ones
where VMEC++ failed to converge "at high fidelity, i.e. the fidelity with
which we score a plasma boundary," reported at lower fidelity instead --
already shows the paper's ALM-NGOpt geometric-problem row is unmarked, i.e.
their own reported 0.0002 norm. constraint violation already converged at
their high-fidelity scoring convention. This module's own `canonical_*`
fields are this project's independent check of the same thing, not
necessarily identical VMEC settings to the paper's -- see plan doc.)

`objective_constraints()` (imported from the library; wrapped only to apply
|iota| to the iota residual, see `objective_constraints_iota_abs`) already
returns both the AL-internal `(objective, constraints)` pair NGOpt's oracle
needs *and* the raw `ConstellarationMetrics` from the forward-model call in
the same return value -- this module uses the metrics half to compute
`problem.compute_feasibility()`/`problem.is_feasible()`/`problem.get_objective()`
for trajectory logging and best-point tracking, so the logged
`constraint_violation`/`feasible` fields are directly comparable to
`ground_truth_optimizer.py`'s scipy trajectories (same ground-truth
semantics), not the AL method's own unnormalized constraint array.
"""

from __future__ import annotations

import dataclasses
import json
import multiprocessing
import os
import random
import time
from collections.abc import Callable
from concurrent import futures
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import nevergrad
import numpy as np
from constellaration import forward_model, problems
from constellaration.geometry import surface_rz_fourier as rz_fourier
from constellaration.optimization import augmented_lagrangian as al
from constellaration.utils import pytree

from constellaration_methods.engines import thread_limits

# Fifth library workaround (see module docstring for the other four):
# `augmented_lagrangian_runner.py` unconditionally calls
# `pytree.register_pydantic_data(SurfaceRZFourier, ...)` at import time, with
# no check for an existing registration. `scipy_minimize_runner.py`
# (imported by `ground_truth_optimizer.py`) does the exact same
# unconditional registration independently -- so importing both modules in
# one process, in *either* order (confirmed directly both ways: `import
# ground_truth_optimizer; import alm_ngopt_optimizer` and the reverse both
# raise), crashes with jax's "Duplicate custom PyTreeDef type registration".
# Not a project bug, but a real ordering hazard anything importing both
# modules will hit -- this project's own test suite included, once both had
# test coverage (pytest's alphabetical collection order decides which one
# "wins" the real registration otherwise). Patch the registration call to be
# idempotent process-wide, permanently, from the moment this module is
# imported -- narrower scoping (e.g. restoring the original after this one
# import) only protects the "alm imported after ground_truth" ordering, not
# the reverse. Re-registering the exact same class via the exact same
# library function is always a harmless no-op, never silently-wrong data,
# so a permanent patch carries no correctness risk.
_original_register_pydantic_data = pytree.register_pydantic_data


def _register_pydantic_data_idempotent(*args, **kwargs):
    try:
        return _original_register_pydantic_data(*args, **kwargs)
    except ValueError as exc:
        if "Duplicate custom PyTreeDef" not in str(exc):
            raise
        return args[0]


pytree.register_pydantic_data = _register_pydantic_data_idempotent

# Silence constellaration's per-call geometry_utils fsolve RuntimeWarning in
# the MAIN process too (workers get the same filter via
# `thread_limits.pin_worker`) -- benign, fires on nearly every forward-model
# call, drowns real signal in run logs. Message-targeted: every other
# RuntimeWarning still surfaces.
import warnings  # noqa: E402

warnings.filterwarnings("ignore", message="The iteration is not making good progress")
from constellaration.optimization.augmented_lagrangian_runner import (  # noqa: E402
    objective_constraints,
)

# Inlined from the excluded `task_template` module (that module's own reason
# for exclusion: it carries LLM prompt-templating and a
# `problem_spec`-derived schema out of this repo's scope). Only the
# mode-budget defaults and the boundary-dict parser this module actually
# calls are reproduced here -- `MAX_POLOIDAL_MODE`/`MAX_TOROIDAL_MODE`/
# `N_FIELD_PERIODS`/`IS_STELLARATOR_SYMMETRIC` are `task_template`'s own
# pinned constants (the historical LLM-prompt-path mode truncation and
# nfp=3 pin), and `_candidate_dict_to_boundary` is a direct copy of
# `task_template.candidate_dict_to_boundary`.
MAX_POLOIDAL_MODE = 4
MAX_TOROIDAL_MODE = 4
N_FIELD_PERIODS = 3
IS_STELLARATOR_SYMMETRIC = True


def _candidate_dict_to_boundary(
    boundary_dict: dict[str, Any],
    max_poloidal_mode: int = MAX_POLOIDAL_MODE,
    max_toroidal_mode: int = MAX_TOROIDAL_MODE,
    n_field_periods: int | None = None,
) -> rz_fourier.SurfaceRZFourier:
    """Parses the `boundary` field of a model response into a validated
    SurfaceRZFourier, enforcing the given mode truncation (defaulting to the
    LLM-prompt path's pinned truncation) and symmetry. Raises
    pydantic.ValidationError (shape/type) or ValueError (wrong shape for
    the given truncation) on malformed output -- callers should treat both
    as a retry signal, not a crash.

    `n_field_periods` (`None` keeps this function's historical behaviour):
    every P1 call site pins `N_FIELD_PERIODS=3` unconditionally; an explicit
    override lets a caller whose seed isn't nfp=3 (e.g. an nfp=1 NAE-seeded
    QI boundary) avoid silently mislabeling its geometry.
    """
    r_cos = np.asarray(boundary_dict["r_cos"], dtype=float)
    z_sin = np.asarray(boundary_dict["z_sin"], dtype=float)

    expected_shape = (max_poloidal_mode + 1, 2 * max_toroidal_mode + 1)
    if r_cos.shape != expected_shape or z_sin.shape != expected_shape:
        raise ValueError(
            f"Expected r_cos/z_sin shape {expected_shape} for "
            f"max_poloidal_mode={max_poloidal_mode}/max_toroidal_mode="
            f"{max_toroidal_mode}, got r_cos={r_cos.shape} z_sin={z_sin.shape}"
        )

    return rz_fourier.SurfaceRZFourier(
        r_cos=r_cos,
        z_sin=z_sin,
        n_field_periods=(n_field_periods if n_field_periods is not None else N_FIELD_PERIODS),
        is_stellarator_symmetric=IS_STELLARATOR_SYMMETRIC,
    )


def iota_abs_constraints(problem, metrics, constraints):
    """Re-express the iota residual with |iota| for P2/P3 (index 1 in both
    branches of the library's objective_constraints), matching the canonical
    `_normalized_constraint_violations`, which use `lower_bound - |iota|`.
    Without this a negative-iota design is pushed through iota = 0 toward
    +lower_bound instead of the nearer -lower_bound (budget ladder seed 176,
    2026-09-13; tests/test_p3_iota_sign_convention.py). Returns the input
    object unchanged when there is nothing to do."""
    if metrics is None or not isinstance(
        problem, (problems.SimpleToBuildQIStellarator, problems.MHDStableQIStellarator)
    ):
        return constraints
    iota = float(metrics.edge_rotational_transform_over_n_field_periods)
    if iota >= 0.0:
        return constraints
    lb = problem._edge_rotational_transform_over_n_field_periods_lower_bound
    return jnp.asarray(constraints).at[1].set(lb - abs(iota))


def objective_constraints_iota_abs(
    x, scale, problem, unravel_and_unmask_fn, settings, aspect_ratio_upper_bound
):
    """The library's objective_constraints with the iota sign fix applied.
    Top-level so it pickles for the worker pool. Every engine call site uses
    this; the library function itself is left untouched."""
    (objective, constraints), metrics = objective_constraints(
        x, scale, problem, unravel_and_unmask_fn, settings, aspect_ratio_upper_bound
    )
    return (objective, iota_abs_constraints(problem, metrics, constraints)), metrics


class CandidateTracker:
    """`claude-code-brief-candidate-logging-fix.md`, change 1: persists
    boundary coefficients for a *broad* set of feasible candidates, not just
    the single strict-best-objective point the 2026-07-13 overnight run
    relied on. That point turned out to be an arbitrary tiebreak pick --
    `record()`'s `is_new_best` uses a strict `<` on objective, so the one
    candidate that got coefficient-logged and canonical-checked was whichever
    happened to nudge the objective record, not the most robust feasible
    point found. 67% of that run's 15,168 feasible calls had constraint
    violation <= 1e-6 and none of them had their coefficients saved.

    Two independent retention policies, both applied to every feasible call:
      - deep-margin reservoir: every feasible call at or below
        `deep_margin_threshold` constraint violation, reservoir-sampled
        (Algorithm R) down to `reservoir_cap` so file size stays bounded
        without biasing toward whichever such calls happened first.
      - best-N shortlist: the `shortlist_n` best feasible calls by
        `(objective ascending, constraint_violation ascending)` for a
        minimize-objective problem (descending on objective for a
        maximize-objective one -- see `consider()`'s `minimize_objective`
        arg), which always includes the strict-best-objective point
        (directly comparable to the one candidate the overnight run saved)
        alongside its deep-margin near-twins.
    """

    def __init__(
        self,
        deep_margin_threshold: float = 1e-6,
        reservoir_cap: int = 500,
        shortlist_n: int = 25,
        rng: random.Random | None = None,
        near_feasible_threshold: float = 0.05,
        near_shortlist_n: int = 25,
    ) -> None:
        self.deep_margin_threshold = deep_margin_threshold
        self.reservoir_cap = reservoir_cap
        self.shortlist_n = shortlist_n
        self._rng = rng if rng is not None else random.Random()
        self._reservoir: list[dict[str, Any]] = []
        self._deep_margin_seen = 0
        self._shortlist: list[dict[str, Any]] = []
        # session-handoff-2026-08-17: infeasible-but-close retention. The
        # seed-5 descent's true best point (k=38, violation 0.008620) was
        # LOST because this tracker only ever admitted feasible calls -- the
        # candidates log was 0 bytes after a 46-iteration descent that never
        # (canonically) crossed feasibility. Best-N by violation, kept in a
        # stream separate from the feasible one so downstream consumers of
        # the feasible candidates file (`confirm_candidates.py`) never see
        # infeasible records mixed in.
        self.near_feasible_threshold = near_feasible_threshold
        self.near_shortlist_n = near_shortlist_n
        self._near_shortlist: list[dict[str, Any]] = []

    def consider(
        self,
        record: dict[str, Any],
        minimize_objective: bool = True,
        feasible: bool = True,
    ) -> None:
        """`record` must have `call_index`, `objective`, and
        `constraint_violation`. `feasible` (default `True`, every pre-gate
        call site's contract) routes the record: feasible calls feed the
        deep-margin reservoir + best-N-by-objective shortlist below;
        `feasible=False` calls feed the near-feasible best-N-by-violation
        shortlist instead (and only when at or below
        `near_feasible_threshold` -- far-from-feasible points carry no
        recovery value).

        `minimize_objective` (default `True`, matching every call site until
        this retarget -- `GeometricalProblem` minimizes elongation): whether
        a *lower* `record["objective"]` is better, per
        `problem.get_objective()`'s own second tuple element. Retargeting at
        `SimpleToBuildQIStellarator` (which *maximizes* the minimum
        normalized magnetic gradient scale length, `get_objective` ->
        `(value, False)`) needs `minimize_objective=False` here, or the
        shortlist would silently keep the worst feasible candidates instead
        of the best."""
        if not feasible:
            if record["constraint_violation"] > self.near_feasible_threshold:
                return
            self._near_shortlist.append(record)
            # Violation is the only ranking that matters for a
            # closest-to-feasible recovery point; objective tiebreaks are
            # direction-ambiguous here, so call_index keeps it deterministic.
            self._near_shortlist.sort(key=lambda r: (r["constraint_violation"], r["call_index"]))
            del self._near_shortlist[self.near_shortlist_n :]
            return

        if record["constraint_violation"] <= self.deep_margin_threshold:
            self._deep_margin_seen += 1
            if len(self._reservoir) < self.reservoir_cap:
                self._reservoir.append(record)
            else:
                j = self._rng.randint(1, self._deep_margin_seen)
                if j <= self.reservoir_cap:
                    self._reservoir[j - 1] = record

        self._shortlist.append(record)
        objective_sign = 1.0 if minimize_objective else -1.0
        self._shortlist.sort(
            key=lambda r: (objective_sign * r["objective"], r["constraint_violation"])
        )
        del self._shortlist[self.shortlist_n :]

    def combined(self) -> list[dict[str, Any]]:
        """Deep-margin reservoir and best-N shortlist, deduped by
        `call_index` (the same call can legitimately qualify for both),
        ordered by `call_index` for a stable, readable file."""
        merged: dict[Any, dict[str, Any]] = {r["call_index"]: r for r in self._reservoir}
        merged.update({r["call_index"]: r for r in self._shortlist})
        return sorted(merged.values(), key=lambda r: r["call_index"])

    def near_feasible(self) -> list[dict[str, Any]]:
        """The near-feasible best-N shortlist, violation ascending -- the
        infeasible-but-close recovery points `feasible=False` `consider()`
        calls accumulated."""
        return list(self._near_shortlist)

    @staticmethod
    def _write_jsonl_atomic(path: Path | str, records: list[dict[str, Any]]) -> None:
        """Same tmp-sidecar + `os.replace` discipline as
        `_write_checkpoint_atomic`: with periodic mid-run flushes
        (qi-canonical-polish-plan Phase 1) a crash mid-write must corrupt
        nothing -- readers always see either the previous or the new file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        os.replace(tmp, path)

    def write(self, path: Path | str, near_feasible_path: Path | str | None = None) -> None:
        self._write_jsonl_atomic(path, self.combined())
        if near_feasible_path is not None:
            self._write_jsonl_atomic(near_feasible_path, self.near_feasible())


def prepare_initial_boundary(
    boundary_dict: dict[str, Any],
    max_poloidal_mode: int,
    max_toroidal_mode: int,
    n_field_periods: int | None = None,
) -> rz_fourier.SurfaceRZFourier:
    """Parses a warm-start/seed boundary and brings it to the run's mode
    budget (G0, session-handoff-2026-07-15: the budget is an explicit
    parameter, no longer the silent 4x4 pin that manufactured the fake ~1.6
    wall). A boundary *below* the budget is zero-padded up -- the
    expand-an-anchor-into-a-wider-representation move. A boundary *above*
    it raises: silently truncating mode content is the exact original
    defect, so it must never happen as a side effect of a default.

    `n_field_periods` (`None` preserves the historical behaviour -- every
    caller before qi-discovery-phase-plan.md Phase 2a is nfp=3-pinned via
    `task_template.candidate_dict_to_boundary`'s own default): forwarded
    straight through so an nfp=1 NAE-seeded QI boundary (the benchmark's own
    recipe for `SimpleToBuildQIStellarator`) isn't silently relabeled nfp=3
    -- see `task_template.candidate_dict_to_boundary`'s docstring for why
    that would be a real geometry-corrupting bug, not just cosmetic.
    """
    r_cos = np.asarray(boundary_dict["r_cos"], dtype=float)
    if r_cos.ndim != 2 or r_cos.shape[1] % 2 == 0:
        raise ValueError(
            f"r_cos must be 2-D with an odd column count (n = -N..N), got shape {r_cos.shape}"
        )
    in_poloidal = r_cos.shape[0] - 1
    in_toroidal = (r_cos.shape[1] - 1) // 2
    if in_poloidal > max_poloidal_mode or in_toroidal > max_toroidal_mode:
        raise ValueError(
            f"initial boundary carries mode content up to m={in_poloidal}/"
            f"n={in_toroidal}, above the run's budget max_poloidal_mode="
            f"{max_poloidal_mode}/max_toroidal_mode={max_toroidal_mode} -- "
            "refusing to silently truncate it; raise the budget instead"
        )
    boundary = _candidate_dict_to_boundary(
        boundary_dict,
        max_poloidal_mode=in_poloidal,
        max_toroidal_mode=in_toroidal,
        n_field_periods=n_field_periods,
    )
    return rz_fourier.set_max_mode_numbers(
        surface=boundary,
        max_poloidal_mode=max_poloidal_mode,
        max_toroidal_mode=max_toroidal_mode,
    )


def _resolve_needs_qi(
    problem: problems.GeometricalProblem | problems.SimpleToBuildQIStellarator,
) -> bool:
    """Whether `problem`'s own evaluation convention requires QI/Boozer
    metrics (`problem._does_it_require_qi`, the exact attribute
    `constellaration.problems.SingleObjectiveProblem.evaluate()` branches on
    itself) -- `False` for `GeometricalProblem` (aspect ratio, triangularity,
    edge iota only), `True` for `SimpleToBuildQIStellarator` (its 5th
    constraint -- log10(qi) <= -4.0 -- *is* the QI residual). Defaults to
    `True` (the safer, QI-active default) if `problem` doesn't carry the
    attribute at all."""
    return bool(getattr(problem, "_does_it_require_qi", True))


def _search_settings_for(
    problem: problems.GeometricalProblem | problems.SimpleToBuildQIStellarator,
    forward_model_settings: forward_model.ConstellarationSettings | None,
) -> forward_model.ConstellarationSettings:
    """Search-time (in-loop) forward-model settings. An explicit
    `forward_model_settings` always wins. Otherwise: bare
    `ConstellarationSettings()` computes QI metrics unconditionally unless
    `qi_settings=None`. For a problem that doesn't need QI
    (`GeometricalProblem`), a rotating-ellipse boundary can trip
    `qi.quasi_isodynamicity_residual`'s "Minimum of B is at the boundary"
    ValueError, which `forward_model.forward_model` swallows into a bare
    non-convergent (metrics=None) result -- a spurious search-signal loss for
    a computation that problem doesn't need. Skipped in that case, same
    convention `problems.py`'s own high-fidelity `evaluate()` uses via
    `_does_it_require_qi`/`default_high_fidelity_skip_qi()`. For a problem
    that DOES need QI (`SimpleToBuildQIStellarator`), QI/Boozer stays active
    -- it's one of the problem's own constraints, not an optional extra."""
    if forward_model_settings is not None:
        return forward_model_settings
    needs_qi = _resolve_needs_qi(problem)
    return forward_model.ConstellarationSettings(
        qi_settings=None if not needs_qi else forward_model.ConstellarationSettings().qi_settings
    )


def _canonical_settings_for(
    problem: problems.GeometricalProblem | problems.SimpleToBuildQIStellarator,
    high_fidelity_settings: forward_model.ConstellarationSettings | None,
) -> forward_model.ConstellarationSettings:
    """Final canonical-confirmation settings. An explicit
    `high_fidelity_settings` always wins. Otherwise matches
    `constellaration.problems.SingleObjectiveProblem.evaluate()`'s own
    convention exactly: `default_high_fidelity_skip_qi()` when
    `_does_it_require_qi` is False (true for `GeometricalProblem`), not
    `default_high_fidelity()` unconditionally -- confirmed this matters, not
    just theoretical: of the 2026-07-14 48-candidate confirmation run's 48
    canonical checks, 2 failed on the QI-only `ValueError: Minimum of B is at
    the boundary` (vs. 46 on a genuine `RuntimeError: VMEC++ did not
    converge`), and both of those 2 flip to canonical_feasible=True under
    `default_high_fidelity_skip_qi()` -- see
    `claude-code-brief-paper-baseline-check.md` Part B. For
    `SimpleToBuildQIStellarator`, `default_high_fidelity()` is used
    unconditionally -- QI/Boozer is one of its own constraints, never
    optional."""
    if high_fidelity_settings is not None:
        return high_fidelity_settings
    needs_qi = _resolve_needs_qi(problem)
    return (
        forward_model.ConstellarationSettings.default_high_fidelity()
        if needs_qi
        else forward_model.ConstellarationSettings.default_high_fidelity_skip_qi()
    )


def _improves_on_incumbent(
    objective_value: float,
    minimize_objective: bool,
    incumbent_objective: float,
) -> bool:
    """Direction-aware "is `objective_value` strictly better than
    `incumbent_objective`" check, per `problem.get_objective()`'s own second
    tuple element -- generalizes `record()`'s original hardcoded `<`
    comparison, which was only ever exercised against `GeometricalProblem`
    (minimizes `max_elongation`, so `<` was correct) until this retarget.
    `SimpleToBuildQIStellarator` *maximizes* the minimum normalized magnetic
    gradient scale length (`get_objective` returns `(value, False)`), so a
    bare `<` would silently keep the WORST feasible candidate seen instead
    of the best -- the same minimize/maximize distinction `seed_buffer.py`/
    `run_loop.py` already thread through to `buffer.Buffer` elsewhere in
    this repo, generalized into this module's own best-tracking."""
    if minimize_objective:
        return objective_value < incumbent_objective
    return objective_value > incumbent_objective


def _should_confirm_canonically(
    *,
    gate_enabled: bool,
    feasible: bool,
    violation: float,
    last_confirm_violation: float,
    min_improvement: float = 0.0,
) -> bool:
    """Canonical feasibility gate trigger (session-handoff-2026-08-17): the
    in-loop forward model's feasibility verdict diverges from the canonical
    scorer's exactly in the endgame -- violation ratio ~1.01 at 0.10+,
    1.19-1.30 at 0.05-0.08, 3.39x at 0.003, where it declared the seed-5
    -3.96 point feasible (0.00296) while canonical says infeasible (0.01004).
    So a search-fidelity `feasible=True` must be confirmed canonically before
    it may flip `best_feasible_found`.

    Confirm only when the claim's violation beats `last_confirm_violation`
    (the best violation we have already paid to confirm) by MORE than the
    relative `min_improvement` fraction: this bounds the ~50-150s canonical
    calls to the descent's MEANINGFUL improvement steps, never per-call --
    67% of the 2026-07-13 run's 15,168 calls were search-feasible, so
    per-call confirmation would dwarf the search itself. `min_improvement=0`
    (the default) is the original strictly-better rule, which the first
    gated run (2026-08-17 afternoon) showed is not enough: a converged
    pocket produced a hairline improvement every ~60-90s, firing 93
    confirmations in ~2h15 (~85% duty cycle) that all landed at the same
    canonical margin (0.0097 +- 0.0001). Pure (like
    `_improves_on_incumbent`) so the trigger logic is unit-testable without
    the physics loop."""
    if not gate_enabled or not feasible:
        return False
    return violation < last_confirm_violation * (1.0 - min_improvement)


def _should_confirm_objective(
    *,
    feasible: bool,
    gate_enabled: bool,
    objective: float | None,
    incumbent_objective: float | None,
    minimize_objective: bool,
    min_improvement: float,
    last_confirm_objective: float | None,
) -> bool:
    """Objective-mode confirmation trigger (qi-family-search-campaign-plan
    Phase C; the B1 watermark-zero lockout): once a run holds a CONFIRMED
    feasible incumbent, a search-feasible claim whose OBJECTIVE beats the
    incumbent by more than the relative `min_improvement` must also be
    confirmable -- B1 drove its violation watermark to 0, the violation
    trigger could never fire again, and the incumbent froze at the
    first-confirmed objective for the rest of the run. Pre-incumbent
    (`incumbent_objective is None`) the violation path owns the regime and
    this never fires. `last_confirm_objective` is the objective watermark,
    symmetric to the violation one: hairline objective improvements must
    not thrash the ~minute-scale canonical calls."""
    if not gate_enabled or not feasible:
        return False
    if incumbent_objective is None or objective is None:
        return False

    def beats(value: float, reference: float) -> bool:
        if minimize_objective:
            return value < reference * (1.0 - min_improvement)
        return value > reference * (1.0 + min_improvement)

    if not beats(objective, incumbent_objective):
        return False
    if last_confirm_objective is not None and not beats(objective, last_confirm_objective):
        return False
    return True


def _updated_violation_watermark(current: float, claim_violation: float) -> float:
    """min-semantics for the violation watermark: an objective-triggered
    confirmation at HIGHER violation than the watermark must never raise
    it, or it reopens the violation-thrash the min-improvement rule closed."""
    return min(current, claim_violation)


@dataclasses.dataclass(frozen=True)
class _GateVerdict:
    """Outcome of `_apply_canonical_gate` for one candidate: the feasibility
    verdict best-tracking may act on, the canonical margin when a
    confirmation actually ran (`None` otherwise -- including a canonical
    non-convergence, which never confirms), and the updated watermarks the
    caller must carry forward."""

    effective_feasible: bool
    canonical_margin: float | None
    last_confirm_violation: float
    last_confirm_objective: float | None = None


class _AsyncConfirmSlot:
    """Single-slot, supersede-stale holding pen for async canonical
    confirmations (qi-family-search-campaign-plan Phase C): while one
    confirmation is in flight the cheap search keeps running; at most ONE
    further claim is held, and only the best seen since. Ranking: lower
    violation wins; ties break on the better objective (direction-aware).
    Pure state machine -- the executor wiring lives in the run loop, per
    the no-physics-in-unit-tests discipline."""

    def __init__(self, minimize_objective: bool = False) -> None:
        self.minimize_objective = minimize_objective
        self.in_flight: dict[str, Any] | None = None
        self.pending: dict[str, Any] | None = None

    def _better(self, a: dict[str, Any], b: dict[str, Any]) -> bool:
        if a["violation"] != b["violation"]:
            return a["violation"] < b["violation"]
        return _improves_on_incumbent(
            a.get("objective", 0.0), self.minimize_objective, b.get("objective", 0.0)
        )

    def offer(self, claim: dict[str, Any]) -> str:
        """Returns what happened: 'submit' (caller must dispatch the
        canonical call), 'held', 'superseded' (replaced the held claim), or
        'dropped'."""
        if self.in_flight is None:
            self.in_flight = claim
            return "submit"
        if self.pending is None:
            self.pending = claim
            return "held"
        if self._better(claim, self.pending):
            self.pending = claim
            return "superseded"
        return "dropped"

    def complete(self) -> dict[str, Any] | None:
        """The in-flight claim has resolved; returns it and frees the slot
        (caller applies the verdict, then asks `take_pending`)."""
        finished = self.in_flight
        self.in_flight = None
        return finished

    def take_pending(self) -> dict[str, Any] | None:
        """Promotes the held claim to in-flight (caller dispatches it)."""
        nxt = self.pending
        self.pending = None
        if nxt is not None:
            self.in_flight = nxt
        return nxt


def _apply_canonical_gate(
    *,
    feasible: bool,
    violation: float,
    last_confirm_violation: float,
    gate_enabled: bool,
    confirm_fn: Callable[[], tuple[bool, float | None]],
    min_improvement: float = 0.0,
    objective: float | None = None,
    incumbent_objective: float | None = None,
    minimize_objective: bool = True,
    objective_min_improvement: float | None = None,
    last_confirm_objective: float | None = None,
) -> _GateVerdict:
    """Applies the canonical feasibility gate to one candidate's
    search-fidelity verdict. Gate off: the search verdict passes through
    unchanged (every pre-gate caller's behaviour). Gate on: a search-feasible
    claim is either confirmed via `confirm_fn` (one canonical forward-model
    call returning `(canonical_feasible, canonical_margin)`), or -- when it
    doesn't beat the confirmation watermark -- DEMOTED to infeasible rather
    than trusted, since an unconfirmed claim is exactly the false-incumbent
    failure this gate exists to stop. A demoted/refuted point still competes
    in best-tracking on its SEARCH-fidelity violation: canonical margins are
    logged for audit but never mixed into the ordering, which would make
    candidates confirmed at different fidelities incomparable."""
    if not gate_enabled:
        return _GateVerdict(
            effective_feasible=feasible,
            canonical_margin=None,
            last_confirm_violation=last_confirm_violation,
            last_confirm_objective=last_confirm_objective,
        )
    if not feasible:
        return _GateVerdict(
            effective_feasible=False,
            canonical_margin=None,
            last_confirm_violation=last_confirm_violation,
            last_confirm_objective=last_confirm_objective,
        )
    trigger_violation = _should_confirm_canonically(
        gate_enabled=gate_enabled,
        feasible=feasible,
        violation=violation,
        last_confirm_violation=last_confirm_violation,
        min_improvement=min_improvement,
    )
    trigger_objective = objective_min_improvement is not None and _should_confirm_objective(
        feasible=feasible,
        gate_enabled=gate_enabled,
        objective=objective,
        incumbent_objective=incumbent_objective,
        minimize_objective=minimize_objective,
        min_improvement=objective_min_improvement,
        last_confirm_objective=last_confirm_objective,
    )
    if not (trigger_violation or trigger_objective):
        return _GateVerdict(
            effective_feasible=False,
            canonical_margin=None,
            last_confirm_violation=last_confirm_violation,
            last_confirm_objective=last_confirm_objective,
        )
    canonical_feasible, canonical_margin = confirm_fn()
    new_obj_watermark = last_confirm_objective
    if objective is not None:
        if new_obj_watermark is None:
            new_obj_watermark = objective
        elif minimize_objective:
            new_obj_watermark = min(new_obj_watermark, objective)
        else:
            new_obj_watermark = max(new_obj_watermark, objective)
    return _GateVerdict(
        effective_feasible=bool(canonical_feasible),
        canonical_margin=canonical_margin,
        last_confirm_violation=_updated_violation_watermark(last_confirm_violation, violation),
        last_confirm_objective=new_obj_watermark,
    )


def _deepens_margin(*, feasible: bool, violation: float, best_margin: float) -> bool:
    """Margin-incumbent predicate (qi-canonical-polish-plan Phase 1): whether
    this point becomes the deepest-margin FEASIBLE point seen. Tracked
    alongside the objective incumbent because the objective chase legally
    wanders to thin-margin winners (run 1: margin 0.00806 -> objective-better
    points at ~0.0097), and the robustness candidate must never be lost to
    that race. Infeasible points never qualify, however small their
    violation -- margin only means anything on the feasible side."""
    return feasible and violation < best_margin


def _objective_converged(
    objective_history: list[float],
    *,
    patience_iters: int | None,
    min_delta: float,
    minimize_objective: bool,
) -> bool:
    """Objective-stall predicate for the polish regime (qi-canonical-polish-
    plan Phase 1): once a run holds a feasible incumbent its violation
    flatlines by construction, so convergence must be judged on the best
    OBJECTIVE per outer iteration instead. Direction-aware
    (`SimpleToBuildQIStellarator` maximizes); maps the history onto the
    minimize convention and reuses `_convergence_stalled`'s window logic
    verbatim so the two stall rules cannot drift apart."""
    if minimize_objective:
        history = list(objective_history)
    else:
        history = [-v for v in objective_history]
    return _convergence_stalled(history, patience_iters=patience_iters, min_delta=min_delta)


def masked_coefficients_and_scale(
    boundary_dict: dict[str, Any],
    *,
    max_poloidal_mode: int,
    max_toroidal_mode: int,
    n_field_periods: int | None = None,
    infinity_norm_spectrum_scaling: float = 1.0,
):
    """The run's boundary-to-search-space machinery, extracted so external
    callers (the polish driver's data-derived trust region,
    qi-canonical-polish-plan Phase 2) measure distances in EXACTLY the
    scaled coordinate space the optimizer's `bounds_initial` lives in --
    duplicating this inline invited silent divergence. Returns
    `(boundary, mask, masked_coefficients, unravel_and_unmask_fn, scale)`;
    the optimizer's `x0` is `masked_coefficients / scale`."""
    boundary = prepare_initial_boundary(
        boundary_dict,
        max_poloidal_mode=max_poloidal_mode,
        max_toroidal_mode=max_toroidal_mode,
        n_field_periods=n_field_periods,
    )
    mask = rz_fourier.build_mask(
        boundary,
        max_poloidal_mode=max_poloidal_mode,
        max_toroidal_mode=max_toroidal_mode,
    )
    initial_guess, unravel_and_unmask_fn = pytree.mask_and_ravel(pytree=boundary, mask=mask)
    scale = rz_fourier.compute_infinity_norm_spectrum_scaling_fun(
        poloidal_modes=boundary.poloidal_modes.flatten(),
        toroidal_modes=boundary.toroidal_modes.flatten(),
        alpha=infinity_norm_spectrum_scaling,
    ).reshape(boundary.poloidal_modes.shape)
    scale = jnp.array(np.concatenate([scale[mask.r_cos], scale[mask.z_sin]]))
    return boundary, mask, initial_guess, unravel_and_unmask_fn, scale


def _bank_gate_verdict(path: Path | str, record: dict[str, Any]) -> None:
    """Event-driven gate banking (2026-08-17 afternoon, live-run finding):
    append one JSON line per canonical-gate verdict, flushed+fsynced before
    returning, so every confirmed boundary is durable the moment it exists.
    The motivating failure: five+ confirmed-feasible boundaries lived ONLY in
    process memory for hours (candidates are written at run end, checkpoints
    per-k), which made the run unstoppable without losing them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


@dataclasses.dataclass(frozen=True)
class _ResumeTrackingState:
    """Best-tracking + budget + convergence-history state a resume starts
    from, per `_resume_tracking_state`. `budget=None` routes through the
    cold-start budget schedule."""

    best_violation: float
    best_objective: float
    best_feasible_found: bool
    budget: int | None
    violation_history: list[float]


def _resume_tracking_state(
    ckpt: dict[str, Any],
    *,
    reset_best_tracking: bool = False,
    clear_feasible_incumbent: bool = False,
    fidelity_switch: bool = False,
    clear_stall_history: bool = False,
) -> _ResumeTrackingState:
    """Resolves the resume best-tracking policies into one state
    (session-handoff-2026-08-17 + qi-canonical-polish-plan Phase 1).
    Default: restore the checkpoint verbatim.

    `reset_best_tracking`: best, budget, AND history re-established from
    scratch (see `run_alm_ngopt_optimization`'s docstring). Kept for
    compatibility, but for a fidelity change `fidelity_switch` below is the
    correct policy -- this one's budget reset is what under-powered the
    polish-5b inner search (1,760 evals in an 80-D box against a 12,680-eval
    incumbent).

    `clear_feasible_incumbent`: demote a FALSE `best_feasible_found` claim
    (the seed-5 k=43 checkpoint records one, made by the unreliable in-loop
    fidelity) while KEEPING best_violation, the matured budget, and the
    convergence history -- the very state the resume exists to preserve.

    `fidelity_switch`: the correct fidelity-change policy. Best-tracking is
    fully re-established (old-fidelity violations do not compare at the new
    fidelity: keeping the proxy's 0.0018 baseline would freeze best_x
    against canonical violations ~4x larger) and the history is cleared
    (the false-stall fix, same reasoning) -- but the matured nevergrad
    BUDGET is kept, which is the polish-5b fix.

    `clear_stall_history` (Phase C climb resume): the incumbent is REAL
    (gate-confirmed) and the budget matured -- keep both verbatim -- but a
    premature stall (patience fired before the duals matured) means the
    histories must clear, or the resume re-imports the flat run and
    re-stalls on its first iteration.

    The policies contradict each other on what they keep, so combining any
    two fails loud rather than picking a quiet precedence."""
    if (
        sum(
            [
                reset_best_tracking,
                clear_feasible_incumbent,
                fidelity_switch,
                clear_stall_history,
            ]
        )
        > 1
    ):
        raise ValueError(
            "resume_reset_best_tracking, resume_clear_feasible_incumbent, "
            "resume_fidelity_switch, and resume_clear_stall_history are "
            "mutually exclusive: they disagree on which of "
            "budget/history/best-tracking a resume keeps"
        )
    if clear_stall_history:
        _bv = ckpt["best_violation"]
        _bo = ckpt["best_objective"]
        return _ResumeTrackingState(
            best_violation=float(_bv) if _bv is not None else float("inf"),
            best_objective=float(_bo) if _bo is not None else float("inf"),
            best_feasible_found=bool(ckpt["best_feasible_found"]),
            budget=int(ckpt["budget"]),
            violation_history=[],
        )
    if reset_best_tracking:
        return _ResumeTrackingState(
            best_violation=float("inf"),
            best_objective=float("inf"),
            best_feasible_found=False,
            budget=None,
            violation_history=[],
        )
    if fidelity_switch:
        return _ResumeTrackingState(
            best_violation=float("inf"),
            best_objective=float("inf"),
            best_feasible_found=False,
            budget=int(ckpt["budget"]),
            violation_history=[],
        )
    _bv = ckpt["best_violation"]
    _bo = ckpt["best_objective"]
    best_violation = float(_bv) if _bv is not None else float("inf")
    best_objective = float(_bo) if _bo is not None else float("inf")
    violation_history = [float(v) for v in ckpt.get("violation_history", [])]
    if clear_feasible_incumbent:
        return _ResumeTrackingState(
            best_violation=best_violation,
            best_objective=float("inf"),
            best_feasible_found=False,
            budget=int(ckpt["budget"]),
            violation_history=violation_history,
        )
    return _ResumeTrackingState(
        best_violation=best_violation,
        best_objective=best_objective,
        best_feasible_found=bool(ckpt["best_feasible_found"]),
        budget=int(ckpt["budget"]),
        violation_history=violation_history,
    )


def _forward_model_call_budget_reached(
    call_count: int, in_flight: int, max_forward_model_calls: int | None
) -> bool:
    """qi-discovery-phase-plan.md Phase 2a: whether `call_count` (completed
    forward-model calls) plus `in_flight` (candidates already submitted to
    the worker pool but not yet completed) has reached
    `max_forward_model_calls` -- the run's "per-seed eval budget" config
    knob the orchestrator (`src/qi_discovery.py`) requires. `None` disables
    the cap entirely (unlimited), matching `max_wall_clock_seconds`'s own
    None-disables convention elsewhere in this function. Extracted as a
    pure function (like `_improves_on_incumbent` above) so the cap logic is
    unit-testable without the real multiprocessing/physics loop -- see this
    module's test file for why that loop itself isn't unit-tested."""
    if max_forward_model_calls is None:
        return False
    return call_count + in_flight >= max_forward_model_calls


def _stall_exceeded(
    last_progress_time: float, now: float, *, max_stall_seconds: float | None
) -> bool:
    """Fail-don't-hang watchdog predicate (session-handoff-2026-07-24): whether
    more than `max_stall_seconds` has elapsed since the last forward-model call
    completed (`last_progress_time`), while evaluations are still in flight --
    i.e. the worker pool has stopped making progress and is likely dead or
    resource-starved rather than merely slow. `None` disables the watchdog
    (matching the `max_wall_clock_seconds`/`max_forward_model_calls`
    None-disables convention). Pure (like `_forward_model_call_budget_reached`)
    so the stall logic is unit-testable without the real forkserver/physics
    loop -- the 2026-07-23 overnight failure hung silently for 9.5h after a
    ~1-minute crash precisely because nothing checked for a stalled pool."""
    if max_stall_seconds is None:
        return False
    return (now - last_progress_time) > max_stall_seconds


def _convergence_stalled(
    violation_history: list[float],
    *,
    patience_iters: int | None,
    min_delta: float,
) -> bool:
    """Smart-cap convergence-plateau predicate (constellaration-smart-cap-
    experiment): whether the best constraint violation has failed to improve by
    more than `min_delta` over the last `patience_iters` completed outer
    iterations -- i.e. the augmented-Lagrangian outer loop has stopped making
    meaningful progress, so the seed should be retired and its remaining budget
    handed to a still-descending one, rather than idling out a flat wall-clock
    cap. `None` (or a non-positive count) disables the check, matching the
    `max_wall_clock_seconds`/`max_forward_model_calls`/`max_stall_seconds`
    None-disables convention elsewhere in this module.

    `violation_history[k]` is the best constraint violation as of the end of
    completed outer iteration `k` (monotonically non-increasing, since it tracks
    a running best). Improvement is measured as the *best before* the trailing
    window minus the *best within* it, so a window that fails to beat the prior
    record -- the definition of a plateau -- yields a non-positive improvement
    and trips the stop. Requires strictly more than `patience_iters` recorded
    iterations before it can trip, so a run is never retired before it has had a
    full patience window to show progress. Pure (like `_stall_exceeded` and
    `_forward_model_call_budget_reached`) so the smart-cap logic is unit-testable
    without the real forkserver/physics loop."""
    if patience_iters is None or patience_iters <= 0:
        return False
    if len(violation_history) <= patience_iters:
        return False
    prior_best = min(violation_history[:-patience_iters])
    recent_best = min(violation_history[-patience_iters:])
    return (prior_best - recent_best) <= min_delta


# Checkpoint/resume of the FULL augmented-Lagrangian state
# (constellaration-smart-cap-experiment): the smart-cap warm-start experiment
# proved that resuming geometry (x) alone re-derives a converged point, because
# the late-stage descent lives in the matured DUAL state -- penalty_parameters
# ramped high, multipliers adapted, bounds shrunk toward bounds_min. Faithful
# "continue where it stopped" therefore needs every AugmentedLagrangianState
# field persisted, plus the best-point tracker, the per-outer-iteration RNG
# stream, the running NGOpt budget, and the outer-iteration index -- everything
# the outer loop reads to produce the next iteration.
CHECKPOINT_SCHEMA = "alm_ngopt_checkpoint_v1"


def _alm_state_to_jsonable(state: al.AugmentedLagrangianState) -> dict[str, Any]:
    """Serialize an AugmentedLagrangianState (all six jax-array fields) to plain
    JSON-able lists/floats. Round-trips with `_alm_state_from_jsonable`."""
    return {
        "x": np.asarray(state.x, dtype=float).tolist(),
        "multipliers": np.asarray(state.multipliers, dtype=float).tolist(),
        "penalty_parameters": np.asarray(state.penalty_parameters, dtype=float).tolist(),
        "objective": float(np.asarray(state.objective)),
        "constraints": np.asarray(state.constraints, dtype=float).tolist(),
        "bounds": np.asarray(state.bounds, dtype=float).tolist(),
    }


def _alm_state_from_jsonable(d: dict[str, Any]) -> al.AugmentedLagrangianState:
    """Inverse of `_alm_state_to_jsonable`. Rebuilds the state at float32 (the
    dtype the live search runs at -- see the field-shape probe in the smart-cap
    experiment notes) so a resumed run matches an uninterrupted one."""
    return al.AugmentedLagrangianState(
        x=jnp.asarray(d["x"], dtype=jnp.float32),
        multipliers=jnp.asarray(d["multipliers"], dtype=jnp.float32),
        penalty_parameters=jnp.asarray(d["penalty_parameters"], dtype=jnp.float32),
        objective=jnp.asarray(d["objective"], dtype=jnp.float32),
        constraints=jnp.asarray(d["constraints"], dtype=jnp.float32),
        bounds=jnp.asarray(d["bounds"], dtype=jnp.float32),
    )


def _rng_state_to_jsonable(rng: np.random.RandomState) -> dict[str, Any]:
    """Serialize a numpy RandomState so the per-outer-iteration sub-seed stream
    (`seed_rng`) continues from the exact same position after a resume, rather
    than restarting and replaying sub-seeds already consumed."""
    key, state, pos, has_gauss, cached = rng.get_state()
    return {
        "key": key,
        "state": np.asarray(state, dtype=np.uint32).tolist(),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached),
    }


def _rng_state_from_jsonable(d: dict[str, Any]) -> np.random.RandomState:
    """Inverse of `_rng_state_to_jsonable`."""
    rng = np.random.RandomState()
    rng.set_state(
        (
            d["key"],
            np.asarray(d["state"], dtype=np.uint32),
            int(d["pos"]),
            int(d["has_gauss"]),
            float(d["cached_gaussian"]),
        )
    )
    return rng


def _write_checkpoint_atomic(path: Path | str, payload: dict[str, Any]) -> None:
    """Write a checkpoint atomically (tmp file + os.replace) so a crash mid-write
    can never leave a truncated, unresumable checkpoint -- the whole point is to
    never lose ALM state again."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def _problem_get_objective(problem, metrics):
    """`get_objective` shim. Single-objective problems (P1/P2) expose it
    directly. MHDStableQIStellarator (P3) is multi-objective and exposes only
    `get_objectives`; the ALM optimises `20 - min_len` (maximise the gradient
    scale length; aspect is handled as a constraint via
    `aspect_ratio_upper_bound`), so mirror that here: raw min_len, minimise=False.
    """
    if hasattr(problem, "get_objective"):
        return problem.get_objective(metrics)
    return (metrics.minimum_normalized_magnetic_gradient_scale_length, False)


def run_alm_ngopt_optimization(
    boundary_dict: dict[str, Any],
    problem: problems.GeometricalProblem | problems.SimpleToBuildQIStellarator,
    max_poloidal_mode: int = MAX_POLOIDAL_MODE,
    max_toroidal_mode: int = MAX_TOROIDAL_MODE,
    n_field_periods: int | None = None,
    forward_model_settings: forward_model.ConstellarationSettings | None = None,
    high_fidelity_settings: forward_model.ConstellarationSettings | None = None,
    infinity_norm_spectrum_scaling: float = 1.0,
    # Algorithm 1 hyperparameters (arXiv 2506.19583 Appendix A.5.1,
    # geometric-problem column).
    penalty_parameters_initial: float = 10.0,
    penalty_parameters_max: float = 1e9,
    penalty_parameters_increase_factor: float = 5.0,
    bounds_initial: float = 0.5,
    bounds_min: float = 0.05,
    bounds_reduction_factor: float = 0.9,
    constraint_violation_tolerance_reduction_factor: float = 0.8,
    # Outer-loop and NGOpt subproblem budget.
    maxit: int = 100,
    nevergrad_budget_initial: int = 1500,
    nevergrad_budget_increment: int = 260,
    nevergrad_budget_max: int = 20000,
    num_workers: int = 2,
    batch_mode: bool = True,
    max_wall_clock_seconds: float | None = None,
    max_forward_model_calls: int | None = None,
    max_stall_seconds: float | None = 600.0,
    convergence_patience_iters: int | None = None,
    convergence_min_delta: float = 0.0,
    checkpoint_path: Path | str | None = None,
    checkpoint_every_iters: int = 1,
    checkpoint_history_dir: Path | str | None = None,
    resume_from_checkpoint: Path | str | None = None,
    resume_reset_best_tracking: bool = False,
    resume_clear_feasible_incumbent: bool = False,
    resume_fidelity_switch: bool = False,
    resume_clear_stall_history: bool = False,
    canonical_confirm_feasible: bool = False,
    canonical_confirm_min_improvement: float = 0.0,
    canonical_confirm_objective_improvement: float | None = None,
    canonical_confirm_async: bool = False,
    gate_log_path: Path | str | None = None,
    feasible_bank_path: Path | str | None = None,
    stall_metric: str = "violation",
    tracker_flush_every: int = 0,
    trajectory_log_path: Path | str | None = None,
    candidates_log_path: Path | str | None = None,
    # MHDStableQIStellarator (P3) only: aspect is an objective there, but the
    # single-design ALM treats it as a constraint with this upper bound (the
    # migration knob). objective_constraints asserts it is not None for P3;
    # None (default) is correct for P1/P2, which ignore it.
    aspect_ratio_upper_bound: float | None = None,
    deep_margin_threshold: float = 1e-6,
    candidates_reservoir_cap: int = 500,
    candidates_shortlist_n: int = 25,
    near_feasible_threshold: float = 0.05,
    candidates_near_shortlist_n: int = 25,
    near_feasible_log_path: Path | str | None = None,
    random_seed: int | None = None,
    progress_every: int = 50,
) -> dict[str, Any]:
    """Runs the augmented Lagrangian / NGOpt method directly against
    `problem`, starting from `boundary_dict`. See module docstring for why
    this reimplements `augmented_lagrangian_runner.run()`'s body instead of
    calling it, and for the hyperparameter-name mapping to the paper's
    Algorithm 1 symbols.

    `max_poloidal_mode`/`max_toroidal_mode` set the run's Fourier mode
    budget -- the search-space dimensionality. Defaults stay at the
    historical 4x4 pin for reproducibility of prior runs, but the value is
    now explicit and stamped into the result dict and every candidate
    record (G0: the silent pin is what manufactured the fake ~1.6 wall;
    see `prepare_initial_boundary` for the warm-start shape rules).

    `n_field_periods` (`None` preserves the historical nfp=3 pin -- see
    `prepare_initial_boundary`/`task_template.candidate_dict_to_boundary`):
    qi-discovery-phase-plan.md Phase 2a's NAE-seeded QI runs need this, since
    the benchmark's own NAE recipe for `SimpleToBuildQIStellarator` is
    nfp=1, not the P1-only nfp=3 pin every earlier caller relied on.

    `max_wall_clock_seconds` (`None` disables): checked before submitting
    each new NGOpt candidate (inner loop) and before starting each outer
    iteration `k` -- once elapsed time exceeds the cap, no further
    candidates are submitted, in-flight ones are allowed to drain (so a
    partially-run subproblem still gets a valid dual/penalty update from
    whatever it completed), and the outer loop then stops. `wall_clock_cap_reached`
    in the return dict distinguishes this from a natural `maxit` exhaustion.

    `max_forward_model_calls` (`None` disables, qi-discovery-phase-plan.md
    Phase 2a): the "per-seed eval budget" config knob the discovery
    orchestrator (`src/qi_discovery.py`) requires -- this method's own
    `maxit`/`nevergrad_budget_*` knobs bound *outer-iteration* structure,
    not a total forward-model-call count directly, so a caller wanting an
    exact evals cap (matching the plan's pre-registered "~500 ALM-NGOpt
    evals per seed") needs this separate stopping condition. Same discipline
    as `max_wall_clock_seconds`: checked before submitting each new
    candidate and before starting each outer iteration via
    `_forward_model_call_budget_reached` (counting both completed calls and
    ones already in flight, so a full batch isn't over-submitted right at
    the boundary); in-flight candidates are allowed to drain rather than
    abandoned. `eval_budget_cap_reached` in the return dict distinguishes
    this from wall-clock or natural `maxit` exhaustion.

    `max_stall_seconds` (default 600s = 10 min; `None` disables): the
    fail-don't-hang watchdog (session-handoff-2026-07-24). While evaluations
    are in flight, the wait on the worker pool is polled rather than blocked
    indefinitely; if no forward-model call completes for `max_stall_seconds`,
    the run kills any surviving child processes and raises `RuntimeError`
    instead of idling forever. This is the safety net for the failure mode
    that wasted the 2026-07-23 overnight run: a resource-starved pool crashed
    ~1 minute in, then the un-timed `futures.wait` hung silently for 9.5h. The
    thread-limit pin (module top) prevents that crash in the first place; this
    watchdog guarantees any *future* wedge fails loud and fast. The default
    must comfortably exceed worker cold-start (forkserver spawn + each worker's
    first XLA compile) and any single forward-model call; 30 min does.

    `convergence_patience_iters` (`None` disables; constellaration-smart-cap-
    experiment): the smart-cap stopping condition -- retire the run once its
    best constraint violation has failed to improve by more than
    `convergence_min_delta` over the last `convergence_patience_iters` completed
    outer iterations (checked via `_convergence_stalled` after each outer
    iteration's dual/penalty update). Unlike `max_stall_seconds` (a *failure*
    watchdog for a dead pool), this is a *convergence* stop for a healthy run
    that has flat-lined, so a caller sweeping many warm-started seeds can hand a
    plateaued seed's remaining budget to a still-descending one instead of
    idling out a flat per-seed wall-clock cap. `convergence_stall_stop` in the
    return dict distinguishes this from wall-clock/eval-budget/`maxit`
    exhaustion. Default-off, so every existing caller is unchanged.

    `checkpoint_path` (`None` disables): after every `checkpoint_every_iters`
    completed outer iterations, atomically write the FULL resumable state --
    the entire `AugmentedLagrangianState` (x, multipliers, penalty_parameters,
    objective, constraints, bounds), the best-point tracker, the per-outer-
    iteration RNG stream, the running NGOpt budget, and the outer-iteration
    index. This is the fix for the smart-cap experiment's finding that
    warm-starting geometry alone re-derives a converged point: the late-stage
    descent lives in the matured dual state, so a faithful resume must restore
    all of it, not just the boundary. Written after the dual/penalty/budget
    update so the checkpoint always describes a clean iteration boundary.

    `checkpoint_history_dir` (`None` disables): when set, ALSO write a
    per-outer-iteration snapshot `k<NNNN>.json` into this directory alongside the
    rolling `checkpoint_path`, so any step is independently resumable/branchable
    and no good intermediate point is overwritten. The motivating loss: once a run
    reaches feasibility it switches from minimizing violation to maximizing the
    objective, and the single rolling checkpoint then tracks the best-objective
    point, silently discarding the earlier best-violation (closest-to-feasible)
    point. Each snapshot is ~13 KB, so keeping every k is cheap.

    `resume_reset_best_tracking` (default `False`): on a resume, re-establish the
    best-point tracker fresh (best unset, not-yet-feasible) instead of restoring it
    from the checkpoint. Use when resuming at a DIFFERENT `forward_model_settings`
    fidelity than the checkpoint was written at -- the restored best/feasibility
    reflect the old fidelity and would otherwise freeze `best_x` and make the run
    polish toward a false feasible incumbent. Ignored on a cold start.

    `resume_clear_feasible_incumbent` (default `False`;
    session-handoff-2026-08-17): on a resume, demote the checkpoint's
    `best_feasible_found`/`best_objective` claim while KEEPING best_violation,
    the matured budget, and the convergence history. Use when the checkpoint's
    feasible claim is known-false (made by the unreliable in-loop fidelity --
    the seed-5 k=43 case) but the descent state itself is good. Mutually
    exclusive with `resume_reset_best_tracking` -- see
    `_resume_tracking_state`.

    `canonical_confirm_feasible` (default `False`, every existing caller
    unchanged; session-handoff-2026-08-17): the canonical feasibility gate.
    The in-loop forward model's violation is faithful to the canonical
    scorer's at ~1.01x when far from feasibility but diverges to 3.39x at
    the wall -- it declared the seed-5 -3.96 point feasible (0.00296) when
    canonical scoring says infeasible (0.01004), silently switching
    best-tracking to objective-improvement ~8 outer iterations before the
    descent's true best. With the gate on, a search-feasible verdict must be
    confirmed by ONE canonical forward-model call before it may flip
    `best_feasible_found`; refuted or unconfirmed claims are demoted to
    infeasible and keep competing on their search-fidelity violation.
    Confirmations only fire on improvements over the last confirmed attempt
    (`_should_confirm_canonically`), bounding the ~50-150s calls to the
    descent's improvement steps. Runs synchronously in the main loop: the
    pool drains behind it, acceptable while confirmations are rare.

    `canonical_confirm_min_improvement` (default 0.0 = strictly-better): the
    RELATIVE improvement over the confirmation watermark a search-feasible
    claim must show before the gate pays for another canonical call. The
    first gated run thrashed at 0.0: a converged pocket emitted hairline
    improvements every ~60-90s -> 93 confirmations in ~2h15 (~85% duty
    cycle), all landing at the same canonical margin. 0.02 (2%) kills that
    failure mode.

    `gate_log_path` (`None` disables): event-driven gate banking -- one
    flushed+fsynced JSON line per canonical verdict (boundary, search
    violation, canonical margin/feasible), written the moment the verdict
    exists. Confirmed-feasible boundaries must never live only in process
    memory again (see `_bank_gate_verdict`).

    `canonical_confirm_objective_improvement` (`None` disables;
    qi-family-search-campaign-plan Phase C): the objective-mode confirmation
    trigger. Once a CONFIRMED feasible incumbent exists, a search-feasible
    claim whose objective beats the incumbent by more than this relative
    fraction is also confirmable -- the fix for B1's watermark-zero lockout
    (violation watermark hit 0, gate closed, incumbent frozen at the
    first-confirmed objective). Has its own objective watermark
    (`_should_confirm_objective`); the violation watermark uses min-
    semantics so objective-triggered confirmations can never reopen
    violation-thrash (`_updated_violation_watermark`).

    `canonical_confirm_async` (default `False` = the proven synchronous
    gate): run canonical confirmations on a dedicated single-worker
    executor so the cheap search keeps its workers during the ~40-150s
    canonical call (B1 measured ~120 evals lost per sync confirmation).
    Claims HOLD in a single-slot supersede-stale pen while one is in
    flight (`_AsyncConfirmSlot`); verdicts are applied in the main loop
    against the CURRENT incumbents; in-flight confirmations are fully
    DRAINED at each outer-iteration boundary so checkpoints only ever
    capture settled state. Watermarks advance at dispatch time (the call
    is paid for), not at verdict time.

    qi-canonical-polish-plan Phase 1 (all default-off; existing callers
    unchanged):

    `resume_fidelity_switch`: the correct fidelity-change resume -- reset
    best-tracking + clear history, KEEP the matured nevergrad budget. See
    `_resume_tracking_state` for why `resume_reset_best_tracking`'s budget
    reset was the polish-5b bug. Mutually exclusive with the other two
    resume policies.

    `feasible_bank_path`: bank EVERY effective-feasible evaluation (one
    durable JSONL line: objective, violation, canonical margin when the
    gate ran, boundary) via the `_bank_gate_verdict` path. Meant for
    all-canonical polish runs where every eval is ground truth; in
    cheap-fidelity runs the banked `violation` is proxy-fidelity -- caller's
    judgement.

    `stall_metric` (`"violation"` default | `"objective"`): which history
    the smart-cap convergence stop watches. `"objective"` is the polish
    regime -- once feasible, violation flatlines by construction, so
    convergence is judged direction-aware on the best objective per outer
    iteration (`_objective_converged`); iterations before the first
    feasible incumbent contribute no objective history.

    `tracker_flush_every` (0 disables): atomically rewrite the candidates +
    near-feasible files every N forward-model calls instead of only at run
    end, so an interrupted run keeps its shortlists.

    `best_margin` / `best_margin_x` (tracked automatically, in the result
    dict and checkpoint): the deepest-margin FEASIBLE point seen, kept
    alongside the objective incumbent so the robustness candidate is never
    lost to the objective race (`_deepens_margin`).

    `near_feasible_threshold` / `candidates_near_shortlist_n` /
    `near_feasible_log_path` (session-handoff-2026-08-17): retention for
    infeasible-but-close points -- best-N by violation at or below the
    threshold, written to `near_feasible_log_path` as JSONL (a SEPARATE file
    from `candidates_log_path`, whose consumers assume feasible records).
    The motivating loss: the seed-5 descent's true best (k=38, violation
    0.008620) was unrecoverable because the feasible-only candidates log
    stayed empty for the whole 46-iteration run. `None` disables the extra
    file; the shortlist is still tracked (cheap) either way.

    `resume_from_checkpoint` (`None` = cold start): load a `checkpoint_path`
    file written by a prior run and continue the descent from exactly that
    outer-iteration boundary -- restored dual state, best-point, RNG position,
    budget, and starting `k`. `boundary_dict` is still used to rebuild the
    (structure-only) mask/scale/unravel machinery, which must match the mode
    budget the checkpoint was written at. `resumed_from_checkpoint` and
    `outer_iteration_start` in the return dict record that a resume happened.

    `trajectory_log_path` (`None` disables): one JSON line per completed
    NGOpt candidate evaluation (outer iteration, running call count, elapsed
    seconds, ground-truth objective/constraint-violation/feasibility per
    `problem.get_objective`/`compute_feasibility`/`is_feasible`, whether it's
    a new best) -- same shape and semantics as
    `ground_truth_optimizer.run_ground_truth_optimization`'s trajectory log,
    for direct comparison against the scipy fleet's trajectories.

    `candidates_log_path` (`None` disables): boundary coefficients for a
    broad set of feasible calls, written once at the end of the run via
    `CandidateTracker` -- see its docstring and
    `claude-code-brief-candidate-logging-fix.md` for why this exists
    (the trajectory log alone only ever preserved one arbitrary candidate's
    coefficients, which undermined the 2026-07-13 overnight run's canonical
    verdict). `deep_margin_threshold`/`candidates_reservoir_cap`/
    `candidates_shortlist_n` configure `CandidateTracker` directly.

    `random_seed` (`None` generates one from OS entropy): seeds NGOpt's
    parametrization (`nevergrad.p.Array.random_state`, confirmed settable
    directly -- `AugmentedLagrangianState`/`al.py` hold no RNG state of
    their own) so the run is deterministically reproducible from the seed
    alone; a fresh sub-seed is drawn per outer iteration `k` from a single
    seeded generator so the whole run -- not just one subproblem -- replays
    identically. Also seeds `CandidateTracker`'s reservoir sampling (a
    separate stream, so search reproducibility and log-retention randomness
    don't interfere). Returned in the result dict either way, so a run that
    wasn't explicitly seeded is still reproducible after the fact.
    """
    boundary, mask, initial_guess, unravel_and_unmask_fn, scale = masked_coefficients_and_scale(
        boundary_dict,
        max_poloidal_mode=max_poloidal_mode,
        max_toroidal_mode=max_toroidal_mode,
        n_field_periods=n_field_periods,
        infinity_norm_spectrum_scaling=infinity_norm_spectrum_scaling,
    )

    x0 = jnp.array(initial_guess) / scale

    # See _search_settings_for's docstring for why this branches on
    # needs_qi -- generalized from the original GeometricalProblem-only
    # inline version so SimpleToBuildQIStellarator (needs_qi=True) gets
    # QI/Boozer active in the search loop, not just at canonical confirmation.
    search_settings = _search_settings_for(problem, forward_model_settings)

    al_settings = al.AugmentedLagrangianSettings(
        constraint_violation_tolerance_reduction_factor=(
            constraint_violation_tolerance_reduction_factor
        ),
        penalty_parameters_increase_factor=penalty_parameters_increase_factor,
        bounds_reduction_factor=bounds_reduction_factor,
        penalty_parameters_max=penalty_parameters_max,
        bounds_min=bounds_min,
    )

    random_seed_actual = (
        random_seed if random_seed is not None else int.from_bytes(os.urandom(4), "big")
    )
    seed_rng = np.random.RandomState(random_seed_actual)
    candidate_tracker = CandidateTracker(
        deep_margin_threshold=deep_margin_threshold,
        reservoir_cap=candidates_reservoir_cap,
        shortlist_n=candidates_shortlist_n,
        rng=random.Random(random_seed_actual),
        near_feasible_threshold=near_feasible_threshold,
        near_shortlist_n=candidates_near_shortlist_n,
    )

    # Canonical-confirmation settings, needed both by the feasibility gate
    # (in-loop, when `canonical_confirm_feasible`) and the final
    # canonical scoring of the returned best point. See
    # _canonical_settings_for's docstring for the fidelity-flip finding this
    # convention protects against.
    canonical_settings = _canonical_settings_for(problem, high_fidelity_settings)

    if stall_metric not in ("violation", "objective"):
        raise ValueError(f"stall_metric must be 'violation' or 'objective', got {stall_metric!r}")

    _resume_ckpt = (
        json.loads(Path(resume_from_checkpoint).read_text())
        if resume_from_checkpoint is not None
        else None
    )

    if _resume_ckpt is None:
        # Cold start: fresh dual state (multipliers 0, penalty rho_i^0, loose
        # bounds), best-point unset, RNG seeded, outer loop begins at k=0.
        (al_objective0, al_constraints0), metrics0 = objective_constraints_iota_abs(
            x0,
            scale,
            problem,
            unravel_and_unmask_fn,
            search_settings,
            aspect_ratio_upper_bound,
        )
        state = al.AugmentedLagrangianState(
            x=jnp.copy(x0),
            multipliers=jnp.zeros_like(al_constraints0),
            # rho_i^0: applied directly, NOT via the dead
            # `penalty_parameters_initial` settings field -- see module docstring.
            penalty_parameters=(jnp.ones_like(al_constraints0) * penalty_parameters_initial),
            objective=al_objective0,
            constraints=al_constraints0,
            bounds=jnp.ones_like(x0) * bounds_initial,
        )
        call_count = 0
        best_x = np.array(x0, copy=True)
        best_violation = float("inf")
        best_objective = float("inf")
        best_feasible_found = False
        best_margin = float("inf")
        best_margin_x: np.ndarray | None = None
        k_start = 0
        budget_resumed: int | None = None
        violation_history_resumed: list[float] = []
        objective_history_resumed: list[float] = []
    else:
        # Resume: restore the FULL matured dual state, best-point, RNG stream,
        # budget, and outer-iteration index so the descent continues exactly
        # where the prior run stopped -- the fix for the warm-start-geometry-only
        # null (which reset the dual state and re-derived a converged point).
        if _resume_ckpt.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError(
                f"resume checkpoint schema {_resume_ckpt.get('schema')!r} != "
                f"expected {CHECKPOINT_SCHEMA!r}"
            )
        state = _alm_state_from_jsonable(_resume_ckpt["alm_state"])
        call_count = int(_resume_ckpt["call_count"])
        best_x = np.asarray(_resume_ckpt["best_x"], dtype=float)
        seed_rng = _rng_state_from_jsonable(_resume_ckpt["seed_rng_state"])
        k_start = int(_resume_ckpt["k_completed"])
        # Best-tracking / budget / convergence-history policy -- see
        # `_resume_tracking_state`'s docstring for the three policies
        # (verbatim restore, fidelity-change full reset, demote-a-false-
        # feasible-incumbent) and why the latter two are mutually exclusive.
        # `best_x` is kept under every policy as the fallback geometry.
        _tracking = _resume_tracking_state(
            _resume_ckpt,
            reset_best_tracking=resume_reset_best_tracking,
            clear_feasible_incumbent=resume_clear_feasible_incumbent,
            fidelity_switch=resume_fidelity_switch,
            clear_stall_history=resume_clear_stall_history,
        )
        best_violation = _tracking.best_violation
        best_objective = _tracking.best_objective
        best_feasible_found = _tracking.best_feasible_found
        budget_resumed = _tracking.budget
        violation_history_resumed = _tracking.violation_history
        # Margin incumbent + objective history restore only on a verbatim
        # resume -- every non-default policy invalidates them for the same
        # reason it invalidates the best-point (fidelity change or demoted
        # incumbent).
        _verbatim = not (
            resume_reset_best_tracking
            or resume_clear_feasible_incumbent
            or resume_fidelity_switch
            or resume_clear_stall_history
        )
        # Margin incumbent survives a stall-history clear (it is REAL,
        # gate-confirmed state); only the histories reset there.
        _keep_margin = _verbatim or resume_clear_stall_history
        _bm = _resume_ckpt.get("best_margin") if _keep_margin else None
        best_margin = float(_bm) if _bm is not None else float("inf")
        _bmx = _resume_ckpt.get("best_margin_x") if _keep_margin else None
        best_margin_x = np.asarray(_bmx, dtype=float) if _bmx is not None else None
        objective_history_resumed = (
            [float(v) for v in _resume_ckpt.get("objective_history", [])] if _verbatim else []
        )

    t_start = time.perf_counter()

    traj_file = open(trajectory_log_path, "a") if trajectory_log_path is not None else None

    mp_context = multiprocessing.get_context("forkserver")
    # Disable the forkserver's default `__main__` preload so it does NOT import
    # jax before workers are forked -- otherwise jaxlib's CPU pool is sized to
    # the full CPU count and each worker's later affinity pin comes too late,
    # leaving ~130 threads/worker and re-hitting the cgroup pid limit
    # (session-handoff-2026-07-24; verified empirically). With the preload off,
    # each worker imports jax fresh at its first task, AFTER `pin_worker` has
    # restricted its affinity -> ~5-8 threads/worker.
    mp_context.set_forkserver_preload([])
    # The preload-less workers don't inherit the launcher's runtime sys.path,
    # so put src on PYTHONPATH (which the forkserver + workers DO inherit) to
    # keep the `pin_worker` initializer reference importable there.
    thread_limits.ensure_on_pythonpath(thread_limits.src_dir())

    last_confirm_violation = float("inf")
    last_confirm_objective: float | None = None
    n_canonical_confirmations = 0
    # Objective direction, captured from the first scored candidate (static
    # per problem class; needed by the objective-stall rule outside record()).
    minimize_objective_flag = True

    # Async canonical confirmations (Phase C): one dedicated worker, one
    # in-flight claim, supersede-stale holding pen. `None` when disabled.
    confirm_slot = (
        _AsyncConfirmSlot() if (canonical_confirm_feasible and canonical_confirm_async) else None
    )
    confirm_executor = (
        futures.ProcessPoolExecutor(
            max_workers=1,
            mp_context=mp_context,
            initializer=thread_limits.pin_worker,
            initargs=(num_workers,),
        )
        if confirm_slot is not None
        else None
    )
    confirm_future: futures.Future | None = None

    def dispatch_confirm(claim: dict[str, Any]) -> None:
        """Submits the claim's canonical evaluation to the confirm worker and
        advances the watermarks (dispatch = the call is paid for)."""
        nonlocal confirm_future, last_confirm_violation, last_confirm_objective
        nonlocal n_canonical_confirmations
        n_canonical_confirmations += 1
        print(
            f"    [canonical-gate] (async) confirming search-feasible claim "
            f"(search violation {claim['violation']:.6g}, objective "
            f"{claim['objective']:.6g})...",
            flush=True,
        )
        confirm_future = confirm_executor.submit(
            objective_constraints_iota_abs,
            jnp.asarray(claim["x"]),
            scale,
            problem,
            unravel_and_unmask_fn,
            canonical_settings,
            aspect_ratio_upper_bound,
        )
        last_confirm_violation = _updated_violation_watermark(
            last_confirm_violation, claim["violation"]
        )
        if claim["objective"] is not None:
            if last_confirm_objective is None:
                last_confirm_objective = claim["objective"]
            elif minimize_objective_flag:
                last_confirm_objective = min(last_confirm_objective, claim["objective"])
            else:
                last_confirm_objective = max(last_confirm_objective, claim["objective"])

    def resolve_confirm(block: bool) -> None:
        """Applies the in-flight claim's verdict once available (always, if
        `block`); banks it, updates incumbents against CURRENT state, and
        promotes+dispatches the held claim. Loops while blocking so an
        iteration-end drain settles the slot completely."""
        nonlocal confirm_future, best_x, best_violation, best_objective
        nonlocal best_feasible_found, best_margin, best_margin_x
        while confirm_future is not None:
            if not block and not confirm_future.done():
                return
            (_ignored, _also_ignored), canonical_metrics = confirm_future.result()
            confirm_future = None
            claim = confirm_slot.complete()
            if canonical_metrics is None:
                canonical_feasible: bool = False
                canonical_margin: float | None = None
            else:
                canonical_feasible = bool(problem.is_feasible(canonical_metrics))
                canonical_margin = float(problem.compute_feasibility(canonical_metrics))
            print(
                f"    [canonical-gate] (async) verdict feasible="
                f"{canonical_feasible} canonical_margin={canonical_margin}",
                flush=True,
            )
            boundary_out = unravel_and_unmask_fn(jnp.asarray(claim["x"]) * scale)
            bank_record = {
                "call_index": claim["call_index"],
                "outer_iteration": claim["k"],
                "elapsed_s": time.perf_counter() - t_start,
                "search_violation": claim["violation"],
                "objective": claim["objective"],
                "canonical_feasible": canonical_feasible,
                "canonical_margin": canonical_margin,
                "canonical_non_convergent": canonical_metrics is None,
                "boundary": {
                    "r_cos": np.asarray(boundary_out.r_cos).tolist(),
                    "z_sin": np.asarray(boundary_out.z_sin).tolist(),
                },
                "max_poloidal_mode": max_poloidal_mode,
                "max_toroidal_mode": max_toroidal_mode,
                "n_field_periods": boundary.n_field_periods,
                "random_seed": random_seed_actual,
            }
            if gate_log_path is not None:
                _bank_gate_verdict(gate_log_path, bank_record)
            if canonical_feasible:
                if not best_feasible_found or _improves_on_incumbent(
                    claim["objective"], minimize_objective_flag, best_objective
                ):
                    best_x = np.array(claim["x"], copy=True)
                    best_violation = claim["violation"]
                    best_objective = claim["objective"]
                    best_feasible_found = True
                if _deepens_margin(
                    feasible=True,
                    violation=claim["violation"],
                    best_margin=best_margin,
                ):
                    best_margin = claim["violation"]
                    best_margin_x = np.array(claim["x"], copy=True)
                if feasible_bank_path is not None:
                    _bank_gate_verdict(feasible_bank_path, bank_record)
            nxt = confirm_slot.take_pending()
            if nxt is not None:
                dispatch_confirm(nxt)

    def confirm_canonically(x: np.ndarray, violation: float, k: int) -> tuple[bool, float | None]:
        """One canonical forward-model call on `x` for the feasibility gate
        (~50-150s at scoring fidelity, so every firing is logged loudly) --
        and, when `gate_log_path` is set, one durable JSONL line per verdict
        BEFORE returning (event-driven banking). A canonical non-convergence
        returns `(False, None)`: an unscoreable point must never become the
        feasible incumbent."""
        nonlocal n_canonical_confirmations
        n_canonical_confirmations += 1
        t0 = time.perf_counter()
        print(
            f"    [canonical-gate] confirming search-feasible claim "
            f"(search violation {violation:.6g}) at canonical fidelity...",
            flush=True,
        )
        _, canonical_metrics = objective_constraints_iota_abs(
            jnp.asarray(x),
            scale,
            problem,
            unravel_and_unmask_fn,
            canonical_settings,
            aspect_ratio_upper_bound,
        )
        if canonical_metrics is None:
            canonical_feasible: bool | None = False
            canonical_margin: float | None = None
            print(
                f"    [canonical-gate] canonical model non-convergent "
                f"({time.perf_counter() - t0:.1f}s) -> claim REFUTED",
                flush=True,
            )
        else:
            canonical_feasible = bool(problem.is_feasible(canonical_metrics))
            canonical_margin = float(problem.compute_feasibility(canonical_metrics))
            print(
                f"    [canonical-gate] verdict feasible={canonical_feasible} "
                f"canonical_margin={canonical_margin:.6g} "
                f"({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )
        if gate_log_path is not None:
            boundary_out = unravel_and_unmask_fn(jnp.asarray(x) * scale)
            _bank_gate_verdict(
                gate_log_path,
                {
                    "outer_iteration": k,
                    "call_index": call_count,
                    "elapsed_s": time.perf_counter() - t_start,
                    "search_violation": violation,
                    "canonical_feasible": canonical_feasible,
                    "canonical_margin": canonical_margin,
                    "canonical_non_convergent": canonical_metrics is None,
                    "boundary": {
                        "r_cos": np.asarray(boundary_out.r_cos).tolist(),
                        "z_sin": np.asarray(boundary_out.z_sin).tolist(),
                    },
                    "max_poloidal_mode": max_poloidal_mode,
                    "max_toroidal_mode": max_toroidal_mode,
                    "n_field_periods": boundary.n_field_periods,
                    "random_seed": random_seed_actual,
                },
            )
        return bool(canonical_feasible), canonical_margin

    def record(k: int, x: np.ndarray, metrics) -> None:
        nonlocal call_count, best_x, best_violation, best_objective
        nonlocal best_feasible_found, last_confirm_violation
        nonlocal last_confirm_objective
        nonlocal best_margin, best_margin_x, minimize_objective_flag
        call_count += 1
        elapsed = time.perf_counter() - t_start

        objective_value = None
        violation = None
        feasible = False
        search_feasible = False
        canonical_margin = None
        is_new_best = False
        if metrics is not None:
            objective_value, minimize_objective = _problem_get_objective(problem, metrics)
            objective_value = float(objective_value)
            minimize_objective_flag = bool(minimize_objective)
            violation = float(problem.compute_feasibility(metrics))
            search_feasible = bool(problem.is_feasible(metrics))
            # The canonical feasibility gate (see the parameter docstrings):
            # `feasible` below is the EFFECTIVE verdict best-tracking acts
            # on; with the gate off it is exactly the search verdict.
            if confirm_slot is not None:
                # Async: apply any settled verdict first, then decide whether
                # THIS claim warrants a confirmation; claims are demoted
                # until their verdict lands.
                resolve_confirm(block=False)
                triggered = _should_confirm_canonically(
                    gate_enabled=True,
                    feasible=search_feasible,
                    violation=violation,
                    last_confirm_violation=last_confirm_violation,
                    min_improvement=canonical_confirm_min_improvement,
                ) or (
                    canonical_confirm_objective_improvement is not None
                    and _should_confirm_objective(
                        feasible=search_feasible,
                        gate_enabled=True,
                        objective=objective_value,
                        incumbent_objective=(best_objective if best_feasible_found else None),
                        minimize_objective=minimize_objective_flag,
                        min_improvement=canonical_confirm_objective_improvement,
                        last_confirm_objective=last_confirm_objective,
                    )
                )
                if triggered:
                    claim = dict(
                        call_index=call_count,
                        k=k,
                        violation=violation,
                        objective=objective_value,
                        x=np.array(x, copy=True),
                    )
                    if confirm_slot.offer(claim) == "submit":
                        dispatch_confirm(claim)
                feasible = False
                canonical_margin = None
            else:
                gate_verdict = _apply_canonical_gate(
                    feasible=search_feasible,
                    violation=violation,
                    last_confirm_violation=last_confirm_violation,
                    gate_enabled=canonical_confirm_feasible,
                    confirm_fn=lambda: confirm_canonically(x, violation, k),
                    min_improvement=canonical_confirm_min_improvement,
                    objective=objective_value,
                    incumbent_objective=(best_objective if best_feasible_found else None),
                    minimize_objective=minimize_objective_flag,
                    objective_min_improvement=(canonical_confirm_objective_improvement),
                    last_confirm_objective=last_confirm_objective,
                )
                last_confirm_violation = gate_verdict.last_confirm_violation
                last_confirm_objective = gate_verdict.last_confirm_objective
                canonical_margin = gate_verdict.canonical_margin
                feasible = gate_verdict.effective_feasible
            if feasible and not best_feasible_found:
                is_new_best = True
            elif (
                feasible
                and best_feasible_found
                and _improves_on_incumbent(objective_value, minimize_objective, best_objective)
            ):
                is_new_best = True
            elif not feasible and not best_feasible_found and violation < best_violation:
                is_new_best = True
            if is_new_best:
                best_x = np.array(x, copy=True)
                best_violation = violation
                if feasible:
                    best_objective = objective_value
                best_feasible_found = best_feasible_found or feasible

            # Margin incumbent (qi-canonical-polish-plan Phase 1): deepest-
            # margin feasible point, kept independently of the objective race.
            if _deepens_margin(feasible=feasible, violation=violation, best_margin=best_margin):
                best_margin = violation
                best_margin_x = np.array(x, copy=True)

            # All-canonical polish banking: every effective-feasible eval is
            # durable the moment it exists (same discipline as the gate log).
            if feasible and feasible_bank_path is not None:
                boundary_bank = unravel_and_unmask_fn(jnp.asarray(x) * scale)
                _bank_gate_verdict(
                    feasible_bank_path,
                    {
                        "call_index": call_count,
                        "outer_iteration": k,
                        "elapsed_s": elapsed,
                        "objective": objective_value,
                        "constraint_violation": violation,
                        "canonical_margin": canonical_margin,
                        "boundary": {
                            "r_cos": np.asarray(boundary_bank.r_cos).tolist(),
                            "z_sin": np.asarray(boundary_bank.z_sin).tolist(),
                        },
                        "max_poloidal_mode": max_poloidal_mode,
                        "max_toroidal_mode": max_toroidal_mode,
                        "n_field_periods": boundary.n_field_periods,
                        "random_seed": random_seed_actual,
                    },
                )

            if candidates_log_path is not None and (
                feasible or violation <= near_feasible_threshold
            ):
                boundary_out = unravel_and_unmask_fn(jnp.asarray(x) * scale)
                candidate_tracker.consider(
                    {
                        "call_index": call_count,
                        "outer_iteration": k,
                        "objective": objective_value,
                        "constraint_violation": violation,
                        # Gate audit trail (session-handoff-2026-08-17):
                        # whether the in-loop model claimed feasibility, and
                        # the canonical margin when a confirmation ran.
                        "search_feasible": search_feasible,
                        "canonical_margin": canonical_margin,
                        "boundary": {
                            "r_cos": np.asarray(boundary_out.r_cos).tolist(),
                            "z_sin": np.asarray(boundary_out.z_sin).tolist(),
                        },
                        # G0 provenance: the representation is part of every
                        # verdict, never an implicit constant again.
                        "max_poloidal_mode": max_poloidal_mode,
                        "max_toroidal_mode": max_toroidal_mode,
                        "n_field_periods": boundary.n_field_periods,
                        "random_seed": random_seed_actual,
                    },
                    minimize_objective=minimize_objective,
                    feasible=feasible,
                )

        if traj_file is not None:
            traj_file.write(
                json.dumps(
                    {
                        "outer_iteration": k,
                        "call": call_count,
                        "elapsed_s": elapsed,
                        "converged": metrics is not None,
                        "objective": objective_value,
                        "constraint_violation": violation,
                        "feasible": feasible,
                        "search_feasible": search_feasible,
                        "canonical_margin": canonical_margin,
                        "is_new_best": is_new_best,
                    }
                )
                + "\n"
            )
            traj_file.flush()

        if progress_every and call_count % progress_every == 0:
            print(
                f"    [alm_ngopt_optimizer] k={k} {call_count} forward-model "
                f"calls, {elapsed:.1f}s elapsed ({elapsed / call_count:.2f}s/call), "
                f"best_violation={best_violation:.4g} "
                f"best_objective={best_objective if best_feasible_found else 'n/a'}",
                flush=True,
            )

        # Periodic shortlist flush (qi-canonical-polish-plan Phase 1): an
        # interrupted run keeps its candidates/near-feasible files. Atomic
        # writes, so a crash mid-flush corrupts nothing.
        if (
            tracker_flush_every
            and candidates_log_path is not None
            and call_count % tracker_flush_every == 0
        ):
            candidate_tracker.write(candidates_log_path, near_feasible_path=near_feasible_log_path)

    if _resume_ckpt is None:
        # Only the cold start has an initial point to record; a resume picks up
        # after the last completed outer iteration, its best-point already
        # restored above.
        record(0, x0, metrics0)

    budget = nevergrad_budget_initial if budget_resumed is None else budget_resumed
    wall_clock_cap_reached = False
    eval_budget_cap_reached = False
    convergence_stall_stop = False
    violation_history: list[float] = list(violation_history_resumed)
    objective_history: list[float] = list(objective_history_resumed)
    k_completed = k_start

    for k in range(k_start, maxit):
        if (
            max_wall_clock_seconds is not None
            and (time.perf_counter() - t_start) >= max_wall_clock_seconds
        ):
            wall_clock_cap_reached = True
            break
        if _forward_model_call_budget_reached(
            call_count, in_flight=0, max_forward_model_calls=max_forward_model_calls
        ):
            eval_budget_cap_reached = True
            break

        parametrization = nevergrad.p.Array(
            init=np.array(state.x),
            lower=np.array(state.x - state.bounds),
            upper=np.array(state.x + state.bounds),
        )
        # A fresh sub-seed per outer iteration, drawn from the single
        # `random_seed_actual`-seeded `seed_rng` -- makes the whole run
        # reproducible from one stored seed, not just one subproblem.
        # Confirmed `random_state` is a directly-settable numpy.random.
        # RandomState on nevergrad's Parameter, not internal/opaque state.
        parametrization.random_state = np.random.RandomState(seed_rng.randint(0, 2**31 - 1))
        oracle = nevergrad.optimizers.NGOpt(
            parametrization=parametrization,
            budget=budget,
            num_workers=num_workers,
        )
        oracle.suggest(np.array(state.x))

        with futures.ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp_context,
            # Pin each worker to a distinct CPU band before it imports jax, so
            # jaxlib's CPU pool sizes to that band (~5-8 threads) instead of the
            # full CPU count (~130). This is the load-bearing fix for the
            # cgroup pid-limit crash; see thread_limits.pin_worker.
            initializer=thread_limits.pin_worker,
            initargs=(num_workers,),
        ) as executor:
            rest_budget = budget
            running_evaluations: list[tuple[futures.Future, Any]] = []
            # Fail-don't-hang watchdog: last time a forward-model call
            # completed. Reset when the pool opens so worker cold-start is
            # measured from here, not from t_start.
            last_progress_time = time.perf_counter()

            while rest_budget or running_evaluations:
                if (
                    max_wall_clock_seconds is not None
                    and (time.perf_counter() - t_start) >= max_wall_clock_seconds
                ):
                    wall_clock_cap_reached = True
                    rest_budget = 0
                    if not running_evaluations:
                        break

                if _forward_model_call_budget_reached(
                    call_count,
                    in_flight=len(running_evaluations),
                    max_forward_model_calls=max_forward_model_calls,
                ):
                    eval_budget_cap_reached = True
                    rest_budget = 0
                    if not running_evaluations:
                        break

                while len(running_evaluations) < min(num_workers, rest_budget):
                    candidate = oracle.ask()
                    future = executor.submit(
                        objective_constraints_iota_abs,
                        jnp.array(candidate.value),
                        scale,
                        problem,
                        unravel_and_unmask_fn,
                        search_settings,
                        aspect_ratio_upper_bound,
                    )
                    running_evaluations.append((future, candidate))
                    rest_budget -= 1

                return_when = futures.ALL_COMPLETED if batch_mode else futures.FIRST_COMPLETED
                # Poll rather than block indefinitely so the stall watchdog
                # (and the wall-clock/eval caps above) stay responsive. `None`
                # disables the watchdog and restores the original blocking
                # wait. A broken pool normally surfaces as a BrokenProcessPool
                # from `future.result()` below (which propagates cleanly and
                # exits non-zero); the watchdog additionally catches a pool
                # that wedges WITHOUT signalling any future done -- the exact
                # 9.5h silent hang of 2026-07-23.
                poll_timeout = None if max_stall_seconds is None else min(max_stall_seconds, 60.0)
                completed, _ = futures.wait(
                    [fut for fut, _ in running_evaluations],
                    timeout=poll_timeout,
                    return_when=return_when,
                )

                now = time.perf_counter()
                if completed:
                    last_progress_time = now
                elif running_evaluations and _stall_exceeded(
                    last_progress_time, now, max_stall_seconds=max_stall_seconds
                ):
                    stall_duration = now - last_progress_time
                    # Kill any surviving children first so the executor's
                    # __exit__ join can't itself hang on a stuck worker, then
                    # fail loud non-zero.
                    for child in multiprocessing.active_children():
                        child.kill()
                    raise RuntimeError(
                        "alm_ngopt_optimizer: no forward-model call completed in "
                        f"{stall_duration:.0f}s (max_stall_seconds="
                        f"{max_stall_seconds}); aborting to avoid a silent hang -- "
                        "the worker pool is dead or resource-starved. See "
                        "session-handoff-2026-07-24-fix-qi-resource-management."
                    )

                for future, candidate in running_evaluations:
                    if future in completed:
                        (al_objective, al_constraints), metrics = future.result()
                        oracle.tell(
                            candidate,
                            al.augmented_lagrangian_function(
                                al_objective, al_constraints, state
                            ).item(),
                        )
                        record(k, np.array(candidate.value), metrics)

                running_evaluations = [
                    (fut, cand) for fut, cand in running_evaluations if fut not in completed
                ]

            recommendation = oracle.provide_recommendation()
            x = jnp.array(recommendation.value)

        # Async-gate drain (Phase C): settle every in-flight/held canonical
        # confirmation BEFORE the dual update and checkpoint, so the persisted
        # state never contains a claim whose verdict is still pending.
        if confirm_slot is not None:
            resolve_confirm(block=True)

        (al_objective, al_constraints), _ = objective_constraints_iota_abs(
            x,
            scale,
            problem,
            unravel_and_unmask_fn,
            search_settings,
            aspect_ratio_upper_bound,
        )
        state = al.update_augmented_lagrangian_state(
            x=jnp.copy(x),
            objective=al_objective,
            constraints=al_constraints,
            state=state,
            settings=al_settings,
        )
        budget = int(min(nevergrad_budget_max, budget + nevergrad_budget_increment))
        k_completed = k + 1

        # Record this outer iteration's best + persist the FULL resumable state
        # BEFORE any break, so the last completed iteration -- including the one
        # a wall-clock or eval-budget cap interrupts -- is always captured. (An
        # earlier ordering wrote the checkpoint after the cap break, leaving it
        # one iteration behind the returned result, so a resume would miss the
        # seed's actual best point.) Written after the dual/penalty/budget update
        # and k_completed bump, so a resume restarts cleanly at the next
        # iteration.
        violation_history.append(best_violation)
        # Objective history only accumulates once a feasible incumbent
        # exists -- before that best_objective is the inf placeholder, and
        # the objective-stall rule (below) is only meaningful in the
        # feasible regime anyway (qi-canonical-polish-plan Phase 1).
        if best_feasible_found:
            objective_history.append(best_objective)
        if checkpoint_path is not None and (k_completed % checkpoint_every_iters == 0):
            _payload = {
                "schema": CHECKPOINT_SCHEMA,
                "alm_state": _alm_state_to_jsonable(state),
                "k_completed": k_completed,
                "budget": budget,
                "seed_rng_state": _rng_state_to_jsonable(seed_rng),
                "best_x": np.asarray(best_x, dtype=float).tolist(),
                "best_violation": (best_violation if best_violation != float("inf") else None),
                "best_objective": (best_objective if best_feasible_found else None),
                "best_feasible_found": best_feasible_found,
                "call_count": call_count,
                "violation_history": violation_history,
                "objective_history": objective_history,
                "best_margin": (best_margin if best_margin != float("inf") else None),
                "best_margin_x": (
                    np.asarray(best_margin_x, dtype=float).tolist()
                    if best_margin_x is not None
                    else None
                ),
                "random_seed_actual": random_seed_actual,
                "max_poloidal_mode": max_poloidal_mode,
                "max_toroidal_mode": max_toroidal_mode,
                "n_field_periods": boundary.n_field_periods,
                "infinity_norm_spectrum_scaling": infinity_norm_spectrum_scaling,
                "boundary_dict": boundary_dict,
            }
            # Rolling "latest" for resume-and-continue.
            _write_checkpoint_atomic(checkpoint_path, _payload)
            # Optional per-outer-iteration history so ANY step is resumable/branchable
            # (and no good intermediate point -- e.g. the best-violation point before
            # a feasible run switches to objective-maximizing -- is overwritten and
            # lost). Same payload, one file per k; ~13 KB each, negligible.
            if checkpoint_history_dir is not None:
                _write_checkpoint_atomic(
                    Path(checkpoint_history_dir) / f"k{k_completed:04d}.json", _payload
                )

        if wall_clock_cap_reached or eval_budget_cap_reached:
            break

        # Smart-cap: retire this seed once its stall metric has plateaued for
        # `convergence_patience_iters` completed outer iterations (default-off,
        # so existing callers are unchanged). `stall_metric='objective'`
        # (qi-canonical-polish-plan Phase 1) judges convergence on the best
        # objective among feasible incumbents instead -- the polish regime,
        # where violation flatlines by construction the moment the run is
        # feasible.
        if stall_metric == "objective":
            stalled = _objective_converged(
                objective_history,
                patience_iters=convergence_patience_iters,
                min_delta=convergence_min_delta,
                minimize_objective=minimize_objective_flag,
            )
        else:
            stalled = _convergence_stalled(
                violation_history,
                patience_iters=convergence_patience_iters,
                min_delta=convergence_min_delta,
            )
        if stalled:
            convergence_stall_stop = True
            break

    if traj_file is not None:
        traj_file.close()

    # Final async-gate settle + worker shutdown (a break out of the k loop --
    # wall-clock, eval-budget, stall -- can leave a confirmation in flight).
    if confirm_slot is not None:
        resolve_confirm(block=True)
    if confirm_executor is not None:
        confirm_executor.shutdown(wait=True)

    if candidates_log_path is not None:
        candidate_tracker.write(candidates_log_path, near_feasible_path=near_feasible_log_path)

    optimized_boundary = unravel_and_unmask_fn(jnp.asarray(best_x) * scale)
    _, final_metrics = objective_constraints_iota_abs(
        jnp.asarray(best_x),
        scale,
        problem,
        unravel_and_unmask_fn,
        search_settings,
        aspect_ratio_upper_bound,
    )

    try:
        canonical_metrics, _ = forward_model.forward_model(
            optimized_boundary, settings=canonical_settings
        )
        canonical_non_convergent = False
    except Exception:
        canonical_metrics = None
        canonical_non_convergent = True

    elapsed_total = time.perf_counter() - t_start

    return {
        "random_seed": random_seed_actual,
        # G0 provenance: the run's representation, recorded in the result
        # itself -- see prepare_initial_boundary.
        "max_poloidal_mode": max_poloidal_mode,
        "max_toroidal_mode": max_toroidal_mode,
        "n_field_periods": boundary.n_field_periods,
        "wall_clock_cap_reached": wall_clock_cap_reached,
        "eval_budget_cap_reached": eval_budget_cap_reached,
        "convergence_stall_stop": convergence_stall_stop,
        "canonical_gate_enabled": canonical_confirm_feasible,
        "canonical_gate_async": canonical_confirm_async,
        "canonical_gate_objective_trigger": (canonical_confirm_objective_improvement is not None),
        "n_canonical_confirmations": n_canonical_confirmations,
        "stall_metric": stall_metric,
        "best_margin": (best_margin if best_margin != float("inf") else None),
        "best_margin_boundary": (
            {
                "r_cos": np.asarray(
                    unravel_and_unmask_fn(jnp.asarray(best_margin_x) * scale).r_cos
                ).tolist(),
                "z_sin": np.asarray(
                    unravel_and_unmask_fn(jnp.asarray(best_margin_x) * scale).z_sin
                ).tolist(),
            }
            if best_margin_x is not None
            else None
        ),
        "resumed_from_checkpoint": str(resume_from_checkpoint)
        if resume_from_checkpoint is not None
        else None,
        "outer_iteration_start": k_start,
        "outer_iterations_completed": k_completed,
        "elapsed_seconds": elapsed_total,
        "n_forward_model_calls": call_count,
        "best_constraint_violation": (best_violation if best_violation != float("inf") else None),
        "best_feasible_objective": best_objective if best_feasible_found else None,
        "optimized_boundary": {
            "r_cos": np.asarray(optimized_boundary.r_cos).tolist(),
            "z_sin": np.asarray(optimized_boundary.z_sin).tolist(),
        },
        "feasible": problem.is_feasible(final_metrics) if final_metrics is not None else None,
        "feasibility_margin": problem.compute_feasibility(final_metrics)
        if final_metrics is not None
        else None,
        "objective_value": _problem_get_objective(problem, final_metrics)[0]
        if final_metrics is not None
        else None,
        "canonical_non_convergent": canonical_non_convergent,
        "canonical_feasible": problem.is_feasible(canonical_metrics)
        if canonical_metrics is not None
        else None,
        "canonical_feasibility_margin": problem.compute_feasibility(canonical_metrics)
        if canonical_metrics is not None
        else None,
        "canonical_objective_value": _problem_get_objective(problem, canonical_metrics)[0]
        if canonical_metrics is not None
        else None,
    }
