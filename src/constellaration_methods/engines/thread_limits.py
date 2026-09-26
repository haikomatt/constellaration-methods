"""Per-worker compute-thread pinning for the parallel discovery run
(session-handoff-2026-07-24-fix-qi-resource-management).

Each ProcessPoolExecutor worker in `alm_ngopt_optimizer.run_alm_ngopt_optimization`
loads the JAX/XLA/VMEC++ stack, and jaxlib's CPU runtime builds a thread pool
sized to the number of CPUs the process can see (`len(sched_getaffinity)`) --
about 130 threads per worker on this 32-logical-CPU box. At N workers that
multiplies past the phase1 container's cgroup pid limit (2048; see
`/sys/fs/cgroup/pids.max`), which is what killed the 2026-07-23 overnight run
~1 minute in with `pthread_create failed: Resource temporarily unavailable`.

Two facts, established empirically while diagnosing the failure, shape the fix:

1. The thread-count env vars (`OMP_NUM_THREADS` etc.) and `XLA_FLAGS=
   --xla_cpu_multi_thread_eigen=false` do NOT cap jaxlib's CPU pool -- it still
   spawns ~97 threads on the first compute. Only restricting the process's CPU
   *affinity* shrinks it (to ~5-8 threads for a 1-2 CPU band), because the pool
   sizes to the visible-CPU count.
2. The pool sizes at the FIRST jax compute, so the affinity restriction must be
   in place before then. A ProcessPoolExecutor `initializer` runs before the
   worker's first task -- but ONLY works if the worker hasn't already imported
   jax. The forkserver preloads `__main__` (which imports the numeric stack),
   so the pool spawner must disable that preload
   (`ctx.set_forkserver_preload([])`); then each worker imports jax fresh at
   first task, after `pin_worker` has set its affinity.

So `pin_worker(num_workers)` is the pool initializer: it assigns each worker a
distinct CPU band (`worker_cpu_band`) and pins the worker's affinity to it,
then applies the thread-count env vars as belt-and-suspenders. The MAIN process
is intentionally left unpinned -- it runs the sequential DESC and canonical
confirmation steps and never multiplies threads, so it can keep full
parallelism.

Because the preload is disabled, workers do not inherit the launcher's runtime
`sys.path`; `ensure_on_pythonpath` puts this module's own directory on
`PYTHONPATH` (inherited by the forkserver and its workers) so the initializer
reference is importable there.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, MutableMapping
from pathlib import Path

# Each numeric library's own thread-count environment knob. Value is the
# thread count as a string. Insurance only -- the affinity band below is the
# load-bearing cap (these env vars do not touch jaxlib's CPU pool).
_THREAD_COUNT_VARS = (
    "OMP_NUM_THREADS",  # OpenMP (libgomp/libiomp): VMEC++, NumPy via OpenBLAS
    "OPENBLAS_NUM_THREADS",  # OpenBLAS directly
    "MKL_NUM_THREADS",  # Intel MKL, if linked
    "NUMEXPR_NUM_THREADS",  # NumExpr
    "VECLIB_MAXIMUM_THREADS",  # Accelerate/vecLib
)
_XLA_FLAGS_VAR = "XLA_FLAGS"
_XLA_SINGLE_THREAD_FLAG = "--xla_cpu_multi_thread_eigen=false"


def thread_limit_env(n_threads: int = 1) -> dict[str, str]:
    """The env-variable map that pins each numeric library's own thread count
    to `n_threads`. Pure -- no I/O -- so the exact variable set is testable.
    `n_threads` must be >= 1."""
    if n_threads < 1:
        raise ValueError(f"n_threads must be >= 1, got {n_threads}")
    env = {var: str(n_threads) for var in _THREAD_COUNT_VARS}
    env[_XLA_FLAGS_VAR] = _XLA_SINGLE_THREAD_FLAG
    return env


def apply_thread_limits(
    n_threads: int = 1,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Apply `thread_limit_env(n_threads)` to `environ` (default the real
    `os.environ`) with setdefault semantics: an existing value is left
    untouched (a deliberate operator override wins), gaps are filled. Returns
    the map of variables actually set. Idempotent."""
    target = os.environ if environ is None else environ
    applied: dict[str, str] = {}
    for key, value in thread_limit_env(n_threads).items():
        if key not in target:
            target[key] = value
            applied[key] = value
    return applied


def worker_cpu_band(pid: int, online_cpus: Iterable[int], num_workers: int) -> set[int]:
    """The distinct CPU band for one worker, given its `pid`, the `online_cpus`
    it may run on, and the total `num_workers`. Pure so the assignment is
    unit-testable without real scheduling.

    Divides the online CPUs into `num_workers` contiguous bands of
    `len(cpus) // num_workers` (>= 1) and returns the band at slot
    `pid % num_workers`. Slot-by-PID needs no cross-process coordination (a
    shared counter is not shareable across forkserver workers); consecutive
    worker PIDs -- the common case -- map to distinct slots, and the worst case
    of a collision only means two workers share a band (mild oversubscription),
    never a crash. With more workers than CPUs the band collapses to a single
    (shared) CPU. Returns all CPUs if `online_cpus` is empty or `num_workers`
    < 1, so a degenerate call never yields an empty affinity mask (which
    `sched_setaffinity` would reject)."""
    cpus = sorted(set(online_cpus))
    if not cpus or num_workers < 1:
        return set(cpus)
    band = max(1, len(cpus) // num_workers)
    slot = pid % num_workers
    start = (slot * band) % len(cpus)
    return {cpus[(start + i) % len(cpus)] for i in range(band)}


def pin_worker(num_workers: int) -> None:
    """ProcessPoolExecutor `initializer` (pass `initargs=(num_workers,)`).
    Runs inside each worker before its first task -- and, with the forkserver
    preload disabled, before it imports jax -- so the affinity restriction
    takes effect when jaxlib sizes its CPU pool. Pins this worker to its
    `worker_cpu_band` (the load-bearing thread cap) and applies the env-var
    thread limits as insurance. Best-effort on affinity: if the platform has
    no `sched_setaffinity` (non-Linux) the env limits still apply.

    Also silences constellaration's per-call `geometry_utils` fsolve
    RuntimeWarning ("The iteration is not making good progress") -- benign,
    emitted on nearly every forward-model call, and responsible for ~2/3 of
    raw run-log volume (2026-08-18 log-noise cleanup). Message-targeted so
    every other RuntimeWarning still surfaces."""
    import warnings

    warnings.filterwarnings("ignore", message="The iteration is not making good progress")
    apply_thread_limits(1)
    setaffinity = getattr(os, "sched_setaffinity", None)
    getaffinity = getattr(os, "sched_getaffinity", None)
    if setaffinity is None or getaffinity is None:
        return
    band = worker_cpu_band(os.getpid(), getaffinity(0), num_workers)
    if band:
        setaffinity(0, band)


def src_dir() -> str:
    """This module's own directory (the repo's `src`), for PYTHONPATH."""
    return str(Path(__file__).resolve().parent)


def ensure_on_pythonpath(path: str, *, environ: MutableMapping[str, str] | None = None) -> str:
    """Prepend `path` to `PYTHONPATH` in `environ` (default `os.environ`) if not
    already present, and return the resulting `PYTHONPATH`. The pool disables
    the forkserver's `__main__` preload, so workers do not inherit the
    launcher's runtime `sys.path`; putting `src` on `PYTHONPATH` (which the
    forkserver and its workers DO inherit) keeps the `pin_worker` initializer
    reference importable in the worker. Pure w.r.t. an injected environ so it
    is unit-testable."""
    target = os.environ if environ is None else environ
    current = target.get("PYTHONPATH", "")
    entries = current.split(os.pathsep) if current else []
    if path not in entries:
        entries.insert(0, path)
    target["PYTHONPATH"] = os.pathsep.join(entries)
    return target["PYTHONPATH"]
