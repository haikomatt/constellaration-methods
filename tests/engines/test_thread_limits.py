"""session-handoff-2026-07-24-fix-qi-resource-management: TDD for
`src/thread_limits.py`, the per-worker compute-thread pin that keeps N
ProcessPoolExecutor workers from each spinning up a ~CPU-count-sized jaxlib
CPU pool and collectively blowing the container's cgroup pid limit
(pids.max=2048 in the phase1 image -- the 2026-07-23 overnight run died
~1 min in with `pthread_create: Resource temporarily unavailable`).

Pure and deterministic -- no numeric imports; the CPU-band assignment, the
env map, and the PYTHONPATH edit are all tested against injected inputs, and
`pin_worker`'s scheduling calls are exercised with monkeypatched os hooks. The
real end-to-end proof (worker thread counts actually drop at 12 workers) is a
gated in-container run, not a unit test.
"""

from __future__ import annotations

import pytest

from constellaration_methods.engines import thread_limits

# ---------------------------------------------------------------------------
# thread_limit_env / apply_thread_limits: the env-var insurance layer
# ---------------------------------------------------------------------------


def test_thread_limit_env_pins_every_known_library_to_one_thread():
    env = thread_limits.thread_limit_env(1)

    assert env["OMP_NUM_THREADS"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["NUMEXPR_NUM_THREADS"] == "1"
    assert env["VECLIB_MAXIMUM_THREADS"] == "1"
    assert env["XLA_FLAGS"] == "--xla_cpu_multi_thread_eigen=false"


def test_thread_limit_env_rejects_a_thread_count_below_one():
    with pytest.raises(ValueError, match="n_threads must be >= 1"):
        thread_limits.thread_limit_env(0)


def test_apply_thread_limits_sets_every_var_on_an_empty_environ():
    environ: dict[str, str] = {}

    applied = thread_limits.apply_thread_limits(1, environ=environ)

    assert environ["OMP_NUM_THREADS"] == "1"
    assert environ["XLA_FLAGS"] == "--xla_cpu_multi_thread_eigen=false"
    assert applied == thread_limits.thread_limit_env(1)


def test_apply_thread_limits_never_clobbers_an_existing_value():
    environ = {"OMP_NUM_THREADS": "4"}

    applied = thread_limits.apply_thread_limits(1, environ=environ)

    assert environ["OMP_NUM_THREADS"] == "4"  # operator override wins
    assert "OMP_NUM_THREADS" not in applied
    assert environ["MKL_NUM_THREADS"] == "1"  # gaps still filled


def test_apply_thread_limits_is_idempotent():
    environ: dict[str, str] = {}

    thread_limits.apply_thread_limits(1, environ=environ)
    second = thread_limits.apply_thread_limits(1, environ=environ)

    assert second == {}


# ---------------------------------------------------------------------------
# worker_cpu_band: the load-bearing distinct-band assignment
# ---------------------------------------------------------------------------


def test_worker_cpu_band_gives_contiguous_distinct_bands_on_this_box():
    # 32 logical CPUs, 12 workers -> band = 32 // 12 = 2 CPUs each.
    cpus = list(range(32))
    assert thread_limits.worker_cpu_band(0, cpus, 12) == {0, 1}
    assert thread_limits.worker_cpu_band(1, cpus, 12) == {2, 3}
    assert thread_limits.worker_cpu_band(11, cpus, 12) == {22, 23}


def test_worker_cpu_band_assigns_by_pid_slot():
    cpus = list(range(32))
    # pid 13 -> slot 13 % 12 == 1 -> same band as pid 1
    assert thread_limits.worker_cpu_band(13, cpus, 12) == thread_limits.worker_cpu_band(1, cpus, 12)


def test_worker_cpu_band_single_worker_gets_all_cpus():
    cpus = list(range(8))
    assert thread_limits.worker_cpu_band(0, cpus, 1) == set(cpus)


def test_worker_cpu_band_collapses_to_one_cpu_when_more_workers_than_cpus():
    cpus = [0, 1, 2, 3]
    band = thread_limits.worker_cpu_band(5, cpus, 8)  # band size max(1, 4//8)=1
    assert len(band) == 1
    assert band <= set(cpus)


def test_worker_cpu_band_empty_online_cpus_returns_empty():
    # a degenerate call must not fabricate CPUs; pin_worker skips setaffinity
    # on an empty band rather than passing an empty mask.
    assert thread_limits.worker_cpu_band(0, [], 12) == set()


def test_worker_cpu_band_non_positive_workers_returns_all_cpus():
    cpus = [0, 1, 2, 3]
    assert thread_limits.worker_cpu_band(0, cpus, 0) == set(cpus)


# ---------------------------------------------------------------------------
# pin_worker: the initializer wiring (monkeypatched scheduling)
# ---------------------------------------------------------------------------


def test_pin_worker_sets_env_and_pins_affinity_to_its_band(monkeypatch):
    fake_environ: dict[str, str] = {}
    recorded: dict[str, object] = {}
    monkeypatch.setattr(thread_limits.os, "environ", fake_environ)
    monkeypatch.setattr(thread_limits.os, "getpid", lambda: 1)
    monkeypatch.setattr(
        thread_limits.os, "sched_getaffinity", lambda pid: set(range(32)), raising=False
    )
    monkeypatch.setattr(
        thread_limits.os,
        "sched_setaffinity",
        lambda pid, mask: recorded.__setitem__("mask", set(mask)),
        raising=False,
    )

    thread_limits.pin_worker(12)

    assert fake_environ["OMP_NUM_THREADS"] == "1"
    # pid 1 -> slot 1 -> CPUs {2, 3}
    assert recorded["mask"] == {2, 3}


def test_pin_worker_without_sched_setaffinity_still_applies_env(monkeypatch):
    fake_environ: dict[str, str] = {}
    monkeypatch.setattr(thread_limits.os, "environ", fake_environ)
    monkeypatch.delattr(thread_limits.os, "sched_setaffinity", raising=False)

    thread_limits.pin_worker(12)  # must not raise on a platform without it

    assert fake_environ["OMP_NUM_THREADS"] == "1"


# ---------------------------------------------------------------------------
# ensure_on_pythonpath: make the initializer importable in preload-less workers
# ---------------------------------------------------------------------------


def test_ensure_on_pythonpath_sets_it_when_empty():
    environ: dict[str, str] = {}

    result = thread_limits.ensure_on_pythonpath("/repo/src", environ=environ)

    assert result == "/repo/src"
    assert environ["PYTHONPATH"] == "/repo/src"


def test_ensure_on_pythonpath_prepends_without_duplicating():
    environ = {"PYTHONPATH": "/other"}

    thread_limits.ensure_on_pythonpath("/repo/src", environ=environ)
    # idempotent second call must not duplicate the entry
    thread_limits.ensure_on_pythonpath("/repo/src", environ=environ)

    assert environ["PYTHONPATH"].split(":") == ["/repo/src", "/other"]


def test_src_dir_points_at_the_module_directory():
    # NOTE (repo split): `src_dir()` returns this module's own directory
    # (docstring: "This module's own directory (the repo's `src`), for
    # PYTHONPATH"). In the original private repo that was a flat `src/`
    # layout, hence the old assertion `.endswith("/src")`. In this ported,
    # installed-package layout the module lives at
    # `src/constellaration_methods/engines/thread_limits.py`, so its own
    # directory is `.../engines` -- the assertion is updated to match, per
    # the Test Integrity Rule (test assumption was wrong for the new
    # layout, not the code).
    assert thread_limits.src_dir().endswith("/engines")
