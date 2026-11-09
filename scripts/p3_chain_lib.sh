#!/usr/bin/env bash
# Shared helpers for the p3 chain runners. Sourced, never executed:
#   source "$(dirname "${BASH_SOURCE[0]}")/p3_chain_lib.sh"
# INVARIANT 1: a chain runner launches detached experiment containers only through
#   launch_detached, never a bare `podman run -d --name`. Guard:
#   tests/test_chain_runner_sh.py::test_breadth_runners_launch_detached_containers_only_via_the_shared_helper
# INVARIANT 2: a refused launch ends the runner non-zero with podman's error text in the log.
#   Guard: ...::test_front_link_exits_the_runner_loudly_when_podman_refuses_the_launch
#   (2026-10-04 23:29: the exited p3-e25 from a first attempt held the name, the refusal was
#   logged as a plain "launch failed", the chain ended silently and E25 never ran.)

log() { echo "$(date -Is) $*"; }

# container_state NAME: podman's State ("running", "exited", ...) for the container of exactly
# that name, empty if none. The name filter is a regex, hence the anchors.
container_state() { podman ps -a --filter "name=^$1\$" --format '{{.State}}'; }

# launch_detached NAME PODMAN_RUN_ARGS...: `podman run -d --name NAME ARGS...`.
# An exited container holding the name (a previous attempt) is removed first and the removal is
# logged. A running one is a live experiment: never touched, the launch is refused. Any podman
# failure is logged with podman's own message and the whole runner exits 1.
launch_detached() {
  local name=$1; shift
  local state err; state=$(container_state "$name")
  if [ "$state" = "running" ] || [ "$state" = "paused" ]; then
    log "$name: a container with this name is still $state; refusing to launch over it"; exit 1
  fi
  if [ -n "$state" ]; then
    log "$name: removing the $state container of that name left by a previous attempt"
    err=$(podman rm "$name" 2>&1 >/dev/null) || { log "$name: podman rm failed: $err"; exit 1; }
  fi
  if ! err=$(podman run -d --name "$name" "$@" 2>&1 >/dev/null); then
    log "$name: podman run failed: $err"; exit 1
  fi
}
