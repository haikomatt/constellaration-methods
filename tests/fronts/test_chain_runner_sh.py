"""Shell tests for the p3 chain runners' detached-container launch.

Background (2026-10-04 23:29): the second
E25 launch failed because the exited container `p3-e25` from the first attempt still held
the name, `front_link()` treated podman's refusal as a plain launch failure and the chain
ended silently, so E25 never ran. These tests run the real shell functions against a fake
`podman` on PATH that reproduces that refusal (fixture: tests/fixtures/podman_name_in_use.txt).

Behaviour pinned:
- an exited container holding the name is removed first, and the removal is logged;
- a *running* container holding the name is never touched: the launch is refused, non-zero;
- a podman failure is logged with podman's own error text and the runner exits non-zero;
- both chain runners launch only through the shared helper (structural invariant).
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
LIB = SCRIPTS / "p3_chain_lib.sh"
RUNNERS = [
    SCRIPTS / "p3_bank_chain_runner.sh",
    SCRIPTS / "p3_nominal_front_runner.sh",
]
NAME_IN_USE = (REPO / "tests/fixtures/podman_name_in_use.txt").read_text()

# The fake podman keeps a one-line-per-container state file "<name> <state>" and appends every
# argv it receives to calls.log, so a test can assert both what podman was asked and in what order.
FAKE_PODMAN = textwrap.dedent(
    r"""
    #!/usr/bin/env bash
    set -u
    STATE=$FAKE_PODMAN_DIR/state
    echo "$*" >> "$FAKE_PODMAN_DIR/calls.log"
    touch "$STATE"
    name_state() { awk -v n="$1" '$1==n {print $2}' "$STATE"; }
    case "$1" in
      ps)
        if [[ "$*" == *"--filter name=^"* ]]; then
          all="$*"; n=${all#*--filter name=^}; n=${n%%\$*}
          name_state "$n"
        elif [[ "$*" == *"{{.Names}} {{.Status}}"* ]]; then
          awk '{print $1 " Exited (0) 1 second ago"}' "$STATE"
        else
          # running names; a running container "finishes" after it has been polled once
          awk '$2=="running" {print $1}' "$STATE"
          sed -i 's/ running$/ exited/' "$STATE"
        fi ;;
      rm)
        [ "${FAKE_PODMAN_RM_FAILS:-0}" = 1 ] && { echo "Error: cannot remove $2" >&2; exit 125; }
        sed -i "/^$2 /d" "$STATE" ;;
      run)
        if [[ "$2" == "--rm" ]]; then exit 0; fi
        n=""; prev=""; for a in "$@"; do [ "$prev" = "--name" ] && n=$a; prev=$a; done
        if [ -n "$(name_state "$n")" ]; then
          sed "s/p3-e25/$n/" "$FAKE_PODMAN_DIR/name_in_use.txt" >&2; exit 125
        fi
        echo "$n running" >> "$STATE"; echo 0123456789abcdef ;;
      logs) ;;
      *) echo "fake podman: unhandled: $*" >&2; exit 99 ;;
    esac
    """
).lstrip()


@pytest.fixture
def fake_podman(tmp_path: Path) -> Path:
    d = tmp_path / "fakebin"
    d.mkdir()
    (d / "podman").write_text(FAKE_PODMAN)
    (d / "podman").chmod((d / "podman").stat().st_mode | stat.S_IEXEC)
    (d / "name_in_use.txt").write_text(NAME_IN_USE)
    return d


def run_bash(
    script: str, fake_podman: Path, cwd: Path, env_extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PATH": f"{fake_podman}:{os.environ['PATH']}",
        "FAKE_PODMAN_DIR": str(fake_podman),
        **(env_extra or {}),
    }
    return subprocess.run(
        ["bash", "-c", script],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def podman_calls(fake_podman: Path) -> list[str]:
    log = fake_podman / "calls.log"
    return log.read_text().splitlines() if log.exists() else []


def set_state(fake_podman: Path, *lines: str) -> None:
    (fake_podman / "state").write_text("".join(f"{line}\n" for line in lines))


# ---------------------------------------------------------------------------------------
# launch_detached (scripts/p3_chain_lib.sh)
# ---------------------------------------------------------------------------------------


def launch(fake_podman: Path, cwd: Path, name: str = "p3-e25") -> subprocess.CompletedProcess[str]:
    return run_bash(
        f"source {LIB}; launch_detached {name} --network none img python3 x.py",
        fake_podman,
        cwd,
    )


def test_launch_detached_removes_exited_container_holding_the_name_and_logs_it(
    fake_podman, tmp_path
):
    set_state(fake_podman, "p3-e25 exited")

    res = launch(fake_podman, tmp_path)

    assert res.returncode == 0, res.stderr
    calls = podman_calls(fake_podman)
    rm_idx = next(i for i, c in enumerate(calls) if c == "rm p3-e25")
    run_idx = next(i for i, c in enumerate(calls) if c.startswith("run -d --name p3-e25 "))
    assert rm_idx < run_idx
    assert (
        "p3-e25" in res.stdout and "remov" in res.stdout.lower() and "exited" in res.stdout.lower()
    )


def test_launch_detached_passes_the_run_arguments_through_unchanged(fake_podman, tmp_path):
    res = launch(fake_podman, tmp_path)

    assert res.returncode == 0, res.stderr
    assert "run -d --name p3-e25 --network none img python3 x.py" in podman_calls(fake_podman)
    assert not any(c.startswith("rm ") for c in podman_calls(fake_podman))


def test_launch_detached_refuses_to_touch_a_running_container_with_the_same_name(
    fake_podman, tmp_path
):
    set_state(fake_podman, "p3-e25 running")

    res = launch(fake_podman, tmp_path)

    assert res.returncode != 0
    assert not any(c.startswith(("rm ", "run ")) for c in podman_calls(fake_podman))
    assert "p3-e25" in res.stdout and "running" in res.stdout


def test_launch_detached_logs_podman_error_text_and_exits_non_zero_when_run_fails(
    fake_podman, tmp_path
):
    # the fake's rm is disabled, so podman run meets the 2026-10-04 name collision verbatim
    set_state(fake_podman, "p3-e25 exited")

    res = run_bash(
        f"source {LIB}; launch_detached p3-e25 img x.py; echo 'MUST NOT REACH'",
        fake_podman,
        tmp_path,
        {"FAKE_PODMAN_RM_FAILS": "1"},
    )

    assert res.returncode != 0
    assert "MUST NOT REACH" not in res.stdout
    assert "p3-e25" in res.stdout and "Error:" in res.stdout


def test_launch_detached_surfaces_the_name_in_use_message_verbatim(fake_podman, tmp_path):
    # fake rm is a no-op here (state keeps the name), so `podman run` prints the fixture text
    (fake_podman / "podman").write_text(FAKE_PODMAN.replace('sed -i "/^$2 /d" "$STATE"', ":"))
    set_state(fake_podman, "p3-e25 exited")

    res = launch(fake_podman, tmp_path)

    assert res.returncode != 0
    assert 'the container name "p3-e25" is already in use' in res.stdout


# ---------------------------------------------------------------------------------------
# front_link() in the chain runners, sourced with a fake podman and a stubbed sleep
# ---------------------------------------------------------------------------------------


def anchors_report(cwd: Path) -> Path:
    (cwd / "results").mkdir(exist_ok=True)
    (cwd / "docs/preregistration").mkdir(parents=True, exist_ok=True)  # read at source time
    p = cwd / "anchors.json"
    members = [
        {
            "retained": True,
            "x": [0.1, 0.2],
            "confirmed_r": [-0.1, 0.0, -0.2, -0.3, -0.1],
        },
        {
            "retained": True,
            "x": [0.3, 0.4],
            "confirmed_r": [-0.2, 0.0, -0.1, -0.3, -0.1],
        },
        {"retained": False, "x": [0.5, 0.6], "confirmed_r": [0.5, 0.5, 0.5, 0.5, 0.5]},
    ]
    p.write_text(json.dumps({"members": members}))
    # Generation-0 evidence for runners with an anchor render check (E28): the anchors'
    # verbatim evaluations reproduce them. Runners without the check ignore the file.
    arch = cwd / "results/p3-e25-nsga2/archive.jsonl"
    arch.parent.mkdir(parents=True, exist_ok=True)
    arch.write_text(
        "".join(
            json.dumps({"tag": f"G0:{i}", "x": m["x"], "r": m["confirmed_r"]}) + "\n"
            for i, m in enumerate(members[:2])
        )
    )
    return p


def source_runner(runner: Path, body: str) -> str:
    # DEADLINE is a plain variable (the real one has passed); sleep is stubbed so wait_exit returns
    return f"source {runner}; DEADLINE=$(( $(date +%s) + 3600 )); sleep() {{ :; }}; {body}"


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda p: p.name)
def test_front_link_replaces_the_exited_container_from_a_previous_attempt(
    runner, fake_podman, tmp_path
):
    set_state(fake_podman, "p3-e25 exited")
    anchors = anchors_report(tmp_path)

    res = run_bash(source_runner(runner, f'front_link e25 {anchors} ""'), fake_podman, tmp_path)

    assert res.returncode == 0, res.stdout + res.stderr
    calls = podman_calls(fake_podman)
    assert "rm p3-e25" in calls
    launch_call = next(c for c in calls if c.startswith("run -d --name p3-e25 "))
    assert "scripts/b3_nsga2_mhd.py" in launch_call and f"--anchors {anchors}" in launch_call
    assert "e25: launching from" in res.stdout and "remov" in res.stdout.lower()


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda p: p.name)
def test_front_link_exits_the_runner_loudly_when_podman_refuses_the_launch(
    runner, fake_podman, tmp_path
):
    (fake_podman / "podman").write_text(FAKE_PODMAN.replace('sed -i "/^$2 /d" "$STATE"', ":"))
    set_state(fake_podman, "p3-e25 exited")
    anchors = anchors_report(tmp_path)

    res = run_bash(
        source_runner(
            runner,
            f'front_link e25 {anchors} "" || echo "SILENT BREAK"; echo "RUNNER DONE"',
        ),
        fake_podman,
        tmp_path,
    )

    assert res.returncode != 0
    assert "SILENT BREAK" not in res.stdout and "RUNNER DONE" not in res.stdout
    assert 'the container name "p3-e25" is already in use' in res.stdout


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda p: p.name)
def test_sourcing_a_runner_defines_front_link_without_starting_the_chain(
    runner, fake_podman, tmp_path
):
    res = run_bash(
        f"source {runner}; declare -F front_link >/dev/null && echo DEFINED",
        fake_podman,
        tmp_path,
    )

    assert res.returncode == 0, res.stderr
    assert res.stdout.strip().endswith("DEFINED")
    assert not podman_calls(fake_podman)


# ---------------------------------------------------------------------------------------
# structural invariant (regression ratchet): no bare detached launch in the chain runners
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda p: p.name)
def test_chain_runners_launch_detached_containers_only_via_the_shared_helper(runner):
    text = runner.read_text()

    assert "podman run -d" not in text, (
        "detached launches must go through launch_detached (p3_chain_lib.sh)"
    )
    assert "p3_chain_lib.sh" in text
