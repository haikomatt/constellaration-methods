"""Anchor render check (scripts/p3_anchor_render_check.py): the E28 guard, run as the CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def test_anchor_render_check_passes_when_generation_zero_reproduces_the_anchors(
    tmp_path,
):
    """E28 guard (2026-10-26): E25 evaluated an anchor's exact x as a different shape and
    nobody noticed for a day. The check compares each anchor's stored residuals with its
    generation-0 evaluation (anchors come first, verbatim, in load order)."""
    rep = {
        "members": [
            {
                "tag": "a",
                "x": [0.5, 0.25],
                "confirmed_r": [0.1, -0.2, 0.0, 0.0, -0.1],
                "retained": True,
            }
        ]
    }
    (tmp_path / "REPORT.json").write_text(json.dumps(rep))
    (tmp_path / "archive.jsonl").write_text(
        json.dumps({"tag": "G0:1", "x": [0.9, 0.1], "r": [9.0, 9.0, 9.0, 9.0, 9.0]})
        + "\n"
        + json.dumps({"tag": "G0:0", "x": [0.5, 0.25], "r": [0.1, -0.2, 0.0, 0.0, -0.1]})
        + "\n"
    )
    out = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "p3_anchor_render_check.py"),
            str(tmp_path / "REPORT.json"),
            str(tmp_path / "archive.jsonl"),
        ],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr


def test_anchor_render_check_fails_loudly_on_a_mismatch(tmp_path):
    rep = {
        "members": [
            {
                "tag": "a",
                "confirmed_r": [-0.276, 0.00997, 0.009995, 0.0073, -0.167],
                "retained": True,
                "x": [0.5, 0.25],
            }
        ]
    }
    (tmp_path / "REPORT.json").write_text(json.dumps(rep))
    (tmp_path / "archive.jsonl").write_text(
        json.dumps({"tag": "G0:0", "x": [0.5, 0.25], "r": [-0.27, 0.026, 0.02, 0.01, -0.16]}) + "\n"
    )
    out = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "p3_anchor_render_check.py"),
            str(tmp_path / "REPORT.json"),
            str(tmp_path / "archive.jsonl"),
        ],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 1
    assert "MISMATCH" in out.stdout


def test_anchor_render_check_resolves_front_members_from_the_source_archive(tmp_path):
    """2026-10-26: E28b's check crashed (KeyError 'x'): a confirmed front's members carry a tag
    and residuals but no x. The x is in the source run's archive under the same tag."""
    src = tmp_path / "src_archive.jsonl"
    src.write_text(
        json.dumps({"tag": "G11:16", "x": [0.5, 0.25], "r": [0.1, -0.2, 0, 0, -0.1]}) + "\n"
    )
    rep = {"members": [{"tag": "G11:16", "confirmed_r": [0.1, -0.2, 0, 0, -0.1], "retained": True}]}
    (tmp_path / "REPORT.json").write_text(json.dumps(rep))
    (tmp_path / "archive.jsonl").write_text(
        json.dumps({"tag": "G0:0", "x": [0.5, 0.25], "r": [0.1, -0.2, 0, 0, -0.1]}) + "\n"
    )
    cmd = [
        sys.executable,
        str(SCRIPTS / "p3_anchor_render_check.py"),
        str(tmp_path / "REPORT.json"),
        str(tmp_path / "archive.jsonl"),
        "--source-archive",
        str(src),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


def test_anchor_render_check_reports_a_member_with_no_coordinates_as_a_mismatch(
    tmp_path,
):
    rep = {"members": [{"tag": "G11:16", "confirmed_r": [0.1, -0.2, 0, 0, -0.1], "retained": True}]}
    (tmp_path / "REPORT.json").write_text(json.dumps(rep))
    (tmp_path / "archive.jsonl").write_text(
        json.dumps({"tag": "G0:0", "x": [0.5], "r": [0]}) + "\n"
    )
    cmd = [
        sys.executable,
        str(SCRIPTS / "p3_anchor_render_check.py"),
        str(tmp_path / "REPORT.json"),
        str(tmp_path / "archive.jsonl"),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    assert out.returncode == 1
    assert "MISMATCH" in out.stdout and "no coordinates" in out.stdout
