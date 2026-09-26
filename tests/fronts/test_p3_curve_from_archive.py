"""Tests for scripts/p3_curve_from_archive.py -- E18 item 3: recompute the
V-vs-evaluations curve from a run's own archive.jsonl after the fact (so a
run that predates this tooling, e.g. results/p3-e17-nsga2, gets one too).
Pure/no VMEC: it only reads already-archived residuals via
p3_front_from_archive.v_curve_from_rows.
"""

import json
from pathlib import Path

from constellaration_methods.fronts.p3_curve_from_archive import (
    build_parser,
    load_archive_rows,
    main,
    render_table,
)


def _write_archive(archive_dir: Path, vs: list) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    with open(archive_dir / "archive.jsonl", "w") as fh:
        for i, v in enumerate(vs):
            fh.write(json.dumps({"tag": f"r{i}", "r": [v, -1, -1, -1, -1]}) + "\n")


def test_load_archive_rows_reads_jsonl_in_order(tmp_path):
    _write_archive(tmp_path, [0.1, 0.2, 0.3])
    rows = load_archive_rows(tmp_path)
    assert [row["tag"] for row in rows] == ["r0", "r1", "r2"]


def test_build_parser_defaults():
    args = build_parser().parse_args(["results/x", "--out", "out.json"])
    assert args.archive_dir == "results/x"
    assert args.out == "out.json"
    assert args.every == 100


def test_main_writes_curve_matching_hand_computed_values(tmp_path, capsys):
    # V per row: 0.10, 0.08, 0.12, 0.03 -> running best at every=2: 0.08, 0.03
    _write_archive(tmp_path, [0.10, 0.08, 0.12, 0.03])
    out_file = tmp_path / "curve.json"
    curve = main([str(tmp_path), "--out", str(out_file), "--every", "2"])
    assert curve == [[2, 0.08], [4, 0.03]]
    assert json.loads(out_file.read_text()) == [[2, 0.08], [4, 0.03]]


def test_main_prints_a_table(tmp_path, capsys):
    _write_archive(tmp_path, [0.10, 0.08])
    main([str(tmp_path), "--out", str(tmp_path / "curve.json"), "--every", "2"])
    out = capsys.readouterr().out
    assert "evaluations" in out
    assert "0.080000" in out


def test_render_table_handles_none_entries():
    table = render_table([[2, None], [4, 0.03]])
    assert "n/a" in table
    assert "0.030000" in table
