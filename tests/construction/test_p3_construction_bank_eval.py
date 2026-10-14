"""E20 / E20b: host-runnable tests for
`constellaration_methods.construction.p3_construction_bank_eval` -- bank
loading, boundary-file writing, rows-satisfied bookkeeping, row selection,
minor-radius re-export bookkeeping, the twist-bin/agreement summaries and CLI
parsing. The canonical (VMEC/forward_model) evaluation paths (`canonical_
once`, `confirm_worst_of_three`) need the pinned image and are exercised only
by the in-image smoke run, same split as the stage-1 driver's tests. The
bit-for-bit re-export ratchet test (E20b registration) also needs the pinned
image (real pyQIC/constellaration `_export_4x4_boundary` call) -- gated with
`pytest.importorskip`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from constellaration_methods.construction import p3_construction_bank_eval as pbe

FIXTURES = Path(__file__).parent / "fixtures"
REAL_BANK_ENTRY = FIXTURES / "e20_bank_entry_e20_nfp3_02473.json"


def _fake_bank_top(tmp_path: Path, n: int) -> Path:
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    rows = [
        {
            "label": f"e20_row{i:03d}",
            "n_field_periods": 3,
            "proxy_v": float(n - i) / n,  # descending proxy_v as i increases
            "boundary": {
                "r_cos": [[0.0, 1.0, 0.0]],
                "r_sin": [[0.0, 0.0, 0.0]],
                "z_cos": [[0.0, 0.0, 0.0]],
                "z_sin": [[0.0, 0.1, 0.0]],
                "n_field_periods": 3,
                "is_stellarator_symmetric": True,
            },
        }
        for i in range(n)
    ]
    (bank_dir / "bank_top.json").write_text(json.dumps(rows))
    return bank_dir


def test_load_top_n_sorts_by_proxy_v_ascending(tmp_path):
    bank_dir = _fake_bank_top(tmp_path, 5)
    top = pbe.load_top_n(bank_dir, 3)
    assert len(top) == 3
    vs = [r["proxy_v"] for r in top]
    assert vs == sorted(vs)


def test_load_top_n_caps_at_available_rows(tmp_path):
    bank_dir = _fake_bank_top(tmp_path, 3)
    top = pbe.load_top_n(bank_dir, 48)
    assert len(top) == 3


def test_write_boundary_files_one_per_row(tmp_path):
    bank_dir = _fake_bank_top(tmp_path, 4)
    rows = pbe.load_top_n(bank_dir, 4)
    out = tmp_path / "boundaries"
    paths = pbe.write_boundary_files(rows, out)
    assert len(paths) == 4
    for p, r in zip(paths, rows):
        assert p.exists()
        loaded = json.loads(p.read_text())
        assert loaded == r["boundary"]


def test_rows_satisfied_at_seed_true_when_residual_non_positive():
    residuals = {"iota": -0.1, "qi": 0.0, "mirror": 0.05, "flux": -0.2, "well": -0.01}
    satisfied = pbe.rows_satisfied_at_seed(residuals)
    assert satisfied == {
        "iota": True,
        "qi": True,
        "mirror": False,
        "flux": True,
        "well": True,
    }


def test_rows_satisfied_at_seed_missing_row_is_unsatisfied():
    satisfied = pbe.rows_satisfied_at_seed({"iota": -0.1})
    assert satisfied["well"] is False


def test_cli_parses_registration_stage2_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("bank_dir", type=Path)
    parser.add_argument("--top", type=int, default=48)
    parser.add_argument("--confirm-top", type=int, default=12)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(
        [
            "results/p3-e20-construction-search",
            "--top",
            "48",
            "--confirm-top",
            "12",
            "--workers",
            "12",
            "--out",
            "results/p3-e20-stage2",
        ]
    )
    assert args.top == 48
    assert args.confirm_top == 12
    assert args.workers == 12
    assert str(args.bank_dir) == "results/p3-e20-construction-search"


def test_cli_smoke_confirm_top_zero_parses():
    parser = argparse.ArgumentParser()
    parser.add_argument("bank_dir", type=Path)
    parser.add_argument("--top", type=int, default=48)
    parser.add_argument("--confirm-top", type=int, default=12)
    args = parser.parse_args(["bank", "--top", "4", "--confirm-top", "0"])
    assert args.confirm_top == 0


# --------------------------------------------------------------------------- #
# E20b: --select-by
# --------------------------------------------------------------------------- #


def test_select_rows_default_proxy_v_matches_load_top_n(tmp_path):
    bank_dir = _fake_bank_top(tmp_path, 5)
    assert pbe.select_rows(bank_dir, top=3, select_by="proxy_v") == pbe.load_top_n(bank_dir, 3)


def test_select_rows_label_list_selects_exact_rows_in_requested_order(tmp_path):
    bank_dir = _fake_bank_top(tmp_path, 5)
    rows = pbe.select_rows(
        bank_dir,
        top=5,
        select_by="label_list",
        labels=["e20_row004", "e20_row000"],
    )
    assert [r["label"] for r in rows] == ["e20_row004", "e20_row000"]


def test_select_rows_label_list_missing_label_raises(tmp_path):
    bank_dir = _fake_bank_top(tmp_path, 3)
    with pytest.raises(ValueError, match="e20_row999"):
        pbe.select_rows(bank_dir, top=3, select_by="label_list", labels=["e20_row999"])


def test_select_rows_label_list_without_labels_raises(tmp_path):
    bank_dir = _fake_bank_top(tmp_path, 3)
    with pytest.raises(ValueError, match="--labels"):
        pbe.select_rows(bank_dir, top=3, select_by="label_list", labels=None)


def test_select_rows_unknown_select_by_raises(tmp_path):
    bank_dir = _fake_bank_top(tmp_path, 3)
    with pytest.raises(ValueError, match="select-by"):
        pbe.select_rows(bank_dir, top=3, select_by="bogus")


# --------------------------------------------------------------------------- #
# E20b: --minor-radius re-export (host-testable via an injected export_fn;
# the real `p3_construction_search._export_4x4_boundary` path is covered by
# the bit-for-bit ratchet test below, which needs the pinned image)
# --------------------------------------------------------------------------- #


def test_reexport_at_minor_radius_calls_export_fn_per_row_and_tags_radius():
    rows = [
        {"label": "a", "vector": [1.0, 2.0], "n_field_periods": 3},
        {"label": "b", "vector": [3.0, 4.0], "n_field_periods": 4},
    ]
    calls = []

    def fake_export(vector, nfp):
        calls.append((vector, nfp))
        return {"r_cos": [[0.0]], "n_field_periods": nfp}

    out = pbe.reexport_at_minor_radius(rows, 0.05, export_fn=fake_export)
    assert calls == [([1.0, 2.0], 3), ([3.0, 4.0], 4)]
    assert [r["minor_radius"] for r in out] == [0.05, 0.05]
    assert out[0]["boundary"] == {"r_cos": [[0.0]], "n_field_periods": 3}
    # original rows (vector, label, etc.) are preserved alongside the new boundary
    assert out[0]["label"] == "a"
    assert out[0]["vector"] == [1.0, 2.0]


def test_reexport_at_minor_radius_does_not_mutate_input_rows():
    rows = [{"label": "a", "vector": [1.0], "n_field_periods": 3, "boundary": "stale"}]
    pbe.reexport_at_minor_radius(rows, 0.05, export_fn=lambda vector, nfp: {"r_cos": "fresh"})
    assert rows[0]["boundary"] == "stale"


def test_reexport_at_minor_radius_bit_for_bit_reproduces_real_bank_entry():
    """E20b registration ratchet test: re-exporting a real bank entry's
    stored `vector`/`n_field_periods` at minor radius 0.1 (the radius E20
    stage 1 used) must reproduce its stored `boundary` exactly -- this is
    what makes stage-2's re-export at a DIFFERENT radius trustworthy (same
    export path, just a different `minor_radius` argument)."""
    pytest.importorskip("constellaration.geometry.surface_rz_fourier")
    entry = json.loads(REAL_BANK_ENTRY.read_text())
    reexported = pbe.reexport_at_minor_radius(
        [
            {
                "label": entry["label"],
                "vector": entry["vector"],
                "n_field_periods": entry["n_field_periods"],
            }
        ],
        0.1,
    )
    assert reexported[0]["boundary"] == entry["boundary"]


# --------------------------------------------------------------------------- #
# E20b: twist bins and the convergence / agreement summaries (synthetic bank
# -- no pinned image needed)
# --------------------------------------------------------------------------- #


def test_twist_bin_boundaries():
    assert pbe.twist_bin(0.0) == "<0.14"
    assert pbe.twist_bin(0.139) == "<0.14"
    assert pbe.twist_bin(0.14) == "0.14-0.25"
    assert pbe.twist_bin(0.24) == "0.14-0.25"
    assert pbe.twist_bin(0.25) == ">=0.25"
    assert pbe.twist_bin(1.0) == ">=0.25"


def _synthetic_per_row():
    """4 rows spanning all three twist bins, one non-convergent, to
    exercise both the convergence-fraction-by-bin table and the
    convergent-only agreement medians."""
    return [
        {
            "label": "low_ok",
            "near_axis_iota_over_nfp": 0.05,
            "near_axis_mirror": 0.10,
            "convergent": True,
            "canonical_iota_over_nfp": 0.10,  # ratio 2.0
            "canonical_mirror": 0.30,  # ratio 3.0
            "canonical_log10_qi": -3.0,
            "twist_bin": "<0.14",
        },
        {
            "label": "mid_ok",
            "near_axis_iota_over_nfp": 0.20,
            "near_axis_mirror": 0.10,
            "convergent": True,
            "canonical_iota_over_nfp": 0.18,  # ratio 0.9
            "canonical_mirror": 0.20,  # ratio 2.0
            "canonical_log10_qi": -1.0,
            "twist_bin": "0.14-0.25",
        },
        {
            "label": "high_fail",
            "near_axis_iota_over_nfp": 0.80,
            "near_axis_mirror": 0.10,
            "convergent": False,
            "canonical_iota_over_nfp": None,
            "canonical_mirror": None,
            "canonical_log10_qi": None,
            "twist_bin": ">=0.25",
        },
        {
            "label": "high_fail2",
            "near_axis_iota_over_nfp": 0.90,
            "near_axis_mirror": 0.10,
            "convergent": False,
            "canonical_iota_over_nfp": None,
            "canonical_mirror": None,
            "canonical_log10_qi": None,
            "twist_bin": ">=0.25",
        },
    ]


def test_convergence_fraction_by_twist_bin():
    table = pbe.convergence_fraction_by_twist_bin(_synthetic_per_row())
    assert table["<0.14"] == {"n": 1, "n_convergent": 1, "convergence_fraction": 1.0}
    assert table["0.14-0.25"] == {
        "n": 1,
        "n_convergent": 1,
        "convergence_fraction": 1.0,
    }
    assert table[">=0.25"] == {"n": 2, "n_convergent": 0, "convergence_fraction": 0.0}


def test_convergent_agreement_summary_medians_over_convergent_rows_only():
    summary = pbe.convergent_agreement_summary(_synthetic_per_row())
    assert summary["n_convergent"] == 2
    # iota ratios [2.0, 0.9] -> median 1.45; mirror ratios [3.0, 2.0] -> median 2.5
    assert summary["median_iota_ratio_canonical_over_near_axis"] == pytest.approx(1.45)
    assert summary["median_mirror_ratio_canonical_over_near_axis"] == pytest.approx(2.5)
    # log10 qi [-3.0, -1.0] -> median -2.0
    assert summary["median_canonical_log10_qi"] == pytest.approx(-2.0)


def test_convergent_agreement_summary_empty_when_nothing_convergent():
    rows = [
        {
            "near_axis_iota_over_nfp": 0.5,
            "near_axis_mirror": 0.1,
            "convergent": False,
            "canonical_iota_over_nfp": None,
            "canonical_mirror": None,
            "canonical_log10_qi": None,
            "twist_bin": ">=0.25",
        }
    ]
    summary = pbe.convergent_agreement_summary(rows)
    assert summary["n_convergent"] == 0
    assert summary["median_iota_ratio_canonical_over_near_axis"] is None
    assert summary["median_mirror_ratio_canonical_over_near_axis"] is None
    assert summary["median_canonical_log10_qi"] is None


def test_build_per_row_comparison_marks_error_rows_non_convergent():
    rows = [
        {
            "label": "a",
            "n_field_periods": 3,
            "proxies": {"iota_over_nfp": 0.05, "mirror": 0.1, "qi_residual": 1e-4},
        }
    ]
    once = [{"error": "RuntimeError: boom"}]
    per_row = pbe.build_per_row_comparison(rows, once, 0.05)
    assert per_row[0]["convergent"] is False
    assert per_row[0]["error_class"] == "RuntimeError: boom"
    assert per_row[0]["near_axis_iota_over_nfp"] == 0.05
    assert per_row[0]["minor_radius"] == 0.05
    assert per_row[0]["twist_bin"] == "<0.14"


def test_build_per_row_comparison_convergent_row_pulls_canonical_raw_fields():
    rows = [
        {
            "label": "a",
            "n_field_periods": 3,
            "proxies": {"iota_over_nfp": 0.20, "mirror": 0.1, "qi_residual": 1e-4},
        }
    ]
    once = [
        {
            "V": 0.4,
            "raw": {"iota_over_nfp": 0.18, "edge_mirror": 0.22, "log10_qi": -1.2},
        }
    ]
    per_row = pbe.build_per_row_comparison(rows, once, 0.05)
    assert per_row[0]["convergent"] is True
    assert per_row[0]["error_class"] is None
    assert per_row[0]["canonical_V"] == 0.4
    assert per_row[0]["canonical_iota_over_nfp"] == 0.18
    assert per_row[0]["canonical_mirror"] == 0.22
    assert per_row[0]["canonical_log10_qi"] == -1.2


# --------------------------------------------------------------------------- #
# E20b: CLI flags
# --------------------------------------------------------------------------- #


def _e20b_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("bank_dir", type=Path)
    parser.add_argument("--top", type=int, default=48)
    parser.add_argument("--confirm-top", type=int, default=12)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minor-radius", type=float, default=None)
    parser.add_argument("--select-by", choices=["proxy_v", "label_list"], default="proxy_v")
    parser.add_argument("--labels", type=str, default=None)
    return parser


def test_cli_parses_minor_radius_flag():
    args = _e20b_parser().parse_args(
        [
            "results/p3-e20-construction",
            "--top",
            "4",
            "--confirm-top",
            "0",
            "--minor-radius",
            "0.05",
            "--out",
            "results/p3-e20b-smoke",
        ]
    )
    assert args.minor_radius == pytest.approx(0.05)


def test_cli_minor_radius_defaults_to_none_unchanged_behaviour():
    args = _e20b_parser().parse_args(["bank", "--out", "out"])
    assert args.minor_radius is None
    assert args.select_by == "proxy_v"
    assert args.labels is None


def test_cli_parses_select_by_label_list_with_labels():
    args = _e20b_parser().parse_args(
        [
            "bank",
            "--out",
            "out",
            "--select-by",
            "label_list",
            "--labels",
            "e20_nfp3_00001,e20_nfp4_00002",
        ]
    )
    assert args.select_by == "label_list"
    assert args.labels == "e20_nfp3_00001,e20_nfp4_00002"


def test_cli_rejects_unknown_select_by():
    with pytest.raises(SystemExit):
        _e20b_parser().parse_args(["bank", "--out", "out", "--select-by", "bogus"])


# --------------------------------------------------------------------------- #
# regression ratchet (mirrored from the private experiment repo): the
# report-level label is EXISTENTIAL over the confirmed rows -- the best label
# ANY confirmed row achieved (goal > substantial > material > null), with the
# determining row named -- not the label of the min-V row. For the pure-V
# `stage2_label` tiers the two coincide on real runs; the rule is pinned here
# so a row-conditioned label (the Phase 2 bars) cannot be masked by a lower-V
# row that fails a required row.
# --------------------------------------------------------------------------- #


def _confirmed_row(label: str, v: float, *, stage2_label: str) -> dict:
    return {"label": label, "worst_of_three_v": v, "stage2_label": stage2_label}


def test_summarise_confirmed_report_label_is_best_label_of_any_confirmed_row():
    confirmed = [
        _confirmed_row("a", 0.15, stage2_label="null"),
        _confirmed_row("b", 0.25, stage2_label="material"),
    ]
    summary = pbe.summarise_confirmed(confirmed)
    assert summary["best_confirmed_label"] == "a"
    assert summary["best_confirmed_worst_of_three_v"] == pytest.approx(0.15)
    assert summary["stage2_label"] == "material"
    assert summary["stage2_label_determined_by"] == "b"


def test_summarise_confirmed_orders_tiers_goal_over_substantial_over_material():
    confirmed = [
        _confirmed_row("m", 0.20, stage2_label="material"),
        _confirmed_row("g", 0.009, stage2_label="goal"),
        _confirmed_row("s", 0.08, stage2_label="substantial"),
    ]
    summary = pbe.summarise_confirmed(confirmed)
    assert summary["stage2_label"] == "goal"
    assert summary["stage2_label_determined_by"] == "g"


def test_summarise_confirmed_empty_when_nothing_confirmed():
    summary = pbe.summarise_confirmed([])
    assert summary["best_confirmed_label"] is None
    assert summary["best_confirmed_worst_of_three_v"] is None
    assert summary["stage2_label"] is None
    assert summary["stage2_label_determined_by"] is None
