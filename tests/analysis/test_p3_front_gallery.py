"""Tests for constellaration_methods.analysis.p3_front_gallery.

Phase 0 evidence: REPORT.json / MANIFEST.json / a `.boundary.json` export
schema were read in full from a real run directory before this module was
written (see p3_front_gallery.py's own docstring for the matching rule that
inspection produced). Tests here build tiny synthetic REPORT.json + boundary
files under tmp_path rather than committing fixtures, since the schema
itself is already pinned by the module docstring and by the real data used
for the `solo_hypervolume` regression value below.

Rendering (matplotlib/PIL, real PNGs) is I/O-heavy and covered by actually
running the script against a real run directory, not by these unit tests --
`test_build_html_*` below exercises `build_html` with fake data-URI strings
so the templating logic is checked without any image I/O.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from constellaration_methods.analysis import p3_front_gallery as gallery


def _tiny_boundary(nfp: int = 3) -> dict:
    """Minimal but schema-valid stellarator-symmetric boundary (mpol=1,
    ntor=1) -- same keys/shape as the real exports, just tiny."""
    return {
        "r_cos": [[1.0, 0.0, 0.0], [0.05, 0.0, 0.0]],
        "z_sin": [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]],
        "r_sin": None,
        "z_cos": None,
        "n_field_periods": nfp,
        "is_stellarator_symmetric": True,
    }


def _synthetic_report() -> dict:
    return {
        "tol": 0.01,
        "confirmed_hypervolume": 42.0,
        "canonical_evaluations_used": 6,
        "confirmed_front_tags": ["A:1", "B:2", "C:3"],
        "members": [
            {
                "tag": "A:1",
                "confirmed_aspect_ratio": 10.0,
                "confirmed_min_len": 10.0,
                "confirmed_r": [-0.1, -0.05, -0.02, -0.01, -0.03],
                "retained": True,
                "strict_nominal": True,
                "solo_hypervolume": 110.0,
            },
            {
                "tag": "B:2",
                "confirmed_aspect_ratio": 10.5,
                "confirmed_min_len": 10.5,
                "confirmed_r": [0.0, 0.005, 0.01, 0.02, -0.01],
                "retained": True,
                "strict_nominal": False,
                "solo_hypervolume": 109.5,
            },
            # dominated by A:1 (worse aspect AND worse min_len) -- must be
            # excluded from the nondominated front used for the plot.
            {
                "tag": "C:3",
                "confirmed_aspect_ratio": 11.0,
                "confirmed_min_len": 9.5,
                "confirmed_r": [-0.2, -0.1, -0.1, -0.1, -0.1],
                "retained": True,
                "strict_nominal": True,
                "solo_hypervolume": 95.0,
            },
        ],
    }


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "run"
    d.mkdir()
    (d / "REPORT.json").write_text(json.dumps(_synthetic_report()))
    (d / "MANIFEST.json").write_text(json.dumps({"tolerance": 0.01, "atol": 1e-4, "files": {}}))
    (d / "A_1.boundary.json").write_text(json.dumps(_tiny_boundary()))
    (d / "A_1.phireflected.boundary.json").write_text(json.dumps(_tiny_boundary()))
    (d / "B_2.boundary.json").write_text(json.dumps(_tiny_boundary()))
    (d / "C_3.boundary.json").write_text(json.dumps(_tiny_boundary()))
    return d


@pytest.fixture
def run_dir_e16(tmp_path: Path) -> Path:
    """A tiny E16-layout run dir (top-level `confirmed_front`, not
    `members`) with a real E16-style multi-colon tag."""
    d = tmp_path / "run_e16"
    d.mkdir()
    report = {
        "final_hypervolume": 1.0,
        "verif_evals": 1,
        "confirmed_front": [
            {
                "tag": "M1:P:L3:t1.0:w10:T0",
                "r": [-0.1, -0.05, -0.02, -0.01, -0.03],
                "aspect_ratio": 10.0,
                "min_len": 10.0,
            }
        ],
    }
    (d / "REPORT.json").write_text(json.dumps(report))
    (d / "M1_P_L3_t1.0_w10_T0.boundary.json").write_text(json.dumps(_tiny_boundary()))
    (d / "M1_P_L3_t1.0_w10_T0.phireflected.boundary.json").write_text(json.dumps(_tiny_boundary()))
    return d


# ---------------------------------------------------------------------------
# Member / file matching
# ---------------------------------------------------------------------------


def test_member_file_stem_replaces_colons_with_underscores():
    assert gallery.member_file_stem("JR:141:+") == "JR_141_+"
    assert gallery.member_file_stem("P:L3:t1.0") == "P_L3_t1.0"


def test_member_file_stem_handles_e16_style_multi_colon_tags():
    # Real E16 run-dir tag/filename pair (results/p3-e16-front/), more
    # colon-separated segments than any E15 tag has.
    assert gallery.member_file_stem("M1:P:L3:t1.0:w10:T0") == "M1_P_L3_t1.0_w10_T0"


def test_find_boundary_file_matches_e16_style_multi_colon_tag(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "M1_P_L3_t1.0_w10_T0.boundary.json").write_text("{}")
    path = gallery.find_boundary_file(d, "M1:P:L3:t1.0:w10:T0")
    assert path.name == "M1_P_L3_t1.0_w10_T0.boundary.json"


def test_find_boundary_file_matches_primary_export_not_phireflected(run_dir):
    path = gallery.find_boundary_file(run_dir, "A:1")
    assert path.name == "A_1.boundary.json"
    assert "phireflected" not in path.name


def test_find_boundary_file_missing_member_raises_file_not_found(run_dir):
    with pytest.raises(FileNotFoundError):
        gallery.find_boundary_file(run_dir, "Z:9")


def test_find_raw_well_returns_none_when_verification_file_absent(run_dir):
    assert gallery.find_raw_well(run_dir, "A:1") is None


def test_find_raw_well_matches_tag_by_file_stem_regardless_of_run_label(run_dir):
    record = {
        "tag": "CONF:E16:A_1:1",
        "raw": {"vacuum_well": -0.0042},
    }
    (run_dir / "verification.jsonl").write_text(json.dumps(record) + "\n")
    assert gallery.find_raw_well(run_dir, "A:1") == pytest.approx(-0.0042)


def test_load_front_members_matches_every_tag_to_its_boundary_file(run_dir):
    report = json.loads((run_dir / "REPORT.json").read_text())
    members = gallery.load_front_members(run_dir, report)
    tags = {member.tag for member in members}
    assert tags == {"A:1", "B:2", "C:3"}
    a1 = next(member for member in members if member.tag == "A:1")
    assert a1.boundary_path.name == "A_1.boundary.json"
    assert a1.residuals["well"] == pytest.approx(-0.03)
    assert a1.residuals["iota"] == pytest.approx(-0.1)


def test_load_front_members_raises_on_wrong_residual_count(run_dir):
    report = json.loads((run_dir / "REPORT.json").read_text())
    report["members"][0]["confirmed_r"] = [-0.1, -0.05]
    with pytest.raises(ValueError, match="residuals"):
        gallery.load_front_members(run_dir, report)


# ---------------------------------------------------------------------------
# REPORT.json layout normalisation (Phase 0 evidence: read against real
# E15-style and E16-style REPORT.json files -- see normalize_report's own
# docstring/comment block for the field mapping this pins down).
# ---------------------------------------------------------------------------


def _e15_style_single_member_report() -> dict:
    return {
        "tol": 0.01,
        "confirmed_hypervolume": 42.0,
        "canonical_evaluations_used": 6,
        "confirmed_front_tags": ["A:1"],
        "members": [
            {
                "tag": "A:1",
                "confirmed_aspect_ratio": 10.0,
                "confirmed_min_len": 10.0,
                "confirmed_r": [-0.1, -0.05, -0.02, -0.01, -0.03],
                "retained": True,
                "strict_nominal": True,
                # matches solo_hypervolume(10.0, 10.0) == 110.0, so the two
                # synthetic reports below normalise to identical records.
                "solo_hypervolume": 110.0,
            }
        ],
    }


def _e16_style_equivalent_report() -> dict:
    return {
        "final_hypervolume": 42.0,
        "verif_evals": 6,
        "confirmed_front": [
            {
                "tag": "A:1",
                "r": [-0.1, -0.05, -0.02, -0.01, -0.03],
                "aspect_ratio": 10.0,
                "min_len": 10.0,
            }
        ],
    }


def test_normalize_report_passes_e15_layout_through_unchanged():
    report = _e15_style_single_member_report()
    assert gallery.normalize_report(report) is report


def test_normalize_report_e16_layout_matches_equivalent_e15_layout():
    e15 = gallery.normalize_report(_e15_style_single_member_report())
    e16 = gallery.normalize_report(_e16_style_equivalent_report())
    assert e16["members"] == e15["members"]
    assert e16["confirmed_hypervolume"] == e15["confirmed_hypervolume"]
    assert e16["canonical_evaluations_used"] == e15["canonical_evaluations_used"]


def test_normalize_report_e16_layout_derives_retained_and_strict_nominal():
    report = {
        "final_hypervolume": 1.0,
        "verif_evals": 1,
        "confirmed_front": [
            {
                "tag": "X:1",
                # first residual is exactly 0.0 -- not strictly negative.
                "r": [0.0, -0.1, -0.1, -0.1, -0.1],
                "aspect_ratio": 5.0,
                "min_len": 5.0,
            }
        ],
    }
    member = gallery.normalize_report(report)["members"][0]
    assert member["retained"] is True
    assert member["strict_nominal"] is False


def test_normalize_report_e16_layout_matches_export_filenames(run_dir_e16):
    report = json.loads((run_dir_e16 / "REPORT.json").read_text())
    normalized = gallery.normalize_report(report)
    members = gallery.load_front_members(run_dir_e16, normalized)
    assert {member.tag for member in members} == {"M1:P:L3:t1.0:w10:T0"}
    assert members[0].boundary_path.name == "M1_P_L3_t1.0_w10_T0.boundary.json"


def test_normalize_report_refuses_unrecognised_layout():
    with pytest.raises(ValueError, match="unrecognised REPORT.json layout"):
        gallery.normalize_report({"foo": 1, "bar": 2})


def test_normalize_report_refusal_names_the_keys_seen():
    with pytest.raises(ValueError, match=r"\['bar', 'foo'\]"):
        gallery.normalize_report({"foo": 1, "bar": 2})


# ---------------------------------------------------------------------------
# Nondominated front ordering
# ---------------------------------------------------------------------------


def test_nondominated_front_excludes_dominated_point():
    points = [
        ("A", 10.0, 10.0),
        ("B", 10.5, 10.5),
        ("C", 11.0, 9.5),  # dominated by A: worse aspect AND worse min_len
    ]
    front = gallery.nondominated_front(points)
    assert [tag for tag, _, _ in front] == ["A", "B"]


def test_nondominated_front_sorts_ascending_by_aspect():
    points = [
        ("B", 10.5, 10.5),
        ("A", 10.0, 10.0),
    ]
    front = gallery.nondominated_front(points)
    assert [tag for tag, _, _ in front] == ["A", "B"]


def test_nondominated_front_keeps_all_points_on_a_genuine_tradeoff():
    points = [("A", 10.0, 10.0), ("B", 10.5, 10.5), ("C", 11.0, 11.0)]
    front = gallery.nondominated_front(points)
    assert len(front) == 3


def test_nondominated_front_drops_exact_duplicate_leaving_one():
    points = [("A", 10.0, 10.0), ("A2", 10.0, 10.0)]
    front = gallery.nondominated_front(points)
    # neither strictly dominates the other (no strict inequality either way)
    assert len(front) == 2


# ---------------------------------------------------------------------------
# Residual colouring thresholds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (-0.5, "green"),
        (-1e-9, "green"),
        (0.0, "amber"),
        (0.005, "amber"),
        (0.01, "amber"),
        (0.0100001, "red"),
        (1.0, "red"),
    ],
)
def test_residual_color_thresholds(value, expected):
    assert gallery.residual_color(value, tol=0.01) == expected


# ---------------------------------------------------------------------------
# Solo hypervolume
# ---------------------------------------------------------------------------


def test_solo_hypervolume_matches_known_report_value():
    # P:L3:t1.0 from a real E15 REPORT.json.
    hv = gallery.solo_hypervolume(aspect=10.532810029811248, min_len=10.440175165412176)
    assert hv == pytest.approx(108.3063115831926, rel=1e-9)


def test_solo_hypervolume_second_known_report_value():
    # JR:159:+ from a real E15 REPORT.json.
    hv = gallery.solo_hypervolume(aspect=10.64281345352038, min_len=10.75984244204361)
    assert hv == pytest.approx(110.03903948741049, rel=1e-9)


# ---------------------------------------------------------------------------
# Benchmark feasibility
# ---------------------------------------------------------------------------


def test_benchmark_feasible_true_when_every_residual_strictly_negative():
    residuals = {
        "iota": -0.1,
        "log10_qi": -0.05,
        "mirror": -0.02,
        "flux": -0.01,
        "well": -0.03,
    }
    assert gallery.benchmark_feasible(residuals) is True


def test_benchmark_feasible_false_when_any_residual_nonnegative():
    residuals = {
        "iota": 0.0,
        "log10_qi": -0.05,
        "mirror": -0.02,
        "flux": -0.01,
        "well": -0.03,
    }
    assert gallery.benchmark_feasible(residuals) is False


# ---------------------------------------------------------------------------
# HTML assembly (pure templating, fake data URIs -- no image I/O)
# ---------------------------------------------------------------------------


def _fake_data_uri() -> str:
    return "data:image/png;base64," + base64.b64encode(b"not a real png").decode("ascii")


def _built_html(run_dir: Path) -> str:
    report = json.loads((run_dir / "REPORT.json").read_text())
    manifest = json.loads((run_dir / "MANIFEST.json").read_text())
    members = gallery.load_front_members(run_dir, report)
    member_images = {
        member.tag: {view: _fake_data_uri() for view in gallery.VIEW_NAMES} for member in members
    }
    return gallery.build_html(
        run_name=run_dir.name,
        report=report,
        manifest=manifest,
        members=members,
        member_images=member_images,
        front_plot_uri=_fake_data_uri(),
        references=gallery.DEFAULT_REFERENCES,
    )


def test_build_html_has_one_img_per_member_per_view_plus_the_front_plot(run_dir):
    report = json.loads((run_dir / "REPORT.json").read_text())
    n_members = len(report["members"])
    html = _built_html(run_dir)
    assert html.count("<img") == n_members * len(gallery.VIEW_NAMES) + 1


def test_build_html_contains_the_footer_line(run_dir):
    html = _built_html(run_dir)
    assert gallery.FOOTER_LINE in html


def test_build_html_footer_text_is_configurable(run_dir):
    report = json.loads((run_dir / "REPORT.json").read_text())
    manifest = json.loads((run_dir / "MANIFEST.json").read_text())
    members = gallery.load_front_members(run_dir, report)
    member_images = {
        member.tag: {view: _fake_data_uri() for view in gallery.VIEW_NAMES} for member in members
    }
    html = gallery.build_html(
        run_name=run_dir.name,
        report=report,
        manifest=manifest,
        members=members,
        member_images=member_images,
        front_plot_uri=_fake_data_uri(),
        references=gallery.DEFAULT_REFERENCES,
        footer_text="a custom footer",
    )
    assert "a custom footer" in html
    assert gallery.FOOTER_LINE not in html


def test_build_html_has_no_network_references(run_dir):
    html = _built_html(run_dir)
    assert "http://" not in html
    assert "https://" not in html


def test_build_html_colours_amber_and_red_residual_cells(run_dir):
    # B:2's residuals include 0.0 (amber) and 0.02 (red, > 0.01 tol).
    html = _built_html(run_dir)
    assert "cell-amber" in html
    assert "cell-red" in html
    assert "cell-green" in html
