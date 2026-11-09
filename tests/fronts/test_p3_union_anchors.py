"""Ratchet tests for p3_union_anchors (E25 anchors)."""

from __future__ import annotations

import json

import pytest

from constellaration_methods.fronts import p3_union_anchors as ua


def _cand(v, aspect, tag, **rep):
    base = {"mpol": 10, "ntor": 10, "nfp": 2, "scaling": 1.0}
    base.update(rep)
    return {
        "tag": tag,
        "x": [0.1, 0.2],
        "r": [-0.2, v, -0.1, -0.05, -0.08],
        "raw": {"aspect_ratio": aspect, "min_len": 7.0},
        **base,
    }


def _hv(aspect, min_len):
    return (20.0 - aspect) * (min_len - 1.0)


def test_union_merges_same_representation_and_marks_strict():
    rep = ua.union_anchors([_cand(-0.01, 10.2, "a"), _cand(0.005, 10.3, "b")], solo_hv_fn=_hv)
    assert abs(rep["members"][0]["solo_hypervolume"] - (20 - 10.2) * 6.0) < 1e-9
    assert rep["n_retained"] == 2 and rep["confirmed_front_tags"] == ["a", "b"]
    assert (
        rep["members"][0]["strict_nominal"] is True and rep["members"][1]["strict_nominal"] is False
    )
    assert rep["members"][1]["feasible"] is True
    assert rep["representation"] == {"mpol": 10, "ntor": 10, "nfp": 2, "scaling": 1.0}


def test_union_refuses_mixed_representations():
    with pytest.raises(ValueError):
        ua.union_anchors([_cand(-0.01, 10.2, "a"), _cand(-0.01, 10.2, "b", nfp=3)], solo_hv_fn=_hv)


def test_union_refuses_missing_raw():
    c = _cand(-0.01, 10.2, "a")
    c["raw"] = None
    with pytest.raises(ValueError):
        ua.union_anchors([c], solo_hv_fn=_hv)


def test_union_writes_every_anchor_boundary_where_the_front_builder_reads_it(tmp_path):
    """2026-10-22: E25 died at once with FileNotFoundError on
    results/p3-e25-anchors-2/anchor_e24r-a85-3.boundary.json. The front builder's
    load_anchor_x rebuilds each anchor from <safe tag>.boundary.json beside REPORT.json;
    the union script wrote the REPORT only."""
    from constellaration_methods.fronts.p3_front_from_archive import _safe_name

    cands = [_cand(-0.01, 10.2, "anchor:e22-a"), _cand(0.005, 7.4, "anchor:e24r-a85-3")]
    rep = ua.union_anchors(cands, solo_hv_fn=_hv)
    written = ua.write_anchor_boundaries(
        cands, tmp_path, boundary_fn=lambda c: {"r_cos": [[1.0]], "tag_seen": c["tag"]}
    )
    for m in rep["members"]:
        path = tmp_path / f"{_safe_name(m['tag'])}.boundary.json"
        assert path.exists(), path
        assert path in written
        assert json.loads(path.read_text())["tag_seen"] == m["tag"]


def test_union_boundary_writer_refuses_to_overwrite_with_a_different_boundary(tmp_path):
    cands = [_cand(-0.01, 10.2, "anchor:e22-a")]
    ua.write_anchor_boundaries(cands, tmp_path, boundary_fn=lambda c: {"v": 1})
    ua.write_anchor_boundaries(cands, tmp_path, boundary_fn=lambda c: {"v": 1})  # same: fine
    with pytest.raises(ValueError, match="differs"):
        ua.write_anchor_boundaries(cands, tmp_path, boundary_fn=lambda c: {"v": 2})


def _with_r00(c, r00):
    """A candidate whose base boundary (the engine's template) has r_cos(0,0) = r00 on a
    1x1 grid; the engine never moves that entry, so it is part of the representation."""
    return dict(
        c,
        boundary_dict={
            "r_cos": [[0.0, r00, 0.0], [0.0, 0.0, 0.0]],
            "z_sin": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "n_field_periods": 2,
        },
    )


def test_union_refuses_anchors_whose_fixed_major_radius_differs():
    """2026-10-26: E25/E25b rendered the E24r designs with E22's major radius (0.8143 vs
    0.8016 and 0.8153): same x, different shape (link 3 evaluated at aspect 7.49, worst
    residual 0.026, not 7.37 / 0.0100). r_cos(0,0) is outside the engine's mask, so it
    belongs to the representation and must agree across anchors."""
    a = _with_r00(_cand(-0.01, 10.2, "anchor:e22"), 0.814321)
    b = _with_r00(_cand(0.005, 7.4, "anchor:a85"), 0.801554)
    with pytest.raises(ValueError, match="major radius"):
        ua.union_anchors([a, b], solo_hv_fn=_hv)
    ok = _with_r00(_cand(0.005, 7.4, "anchor:a85b"), 0.814321)
    assert ua.union_anchors([a, ok], solo_hv_fn=_hv)["n_retained"] == 2
