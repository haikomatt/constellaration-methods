import json

import pytest
from check_no_boundaries import check_tree, find_boundary_keys


def test_find_boundary_keys_returns_empty_for_clean_data():
    assert find_boundary_keys({"label": "E13", "n_front_members": 25}) == []


def test_find_boundary_keys_finds_top_level_hit():
    assert find_boundary_keys({"boundary": {}}) == ["$.boundary"]


def test_find_boundary_keys_finds_nested_hits_with_list_indices():
    data = {"members": [{"tag": "a"}, {"boundary_dict": {}}]}

    hits = find_boundary_keys(data)

    assert hits == ["$.members[1].boundary_dict"]


def test_check_tree_missing_root_is_clean(tmp_path):
    assert check_tree(tmp_path / "does-not-exist") == {}


def test_check_tree_flags_a_dirty_file(tmp_path):
    root = tmp_path / "p3"
    root.mkdir()
    (root / "REPORT.json").write_text(json.dumps({"boundary": {"r_cos": [[1.0]]}}))

    offenders = check_tree(root)

    assert list(offenders) == [str(root / "REPORT.json")]
    # both the outer "boundary" key and the "r_cos" key nested inside its
    # value are reported -- find_boundary_keys keeps recursing into a hit's
    # own value so a boundary dict's contents are never silently skipped.
    assert offenders[str(root / "REPORT.json")] == ["$.boundary", "$.boundary.r_cos"]


def test_check_tree_passes_a_clean_file(tmp_path):
    root = tmp_path / "p3"
    root.mkdir()
    (root / "REPORT.json").write_text(json.dumps({"label": "E13", "n_front_members": 25}))

    assert check_tree(root) == {}


def test_check_tree_scans_nested_subdirectories(tmp_path):
    root = tmp_path / "p3"
    (root / "e17-front").mkdir(parents=True)
    (root / "e17-front" / "REPORT.json").write_text(json.dumps({"x": [1, 2]}))

    offenders = check_tree(root)

    assert str(root / "e17-front" / "REPORT.json") in offenders


@pytest.mark.parametrize(
    "bad_key", ["x", "boundary", "boundary_dict", "r_cos", "z_sin", "r_sin", "z_cos"]
)
def test_check_tree_catches_every_banned_key(tmp_path, bad_key):
    root = tmp_path / "p3"
    root.mkdir()
    (root / "REPORT.json").write_text(json.dumps({bad_key: [1.0]}))

    offenders = check_tree(root)

    assert offenders


def test_check_tree_exempts_exactly_the_e24r_b14_2_own_lineage_exports(tmp_path):
    root = tmp_path / "p3"
    own = root / "E24r-b14-2" / "own-lineage"
    own.mkdir(parents=True)
    (own / "feasible_boundary_P1L1.json").write_text(json.dumps({"r_cos": [[1.0]]}))
    (root / "E24r-b14-2" / "REPORT.json").write_text(json.dumps({"r_cos": [[1.0]]}))
    (root / "E24r-a85-2").mkdir()
    (root / "E24r-a85-2" / "best_boundary.json").write_text(json.dumps({"r_cos": [[1.0]]}))

    offenders = check_tree(root)

    assert set(offenders) == {
        str(root / "E24r-b14-2" / "REPORT.json"),
        str(root / "E24r-a85-2" / "best_boundary.json"),
    }
