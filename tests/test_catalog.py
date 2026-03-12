"""Tests for the trait catalog."""

from super_brain.catalog import (
    TRAIT_CATALOG,
    ALL_DIMENSIONS,
    CONSISTENCY_RULES,
    TRAIT_MAP,
    get_traits_for_dimension,
    get_trait_by_name,
)


def test_catalog_has_69_traits():
    assert len(TRAIT_CATALOG) == 69


def test_all_dimensions_present():
    dims_in_catalog = {t["dimension"] for t in TRAIT_CATALOG}
    expected = {"OPN", "CON", "EXT", "AGR", "NEU", "HON", "DRK", "EMO", "SOC", "COG", "VAL", "STR", "HUM"}
    assert dims_in_catalog == expected


def test_all_dimensions_described():
    for dim in {"OPN", "CON", "EXT", "AGR", "NEU", "HON", "DRK", "EMO", "SOC", "COG", "VAL", "STR", "HUM"}:
        assert dim in ALL_DIMENSIONS


def test_trait_counts_per_dimension():
    counts = {}
    for t in TRAIT_CATALOG:
        counts[t["dimension"]] = counts.get(t["dimension"], 0) + 1
    assert counts["OPN"] == 7
    assert counts["CON"] == 7
    assert counts["EXT"] == 7
    assert counts["AGR"] == 7
    assert counts["NEU"] == 6
    assert counts["HON"] == 4
    assert counts["DRK"] == 4
    assert counts["EMO"] == 5
    assert counts["SOC"] == 6
    assert counts["COG"] == 4
    assert counts["VAL"] == 4
    assert counts["STR"] == 4
    assert counts["HUM"] == 4


def test_unique_trait_names():
    names = [t["name"] for t in TRAIT_CATALOG]
    assert len(names) == len(set(names)), f"Duplicate names: {[n for n in names if names.count(n) > 1]}"


def test_each_trait_has_required_fields():
    for t in TRAIT_CATALOG:
        assert "dimension" in t
        assert "name" in t
        assert "description" in t
        assert "detection_hint" in t
        assert "value_anchors" in t
        anchors = t["value_anchors"]
        for key in ["0.0", "0.25", "0.50", "0.75", "1.0"]:
            assert key in anchors, f"Missing anchor {key} for {t['name']}"


def test_consistency_rules_reference_valid_traits():
    all_names = {t["name"] for t in TRAIT_CATALOG}
    for name_a, name_b, max_sum in CONSISTENCY_RULES:
        assert name_a in all_names, f"Invalid trait in consistency rule: {name_a}"
        assert name_b in all_names, f"Invalid trait in consistency rule: {name_b}"
        assert 0 < max_sum <= 2.0


def test_trait_map_complete():
    assert len(TRAIT_MAP) == 69
    for t in TRAIT_CATALOG:
        assert (t["dimension"], t["name"]) in TRAIT_MAP


def test_get_traits_for_dimension():
    opn = get_traits_for_dimension("OPN")
    assert len(opn) == 7
    assert all(t["dimension"] == "OPN" for t in opn)


def test_new_traits_exist():
    names = {t["name"] for t in TRAIT_CATALOG}
    for expected in ["curiosity", "decisiveness", "verbosity", "politeness", "optimism"]:
        assert expected in names, f"Missing new trait: {expected}"


def test_get_trait_by_name():
    t = get_trait_by_name("narcissism")
    assert t is not None
    assert t["dimension"] == "DRK"
    assert get_trait_by_name("nonexistent") is None
