"""Tests for scenario engine."""
import pytest
from super_brain.scenarios import SCENARIOS, Scenario, get_coverage_matrix, get_scenario_sequence
from super_brain.catalog import TRAIT_CATALOG


def test_all_69_traits_covered():
    """Every trait in the catalog must be covered by at least one scenario."""
    coverage = get_coverage_matrix()
    catalog_traits = {t["name"] for t in TRAIT_CATALOG}
    covered_traits = set(coverage.keys())
    missing = catalog_traits - covered_traits
    assert missing == set(), f"Uncovered traits: {missing}"


def test_coverage_is_complete():
    """Coverage matrix should have exactly 69 traits."""
    coverage = get_coverage_matrix()
    assert len(coverage) == 69


def test_scenario_count():
    """Should have 16 scenario packs."""
    assert len(SCENARIOS) == 16


def test_each_scenario_has_target_traits():
    """Every scenario should target at least 3 traits."""
    for s in SCENARIOS:
        assert len(s.target_traits) >= 3, f"{s.id} has only {len(s.target_traits)} traits"


def test_scenario_ids_unique():
    """All scenario IDs should be unique."""
    ids = [s.id for s in SCENARIOS]
    assert len(ids) == len(set(ids)), f"Duplicate IDs found"


def test_get_scenario_sequence_shuffles():
    """Different seeds should produce different orderings."""
    seq1 = get_scenario_sequence(seed=0)
    seq2 = get_scenario_sequence(seed=42)
    ids1 = [s.id for s in seq1]
    ids2 = [s.id for s in seq2]
    # Same scenarios but different order
    assert set(ids1) == set(ids2)
    assert ids1 != ids2  # very unlikely to be same with different seeds


def test_get_scenario_sequence_deterministic():
    """Same seed should produce same ordering."""
    seq1 = get_scenario_sequence(seed=123)
    seq2 = get_scenario_sequence(seed=123)
    assert [s.id for s in seq1] == [s.id for s in seq2]


def test_all_target_traits_exist_in_catalog():
    """Every trait referenced in scenarios must exist in the catalog."""
    catalog_traits = {t["name"] for t in TRAIT_CATALOG}
    for s in SCENARIOS:
        for t in s.target_traits:
            assert t in catalog_traits, f"Scenario {s.id} references unknown trait: {t}"


def test_scenario_turns_reasonable():
    """Each scenario should have 2-5 turns."""
    for s in SCENARIOS:
        assert 2 <= s.turns <= 5, f"{s.id} has {s.turns} turns (expected 2-5)"


def test_total_turns_across_scenarios():
    """Total turns should be reasonable (40-80 range)."""
    total = sum(s.turns for s in SCENARIOS)
    assert 40 <= total <= 80, f"Total turns: {total}"
