"""Tests for trait → natural conversation topic mapping."""

from super_brain.trait_topic_map import TRAIT_TOPIC_MAP, get_topics_for_traits


def test_trait_topic_map_covers_stubborn_traits():
    """Map must cover the 5 stubbornly hard traits from EVAL_HISTORY."""
    stubborn = [
        "humor_self_enhancing",
        "social_dominance",
        "mirroring_ability",
        "information_control",
        "competence",
    ]
    for trait in stubborn:
        assert trait in TRAIT_TOPIC_MAP, f"Missing stubborn trait: {trait}"
        assert len(TRAIT_TOPIC_MAP[trait]) >= 2, f"Need ≥2 topics for {trait}"


def test_trait_topic_map_has_minimum_coverage():
    """Map should cover at least 30 traits (most impactful ones)."""
    assert len(TRAIT_TOPIC_MAP) >= 30


def test_new_traits_have_topics():
    for name in ["verbosity", "curiosity", "politeness", "optimism", "decisiveness"]:
        assert name in TRAIT_TOPIC_MAP, f"Missing topic map for: {name}"
        assert len(TRAIT_TOPIC_MAP[name]) >= 2


def test_get_topics_for_traits():
    """get_topics_for_traits should return natural conversation starters."""
    topics = get_topics_for_traits(["social_dominance", "trust"], max_per_trait=2)
    assert len(topics) >= 2
    assert len(topics) <= 4
    assert all(isinstance(t, str) for t in topics)
