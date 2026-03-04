"""Tests for archetype derivation."""

from super_brain.models import PersonalityDNA, Trait, SampleSummary
from super_brain.archetypes import derive_archetypes, top_archetypes


def _make_profile(traits: list[dict]) -> PersonalityDNA:
    return PersonalityDNA(
        id="test",
        sample_summary=SampleSummary(
            total_tokens=0, conversation_count=0,
            date_range=["unknown", "unknown"],
            contexts=["test"], confidence_overall=1.0,
        ),
        traits=[
            Trait(dimension=t["dim"], name=t["name"], value=t["value"], confidence=1.0)
            for t in traits
        ],
    )


def test_derive_all_systems():
    profile = _make_profile([
        {"dim": "OPN", "name": "fantasy", "value": 0.5},
        {"dim": "EXT", "name": "assertiveness", "value": 0.5},
    ])
    results = derive_archetypes(profile)
    assert "MBTI" in results
    assert "Enneagram" in results
    assert "Seduction" in results
    assert "Jung" in results
    assert "DISC" in results
    assert "Alignment" in results


def test_derive_specific_system():
    profile = _make_profile([
        {"dim": "EXT", "name": "assertiveness", "value": 0.9},
        {"dim": "SOC", "name": "social_dominance", "value": 0.9},
    ])
    results = derive_archetypes(profile, systems=["DISC"])
    assert "DISC" in results
    assert "MBTI" not in results


def test_mbti_types_count():
    profile = _make_profile([])
    results = derive_archetypes(profile, systems=["MBTI"])
    assert len(results["MBTI"]) == 16


def test_enneagram_types_count():
    profile = _make_profile([])
    results = derive_archetypes(profile, systems=["Enneagram"])
    assert len(results["Enneagram"]) == 9


def test_seduction_types_count():
    profile = _make_profile([])
    results = derive_archetypes(profile, systems=["Seduction"])
    assert len(results["Seduction"]) == 9


def test_jung_types_count():
    profile = _make_profile([])
    results = derive_archetypes(profile, systems=["Jung"])
    assert len(results["Jung"]) == 12


def test_disc_types_count():
    profile = _make_profile([])
    results = derive_archetypes(profile, systems=["DISC"])
    assert len(results["DISC"]) == 4


def test_alignment_types_count():
    profile = _make_profile([])
    results = derive_archetypes(profile, systems=["Alignment"])
    assert len(results["Alignment"]) == 9


def test_top_archetypes():
    profile = _make_profile([
        {"dim": "EXT", "name": "assertiveness", "value": 0.9},
    ])
    tops = top_archetypes(profile, n=1)
    for system, matches in tops.items():
        assert len(matches) == 1


def test_results_sorted_by_score():
    profile = _make_profile([
        {"dim": "OPN", "name": "fantasy", "value": 0.5},
    ])
    results = derive_archetypes(profile)
    for system, matches in results.items():
        for i in range(len(matches) - 1):
            assert matches[i].score >= matches[i + 1].score, (
                f"{system}: {matches[i].name}({matches[i].score}) < {matches[i+1].name}({matches[i+1].score})"
            )


def test_charismatic_leader_matches_expected():
    """A charismatic leader profile should match ENTJ/ENFJ in MBTI and Charismatic in Seduction."""
    profile = _make_profile([
        {"dim": "EXT", "name": "warmth", "value": 0.80},
        {"dim": "EXT", "name": "gregariousness", "value": 0.75},
        {"dim": "EXT", "name": "assertiveness", "value": 0.90},
        {"dim": "EXT", "name": "activity_level", "value": 0.85},
        {"dim": "EXT", "name": "positive_emotions", "value": 0.85},
        {"dim": "STR", "name": "self_mythologizing", "value": 0.75},
        {"dim": "SOC", "name": "charm_influence", "value": 0.85},
        {"dim": "COG", "name": "locus_of_control", "value": 0.90},
        {"dim": "AGR", "name": "modesty", "value": 0.20},
        {"dim": "NEU", "name": "anxiety", "value": 0.10},
    ])
    results = derive_archetypes(profile)

    # MBTI: should be E*J type
    top_mbti = results["MBTI"][0].name
    assert top_mbti[0] == "E", f"Expected E***, got {top_mbti}"

    # Seduction: Charismatic should be in top 3
    seduction_names = [m.name for m in results["Seduction"][:3]]
    assert "Charismatic" in seduction_names, f"Expected Charismatic in top 3, got {seduction_names}"


def test_empathetic_healer_matches_expected():
    """An empathetic healer should match Caregiver in Jung and Ideal Lover in Seduction."""
    profile = _make_profile([
        {"dim": "EMO", "name": "empathy_cognitive", "value": 0.90},
        {"dim": "EMO", "name": "empathy_affective", "value": 0.90},
        {"dim": "AGR", "name": "altruism", "value": 0.90},
        {"dim": "AGR", "name": "tender_mindedness", "value": 0.90},
        {"dim": "EXT", "name": "warmth", "value": 0.90},
        {"dim": "VAL", "name": "care_harm", "value": 0.95},
    ])
    results = derive_archetypes(profile)

    # Jung: Caregiver should be #1
    top_jung = results["Jung"][0].name
    assert top_jung == "Caregiver", f"Expected Caregiver, got {top_jung}"

    # Seduction: Ideal Lover should be in top 3
    seduction_names = [m.name for m in results["Seduction"][:3]]
    assert "Ideal Lover" in seduction_names or "Charmer" in seduction_names
