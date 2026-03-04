"""Tests for the speaker module."""

from super_brain.models import PersonalityDNA, Trait, SampleSummary
from super_brain.speaker import (
    _value_to_instruction,
    profile_to_style_instructions,
    _generate_boundary_constraints,
    _generate_interaction_warnings,
)


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


def test_value_to_instruction_levels():
    """Test that different values produce different level descriptions."""
    low = Trait(dimension="OPN", name="fantasy", value=0.1, confidence=0.9)
    mid = Trait(dimension="OPN", name="fantasy", value=0.5, confidence=0.9)
    high = Trait(dimension="OPN", name="fantasy", value=0.9, confidence=0.9)

    low_desc = _value_to_instruction(low)
    mid_desc = _value_to_instruction(mid)
    high_desc = _value_to_instruction(high)

    assert "very low" in low_desc
    assert "moderate" in mid_desc
    assert "very high" in high_desc


def test_value_to_instruction_includes_anchor():
    """Anchor text from catalog should appear in instruction."""
    t = Trait(dimension="OPN", name="fantasy", value=0.75, confidence=0.9)
    desc = _value_to_instruction(t)
    assert "imagin" in desc.lower()  # Should include anchor about imagination


def test_profile_to_style_instructions_structure():
    profile = _make_profile([
        {"dim": "OPN", "name": "fantasy", "value": 0.8},
        {"dim": "DRK", "name": "narcissism", "value": 0.7},
    ])
    instructions = profile_to_style_instructions(profile)
    assert "<personality_profile>" in instructions
    assert "</personality_profile>" in instructions
    assert "fantasy" in instructions
    assert "narcissism" in instructions


def test_profile_to_style_instructions_intensity():
    profile = _make_profile([
        {"dim": "OPN", "name": "fantasy", "value": 0.5},
    ])
    normal = profile_to_style_instructions(profile, intensity_scale=1.0)
    amplified = profile_to_style_instructions(profile, intensity_scale=1.5)
    # Amplified should show higher value
    assert "0.50" in normal
    assert "0.75" in amplified


def test_boundary_constraints_mid_range():
    profile = _make_profile([
        {"dim": "OPN", "name": "fantasy", "value": 0.50},
    ])
    constraints = _generate_boundary_constraints(profile)
    assert "AIM FOR" in constraints
    assert "TOO LOW" in constraints
    assert "TOO HIGH" in constraints


def test_boundary_constraints_extreme():
    profile = _make_profile([
        {"dim": "OPN", "name": "fantasy", "value": 0.05},
    ])
    constraints = _generate_boundary_constraints(profile)
    assert "Keep very low" in constraints


def test_interaction_warnings_narcissism_modesty():
    profile = _make_profile([
        {"dim": "DRK", "name": "narcissism", "value": 0.8},
        {"dim": "AGR", "name": "modesty", "value": 0.1},
    ])
    warnings = _generate_interaction_warnings(profile)
    assert "narcissism" in warnings.lower() or "WARNING" in warnings


def test_interaction_warnings_none_when_normal():
    profile = _make_profile([
        {"dim": "OPN", "name": "fantasy", "value": 0.5},
        {"dim": "CON", "name": "order", "value": 0.5},
    ])
    warnings = _generate_interaction_warnings(profile)
    assert warnings == ""
