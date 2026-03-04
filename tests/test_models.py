"""Tests for data models."""

import json

from pydantic import ValidationError
import pytest

from super_brain.models import (
    Trait,
    Evidence,
    TraitRelation,
    SampleSummary,
    PersonalityDNA,
)


def test_trait_creation():
    t = Trait(dimension="OPN", name="fantasy", value=0.75, confidence=0.9)
    assert t.value == 0.75
    assert t.confidence == 0.9
    assert t.evidence == []


def test_trait_value_bounds():
    with pytest.raises(ValidationError):
        Trait(dimension="OPN", name="fantasy", value=1.5, confidence=0.9)
    with pytest.raises(ValidationError):
        Trait(dimension="OPN", name="fantasy", value=-0.1, confidence=0.9)


def test_trait_with_evidence():
    t = Trait(
        dimension="DRK", name="narcissism", value=0.3, confidence=0.8,
        evidence=[Evidence(text="I'm the best", source="input_text")],
    )
    assert len(t.evidence) == 1
    assert t.evidence[0].text == "I'm the best"


def test_trait_relation():
    r = TraitRelation(source="narcissism", target="modesty", correlation=-0.8, direction="negative")
    assert r.correlation == -0.8


def test_sample_summary():
    s = SampleSummary(
        total_tokens=500, conversation_count=1,
        date_range=["2024-01-01", "2024-01-01"],
        contexts=["general"], confidence_overall=0.85,
    )
    assert s.total_tokens == 500


def test_personality_dna_creation():
    dna = PersonalityDNA(
        id="test_person",
        sample_summary=SampleSummary(
            total_tokens=100, conversation_count=1,
            date_range=["unknown", "unknown"],
            contexts=["test"], confidence_overall=0.9,
        ),
        traits=[
            Trait(dimension="OPN", name="fantasy", value=0.7, confidence=0.9),
            Trait(dimension="DRK", name="narcissism", value=0.2, confidence=0.8),
        ],
    )
    assert len(dna.traits) == 2
    assert dna.version == "0.1"


def test_personality_dna_serialization():
    dna = PersonalityDNA(
        id="test_person",
        sample_summary=SampleSummary(
            total_tokens=100, conversation_count=1,
            date_range=["unknown", "unknown"],
            contexts=["test"], confidence_overall=0.9,
        ),
        traits=[
            Trait(dimension="OPN", name="fantasy", value=0.7, confidence=0.9),
        ],
    )
    json_str = dna.model_dump_json()
    loaded = PersonalityDNA.model_validate_json(json_str)
    assert loaded.id == "test_person"
    assert loaded.traits[0].value == 0.7
    assert loaded.traits[0].name == "fantasy"


def test_personality_dna_empty_traits():
    dna = PersonalityDNA(
        id="empty",
        sample_summary=SampleSummary(
            total_tokens=0, conversation_count=0,
            date_range=["unknown", "unknown"],
            contexts=[], confidence_overall=0.0,
        ),
    )
    assert dna.traits == []
    assert dna.trait_relations == []
