"""Tests for storage module."""

import tempfile
from pathlib import Path

from super_brain.models import PersonalityDNA, Trait, SampleSummary
from super_brain.storage import save_profile, load_profile


def _make_profile() -> PersonalityDNA:
    return PersonalityDNA(
        id="test_person",
        sample_summary=SampleSummary(
            total_tokens=500, conversation_count=1,
            date_range=["2024-01-01", "2024-01-01"],
            contexts=["test"], confidence_overall=0.85,
        ),
        traits=[
            Trait(dimension="OPN", name="fantasy", value=0.7, confidence=0.9),
            Trait(dimension="DRK", name="narcissism", value=0.2, confidence=0.8),
        ],
    )


def test_save_and_load_roundtrip():
    profile = _make_profile()
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "profile.json"
        save_profile(profile, filepath)
        loaded = load_profile(filepath)
        assert loaded.id == "test_person"
        assert len(loaded.traits) == 2
        assert loaded.traits[0].value == 0.7
        assert loaded.traits[1].name == "narcissism"


def test_save_creates_directories():
    profile = _make_profile()
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "nested" / "deep" / "profile.json"
        save_profile(profile, filepath)
        assert filepath.exists()
        loaded = load_profile(filepath)
        assert loaded.id == "test_person"


def test_save_overwrites():
    profile1 = _make_profile()
    profile2 = PersonalityDNA(
        id="different_person",
        sample_summary=SampleSummary(
            total_tokens=100, conversation_count=1,
            date_range=["unknown", "unknown"],
            contexts=["test"], confidence_overall=0.5,
        ),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "profile.json"
        save_profile(profile1, filepath)
        save_profile(profile2, filepath)
        loaded = load_profile(filepath)
        assert loaded.id == "different_person"
