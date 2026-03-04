"""Save and load PersonalityDNA profiles to/from JSON files."""

from __future__ import annotations

from pathlib import Path

from super_brain.models import PersonalityDNA


def save_profile(profile: PersonalityDNA, filepath: str | Path) -> None:
    """Save a PersonalityDNA profile to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.model_dump_json(indent=2))


def load_profile(filepath: str | Path) -> PersonalityDNA:
    """Load a PersonalityDNA profile from a JSON file."""
    path = Path(filepath)
    return PersonalityDNA.model_validate_json(path.read_text())
