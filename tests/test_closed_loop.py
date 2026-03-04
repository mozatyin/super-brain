"""Closed-loop integration test: profile → speaker → detector → compare.

Requires ANTHROPIC_API_KEY environment variable.
"""

import os
import pytest

from super_brain.models import PersonalityDNA, Trait, SampleSummary
from super_brain.speaker import Speaker
from super_brain.detector import Detector


pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


def _make_profile(traits: list[dict]) -> PersonalityDNA:
    return PersonalityDNA(
        id="closed_loop_test",
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


EXTREME_PROFILE = _make_profile([
    {"dim": "NEU", "name": "anxiety", "value": 0.90},
    {"dim": "EXT", "name": "assertiveness", "value": 0.10},
    {"dim": "EMO", "name": "empathy_affective", "value": 0.85},
    {"dim": "DRK", "name": "narcissism", "value": 0.05},
    {"dim": "OPN", "name": "fantasy", "value": 0.85},
    {"dim": "CON", "name": "order", "value": 0.15},
])

PROMPTS = [
    "Describe how you handle a conflict at work",
    "What's your opinion on people who break rules?",
    "Tell me about something you're worried about",
    "Describe your ideal day off",
]


@pytest.fixture(scope="module")
def api_key():
    return os.environ["ANTHROPIC_API_KEY"]


@pytest.fixture(scope="module")
def generated_text(api_key):
    speaker = Speaker(api_key=api_key)
    lines = []
    for prompt in PROMPTS:
        text = speaker.generate(profile=EXTREME_PROFILE, content=prompt)
        lines.append(f"Speaker: {text}")
    return "\n\n".join(lines)


@pytest.fixture(scope="module")
def detected_profile(api_key, generated_text):
    detector = Detector(api_key=api_key)
    return detector.analyze(
        text=generated_text,
        speaker_id="closed_loop_test",
        speaker_label="Speaker",
    )


def test_detected_profile_has_traits(detected_profile):
    """Detector should return traits for all 68 dimensions."""
    assert len(detected_profile.traits) >= 60  # Allow some margin


def test_extreme_traits_detected_within_tolerance(detected_profile):
    """Extreme traits (>0.8 or <0.2) should be detected within ±0.35."""
    original_map = {t.name: t.value for t in EXTREME_PROFILE.traits}
    detected_map = {t.name: t.value for t in detected_profile.traits}

    errors = []
    for name, original in original_map.items():
        if name not in detected_map:
            continue
        error = abs(original - detected_map[name])
        errors.append((name, original, detected_map[name], error))

    # At least half of the traits should be within 0.35
    within_tolerance = sum(1 for _, _, _, e in errors if e <= 0.35)
    total = len(errors)
    ratio = within_tolerance / total if total > 0 else 0

    assert ratio >= 0.5, (
        f"Only {within_tolerance}/{total} traits within ±0.35. "
        f"Errors: {[(n, f'{o:.2f}→{d:.2f} (err={e:.2f})') for n, o, d, e in errors]}"
    )
