"""Tests for V2.6 Soul-informed detection context builder."""

from eval_conversation import _build_detector_soul_context
from super_brain.models import (
    Soul, Fact, Reality, Intention, Gap,
    PersonalityDNA, SampleSummary,
)


def _make_profile():
    return PersonalityDNA(
        id="test",
        sample_summary=SampleSummary(
            total_tokens=0, conversation_count=0,
            date_range=["u", "u"], contexts=["t"], confidence_overall=0.5,
        ),
    )


def test_empty_soul_returns_empty():
    soul = Soul(id="test", character=_make_profile())
    result = _build_detector_soul_context(soul)
    assert result == ""


def test_soul_with_facts_only():
    soul = Soul(
        id="test",
        character=_make_profile(),
        facts=[
            Fact(category="career", content="software engineer", confidence=0.9, source_turn=3),
            Fact(category="hobby", content="plays guitar", confidence=0.7, source_turn=5),
        ],
    )
    result = _build_detector_soul_context(soul)
    assert "software engineer" in result
    assert "guitar" in result
    assert "Background Information" in result
    assert "OBSERVED BEHAVIOR" in result


def test_soul_with_reality():
    soul = Soul(
        id="test",
        character=_make_profile(),
        facts=[Fact(category="career", content="engineer", confidence=0.9, source_turn=1)],
        reality=Reality(
            summary="Mid-career engineer feeling stuck",
            domains={"career": "engineering"},
            constraints=["mortgage", "limited savings"],
            resources=["technical skills", "network"],
        ),
    )
    result = _build_detector_soul_context(soul)
    assert "feeling stuck" in result
    assert "mortgage" in result
    assert "technical skills" in result


def test_soul_with_intentions():
    soul = Soul(
        id="test",
        character=_make_profile(),
        facts=[Fact(category="career", content="engineer", confidence=0.9, source_turn=1)],
        intentions=[
            Intention(description="start own business", domain="career", strength=0.85),
            Intention(description="travel more", domain="personal_growth", strength=0.6),
            Intention(description="learn cooking", domain="creative", strength=0.4),
            Intention(description="exercise daily", domain="health", strength=0.3),
        ],
    )
    result = _build_detector_soul_context(soul)
    # Should include top 3 by strength
    assert "start own business" in result
    assert "travel more" in result
    assert "learn cooking" in result
    # 4th intention (lowest strength) should NOT be included
    assert "exercise daily" not in result


def test_facts_limited_to_10():
    soul = Soul(
        id="test",
        character=_make_profile(),
        facts=[
            Fact(category="misc", content=f"fact_{i}", confidence=0.5 + i * 0.01, source_turn=i)
            for i in range(20)
        ],
    )
    result = _build_detector_soul_context(soul)
    # Should have at most 10 fact lines
    fact_lines = [line for line in result.split("\n") if line.startswith("- [")]
    assert len(fact_lines) <= 10


def test_detector_analyze_accepts_soul_context():
    """Verify Detector.analyze() signature accepts soul_context parameter."""
    import inspect
    from super_brain.detector import Detector
    sig = inspect.signature(Detector.analyze)
    assert "soul_context" in sig.parameters
    assert sig.parameters["soul_context"].default is None


def test_detect_and_compare_accepts_soul_parameter():
    """Verify detect_and_compare() signature accepts soul parameter."""
    import inspect
    from eval_conversation import detect_and_compare
    sig = inspect.signature(detect_and_compare)
    assert "soul" in sig.parameters
    assert sig.parameters["soul"].default is None
