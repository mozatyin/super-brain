"""Tests for V2.5 data models: Intention, Gap, ThinkDeepResult."""

from super_brain.models import Intention, Gap, ThinkDeepResult


def test_intention_creation():
    i = Intention(
        description="wants to start own business",
        domain="career",
        strength=0.8,
        blockers=["limited savings", "risk aversion"],
    )
    assert i.description == "wants to start own business"
    assert i.domain == "career"
    assert i.strength == 0.8
    assert len(i.blockers) == 2


def test_intention_defaults():
    i = Intention(description="learn guitar", domain="hobby", strength=0.5)
    assert i.blockers == []


def test_gap_creation():
    g = Gap(
        intention="wants to start own business",
        reality="currently employed full-time with mortgage",
        bridge_question="What would need to change for you to take that leap?",
        priority=0.9,
    )
    assert "business" in g.intention
    assert g.priority == 0.9
    assert "leap" in g.bridge_question


def test_think_deep_result():
    result = ThinkDeepResult(
        soul_narrative="This person is at a crossroads between security and ambition",
        intentions=[
            Intention(description="start a business", domain="career", strength=0.8),
        ],
        gaps=[
            Gap(
                intention="start a business",
                reality="employed full-time",
                bridge_question="What's holding you back?",
                priority=0.9,
            ),
        ],
        critical_question="What would you do if money weren't a concern?",
        conversation_strategy="Shift from listening to exploring risk tolerance",
    )
    assert len(result.intentions) == 1
    assert len(result.gaps) == 1
    assert result.critical_question != ""
    assert "crossroads" in result.soul_narrative


def test_think_deep_result_empty():
    result = ThinkDeepResult(
        soul_narrative="Not enough information yet",
        intentions=[],
        gaps=[],
        critical_question="Tell me more about what matters to you",
        conversation_strategy="Continue listening",
    )
    assert len(result.intentions) == 0
    assert len(result.gaps) == 0
