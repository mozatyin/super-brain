"""Tests for V2.5 Soul Coverage scoring."""

from super_brain.models import (
    Soul, Fact, Reality, PersonalityDNA, SampleSummary,
)
from super_brain.soul_coverage import compute_soul_coverage


def _make_profile():
    return PersonalityDNA(
        id="test",
        sample_summary=SampleSummary(
            total_tokens=0, conversation_count=0,
            date_range=["u", "u"], contexts=["t"], confidence_overall=0.5,
        ),
    )


def test_empty_soul_coverage():
    soul = Soul(id="test", character=_make_profile())
    score = compute_soul_coverage(soul)
    assert score == 0.0


def test_full_soul_coverage():
    from super_brain.models import Intention, Gap
    soul = Soul(
        id="test",
        character=_make_profile(),
        facts=[
            Fact(category="c", content=f"fact_{i}", confidence=0.9, source_turn=i)
            for i in range(10)
        ],
        reality=Reality(
            summary="Full reality",
            domains={"career": "engineer"},
            constraints=["time"],
            resources=["skills"],
        ),
        secrets=["s1", "s2", "s3"],
        intentions=[
            Intention(description="i1", domain="career", strength=0.8),
            Intention(description="i2", domain="health", strength=0.6),
            Intention(description="i3", domain="creative", strength=0.5),
        ],
        gaps=[
            Gap(intention="i1", reality="r1", bridge_question="q?", priority=0.9),
            Gap(intention="i2", reality="r2", bridge_question="q?", priority=0.7),
        ],
    )
    score = compute_soul_coverage(soul)
    assert score == 1.0


def test_partial_soul_coverage():
    soul = Soul(
        id="test",
        character=_make_profile(),
        facts=[
            Fact(category="c", content=f"fact_{i}", confidence=0.9, source_turn=i)
            for i in range(5)  # 5/10 = 0.5
        ],
        reality=Reality(
            summary="Partial",
            domains={},
            constraints=[],
            resources=[],
        ),
        # reality populated = 1.0
        secrets=["s1"],  # 1/3 = 0.333
    )
    score = compute_soul_coverage(soul)
    # (0.5 + 1.0 + 0.333 + 0.0 + 0.0) / 5 = 0.367
    assert abs(score - 0.367) < 0.01


def test_coverage_facts_cap():
    """More than 10 facts still gives 1.0 for facts component."""
    soul = Soul(
        id="test",
        character=_make_profile(),
        facts=[
            Fact(category="c", content=f"fact_{i}", confidence=0.9, source_turn=i)
            for i in range(20)
        ],
    )
    # 20 facts → 1.0, reality=0, secrets=0, intentions=0, gaps=0
    score = compute_soul_coverage(soul)
    assert abs(score - 1.0 / 5) < 0.01


def test_v25_coverage_with_intentions_and_gaps():
    from super_brain.models import Intention, Gap
    soul = Soul(
        id="test",
        character=_make_profile(),
        facts=[Fact(category="c", content=f"f{i}", confidence=0.9, source_turn=i) for i in range(10)],
        reality=Reality(summary="Full", domains={}, constraints=[], resources=[]),
        secrets=["s1", "s2", "s3"],
        intentions=[
            Intention(description="start biz", domain="career", strength=0.8),
            Intention(description="travel", domain="personal_growth", strength=0.6),
            Intention(description="learn guitar", domain="creative", strength=0.5),
        ],
        gaps=[
            Gap(intention="start biz", reality="employed", bridge_question="q?", priority=0.9),
            Gap(intention="travel", reality="no time", bridge_question="q?", priority=0.7),
        ],
    )
    score = compute_soul_coverage(soul)
    assert score == 1.0  # all 5 components maxed


def test_v25_coverage_partial_intentions():
    from super_brain.models import Intention
    soul = Soul(
        id="test",
        character=_make_profile(),
        facts=[Fact(category="c", content=f"f{i}", confidence=0.9, source_turn=i) for i in range(10)],
        reality=Reality(summary="Full", domains={}, constraints=[], resources=[]),
        secrets=["s1", "s2", "s3"],
        intentions=[
            Intention(description="start biz", domain="career", strength=0.8),
        ],
        # 1 intention / 3 = 0.333, 0 gaps / 2 = 0.0
    )
    score = compute_soul_coverage(soul)
    # (1.0 + 1.0 + 1.0 + 0.333 + 0.0) / 5 = 0.667
    assert abs(score - 0.667) < 0.01
