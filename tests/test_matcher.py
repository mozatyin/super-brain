"""Tests for the matcher module."""

from super_brain.matcher import DepthLevel, MatcherResponse


def test_depth_level_ordering():
    assert DepthLevel.PLEASANTRY < DepthLevel.FACTUAL
    assert DepthLevel.FACTUAL < DepthLevel.OPINION
    assert DepthLevel.OPINION < DepthLevel.EMOTIONAL
    assert DepthLevel.EMOTIONAL < DepthLevel.BELIEF
    assert DepthLevel.BELIEF < DepthLevel.INSIGHT


def test_depth_level_values():
    assert DepthLevel.PLEASANTRY == 0
    assert DepthLevel.INSIGHT == 5


def test_matcher_response_creation():
    r = MatcherResponse(
        response_text="Tell me more about that.",
        assessed_depth=DepthLevel.OPINION,
        target_depth=DepthLevel.EMOTIONAL,
        strategy_used="SPECIFIC_QUESTION",
    )
    assert r.assessed_depth == 2
    assert r.target_depth == 3
    assert "Tell me more" in r.response_text
