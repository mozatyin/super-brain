"""Tests for V2.5 ThinkDeep — strategic intention/gap analysis."""

import json
from unittest.mock import MagicMock, patch

from super_brain.think_deep import ThinkDeep, _parse_think_deep_response, _build_soul_context
from super_brain.models import (
    Soul, Fact, Reality, ThinkDeepResult, Intention, Gap,
    PersonalityDNA, SampleSummary, Trait,
)


def _make_soul(facts=None, reality=None, secrets=None, contradictions=None):
    profile = PersonalityDNA(
        id="test",
        sample_summary=SampleSummary(
            total_tokens=0, conversation_count=0,
            date_range=["u", "u"], contexts=["t"], confidence_overall=0.5,
        ),
        traits=[
            Trait(dimension="EXT", name="assertiveness", value=0.7, confidence=0.6),
            Trait(dimension="AGR", name="trust", value=0.3, confidence=0.5),
        ],
    )
    return Soul(
        id="test_soul",
        character=profile,
        facts=facts or [],
        reality=reality,
        secrets=secrets or [],
        contradictions=contradictions or [],
    )


def test_parse_think_deep_response_valid():
    raw = json.dumps({
        "soul_narrative": "A person at a crossroads",
        "intentions": [
            {"description": "start a business", "domain": "career", "strength": 0.8, "blockers": ["money"]},
        ],
        "gaps": [
            {
                "intention": "start a business",
                "reality": "employed full-time",
                "bridge_question": "What's stopping you?",
                "priority": 0.9,
            },
        ],
        "critical_question": "What would you do if money weren't a concern?",
        "conversation_strategy": "Explore risk tolerance",
    })
    data = _parse_think_deep_response(raw)
    assert data["soul_narrative"] == "A person at a crossroads"
    assert len(data["intentions"]) == 1
    assert len(data["gaps"]) == 1
    assert data["critical_question"] != ""


def test_parse_think_deep_response_code_block():
    raw = "```json\n" + json.dumps({
        "soul_narrative": "narrative",
        "intentions": [],
        "gaps": [],
        "critical_question": "question",
        "conversation_strategy": "strategy",
    }) + "\n```"
    data = _parse_think_deep_response(raw)
    assert data["soul_narrative"] == "narrative"


def test_parse_think_deep_response_invalid():
    data = _parse_think_deep_response("not json at all")
    assert data["soul_narrative"] == ""
    assert data["intentions"] == []
    assert data["gaps"] == []
    assert data["critical_question"] == ""
    assert data["conversation_strategy"] == ""


def test_build_soul_context_includes_facts_and_reality():
    soul = _make_soul(
        facts=[
            Fact(category="career", content="software engineer", confidence=0.9, source_turn=3),
            Fact(category="hobby", content="plays guitar", confidence=0.7, source_turn=5),
        ],
        reality=Reality(
            summary="A software engineer who plays guitar",
            domains={"career": "software engineer"},
            constraints=["limited time"],
            resources=["technical skills"],
        ),
        secrets=["avoids discussing family"],
        contradictions=["said values independence but hates making decisions"],
    )
    context = _build_soul_context(soul)
    assert "software engineer" in context
    assert "guitar" in context
    assert "avoids discussing family" in context
    assert "independence" in context
    assert "assertiveness" in context  # from character traits


def test_build_soul_context_minimal_soul():
    soul = _make_soul()
    context = _build_soul_context(soul)
    assert "assertiveness" in context  # character traits always included
    assert "No facts" in context or "facts" in context.lower()


def test_think_deep_analyze_with_mock():
    """Test ThinkDeep.analyze() with a mocked LLM response."""
    with patch("super_brain.think_deep.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "soul_narrative": "An ambitious engineer constrained by financial fears",
            "intentions": [
                {"description": "start own company", "domain": "career", "strength": 0.85, "blockers": ["savings"]},
                {"description": "travel more", "domain": "personal_growth", "strength": 0.6, "blockers": ["time"]},
            ],
            "gaps": [
                {
                    "intention": "start own company",
                    "reality": "employed with mortgage",
                    "bridge_question": "What level of financial security would you need to make the jump?",
                    "priority": 0.9,
                },
            ],
            "critical_question": "If you knew you couldn't fail, what would you do differently tomorrow?",
            "conversation_strategy": "Explore the tension between security and ambition",
        }))]
        mock_client.messages.create.return_value = mock_response

        think_deep = ThinkDeep(api_key="test-key")
        soul = _make_soul(
            facts=[
                Fact(category="career", content="software engineer", confidence=0.9, source_turn=3),
            ],
            reality=Reality(
                summary="Employed engineer with mortgage",
                domains={"career": "software engineer"},
                constraints=["mortgage"],
                resources=["technical skills"],
            ),
        )
        conversation = [
            {"role": "chatter", "text": "What's on your mind lately?"},
            {"role": "speaker", "text": "I've been thinking about starting my own company."},
        ]
        result = think_deep.analyze(soul=soul, conversation=conversation)

        assert isinstance(result, ThinkDeepResult)
        assert len(result.intentions) == 2
        assert result.intentions[0].description == "start own company"
        assert result.intentions[0].strength == 0.85
        assert len(result.gaps) == 1
        assert "fail" in result.critical_question
        assert result.conversation_strategy != ""

        mock_client.messages.create.assert_called_once()
