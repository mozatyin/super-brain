"""Tests for V2.1 Think Slow periodic extraction."""

from unittest.mock import MagicMock, patch
import json

from super_brain.models import ThinkSlowResult, PersonalityDNA, SampleSummary


def test_think_slow_result_creation():
    """ThinkSlowResult should hold partial profile, confidence map, and focus list."""
    profile = PersonalityDNA(
        id="test",
        sample_summary=SampleSummary(
            total_tokens=0, conversation_count=0,
            date_range=["unknown", "unknown"],
            contexts=["test"], confidence_overall=0.5,
        ),
    )
    result = ThinkSlowResult(
        partial_profile=profile,
        confidence_map={"anxiety": 0.8, "trust": 0.3},
        low_confidence_traits=["trust"],
        observations=["Speaker avoids personal topics"],
    )
    assert result.low_confidence_traits == ["trust"]
    assert result.confidence_map["anxiety"] == 0.8
    assert len(result.observations) == 1


def test_think_slow_result_defaults():
    """ThinkSlowResult should have sensible defaults."""
    profile = PersonalityDNA(
        id="test",
        sample_summary=SampleSummary(
            total_tokens=0, conversation_count=0,
            date_range=["unknown", "unknown"],
            contexts=["test"], confidence_overall=0.5,
        ),
    )
    result = ThinkSlowResult(
        partial_profile=profile,
        confidence_map={},
        low_confidence_traits=[],
        observations=[],
    )
    assert result.low_confidence_traits == []
    assert result.confidence_map == {}


def test_think_slow_extract_returns_result():
    """ThinkSlow.extract() should return a ThinkSlowResult from conversation."""
    mock_response_data = {
        "observations": [
            "Speaker uses short, direct sentences — possible low gregariousness",
            "Avoids emotional topics — possible low feelings openness",
        ],
        "trait_estimates": [
            {"dimension": "EXT", "name": "gregariousness", "value": 0.30, "confidence": 0.6},
            {"dimension": "OPN", "name": "feelings", "value": 0.35, "confidence": 0.4},
            {"dimension": "NEU", "name": "anxiety", "value": 0.50, "confidence": 0.3},
        ],
    }

    with patch("super_brain.think_slow.anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=json.dumps(mock_response_data))]
        mock_client.messages.create.return_value = mock_resp

        from super_brain.think_slow import ThinkSlow
        ts = ThinkSlow(api_key="test-key")

        conversation = [
            {"role": "chatter", "text": "How's your day going?"},
            {"role": "speaker", "text": "Fine."},
            {"role": "chatter", "text": "Do anything fun?"},
            {"role": "speaker", "text": "Not really. Just work."},
        ]

        result = ts.extract(conversation, focus_traits=None, previous=None)

        assert result.partial_profile is not None
        assert len(result.partial_profile.traits) == 3
        assert "anxiety" in result.confidence_map
        assert "anxiety" in result.low_confidence_traits
        assert "feelings" in result.low_confidence_traits
        assert "gregariousness" not in result.low_confidence_traits


def test_think_slow_extract_with_focus_traits():
    """When focus_traits are provided, they should appear in the LLM prompt."""
    with patch("super_brain.think_slow.anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=json.dumps({
            "observations": [],
            "trait_estimates": [
                {"dimension": "SOC", "name": "social_dominance", "value": 0.50, "confidence": 0.5},
            ],
        }))]
        mock_client.messages.create.return_value = mock_resp

        from super_brain.think_slow import ThinkSlow
        ts = ThinkSlow(api_key="test-key")

        result = ts.extract(
            conversation=[{"role": "chatter", "text": "Hi"}, {"role": "speaker", "text": "Hello"}],
            focus_traits=["social_dominance", "humor_self_enhancing"],
            previous=None,
        )

        call_args = mock_client.messages.create.call_args
        user_msg = call_args[1]["messages"][0]["content"]
        assert "social_dominance" in user_msg
        assert "humor_self_enhancing" in user_msg
