"""Tests for Soul-Aware Diagnostic Questions (V3.0)."""

import json
import pytest

from super_brain.diagnostic_questions import (
    _build_soul_context,
    _build_target_section,
    _parse_diagnostic_response,
    generate_diagnostic_questions,
)
from super_brain.models import IncisiveQuestion


class TestBuildSoulContext:
    def test_empty_context(self):
        result = _build_soul_context(confidence_map={})
        assert result == "No prior context available."

    def test_high_confidence_traits_with_values(self):
        conf = {"warmth": 0.7, "anxiety": 0.6, "trust": 0.3}
        vals = {"warmth": 0.8, "anxiety": 0.2, "trust": 0.5}
        result = _build_soul_context(confidence_map=conf, trait_values=vals)
        assert "What We Already Know" in result
        assert "warmth" in result
        assert "high" in result  # warmth=0.8 → "high"
        assert "low" in result  # anxiety=0.2 → "low"
        # trust confidence < 0.5, should NOT appear
        assert "trust" not in result

    def test_high_confidence_without_values(self):
        conf = {"warmth": 0.7}
        result = _build_soul_context(confidence_map=conf)
        assert "warmth" in result
        assert "confidence=0.70" in result

    def test_known_facts(self):
        result = _build_soul_context(
            confidence_map={},
            known_facts=["software engineer", "lives in NYC"],
        )
        assert "Known Facts" in result
        assert "software engineer" in result
        assert "NYC" in result

    def test_reality_summary(self):
        result = _build_soul_context(
            confidence_map={},
            reality_summary="Mid-career professional exploring new opportunities",
        )
        assert "Life Situation" in result
        assert "Mid-career" in result

    def test_conversation_context(self):
        conv = [
            {"role": "chatter", "text": "How are you?"},
            {"role": "speaker", "text": "I'm doing great, thanks!"},
        ]
        result = _build_soul_context(confidence_map={}, conversation=conv)
        assert "Recent Conversation" in result
        assert "Person A" in result
        assert "Person B (target)" in result
        assert "I'm doing great" in result

    def test_conversation_truncated_to_last_8(self):
        conv = [{"role": "speaker", "text": f"Turn {i}"} for i in range(20)]
        result = _build_soul_context(confidence_map={}, conversation=conv)
        assert "Turn 12" in result
        assert "Turn 19" in result
        assert "Turn 0" not in result

    def test_combined_context(self):
        result = _build_soul_context(
            confidence_map={"warmth": 0.8},
            trait_values={"warmth": 0.7},
            known_facts=["teacher"],
            reality_summary="Enjoys mentoring",
            conversation=[{"role": "speaker", "text": "Hello"}],
        )
        assert "What We Already Know" in result
        assert "Known Facts" in result
        assert "Life Situation" in result
        assert "Recent Conversation" in result


class TestBuildTargetSection:
    def test_basic_targets(self):
        result = _build_target_section(
            low_confidence_traits=["narcissism", "warmth"],
            confidence_map={"narcissism": 0.1, "warmth": 0.3},
        )
        assert "Target Traits" in result
        assert "narcissism" in result
        assert "warmth" in result
        # narcissism has lower confidence, should appear first
        narc_pos = result.find("narcissism")
        warmth_pos = result.find("warmth")
        assert narc_pos < warmth_pos

    def test_sorts_by_confidence(self):
        result = _build_target_section(
            low_confidence_traits=["warmth", "anxiety", "trust"],
            confidence_map={"warmth": 0.4, "anxiety": 0.1, "trust": 0.2},
        )
        # anxiety (0.1) < trust (0.2) < warmth (0.4)
        anx_pos = result.find("anxiety")
        trust_pos = result.find("trust")
        warmth_pos = result.find("warmth")
        assert anx_pos < trust_pos < warmth_pos

    def test_includes_detection_hints(self):
        result = _build_target_section(
            low_confidence_traits=["narcissism"],
            confidence_map={"narcissism": 0.1},
        )
        assert "Detection hint" in result

    def test_max_targets_limit(self):
        traits = [f"trait_{i}" for i in range(20)]
        conf = {t: 0.1 for t in traits}
        result = _build_target_section(traits, conf, max_targets=3)
        # Only first 3 should appear (trait_0, trait_1, trait_2 after sorting)
        assert result.count("confidence=") == 3

    def test_unknown_trait(self):
        result = _build_target_section(
            low_confidence_traits=["completely_made_up_trait"],
            confidence_map={"completely_made_up_trait": 0.1},
        )
        assert "completely_made_up_trait" in result
        assert "confidence=0.10" in result


class TestParseDiagnosticResponse:
    def test_valid_json_array(self):
        raw = json.dumps([
            {
                "question": "How do you handle stress?",
                "target_traits": ["anxiety"],
                "question_type": "attribution",
                "rationale": "reveals stress response",
            }
        ])
        result = _parse_diagnostic_response(raw)
        assert len(result) == 1
        assert result[0]["question"] == "How do you handle stress?"

    def test_markdown_code_block(self):
        raw = '```json\n[{"question": "Test?", "target_traits": ["warmth"]}]\n```'
        result = _parse_diagnostic_response(raw)
        assert len(result) == 1
        assert result[0]["question"] == "Test?"

    def test_wrapped_in_object(self):
        raw = json.dumps({
            "questions": [
                {"question": "Q1?", "target_traits": ["trust"]},
                {"question": "Q2?", "target_traits": ["warmth"]},
            ]
        })
        result = _parse_diagnostic_response(raw)
        assert len(result) == 2

    def test_json_with_surrounding_text(self):
        raw = 'Here are the questions:\n[{"question": "Test?", "target_traits": ["warmth"]}]\nDone.'
        result = _parse_diagnostic_response(raw)
        assert len(result) == 1

    def test_invalid_json_returns_empty(self):
        result = _parse_diagnostic_response("This is not JSON at all")
        assert result == []

    def test_empty_string(self):
        result = _parse_diagnostic_response("")
        assert result == []

    def test_multiple_questions(self):
        raw = json.dumps([
            {"question": f"Q{i}?", "target_traits": [f"trait_{i}"]}
            for i in range(5)
        ])
        result = _parse_diagnostic_response(raw)
        assert len(result) == 5


class TestGenerateDiagnosticQuestions:
    def test_empty_traits_returns_empty(self):
        result = generate_diagnostic_questions(
            low_confidence_traits=[],
            confidence_map={},
            api_key="test-key",
        )
        assert result == []

    def test_no_api_key_returns_empty(self):
        result = generate_diagnostic_questions(
            low_confidence_traits=["warmth"],
            confidence_map={"warmth": 0.1},
            api_key="",
        )
        assert result == []
