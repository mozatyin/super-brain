"""Tests for V2.0 Deep Listening Chatter."""

from eval_conversation import _build_chatter_system


def test_deep_listening_prompt_contains_10_components():
    """V2.0 chatter system prompt should reference Deep Listening principles."""
    prompt = _build_chatter_system(turn_number=1, total_turns=20)
    assert "attention" in prompt.lower() or "presence" in prompt.lower()
    assert "ease" in prompt.lower() or "no rush" in prompt.lower()
    assert "appreciation" in prompt.lower()


def test_deep_listening_phase_split():
    """Turns 1-14 = Deep Listening, Turns 15-20 = Incisive Questions."""
    early = _build_chatter_system(turn_number=3, total_turns=20)
    late = _build_chatter_system(turn_number=16, total_turns=20)
    assert "incisive" not in early.lower()
    assert "incisive" in late.lower() or "targeted" in late.lower()


def test_chatter_prompt_short_response_instruction():
    """Chatter should produce 1-2 sentence responses to maximize speaker output."""
    prompt = _build_chatter_system(turn_number=5, total_turns=20)
    assert "1-2 sentence" in prompt.lower() or "short" in prompt.lower()


def test_chatter_no_personality_probing():
    """Deep Listening should never directly probe personality."""
    for turn in [1, 5, 10, 15, 20]:
        prompt = _build_chatter_system(turn_number=turn, total_turns=20)
        assert "what kind of person" not in prompt.lower()
        assert "describe yourself" not in prompt.lower()
