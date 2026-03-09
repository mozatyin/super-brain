"""Tests for simulate_conversation with FactExtractor (V2.4+)."""

from unittest.mock import patch

from super_brain.models import (
    Fact, Reality, FactExtractionResult, Soul, ThinkSlowResult,
    PersonalityDNA, SampleSummary,
)


def test_simulate_with_fact_extractor_returns_soul_with_facts():
    """When fact_extractor is provided, Soul should accumulate facts."""
    from eval_conversation import simulate_conversation, Chatter, PersonalitySpeaker
    from super_brain.profile_gen import generate_profile
    from super_brain.think_slow import ThinkSlow
    from super_brain.fact_extractor import FactExtractor

    with patch.object(Chatter, "next_message", return_value="Tell me more about that."):
        with patch.object(PersonalitySpeaker, "respond", return_value="I work as an engineer."):
            with patch.object(ThinkSlow, "extract") as mock_ts:
                with patch.object(FactExtractor, "extract") as mock_fe:
                    mock_profile = PersonalityDNA(
                        id="partial",
                        sample_summary=SampleSummary(
                            total_tokens=0, conversation_count=0,
                            date_range=["u", "u"], contexts=["t"],
                            confidence_overall=0.5,
                        ),
                    )
                    mock_ts.return_value = ThinkSlowResult(
                        partial_profile=mock_profile,
                        confidence_map={},
                        low_confidence_traits=[],
                        observations=[],
                        incisive_questions=[],
                        info_staleness=0.0,
                    )
                    mock_fe.return_value = FactExtractionResult(
                        new_facts=[
                            Fact(category="career", content="engineer", confidence=0.9, source_turn=1),
                        ],
                        reality=Reality(
                            summary="An engineer",
                            domains={"career": "engineer"},
                            constraints=[],
                            resources=[],
                        ),
                        secrets=[],
                        contradictions=[],
                    )

                    profile = generate_profile("test", seed=0)
                    chatter = Chatter.__new__(Chatter)
                    speaker = PersonalitySpeaker.__new__(PersonalitySpeaker)
                    think_slow = ThinkSlow.__new__(ThinkSlow)
                    fact_extractor = FactExtractor.__new__(FactExtractor)

                    result = simulate_conversation(
                        chatter, speaker, profile, n_turns=12, seed=0,
                        think_slow=think_slow,
                        fact_extractor=fact_extractor,
                    )

                    conversation, ts_results, soul = result
                    assert isinstance(soul, Soul)
                    assert len(soul.facts) >= 1
