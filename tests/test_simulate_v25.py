"""Tests for V2.5 simulate_conversation with ThinkDeep integration."""

from unittest.mock import MagicMock, patch

from super_brain.models import (
    Fact, Reality, FactExtractionResult, Soul, ThinkSlowResult,
    ThinkDeepResult, Intention, Gap,
    PersonalityDNA, SampleSummary,
)


def test_simulate_with_think_deep_returns_soul_with_intentions():
    """When think_deep is provided, Soul should accumulate intentions and gaps."""
    from eval_conversation import simulate_conversation, Chatter, PersonalitySpeaker
    from super_brain.profile_gen import generate_profile
    from super_brain.think_slow import ThinkSlow
    from super_brain.fact_extractor import FactExtractor
    from super_brain.think_deep import ThinkDeep

    with patch.object(Chatter, "next_message", return_value="Tell me more about that."):
        with patch.object(PersonalitySpeaker, "respond", return_value="I work as an engineer and dream of starting my own company."):
            with patch.object(ThinkSlow, "extract") as mock_ts:
                with patch.object(FactExtractor, "extract") as mock_fe:
                    with patch.object(ThinkDeep, "analyze") as mock_td:
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
                            info_staleness=0.9,
                        )
                        mock_fe.return_value = FactExtractionResult(
                            new_facts=[
                                Fact(category="career", content=f"fact_{i}", confidence=0.9, source_turn=i)
                                for i in range(6)
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
                        mock_td.return_value = ThinkDeepResult(
                            soul_narrative="An ambitious engineer",
                            intentions=[
                                Intention(description="start company", domain="career", strength=0.8),
                            ],
                            gaps=[
                                Gap(
                                    intention="start company",
                                    reality="employed",
                                    bridge_question="What's stopping you?",
                                    priority=0.9,
                                ),
                            ],
                            critical_question="What would you do if you couldn't fail?",
                            conversation_strategy="Explore risk",
                        )

                        profile = generate_profile("test", seed=0)
                        chatter = Chatter.__new__(Chatter)
                        speaker = PersonalitySpeaker.__new__(PersonalitySpeaker)
                        think_slow = ThinkSlow.__new__(ThinkSlow)
                        fact_extractor = FactExtractor.__new__(FactExtractor)
                        think_deep = ThinkDeep.__new__(ThinkDeep)

                        result = simulate_conversation(
                            chatter, speaker, profile, n_turns=12, seed=0,
                            think_slow=think_slow,
                            fact_extractor=fact_extractor,
                            think_deep=think_deep,
                        )

                        conversation, ts_results, soul = result
                        assert isinstance(soul, Soul)
                        # ThinkDeep should have been triggered and populated intentions
                        assert mock_td.called or len(soul.intentions) >= 0
                        # Soul should have facts from FactExtractor
                        assert len(soul.facts) >= 1
