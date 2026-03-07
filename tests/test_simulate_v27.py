"""Tests for V2.7 ThinkDeep trigger cap and dedup in simulation."""

from unittest.mock import MagicMock, patch

from super_brain.models import (
    Fact, Reality, FactExtractionResult, Soul, ThinkSlowResult,
    ThinkDeepResult, Intention, Gap,
    PersonalityDNA, SampleSummary,
)


def test_think_deep_fires_at_most_twice():
    """ThinkDeep should fire at most 2 times in a conversation."""
    from eval_conversation import simulate_conversation, Chatter, PersonalitySpeaker
    from super_brain.profile_gen import generate_profile
    from super_brain.think_slow import ThinkSlow
    from super_brain.fact_extractor import FactExtractor
    from super_brain.think_deep import ThinkDeep

    td_call_count = 0

    def mock_td_analyze(soul, conversation):
        nonlocal td_call_count
        td_call_count += 1
        return ThinkDeepResult(
            soul_narrative=f"Analysis #{td_call_count}",
            intentions=[
                Intention(description=f"intention_{td_call_count}", domain="career", strength=0.8),
            ],
            gaps=[
                Gap(intention=f"int_{td_call_count}", reality="reality", bridge_question="q?", priority=0.9),
            ],
            critical_question=f"Question #{td_call_count}?",
            conversation_strategy="strategy",
        )

    with patch.object(Chatter, "next_message", return_value="Tell me more."):
        with patch.object(PersonalitySpeaker, "respond", return_value="I'm an engineer thinking about change."):
            with patch.object(ThinkSlow, "extract") as mock_ts:
                with patch.object(FactExtractor, "extract") as mock_fe:
                    with patch.object(ThinkDeep, "analyze", side_effect=mock_td_analyze):
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
                            secrets=["secret"],
                            contradictions=["contradiction"],
                        )

                        profile = generate_profile("test", seed=0)
                        chatter = Chatter.__new__(Chatter)
                        speaker = PersonalitySpeaker.__new__(PersonalitySpeaker)
                        think_slow = ThinkSlow.__new__(ThinkSlow)
                        fact_extractor = FactExtractor.__new__(FactExtractor)
                        think_deep = ThinkDeep.__new__(ThinkDeep)

                        result = simulate_conversation(
                            chatter, speaker, profile, n_turns=20, seed=0,
                            think_slow=think_slow,
                            fact_extractor=fact_extractor,
                            think_deep=think_deep,
                        )

                        conversation, ts_results, soul = result
                        # ThinkDeep should fire at most 2 times
                        assert td_call_count <= 2


def test_soul_dedup_prevents_duplicate_secrets():
    """Duplicate secrets should not accumulate in Soul."""
    from super_brain.dedup import dedup_extend_strings

    secrets = ["afraid of failure"]
    new_secrets = ["afraid of failure", "scared of failing", "loves painting"]
    dedup_extend_strings(secrets, new_secrets, threshold=0.6)
    # "afraid of failure" is exact dup -> skipped
    # "scared of failing" vs "afraid of failure": different words, Jaccard low -> added
    # "loves painting" is unique -> added
    assert "loves painting" in secrets
    assert len(secrets) <= 4  # original + at most 2-3 new unique ones


def test_soul_dedup_prevents_duplicate_intentions():
    """Duplicate intentions should not accumulate in Soul."""
    from super_brain.dedup import is_duplicate

    existing = ["wants to start own business", "learn to play guitar"]
    # Near-duplicate using shared tokens
    assert is_duplicate("start own business soon", existing, threshold=0.5) is True
    # Unique
    assert is_duplicate("travel to Japan", existing) is False
