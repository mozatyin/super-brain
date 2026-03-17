"""Tests for behavioral feature extraction and trait adjustments."""

import pytest
from super_brain.behavioral_features import (
    BehavioralFeatures,
    extract_features,
    compute_adjustments,
    apply_adjustments,
    compute_direct_scores,
    RULE_BASED_TRAITS,
)
from super_brain.models import PersonalityDNA, Trait, SampleSummary


def _make_conversation(speaker_texts: list[str], chatter_texts: list[str] | None = None):
    """Build a conversation list from speaker texts."""
    conv = []
    if chatter_texts is None:
        chatter_texts = ["Hey, tell me more."] * len(speaker_texts)
    for c, s in zip(chatter_texts, speaker_texts):
        conv.append({"role": "chatter", "text": c})
        conv.append({"role": "speaker", "text": s})
    return conv


def _make_profile(trait_values: dict[str, float]) -> PersonalityDNA:
    """Build a minimal PersonalityDNA from trait name→value dict."""
    traits = [
        Trait(dimension="TEST", name=name, value=val, confidence=0.7)
        for name, val in trait_values.items()
    ]
    return PersonalityDNA(
        id="test",
        sample_summary=SampleSummary(
            total_tokens=100, conversation_count=1,
            date_range=["2024-01-01"], contexts=["test"],
            confidence_overall=0.7,
        ),
        traits=traits,
    )


class TestExtractFeatures:
    def test_empty_conversation(self):
        result = extract_features([], speaker_role="speaker")
        assert result.turn_count == 0
        assert result.total_words == 0

    def test_basic_word_count(self):
        conv = _make_conversation(["Hello world", "This is a test"])
        result = extract_features(conv)
        assert result.turn_count == 2
        assert result.total_words == 6
        assert result.avg_words_per_turn == 3.0

    def test_self_reference_ratio(self):
        conv = _make_conversation([
            "I think I should go. My plan is to leave myself alone."
        ])
        result = extract_features(conv)
        # words: i, think, i, should, go, my, plan, is, to, leave, myself, alone = 12
        # self refs: i, i, my, myself = 4
        assert result.self_ref_ratio == pytest.approx(4 / 12, abs=0.01)

    def test_other_reference_ratio(self):
        conv = _make_conversation([
            "You should tell your friend that they need our help."
        ])
        result = extract_features(conv)
        # you, your, they, our = 4 other-refs
        total = result.total_words
        assert result.other_ref_ratio == pytest.approx(4 / total, abs=0.01)

    def test_hedging_words(self):
        conv = _make_conversation([
            "Maybe I should probably go. I think it could be fine."
        ])
        result = extract_features(conv)
        # hedging words: maybe, probably
        # hedging phrases: "i think", "could be"
        assert result.hedging_ratio > 0.1

    def test_absolutist_words(self):
        conv = _make_conversation([
            "I always do that. Everyone definitely knows. It's absolutely never wrong."
        ])
        result = extract_features(conv)
        # always, everyone, definitely, absolutely, never = 5
        assert result.absolutist_ratio > 0.1

    def test_question_ratio(self):
        conv = _make_conversation([
            "Really? Is that true? I had no idea."
        ])
        result = extract_features(conv)
        # 3 sentences, 2 questions
        assert result.question_ratio == pytest.approx(2 / 3, abs=0.1)

    def test_exclamation_ratio(self):
        conv = _make_conversation([
            "Wow! That's amazing! I can't believe it."
        ])
        result = extract_features(conv)
        # 3 sentences, 2 exclamations
        assert result.exclamation_ratio == pytest.approx(2 / 3, abs=0.1)

    def test_emotion_words(self):
        conv = _make_conversation([
            "I love this amazing day! But I hate the terrible traffic."
        ])
        result = extract_features(conv)
        # pos: love, amazing = 2; neg: hate, terrible = 2
        assert result.pos_emotion_ratio > 0
        assert result.neg_emotion_ratio > 0

    def test_only_speaker_turns(self):
        conv = [
            {"role": "chatter", "text": "I always definitely think about myself."},
            {"role": "speaker", "text": "Hello."},
        ]
        result = extract_features(conv)
        assert result.total_words == 1
        assert result.self_ref_ratio == 0.0

    def test_words_std(self):
        conv = _make_conversation(["Hello", "This is a much longer sentence with many words"])
        result = extract_features(conv)
        assert result.words_std > 0


class TestComputeAdjustments:
    def test_high_self_ref_increases_narcissism(self):
        features = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.10, other_ref_ratio=0.02, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        adj = compute_adjustments(features)
        assert "narcissism" in adj
        assert adj["narcissism"] > 0

    def test_low_self_ref_decreases_narcissism(self):
        features = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.02, other_ref_ratio=0.05, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        adj = compute_adjustments(features)
        assert adj.get("narcissism", 0) < 0

    def test_high_hedging_reduces_assertiveness(self):
        features = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.03,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        adj = compute_adjustments(features)
        assert adj.get("assertiveness", 0) < 0
        assert adj.get("modesty", 0) > 0

    def test_no_rules_fire_for_moderate_values(self):
        features = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=120, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.010,
            absolutist_ratio=0.005, question_ratio=0.15, exclamation_ratio=0.10,
            pos_emotion_ratio=0.010, neg_emotion_ratio=0.008,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        adj = compute_adjustments(features)
        # Most rules shouldn't fire for moderate values
        assert len(adj) <= 3  # at most a few edge cases

    def test_multiple_rules_stack(self):
        features = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=250, words_std=10,
            self_ref_ratio=0.10, other_ref_ratio=0.01, hedging_ratio=0.001,
            absolutist_ratio=0.020, question_ratio=0.3, exclamation_ratio=0.4,
            pos_emotion_ratio=0.03, neg_emotion_ratio=0.02,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        adj = compute_adjustments(features)
        # assertiveness should get boost from both low hedging + high absolutist
        assert adj.get("assertiveness", 0) > 0.10


class TestApplyAdjustments:
    def test_no_adjustments_returns_same(self):
        profile = _make_profile({"narcissism": 0.5, "assertiveness": 0.5})
        result = apply_adjustments(profile, {})
        assert result.traits[0].value == 0.5

    def test_positive_adjustment(self):
        profile = _make_profile({"narcissism": 0.5})
        result = apply_adjustments(profile, {"narcissism": 0.06})
        assert result.traits[0].value == pytest.approx(0.56, abs=0.01)

    def test_negative_adjustment(self):
        profile = _make_profile({"assertiveness": 0.4})
        result = apply_adjustments(profile, {"assertiveness": -0.07})
        assert result.traits[0].value == pytest.approx(0.33, abs=0.01)

    def test_clamp_to_bounds(self):
        profile = _make_profile({"trait_a": 0.95, "trait_b": 0.05})
        result = apply_adjustments(profile, {"trait_a": 0.10, "trait_b": -0.10})
        vals = {t.name: t.value for t in result.traits}
        assert vals["trait_a"] == 1.0
        assert vals["trait_b"] == 0.0

    def test_unadjusted_traits_unchanged(self):
        profile = _make_profile({"narcissism": 0.5, "warmth": 0.7})
        result = apply_adjustments(profile, {"narcissism": 0.06})
        vals = {t.name: t.value for t in result.traits}
        assert vals["warmth"] == 0.7


class TestNewBehavioralFeatures:
    """Tests for V3.2 politeness, curiosity, and decisiveness features."""

    def test_politeness_ratio(self):
        conv = _make_conversation([
            "Please help me. Thank you so much, I really appreciate it."
        ])
        result = extract_features(conv)
        # polite words: please, thank, appreciate = 3
        # polite phrases: "thank you" = 1
        assert result.politeness_ratio > 0.1

    def test_curiosity_ratio(self):
        conv = _make_conversation([
            "I wonder how does that work? Why is it like that? Tell me more."
        ])
        result = extract_features(conv)
        # curiosity phrases: "i wonder", "how does", "why is", "tell me more"
        assert result.curiosity_ratio > 0.1

    def test_decisiveness_ratio(self):
        conv = _make_conversation([
            "I've decided to go. I will absolutely do it, definitely."
        ])
        result = extract_features(conv)
        # decisive words: decided, absolutely, definitely = 3
        # decisive phrases: "i've decided", "i will" = 2
        assert result.decisiveness_ratio > 0.1

    def test_new_features_zero_for_empty(self):
        result = extract_features([], speaker_role="speaker")
        assert result.politeness_ratio == 0
        assert result.curiosity_ratio == 0
        assert result.decisiveness_ratio == 0

    def test_new_features_present_in_extraction(self):
        conv = _make_conversation(["Hello world"])
        result = extract_features(conv)
        assert hasattr(result, "politeness_ratio")
        assert hasattr(result, "curiosity_ratio")
        assert hasattr(result, "decisiveness_ratio")


class TestV32AdjustmentRules:
    """Tests for the 15 new V3.2 adjustment rules."""

    def test_competence_long_turns(self):
        features = BehavioralFeatures(
            turn_count=10, total_words=2000, avg_words_per_turn=200, words_std=20,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.015, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        adj = compute_adjustments(features)
        assert adj.get("competence", 0) > 0  # long turns + high absolutist

    def test_competence_reduced_by_hedging(self):
        features = BehavioralFeatures(
            turn_count=10, total_words=2000, avg_words_per_turn=200, words_std=20,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.03,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        adj = compute_adjustments(features)
        # competence gets +0.04 from long turns but -0.05 from hedging
        assert adj.get("competence", 0) < 0

    def test_social_dominance_questions_reduce(self):
        features = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.35, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        adj = compute_adjustments(features)
        assert adj.get("social_dominance", 0) < 0

    def test_social_dominance_long_self_ref(self):
        features = BehavioralFeatures(
            turn_count=10, total_words=2000, avg_words_per_turn=200, words_std=20,
            self_ref_ratio=0.09, other_ref_ratio=0.02, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        adj = compute_adjustments(features)
        assert adj.get("social_dominance", 0) > 0

    def test_self_consciousness_hedging_removed(self):
        """Self_consciousness hedging rule was removed (compounded LLM over-detection)."""
        features = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.03,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        adj = compute_adjustments(features)
        assert "self_consciousness" not in adj

    def test_hot_cold_oscillation_high_std(self):
        features = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=100,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        adj = compute_adjustments(features)
        assert adj.get("hot_cold_oscillation", 0) > 0

    def test_hot_cold_oscillation_low_std(self):
        features = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        adj = compute_adjustments(features)
        assert adj.get("hot_cold_oscillation", 0) < 0

    def test_decisiveness_hedging_reduces(self):
        features = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.03,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        adj = compute_adjustments(features)
        assert adj.get("decisiveness", 0) < 0

    def test_decisiveness_absolutist_increases(self):
        features = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.005,
            absolutist_ratio=0.015, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        adj = compute_adjustments(features)
        assert adj.get("decisiveness", 0) > 0

    def test_curiosity_high_questions(self):
        features = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.30, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        adj = compute_adjustments(features)
        assert adj.get("curiosity", 0) > 0

    def test_verbosity_long_turns_no_adjustment(self):
        """Verbosity high-end rule removed (compounded with LLM over-detection)."""
        features = BehavioralFeatures(
            turn_count=10, total_words=2000, avg_words_per_turn=200, words_std=20,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        adj = compute_adjustments(features)
        assert "verbosity" not in adj  # no upward adjustment

    def test_verbosity_short_turns(self):
        features = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        adj = compute_adjustments(features)
        assert adj.get("verbosity", 0) < 0


class TestComputeDirectScores:
    """Tests for rule-based direct trait scoring."""

    def test_returns_all_seven_traits(self):
        bf = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=40,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.15, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        scores = compute_direct_scores(bf)
        assert set(scores.keys()) == RULE_BASED_TRAITS

    def test_verbosity_short_turns(self):
        bf = BehavioralFeatures(
            turn_count=10, total_words=200, avg_words_per_turn=20, words_std=5,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        scores = compute_direct_scores(bf)
        assert 0.10 <= scores["verbosity"] <= 0.25

    def test_verbosity_medium_turns(self):
        bf = BehavioralFeatures(
            turn_count=10, total_words=600, avg_words_per_turn=60, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        scores = compute_direct_scores(bf)
        assert 0.25 <= scores["verbosity"] <= 0.45

    def test_verbosity_long_turns(self):
        bf = BehavioralFeatures(
            turn_count=10, total_words=2000, avg_words_per_turn=200, words_std=30,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        scores = compute_direct_scores(bf)
        assert scores["verbosity"] >= 0.65

    def test_politeness_low(self):
        bf = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.002, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        scores = compute_direct_scores(bf)
        assert scores["politeness"] < 0.25

    def test_politeness_high(self):
        bf = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.040, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        scores = compute_direct_scores(bf)
        assert scores["politeness"] >= 0.58

    def test_decisiveness_high_hedging(self):
        bf = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.035,
            absolutist_ratio=0.002, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.002,
        )
        scores = compute_direct_scores(bf)
        assert scores["decisiveness"] < 0.40

    def test_decisiveness_high_absolutist(self):
        bf = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.003,
            absolutist_ratio=0.020, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.015,
        )
        scores = compute_direct_scores(bf)
        assert scores["decisiveness"] > 0.55

    def test_curiosity_many_questions(self):
        bf = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.35, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.03, decisiveness_ratio=0.01,
        )
        scores = compute_direct_scores(bf)
        assert scores["curiosity"] > 0.55

    def test_hot_cold_high_variance(self):
        bf = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=120,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        scores = compute_direct_scores(bf)
        assert scores["hot_cold_oscillation"] > 0.55

    def test_hot_cold_low_variance(self):
        bf = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        scores = compute_direct_scores(bf)
        assert scores["hot_cold_oscillation"] < 0.30

    def test_self_mythologizing_high_self_ref(self):
        bf = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.12, other_ref_ratio=0.01, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        scores = compute_direct_scores(bf)
        assert scores["self_mythologizing"] > 0.50

    def test_optimism_positive_dominant(self):
        bf = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.030, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        scores = compute_direct_scores(bf)
        assert scores["optimism"] > 0.55

    def test_optimism_negative_dominant(self):
        bf = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.003, neg_emotion_ratio=0.025,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        scores = compute_direct_scores(bf)
        assert scores["optimism"] < 0.35

    def test_empty_features_returns_baselines(self):
        bf = BehavioralFeatures(
            turn_count=0, total_words=0, avg_words_per_turn=0, words_std=0,
            self_ref_ratio=0, other_ref_ratio=0, hedging_ratio=0,
            absolutist_ratio=0, question_ratio=0, exclamation_ratio=0,
            pos_emotion_ratio=0, neg_emotion_ratio=0,
            politeness_ratio=0, curiosity_ratio=0, decisiveness_ratio=0,
        )
        scores = compute_direct_scores(bf)
        assert len(scores) == 7
        for v in scores.values():
            assert 0.0 <= v <= 1.0

    def test_interpolation_is_continuous(self):
        """Adjacent inputs should produce scores within 0.05 of each other."""
        for wpt in range(10, 300, 5):
            bf1 = BehavioralFeatures(
                turn_count=10, total_words=wpt*10, avg_words_per_turn=wpt, words_std=10,
                self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
                absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
                pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
                politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
            )
            bf2 = BehavioralFeatures(
                turn_count=10, total_words=(wpt+5)*10, avg_words_per_turn=wpt+5, words_std=10,
                self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
                absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
                pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
                politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
            )
            s1 = compute_direct_scores(bf1)["verbosity"]
            s2 = compute_direct_scores(bf2)["verbosity"]
            assert abs(s1 - s2) < 0.05, f"Discontinuity at wpt={wpt}: {s1} vs {s2}"

    def test_rule_based_traits_constant(self):
        assert RULE_BASED_TRAITS == {
            "verbosity", "politeness", "decisiveness", "curiosity",
            "hot_cold_oscillation", "self_mythologizing", "optimism",
        }
