"""Tests for V8x behavioral→strategy mapping engine."""

import pytest
from super_brain.v8x_strategy import ReplyStrategy, V8xStrategyEngine


@pytest.fixture
def engine():
    return V8xStrategyEngine()


class TestComputeStrategy:
    def test_warm_tone_on_distress(self, engine):
        """High neg_emotion_ratio → tone='warm', empathy_level >= 0.8."""
        features = {"neg_emotion_ratio": 0.15, "word_count": 50}
        strategy = engine.compute_strategy(features, detector_results=[], turn_number=3)
        assert strategy.tone == "warm"
        assert strategy.empathy_level >= 0.8

    def test_playful_on_short_input(self, engine):
        """Low word_count → tone='playful' or 'neutral', question_type='specific'."""
        features = {"word_count": 10}
        strategy = engine.compute_strategy(features, detector_results=[], turn_number=3)
        assert strategy.tone in ("playful", "neutral")
        assert strategy.question_type == "specific"

    def test_deep_depth_after_turn_5(self, engine):
        """Turn >= 5 + high self_ref → depth='deep'."""
        features = {"self_ref_ratio": 0.25, "word_count": 80}
        strategy = engine.compute_strategy(features, detector_results=[], turn_number=5)
        assert strategy.depth == "deep"

    def test_surface_depth_turn_1(self, engine):
        """Turn 1 → depth='surface' regardless of other signals."""
        features = {"self_ref_ratio": 0.3, "word_count": 200}
        strategy = engine.compute_strategy(features, detector_results=[], turn_number=1)
        assert strategy.depth == "surface"

    def test_mirror_high_turn_1(self, engine):
        """Turn 1 → mirror_ratio >= 0.7."""
        features = {"word_count": 50}
        strategy = engine.compute_strategy(features, detector_results=[], turn_number=1)
        assert strategy.mirror_ratio >= 0.7

    def test_firm_tone_on_absolutist(self, engine):
        """High absolutist_ratio → tone='firm'."""
        features = {"absolutist_ratio": 0.15, "word_count": 80}
        strategy = engine.compute_strategy(features, detector_results=[], turn_number=3)
        assert strategy.tone == "firm"

    def test_high_question_ratio_mirror(self, engine):
        """High question_ratio → question_type='specific', mirror_ratio=0.6."""
        features = {"question_ratio": 0.4, "word_count": 80}
        strategy = engine.compute_strategy(features, detector_results=[], turn_number=3)
        assert strategy.question_type == "specific"
        assert strategy.mirror_ratio >= 0.5

    def test_hedging_warm_reflective(self, engine):
        """High hedging_ratio → tone='warm', question_type='reflective'."""
        features = {"hedging_ratio": 0.15, "word_count": 80}
        strategy = engine.compute_strategy(features, detector_results=[], turn_number=3)
        assert strategy.tone == "warm"
        assert strategy.question_type == "reflective"

    def test_high_self_ref_deep_empathy(self, engine):
        """High self_ref_ratio → empathy_level >= 0.7, depth='deep' (after turn 5)."""
        features = {"self_ref_ratio": 0.25, "word_count": 80}
        strategy = engine.compute_strategy(features, detector_results=[], turn_number=6)
        assert strategy.empathy_level >= 0.7
        assert strategy.depth == "deep"

    def test_strategy_output_format(self, engine):
        """All fields present with valid values."""
        features = {"word_count": 50}
        strategy = engine.compute_strategy(features, detector_results=[], turn_number=2)
        assert strategy.tone in ("warm", "direct", "playful", "neutral", "firm")
        assert strategy.depth in ("surface", "medium", "deep")
        assert strategy.question_type in ("specific", "open", "reflective", "none")
        assert 0.0 <= strategy.empathy_level <= 1.0
        assert 0.0 <= strategy.mirror_ratio <= 1.0


class TestComposerDirective:
    def test_composer_directive_under_50_tokens(self, engine):
        """Generated directive word count < 50."""
        strategy = ReplyStrategy(
            tone="warm", depth="medium", question_type="reflective",
            empathy_level=0.8, mirror_ratio=0.7,
        )
        directive = engine.generate_composer_directive(strategy, turn_number=3)
        assert isinstance(directive, str)
        assert len(directive.split()) < 50

    def test_directive_mentions_tone(self, engine):
        """Directive should reference the tone."""
        strategy = ReplyStrategy(
            tone="direct", depth="surface", question_type="specific",
            empathy_level=0.4, mirror_ratio=0.5,
        )
        directive = engine.generate_composer_directive(strategy, turn_number=2)
        assert "direct" in directive.lower()

    def test_directive_high_empathy_mentions_empathy(self, engine):
        """High empathy → directive mentions empathy or warmth."""
        strategy = ReplyStrategy(
            tone="warm", depth="medium", question_type="reflective",
            empathy_level=0.9, mirror_ratio=0.7,
        )
        directive = engine.generate_composer_directive(strategy, turn_number=3)
        lower = directive.lower()
        assert "empathy" in lower or "empathize" in lower or "warm" in lower

    def test_directive_low_empathy_no_over_empathize(self, engine):
        """Low empathy → directive warns not to over-empathize."""
        strategy = ReplyStrategy(
            tone="direct", depth="surface", question_type="specific",
            empathy_level=0.3, mirror_ratio=0.3,
        )
        directive = engine.generate_composer_directive(strategy, turn_number=2)
        assert "over-empathize" in directive.lower() or "don't" in directive.lower()


class TestD1Hook:
    def test_d1_hook_passive_user(self, engine):
        """Low engagement → hook_type='passive'."""
        features = {"word_count": 8, "question_ratio": 0.0}
        result = engine.generate_d1_hook(features, detector_results=[])
        assert result["hook_type"] == "passive"
        assert "hook_text" in result
        assert "push_text" in result

    def test_d1_hook_curious_user(self, engine):
        """High questions → hook_type='curious'."""
        features = {"question_ratio": 0.5, "word_count": 100, "topic_count": 4}
        result = engine.generate_d1_hook(features, detector_results=[])
        assert result["hook_type"] == "curious"

    def test_d1_hook_action_user(self, engine):
        """Action intent → hook_type='action'."""
        features = {"action_intent": True, "word_count": 80}
        result = engine.generate_d1_hook(features, detector_results=[])
        assert result["hook_type"] == "action"

    def test_d1_hook_defensive_user(self, engine):
        """High absolutist + low self_ref → hook_type='defensive'."""
        features = {"absolutist_ratio": 0.15, "self_ref_ratio": 0.02, "word_count": 80}
        result = engine.generate_d1_hook(features, detector_results=[])
        assert result["hook_type"] == "defensive"

    def test_d1_hook_has_all_keys(self, engine):
        """Hook dict has hook_type, hook_text, push_text."""
        features = {"word_count": 50}
        result = engine.generate_d1_hook(features, detector_results=[])
        assert set(result.keys()) >= {"hook_type", "hook_text", "push_text"}
