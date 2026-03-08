"""Tests for V2.8 ensemble blending."""

from super_brain.ensemble import blend_with_trajectory, _weighted_mean
from super_brain.models import (
    PersonalityDNA, Trait, Evidence, SampleSummary, ThinkSlowResult,
)


def _make_detected(traits_dict: dict[str, float]) -> PersonalityDNA:
    """Create a detected profile from {name: value} dict."""
    traits = [
        Trait(dimension="TEST", name=name, value=value, confidence=0.7,
              evidence=[Evidence(text="test", source="detector")])
        for name, value in traits_dict.items()
    ]
    return PersonalityDNA(
        id="detected",
        sample_summary=SampleSummary(
            total_tokens=100, conversation_count=1,
            date_range=["u", "u"], contexts=["eval"],
            confidence_overall=0.7,
        ),
        traits=traits,
    )


def _make_ts_result(traits_dict: dict[str, tuple[float, float]]) -> ThinkSlowResult:
    """Create ThinkSlowResult from {name: (value, confidence)} dict."""
    traits = [
        Trait(dimension="TEST", name=name, value=val, confidence=conf)
        for name, (val, conf) in traits_dict.items()
    ]
    conf_map = {name: conf for name, (val, conf) in traits_dict.items()}
    return ThinkSlowResult(
        partial_profile=PersonalityDNA(
            id="ts",
            sample_summary=SampleSummary(
                total_tokens=50, conversation_count=1,
                date_range=["u", "u"], contexts=["ts"],
                confidence_overall=0.5,
            ),
            traits=traits,
        ),
        confidence_map=conf_map,
        low_confidence_traits=[],
        observations=[],
        incisive_questions=[],
        info_staleness=0.5,
    )


def test_weighted_mean():
    assert abs(_weighted_mean([0.8, 0.2], [1.0, 1.0]) - 0.5) < 0.01
    assert abs(_weighted_mean([0.8, 0.2], [0.9, 0.1]) - 0.74) < 0.01


def test_weighted_mean_zero_weights():
    result = _weighted_mean([0.5, 0.6], [0.0, 0.0])
    assert abs(result - 0.55) < 0.01  # falls back to simple mean


def test_weighted_mean_empty():
    result = _weighted_mean([], [])
    assert result == 0.5  # fallback for empty input


def test_blend_no_ts_results():
    detected = _make_detected({"trust": 0.7, "anxiety": 0.3})
    result = blend_with_trajectory(detected, [])
    assert result.traits[0].value == 0.7
    assert result.traits[1].value == 0.3


def test_blend_with_single_ts():
    detected = _make_detected({"trust": 0.7})
    ts = [_make_ts_result({"trust": (0.5, 0.8)})]
    result = blend_with_trajectory(detected, ts)
    # ts_weight = 0.8 * 0.4 = 0.32, det_weight = 0.68
    # blended = 0.68 * 0.7 + 0.32 * 0.5 = 0.476 + 0.16 = 0.636
    assert abs(result.traits[0].value - 0.636) < 0.01


def test_blend_with_multiple_ts():
    detected = _make_detected({"trust": 0.7})
    ts = [
        _make_ts_result({"trust": (0.4, 0.6)}),
        _make_ts_result({"trust": (0.5, 0.8)}),
        _make_ts_result({"trust": (0.6, 0.7)}),
    ]
    result = blend_with_trajectory(detected, ts)
    # ts_avg = weighted_mean([0.4, 0.5, 0.6], [0.6, 0.8, 0.7])
    #        = (0.24 + 0.40 + 0.42) / (0.6 + 0.8 + 0.7)
    #        = 1.06 / 2.1 ≈ 0.5048
    # mean_conf = (0.6 + 0.8 + 0.7) / 3 ≈ 0.7
    # ts_weight = 0.7 * 0.4 = 0.28, det_weight = 0.72
    # blended = 0.72 * 0.7 + 0.28 * 0.5048 = 0.504 + 0.1413 ≈ 0.645
    assert abs(result.traits[0].value - 0.645) < 0.02


def test_blend_trait_not_in_ts():
    detected = _make_detected({"trust": 0.7, "anxiety": 0.3})
    ts = [_make_ts_result({"trust": (0.5, 0.8)})]  # no anxiety in TS
    result = blend_with_trajectory(detected, ts)
    trust_trait = [t for t in result.traits if t.name == "trust"][0]
    anxiety_trait = [t for t in result.traits if t.name == "anxiety"][0]
    assert trust_trait.value != 0.7  # blended
    assert anxiety_trait.value == 0.3  # unchanged — no TS data


def test_blend_low_confidence_ts():
    detected = _make_detected({"trust": 0.7})
    ts = [_make_ts_result({"trust": (0.3, 0.1)})]  # very low confidence
    result = blend_with_trajectory(detected, ts)
    # ts_weight = 0.1 * 0.4 = 0.04 — almost no influence
    # blended ≈ 0.96 * 0.7 + 0.04 * 0.3 = 0.672 + 0.012 = 0.684
    assert abs(result.traits[0].value - 0.684) < 0.01


def test_blend_preserves_profile_metadata():
    detected = _make_detected({"trust": 0.7})
    detected.id = "my_profile"
    detected.version = "2.8"
    ts = [_make_ts_result({"trust": (0.5, 0.8)})]
    result = blend_with_trajectory(detected, ts)
    assert result.id == "my_profile"
    assert result.version == "2.8"


def test_blend_clamps_to_0_1():
    detected = _make_detected({"trust": 0.95})
    ts = [_make_ts_result({"trust": (0.99, 0.9)})]
    result = blend_with_trajectory(detected, ts)
    assert result.traits[0].value <= 1.0
    assert result.traits[0].value >= 0.0


def test_detect_and_compare_accepts_ts_results_parameter():
    """Verify detect_and_compare() signature accepts ts_results parameter."""
    import inspect
    from eval_conversation import detect_and_compare
    sig = inspect.signature(detect_and_compare)
    assert "ts_results" in sig.parameters
    assert sig.parameters["ts_results"].default is None
