"""Tests for the detector module."""

import pytest

from super_brain.detector import (
    DIMENSION_BATCHES,
    _parse_batch_response,
    _validate_consistency,
    _clamp,
    _get_traits_for_batch,
    _build_trait_prompt,
)
from super_brain.models import Trait


def test_dimension_batches_cover_all_dimensions():
    all_dims = set()
    for batch in DIMENSION_BATCHES:
        all_dims.update(batch)
    expected = {"OPN", "CON", "EXT", "AGR", "NEU", "HON", "DRK", "EMO", "SOC", "COG", "VAL", "STR", "HUM"}
    assert all_dims == expected


def test_batch_trait_counts():
    """Each batch should have the expected number of traits."""
    expected_counts = [12, 12, 10, 10, 10, 8, 4]  # total = 66? Let me verify
    for i, batch_dims in enumerate(DIMENSION_BATCHES):
        traits = _get_traits_for_batch(batch_dims)
        assert len(traits) == expected_counts[i], (
            f"Batch {i+1} ({batch_dims}): expected {expected_counts[i]}, got {len(traits)}"
        )


def test_total_traits_across_batches():
    total = sum(len(_get_traits_for_batch(b)) for b in DIMENSION_BATCHES)
    assert total == 66


def test_parse_batch_response_valid_json():
    raw = '{"reasoning": [], "scores": [{"dimension": "OPN", "name": "fantasy", "value": 0.75, "confidence": 0.9, "evidence_quote": "test"}]}'
    result = _parse_batch_response(raw)
    assert len(result) == 1
    assert result[0]["name"] == "fantasy"
    assert result[0]["value"] == 0.75


def test_parse_batch_response_markdown_fenced():
    raw = '```json\n{"reasoning": [], "scores": [{"dimension": "OPN", "name": "fantasy", "value": 0.5, "confidence": 0.8, "evidence_quote": "x"}]}\n```'
    result = _parse_batch_response(raw)
    assert len(result) == 1


def test_parse_batch_response_list():
    raw = '[{"dimension": "OPN", "name": "fantasy", "value": 0.5, "confidence": 0.8}]'
    result = _parse_batch_response(raw)
    assert len(result) == 1


def test_parse_batch_response_invalid():
    with pytest.raises(ValueError):
        _parse_batch_response("this is not json at all")


def test_clamp():
    assert _clamp(0.5) == 0.5
    assert _clamp(-0.1) == 0.0
    assert _clamp(1.5) == 1.0
    assert _clamp(0.0) == 0.0
    assert _clamp(1.0) == 1.0


def test_validate_consistency_narcissism_humility():
    """narcissism + humility_hexaco should be <= 1.3"""
    traits = [
        Trait(dimension="DRK", name="narcissism", value=0.9, confidence=0.8),
        Trait(dimension="HON", name="humility_hexaco", value=0.9, confidence=0.9),
    ]
    result = _validate_consistency(traits)
    tmap = {t.name: t.value for t in result}
    assert tmap["narcissism"] + tmap["humility_hexaco"] <= 1.3 + 0.01


def test_validate_consistency_sincerity_machiavellianism():
    """sincerity + machiavellianism should be <= 1.2"""
    traits = [
        Trait(dimension="HON", name="sincerity", value=0.9, confidence=0.8),
        Trait(dimension="DRK", name="machiavellianism", value=0.9, confidence=0.9),
    ]
    result = _validate_consistency(traits)
    tmap = {t.name: t.value for t in result}
    assert tmap["sincerity"] + tmap["machiavellianism"] <= 1.2 + 0.01


def test_validate_consistency_no_change_when_valid():
    """No change needed if traits are already consistent."""
    traits = [
        Trait(dimension="DRK", name="narcissism", value=0.3, confidence=0.8),
        Trait(dimension="HON", name="humility_hexaco", value=0.5, confidence=0.9),
    ]
    result = _validate_consistency(traits)
    tmap = {t.name: t.value for t in result}
    assert tmap["narcissism"] == 0.3
    assert tmap["humility_hexaco"] == 0.5


def test_build_trait_prompt_includes_anchors():
    from super_brain.catalog import get_traits_for_dimension
    opn_traits = get_traits_for_dimension("OPN")
    prompt = _build_trait_prompt(opn_traits)
    assert "fantasy" in prompt
    assert "0.0 =" in prompt
    assert "1.0 =" in prompt
