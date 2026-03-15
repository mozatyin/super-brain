"""Tests for literary dialogue extraction and segmentation."""

import pytest
from super_brain.literary import segment_dialogue, compute_mae


class TestSegmentDialogue:
    """Test dialogue segmentation into fixed-size chunks."""

    def test_basic_segmentation(self):
        """Splits quotes into segments of target size."""
        quotes = [f"Quote {i}" for i in range(25)]
        segments = segment_dialogue(quotes, segment_size=10)
        assert len(segments) == 3  # 10 + 10 + 5
        assert len(segments[0]) == 10
        assert len(segments[2]) == 5

    def test_small_input(self):
        """Fewer quotes than segment_size -> single segment."""
        quotes = ["Hello", "World"]
        segments = segment_dialogue(quotes, segment_size=10)
        assert len(segments) == 1
        assert len(segments[0]) == 2

    def test_empty_input(self):
        """Empty list -> empty result."""
        segments = segment_dialogue([], segment_size=10)
        assert segments == []

    def test_exact_multiple(self):
        """Exact multiple of segment_size -> no remainder segment."""
        quotes = [f"Q{i}" for i in range(20)]
        segments = segment_dialogue(quotes, segment_size=10)
        assert len(segments) == 2


class TestComputeMAE:
    """Test MAE computation between detected profile and ground truth."""

    def test_perfect_match(self):
        """Identical profiles -> MAE = 0."""
        detected = {"a": {"value": 0.5, "confidence": 0.8}}
        gt = {"a": 0.5}
        assert compute_mae(detected, gt) == pytest.approx(0.0)

    def test_known_error(self):
        """Known difference -> correct MAE."""
        detected = {
            "a": {"value": 0.3, "confidence": 0.8},
            "b": {"value": 0.7, "confidence": 0.8},
        }
        gt = {"a": 0.5, "b": 0.5}
        # errors: 0.2 + 0.2 = 0.4, MAE = 0.2
        assert compute_mae(detected, gt) == pytest.approx(0.2)

    def test_missing_traits_excluded(self):
        """Only traits in both detected and GT count."""
        detected = {"a": {"value": 0.5, "confidence": 0.8}}
        gt = {"a": 0.5, "b": 0.9}
        assert compute_mae(detected, gt) == pytest.approx(0.0)
