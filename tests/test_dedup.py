"""Tests for V2.7 text deduplication."""

from super_brain.dedup import is_duplicate, dedup_extend_strings


def test_exact_duplicate():
    assert is_duplicate("start a business", ["start a business"]) is True


def test_near_duplicate():
    assert is_duplicate(
        "start a new business",
        ["start a business"],
        threshold=0.5,
    ) is True


def test_not_duplicate():
    assert is_duplicate("loves playing guitar", ["start a business"]) is False


def test_empty_existing():
    assert is_duplicate("anything", []) is False


def test_empty_new():
    assert is_duplicate("", ["something"]) is False


def test_high_threshold_rejects():
    assert is_duplicate(
        "wants to start own business",
        ["start a business"],
        threshold=0.9,
    ) is False


def test_dedup_extend_strings():
    target = ["start a business", "learn guitar"]
    new = ["start a new business", "play piano", "start a business"]
    added = dedup_extend_strings(target, new, threshold=0.5)
    # "start a new business" is similar to "start a business" (Jaccard 0.75) -> skipped
    # "play piano" is unique -> added
    # "start a business" is exact duplicate -> skipped
    assert added == 1
    assert "play piano" in target
    assert len(target) == 3


def test_dedup_extend_strings_all_unique():
    target = ["fact one"]
    new = ["fact two", "fact three"]
    added = dedup_extend_strings(target, new)
    assert added == 2
    assert len(target) == 3
