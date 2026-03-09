"""Tests for V2.7 dedup utilities used by Soul accumulation."""


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
