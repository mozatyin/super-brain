"""Text deduplication utilities for Soul state management (V2.7)."""

from __future__ import annotations


def _tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase word set."""
    return set(text.lower().split())


def is_duplicate(new_text: str, existing_texts: list[str], threshold: float = 0.6) -> bool:
    """Check if new_text is a duplicate of any existing text.

    Uses Jaccard similarity (token set overlap ratio).
    Returns True if similarity with any existing text exceeds threshold.
    """
    new_tokens = _tokenize(new_text)
    if not new_tokens:
        return False
    for existing in existing_texts:
        existing_tokens = _tokenize(existing)
        if not existing_tokens:
            continue
        intersection = len(new_tokens & existing_tokens)
        union = len(new_tokens | existing_tokens)
        if union > 0 and intersection / union >= threshold:
            return True
    return False


def dedup_extend_strings(target: list[str], new_items: list[str], threshold: float = 0.6) -> int:
    """Extend target list with non-duplicate new items. Returns count added."""
    added = 0
    for item in new_items:
        if not is_duplicate(item, target, threshold):
            target.append(item)
            added += 1
    return added
