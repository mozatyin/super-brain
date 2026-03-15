"""Utilities for literary character personality experiments."""

from __future__ import annotations


def segment_dialogue(quotes: list[str], segment_size: int = 12) -> list[list[str]]:
    """Split a list of quotes into fixed-size segments.

    Args:
        quotes: List of character dialogue quotes.
        segment_size: Target number of quotes per segment.

    Returns:
        List of segments, each a list of quotes.
    """
    if not quotes:
        return []
    return [
        quotes[i : i + segment_size]
        for i in range(0, len(quotes), segment_size)
    ]


def compute_mae(
    detected: dict[str, dict],
    ground_truth: dict[str, float],
) -> float:
    """Compute MAE between detected profile and ground truth.

    Args:
        detected: {trait_name: {"value": float, "confidence": float}}
        ground_truth: {trait_name: float}

    Returns:
        Mean absolute error over shared traits.
    """
    errors = []
    for name, gt_val in ground_truth.items():
        if name in detected:
            errors.append(abs(detected[name]["value"] - gt_val))
    if not errors:
        return 1.0
    return sum(errors) / len(errors)
