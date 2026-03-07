"""Soul Coverage scoring for V2.5 evaluation.

Measures how populated the Soul model is after a conversation.
Components (V2.5 — facts + reality + secrets + intentions + gaps):
- facts: min(count / 10, 1.0)
- reality: 1.0 if populated, else 0.0
- secrets: min(count / 3, 1.0)
- intentions: min(count / 3, 1.0)
- gaps: min(count / 2, 1.0)
"""

from __future__ import annotations

from super_brain.models import Soul


def compute_soul_coverage(soul: Soul) -> float:
    """Compute Soul Coverage score (0.0-1.0).

    V2.5 components (5 items, equally weighted):
    - facts: min(len / 10, 1.0) — 10+ facts = full
    - reality: 1.0 if populated, else 0.0
    - secrets: min(len / 3, 1.0) — 3+ secrets = full
    - intentions: min(len / 3, 1.0) — 3+ intentions = full
    - gaps: min(len / 2, 1.0) — 2+ gaps = full
    """
    scores: list[float] = []
    scores.append(min(len(soul.facts) / 10.0, 1.0))
    scores.append(1.0 if soul.reality else 0.0)
    scores.append(min(len(soul.secrets) / 3.0, 1.0))
    scores.append(min(len(soul.intentions) / 3.0, 1.0))
    scores.append(min(len(soul.gaps) / 2.0, 1.0))
    return sum(scores) / len(scores) if scores else 0.0
