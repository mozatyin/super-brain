"""Progressive personality detection with Bayesian belief updating.

Wraps the existing Detector to incrementally build a personality profile
from sequential text segments, inspired by SoulGraph's Dual Soul architecture.
"""

from __future__ import annotations

from super_brain.catalog import TRAIT_CATALOG
from super_brain.detector import Detector
from super_brain.models import PersonalityDNA, Trait, SampleSummary


def bayesian_update(
    prior_val: float,
    prior_conf: float,
    obs_val: float,
    obs_conf: float,
) -> tuple[float, float]:
    """Bayesian merge of prior belief with new observation.

    Args:
        prior_val: Current estimated trait value (0.0-1.0).
        prior_conf: Confidence in current estimate (0.0-1.0).
        obs_val: Newly observed trait value (0.0-1.0).
        obs_conf: Confidence of new observation (0.0-1.0).

    Returns:
        (new_value, new_confidence) tuple.
    """
    total_conf = prior_conf + obs_conf
    if total_conf == 0:
        return 0.5, 0.0

    new_val = (prior_val * prior_conf + obs_val * obs_conf) / total_conf
    # Asymptotic confidence growth: diminishing returns from new evidence
    new_conf = min(0.95, prior_conf + obs_conf * (1.0 - prior_conf) * 0.5)
    return new_val, new_conf


class ProgressiveDetector:
    """Incrementally builds a personality profile from sequential text segments.

    Each call to update() runs the Detector on new text, then Bayesian-merges
    the results with accumulated priors. Tracks full history for convergence analysis.
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self._detector = Detector(api_key=api_key, model=model)
        self._priors: dict[str, tuple[float, float]] = {}  # trait_name -> (value, conf)
        self._trait_dims: dict[str, str] = {
            t["name"]: t["dimension"] for t in TRAIT_CATALOG
        }
        self._history: list[dict] = []
        self._segment_count = 0

    def update(self, text: str, speaker_label: str = "Speaker") -> dict:
        """Detect traits from a text segment and merge with priors.

        Args:
            text: The dialogue text segment to analyze.
            speaker_label: Label identifying the target speaker in the text.

        Returns:
            Snapshot dict with segment_id, per-trait values, and confidence.
        """
        # Run single-shot detection on this segment
        result = self._detector.analyze(
            text=text,
            speaker_id=f"progressive_seg_{self._segment_count}",
            speaker_label=speaker_label,
        )

        # Merge each detected trait with prior
        for trait in result.traits:
            prior_val, prior_conf = self._priors.get(trait.name, (0.5, 0.0))
            new_val, new_conf = bayesian_update(
                prior_val, prior_conf, trait.value, trait.confidence,
            )
            self._priors[trait.name] = (new_val, new_conf)

        # Build snapshot
        snapshot = {
            "segment_id": self._segment_count,
            "traits": {
                name: {"value": round(v, 3), "confidence": round(c, 3)}
                for name, (v, c) in self._priors.items()
            },
        }
        self._history.append(snapshot)
        self._segment_count += 1
        return snapshot

    def get_profile_dict(self) -> dict[str, dict]:
        """Return current accumulated profile as {trait: {value, confidence}}."""
        return {
            name: {"value": round(v, 3), "confidence": round(c, 3)}
            for name, (v, c) in self._priors.items()
        }

    def get_profile(self) -> PersonalityDNA:
        """Return current accumulated profile as PersonalityDNA."""
        traits = []
        for name, (val, conf) in self._priors.items():
            dim = self._trait_dims.get(name, "UNK")
            traits.append(Trait(dimension=dim, name=name, value=val, confidence=conf))
        return PersonalityDNA(
            id="progressive",
            sample_summary=SampleSummary(
                total_tokens=0,
                conversation_count=self._segment_count,
                date_range=["unknown", "unknown"],
                contexts=["literary"],
                confidence_overall=(
                    sum(c for _, c in self._priors.values()) / max(len(self._priors), 1)
                ),
            ),
            traits=traits,
        )

    def get_history(self) -> list[dict]:
        """Return all historical snapshots."""
        return list(self._history)

    def reset(self):
        """Clear all priors and history."""
        self._priors.clear()
        self._history.clear()
        self._segment_count = 0
