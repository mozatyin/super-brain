"""Ensemble blending of Detector + ThinkSlow trajectory (V2.8)."""

from __future__ import annotations

from super_brain.models import PersonalityDNA, Trait, ThinkSlowResult


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    """Compute weighted mean. Falls back to simple mean if weights sum to 0."""
    total_weight = sum(weights)
    if total_weight == 0:
        return sum(values) / len(values) if values else 0.5
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def blend_with_trajectory(
    detected: PersonalityDNA,
    ts_results: list[ThinkSlowResult],
    max_ts_weight: float = 0.4,
) -> PersonalityDNA:
    """Blend Detector results with ThinkSlow trajectory estimates.

    For each trait:
    - Collect all ThinkSlow estimates and their confidence scores
    - Compute confidence-weighted mean of ThinkSlow estimates
    - Blend: final = det_weight * detector_value + ts_weight * ts_avg
    - ts_weight = mean(confidences) * max_ts_weight (capped)

    Traits not estimated by ThinkSlow keep pure Detector values.
    """
    if not ts_results:
        return detected

    # Collect ThinkSlow trajectory per trait: name -> [(value, conf), ...]
    ts_trajectory: dict[str, list[tuple[float, float]]] = {}
    for ts in ts_results:
        for trait in ts.partial_profile.traits:
            conf = ts.confidence_map.get(trait.name, trait.confidence)
            ts_trajectory.setdefault(trait.name, []).append((trait.value, conf))

    # Blend each detected trait
    blended_traits = []
    for trait in detected.traits:
        trajectory = ts_trajectory.get(trait.name)
        if trajectory:
            values = [v for v, _ in trajectory]
            confs = [c for _, c in trajectory]
            ts_avg = _weighted_mean(values, confs)
            mean_conf = sum(confs) / len(confs)
            ts_weight = min(mean_conf * max_ts_weight, max_ts_weight)
            det_weight = 1.0 - ts_weight
            blended_value = det_weight * trait.value + ts_weight * ts_avg
            blended_value = max(0.0, min(1.0, blended_value))
            blended_traits.append(Trait(
                dimension=trait.dimension,
                name=trait.name,
                value=round(blended_value, 3),
                confidence=trait.confidence,
                evidence=trait.evidence,
            ))
        else:
            blended_traits.append(trait)

    return PersonalityDNA(
        id=detected.id,
        version=detected.version,
        created=detected.created,
        updated=detected.updated,
        sample_summary=detected.sample_summary,
        traits=blended_traits,
        trait_relations=detected.trait_relations,
    )
