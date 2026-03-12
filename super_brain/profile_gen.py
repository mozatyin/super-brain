"""Generate random but internally consistent 69-trait personality profiles."""

from __future__ import annotations

import random

from super_brain.catalog import TRAIT_CATALOG, CONSISTENCY_RULES
from super_brain.models import PersonalityDNA, Trait, SampleSummary


# ── Correlation clusters: traits that should co-vary ─────────────────────────
# (trait_a, trait_b, direction) — "positive" means they move together,
# "negative" means they move opposite.  Used to generate realistic profiles.

_SOFT_CORRELATIONS: list[tuple[str, str, str]] = [
    # Big Five inter-facet correlations
    ("fantasy", "ideas", "positive"),
    ("fantasy", "aesthetics", "positive"),
    ("aesthetics", "feelings", "positive"),
    ("actions", "excitement_seeking", "positive"),
    ("order", "self_discipline", "positive"),
    ("order", "deliberation", "positive"),
    ("achievement_striving", "self_discipline", "positive"),
    ("competence", "locus_of_control", "positive"),
    ("warmth", "gregariousness", "positive"),
    ("warmth", "positive_emotions", "positive"),
    ("assertiveness", "activity_level", "positive"),
    ("trust", "straightforwardness", "positive"),
    ("altruism", "tender_mindedness", "positive"),
    ("compliance", "modesty", "positive"),
    ("anxiety", "vulnerability", "positive"),
    ("anxiety", "self_consciousness", "positive"),
    ("depression", "vulnerability", "positive"),
    ("angry_hostility", "impulsiveness", "positive"),
    # Cross-layer correlations
    ("empathy_cognitive", "empathy_affective", "positive"),

    ("emotional_volatility", "impulsiveness", "positive"),
    ("emotional_regulation", "deliberation", "positive"),
    ("narcissism", "self_mythologizing", "positive"),
    ("machiavellianism", "information_control", "positive"),
    ("psychopathy", "sadism", "positive"),
    ("warmth", "empathy_affective", "positive"),
    ("assertiveness", "social_dominance", "positive"),
    ("assertiveness", "conflict_assertiveness", "positive"),
    ("compliance", "conflict_cooperativeness", "positive"),
    ("humor_affiliative", "warmth", "positive"),
    ("humor_aggressive", "angry_hostility", "positive"),
    # Negative correlations
    ("assertiveness", "compliance", "negative"),
    ("trust", "machiavellianism", "negative"),
    ("modesty", "narcissism", "negative"),
    ("altruism", "psychopathy", "negative"),
    ("anxiety", "locus_of_control", "negative"),
    ("depression", "positive_emotions", "negative"),
    ("emotional_regulation", "emotional_volatility", "negative"),
    ("impulsiveness", "deliberation", "negative"),
    ("sincerity", "machiavellianism", "negative"),
    ("values_openness", "authority_respect", "negative"),
]


def generate_profile(
    profile_id: str,
    seed: int | None = None,
    archetype_bias: dict[str, float] | None = None,
) -> PersonalityDNA:
    """Generate a random but internally consistent 69-trait personality profile.

    Args:
        profile_id: Unique identifier for the profile.
        seed: Random seed for reproducibility.
        archetype_bias: Optional dict of {trait_name: target_value} to bias
            certain traits toward specific values (useful for creating
            recognizable personality types).

    Returns:
        A complete PersonalityDNA with all 69 traits.
    """
    rng = random.Random(seed)

    # Step 1: Generate base random values
    values: dict[str, float] = {}
    for t in TRAIT_CATALOG:
        if archetype_bias and t["name"] in archetype_bias:
            # Bias toward target with some noise
            target = archetype_bias[t["name"]]
            noise = rng.gauss(0, 0.08)
            values[t["name"]] = max(0.0, min(1.0, target + noise))
        else:
            # Random value with mild central tendency (beta distribution)
            values[t["name"]] = rng.betavariate(2.0, 2.0)

    # Step 2: Apply soft correlations (nudge correlated traits toward each other)
    for _ in range(3):  # iterate a few times to converge
        for trait_a, trait_b, direction in _SOFT_CORRELATIONS:
            if trait_a not in values or trait_b not in values:
                continue
            va, vb = values[trait_a], values[trait_b]
            if direction == "positive":
                # Nudge toward each other
                mid = (va + vb) / 2
                values[trait_a] = va * 0.7 + mid * 0.3
                values[trait_b] = vb * 0.7 + mid * 0.3
            else:  # negative
                # Nudge apart (toward summing to 1.0)
                target_a = 1.0 - vb
                target_b = 1.0 - va
                values[trait_a] = va * 0.7 + target_a * 0.3
                values[trait_b] = vb * 0.7 + target_b * 0.3

    # Step 3: Enforce hard consistency rules
    for name_a, name_b, max_sum in CONSISTENCY_RULES:
        if name_a not in values or name_b not in values:
            continue
        total = values[name_a] + values[name_b]
        if total > max_sum:
            excess = total - max_sum
            # Reduce both proportionally
            ratio_a = values[name_a] / total
            values[name_a] -= excess * ratio_a
            values[name_b] -= excess * (1 - ratio_a)

    # Step 4: Clamp all values
    for name in values:
        values[name] = max(0.0, min(1.0, values[name]))

    # Step 5: Build traits
    dim_map = {t["name"]: t["dimension"] for t in TRAIT_CATALOG}
    traits = [
        Trait(
            dimension=dim_map[name],
            name=name,
            value=round(values[name], 2),
            confidence=1.0,
        )
        for name in values
        if name in dim_map
    ]

    return PersonalityDNA(
        id=profile_id,
        sample_summary=SampleSummary(
            total_tokens=0,
            conversation_count=0,
            date_range=["unknown", "unknown"],
            contexts=["generated"],
            confidence_overall=1.0,
        ),
        traits=traits,
    )
