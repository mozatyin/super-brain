"""Trait-to-archetype mapping: derive personality archetypes from trait combinations.

Maps 68 continuous traits to categorical type systems:
- MBTI 16 types
- Enneagram 9 types
- Art of Seduction 9 archetypes
- Jung 12 archetypes
- DISC 4 types
- D&D Alignment 9 types
"""

from __future__ import annotations

from dataclasses import dataclass

from super_brain.models import PersonalityDNA


@dataclass
class ArchetypeMatch:
    """A matched archetype with confidence score."""
    system: str      # e.g., "MBTI", "Enneagram", "Seduction"
    name: str        # e.g., "INTJ", "Type 4", "Coquette"
    score: float     # 0.0-1.0 match confidence
    description: str


def _trait_map(profile: PersonalityDNA) -> dict[str, float]:
    """Build name->value lookup from a profile."""
    return {t.name: t.value for t in profile.traits}


# ═══════════════════════════════════════════════════════════════════════════
# MBTI — derived from Big Five facets
# ═══════════════════════════════════════════════════════════════════════════

_MBTI_TYPES: dict[str, dict[str, str]] = {
    "INTJ": {"desc": "The Architect — strategic, independent, determined"},
    "INTP": {"desc": "The Logician — analytical, objective, reserved"},
    "ENTJ": {"desc": "The Commander — decisive, ambitious, strategic"},
    "ENTP": {"desc": "The Debater — clever, curious, challenging"},
    "INFJ": {"desc": "The Advocate — insightful, principled, compassionate"},
    "INFP": {"desc": "The Mediator — idealistic, empathetic, creative"},
    "ENFJ": {"desc": "The Protagonist — charismatic, empathetic, organized"},
    "ENFP": {"desc": "The Campaigner — enthusiastic, creative, sociable"},
    "ISTJ": {"desc": "The Logistician — responsible, thorough, dependable"},
    "ISFJ": {"desc": "The Defender — supportive, reliable, patient"},
    "ESTJ": {"desc": "The Executive — organized, logical, assertive"},
    "ESFJ": {"desc": "The Consul — caring, sociable, traditional"},
    "ISTP": {"desc": "The Virtuoso — bold, practical, experimental"},
    "ISFP": {"desc": "The Adventurer — flexible, charming, sensitive"},
    "ESTP": {"desc": "The Entrepreneur — energetic, perceptive, direct"},
    "ESFP": {"desc": "The Entertainer — spontaneous, energetic, enthusiastic"},
}


def _mbti_scores(tm: dict[str, float]) -> list[ArchetypeMatch]:
    """Compute MBTI type scores from Big Five traits.

    Mapping based on established research:
    - E/I: extraversion (warmth, gregariousness, assertiveness, activity_level)
    - S/N: openness (fantasy, ideas, actions, values_openness)
    - T/F: agreeableness (trust, tender_mindedness, altruism) + feelings
    - J/P: conscientiousness (order, self_discipline, deliberation)
    """
    # E-I axis: high = Extraversion
    e_score = (
        tm.get("warmth", 0.5) * 0.25
        + tm.get("gregariousness", 0.5) * 0.30
        + tm.get("assertiveness", 0.5) * 0.25
        + tm.get("activity_level", 0.5) * 0.20
    )

    # N-S axis: high = iNtuition (openness)
    n_score = (
        tm.get("fantasy", 0.5) * 0.25
        + tm.get("ideas", 0.5) * 0.30
        + tm.get("actions", 0.5) * 0.20
        + tm.get("values_openness", 0.5) * 0.25
    )

    # F-T axis: high = Feeling (agreeableness + emotional openness)
    f_score = (
        tm.get("trust", 0.5) * 0.20
        + tm.get("tender_mindedness", 0.5) * 0.25
        + tm.get("altruism", 0.5) * 0.20
        + tm.get("feelings", 0.5) * 0.20
        + tm.get("empathy_affective", 0.5) * 0.15
    )

    # J-P axis: high = Judging (conscientiousness)
    j_score = (
        tm.get("order", 0.5) * 0.30
        + tm.get("self_discipline", 0.5) * 0.25
        + tm.get("deliberation", 0.5) * 0.25
        + tm.get("dutifulness", 0.5) * 0.20
    )

    results = []
    for mbti_type, info in _MBTI_TYPES.items():
        e, n, f, j = mbti_type[0], mbti_type[1], mbti_type[2], mbti_type[3]

        score = 1.0
        score *= e_score if e == "E" else (1 - e_score)
        score *= n_score if n == "N" else (1 - n_score)
        score *= f_score if f == "F" else (1 - f_score)
        score *= j_score if j == "J" else (1 - j_score)

        # Normalize: max possible is 1.0^4 = 1.0, but typical is much lower
        # Scale to 0-1 range (multiply by 16 since there are 16 types)
        score = min(1.0, score * 16)

        results.append(ArchetypeMatch(
            system="MBTI",
            name=mbti_type,
            score=round(score, 3),
            description=info["desc"],
        ))

    results.sort(key=lambda x: -x.score)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Enneagram — derived from Big Five + Dark + Values + Emotional
# ═══════════════════════════════════════════════════════════════════════════

_ENNEAGRAM_PROFILES: dict[str, dict[str, tuple[str, float]]] = {
    # type: {trait_name: (direction, weight)} — direction is "high" or "low"
    "Type 1 — The Reformer": {
        "order": ("high", 0.20), "dutifulness": ("high", 0.20),
        "fairness_justice": ("high", 0.20), "angry_hostility": ("high", 0.15),
        "self_discipline": ("high", 0.15), "authority_respect": ("high", 0.10),
    },
    "Type 2 — The Helper": {
        "altruism": ("high", 0.25), "warmth": ("high", 0.20),
        "empathy_affective": ("high", 0.20), "attachment_anxiety": ("high", 0.15),
        "modesty": ("low", 0.10), "care_harm": ("high", 0.10),
    },
    "Type 3 — The Achiever": {
        "achievement_striving": ("high", 0.25), "competence": ("high", 0.20),
        "narcissism": ("high", 0.15), "self_mythologizing": ("high", 0.15),
        "assertiveness": ("high", 0.15), "modesty": ("low", 0.10),
    },
    "Type 4 — The Individualist": {
        "emotional_granularity": ("high", 0.20), "aesthetics": ("high", 0.15),
        "fantasy": ("high", 0.15), "depression": ("high", 0.15),
        "self_consciousness": ("high", 0.15), "emotional_expressiveness": ("high", 0.10),
        "values_openness": ("high", 0.10),
    },
    "Type 5 — The Investigator": {
        "need_for_cognition": ("high", 0.25), "ideas": ("high", 0.20),
        "gregariousness": ("low", 0.15), "attachment_avoidance": ("high", 0.15),
        "emotional_expressiveness": ("low", 0.15), "intuitive_vs_analytical": ("high", 0.10),
    },
    "Type 6 — The Loyalist": {
        "anxiety": ("high", 0.20), "loyalty_group": ("high", 0.20),
        "trust": ("low", 0.15), "authority_respect": ("high", 0.15),
        "deliberation": ("high", 0.15), "vulnerability": ("high", 0.15),
    },
    "Type 7 — The Enthusiast": {
        "positive_emotions": ("high", 0.20), "excitement_seeking": ("high", 0.20),
        "actions": ("high", 0.15), "impulsiveness": ("high", 0.15),
        "humor_affiliative": ("high", 0.15), "anxiety": ("low", 0.15),
    },
    "Type 8 — The Challenger": {
        "assertiveness": ("high", 0.20), "social_dominance": ("high", 0.20),
        "conflict_assertiveness": ("high", 0.15), "vulnerability": ("low", 0.15),
        "compliance": ("low", 0.15), "locus_of_control": ("high", 0.15),
    },
    "Type 9 — The Peacemaker": {
        "compliance": ("high", 0.20), "conflict_cooperativeness": ("high", 0.20),
        "angry_hostility": ("low", 0.15), "conflict_assertiveness": ("low", 0.15),
        "emotional_regulation": ("high", 0.15), "warmth": ("high", 0.15),
    },
}


def _enneagram_scores(tm: dict[str, float]) -> list[ArchetypeMatch]:
    """Compute Enneagram match scores."""
    results = []
    for etype, traits in _ENNEAGRAM_PROFILES.items():
        score = 0.0
        for trait_name, (direction, weight) in traits.items():
            val = tm.get(trait_name, 0.5)
            if direction == "high":
                score += val * weight
            else:  # "low"
                score += (1 - val) * weight
        results.append(ArchetypeMatch(
            system="Enneagram", name=etype, score=round(score, 3),
            description=etype.split(" — ")[1] if " — " in etype else etype,
        ))
    results.sort(key=lambda x: -x.score)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Art of Seduction 9 Archetypes
# ═══════════════════════════════════════════════════════════════════════════

_SEDUCTION_PROFILES: dict[str, dict[str, tuple[str, float]]] = {
    "Siren": {
        "aesthetics": ("high", 0.20), "emotional_expressiveness": ("high", 0.20),
        "charm_influence": ("high", 0.20), "straightforwardness": ("low", 0.15),
        "excitement_seeking": ("high", 0.15), "self_consciousness": ("low", 0.10),
    },
    "Rake": {
        "emotional_expressiveness": ("high", 0.20), "impulsiveness": ("high", 0.20),
        "attachment_anxiety": ("high", 0.15), "deliberation": ("low", 0.15),
        "emotional_regulation": ("low", 0.15), "emotional_volatility": ("high", 0.15),
    },
    "Ideal Lover": {
        "empathy_cognitive": ("high", 0.25), "mirroring_ability": ("high", 0.25),
        "empathy_affective": ("high", 0.20), "narcissism": ("low", 0.15),
        "warmth": ("high", 0.15),
    },
    "Dandy": {
        "values_openness": ("high", 0.20), "cognitive_flexibility": ("high", 0.20),
        "self_mythologizing": ("high", 0.20), "compliance": ("low", 0.15),
        "authority_respect": ("low", 0.15), "aesthetics": ("high", 0.10),
    },
    "Natural": {
        "positive_emotions": ("high", 0.20), "trust": ("high", 0.20),
        "emotional_expressiveness": ("high", 0.20), "machiavellianism": ("low", 0.20),
        "information_control": ("low", 0.20),
    },
    "Coquette": {
        "hot_cold_oscillation": ("high", 0.25), "information_control": ("high", 0.20),
        "self_discipline": ("high", 0.20), "attachment_anxiety": ("low", 0.15),
        "charm_influence": ("high", 0.10), "attachment_avoidance": ("high", 0.10),
    },
    "Charmer": {
        "charm_influence": ("high", 0.25), "empathy_cognitive": ("high", 0.20),
        "conflict_cooperativeness": ("high", 0.20), "angry_hostility": ("low", 0.15),
        "warmth": ("high", 0.10), "compliance": ("high", 0.10),
    },
    "Charismatic": {
        "assertiveness": ("high", 0.20), "self_mythologizing": ("high", 0.20),
        "locus_of_control": ("high", 0.20), "modesty": ("low", 0.15),
        "anxiety": ("low", 0.15), "positive_emotions": ("high", 0.10),
    },
    "Star": {
        "self_mythologizing": ("high", 0.25), "aesthetics": ("high", 0.20),
        "attachment_avoidance": ("high", 0.20), "gregariousness": ("low", 0.15),
        "information_control": ("high", 0.10), "narcissism": ("high", 0.10),
    },
}


def _seduction_scores(tm: dict[str, float]) -> list[ArchetypeMatch]:
    """Compute Art of Seduction archetype match scores."""
    results = []
    for stype, traits in _SEDUCTION_PROFILES.items():
        score = 0.0
        for trait_name, (direction, weight) in traits.items():
            val = tm.get(trait_name, 0.5)
            if direction == "high":
                score += val * weight
            else:
                score += (1 - val) * weight
        results.append(ArchetypeMatch(
            system="Seduction", name=stype, score=round(score, 3),
            description=f"Art of Seduction: {stype}",
        ))
    results.sort(key=lambda x: -x.score)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Jung 12 Archetypes
# ═══════════════════════════════════════════════════════════════════════════

_JUNG_PROFILES: dict[str, dict[str, tuple[str, float]]] = {
    "Hero": {
        "assertiveness": ("high", 0.20), "achievement_striving": ("high", 0.20),
        "competence": ("high", 0.20), "vulnerability": ("low", 0.20),
        "locus_of_control": ("high", 0.20),
    },
    "Caregiver": {
        "altruism": ("high", 0.25), "empathy_affective": ("high", 0.25),
        "warmth": ("high", 0.20), "care_harm": ("high", 0.20),
        "tender_mindedness": ("high", 0.10),
    },
    "Explorer": {
        "actions": ("high", 0.25), "excitement_seeking": ("high", 0.20),
        "values_openness": ("high", 0.20), "attachment_avoidance": ("high", 0.15),
        "compliance": ("low", 0.20),
    },
    "Rebel": {
        "values_openness": ("high", 0.20), "authority_respect": ("low", 0.25),
        "compliance": ("low", 0.20), "conflict_assertiveness": ("high", 0.15),
        "angry_hostility": ("high", 0.10), "excitement_seeking": ("high", 0.10),
    },
    "Lover": {
        "aesthetics": ("high", 0.20), "emotional_expressiveness": ("high", 0.20),
        "warmth": ("high", 0.20), "empathy_affective": ("high", 0.20),
        "feelings": ("high", 0.20),
    },
    "Creator": {
        "fantasy": ("high", 0.25), "aesthetics": ("high", 0.20),
        "ideas": ("high", 0.20), "cognitive_flexibility": ("high", 0.15),
        "values_openness": ("high", 0.10), "self_mythologizing": ("high", 0.10),
    },
    "Jester": {
        "humor_affiliative": ("high", 0.25), "positive_emotions": ("high", 0.20),
        "humor_self_enhancing": ("high", 0.15), "impulsiveness": ("high", 0.15),
        "warmth": ("high", 0.15), "anxiety": ("low", 0.10),
    },
    "Sage": {
        "need_for_cognition": ("high", 0.25), "ideas": ("high", 0.20),
        "intuitive_vs_analytical": ("high", 0.20), "cognitive_flexibility": ("high", 0.15),
        "deliberation": ("high", 0.10), "emotional_regulation": ("high", 0.10),
    },
    "Magician": {
        "fantasy": ("high", 0.20), "charm_influence": ("high", 0.20),
        "self_mythologizing": ("high", 0.20), "cognitive_flexibility": ("high", 0.15),
        "locus_of_control": ("high", 0.15), "information_control": ("high", 0.10),
    },
    "Ruler": {
        "social_dominance": ("high", 0.25), "assertiveness": ("high", 0.20),
        "order": ("high", 0.15), "competence": ("high", 0.15),
        "authority_respect": ("high", 0.15), "locus_of_control": ("high", 0.10),
    },
    "Innocent": {
        "trust": ("high", 0.25), "positive_emotions": ("high", 0.20),
        "sincerity": ("high", 0.15), "anxiety": ("low", 0.15),
        "machiavellianism": ("low", 0.15), "vulnerability": ("high", 0.10),
    },
    "Everyman": {
        "gregariousness": ("high", 0.20), "compliance": ("high", 0.15),
        "modesty": ("high", 0.15), "loyalty_group": ("high", 0.15),
        "warmth": ("high", 0.15), "conflict_cooperativeness": ("high", 0.10),
        "trust": ("high", 0.10),
    },
}


def _jung_scores(tm: dict[str, float]) -> list[ArchetypeMatch]:
    """Compute Jung 12 archetype match scores."""
    results = []
    for jtype, traits in _JUNG_PROFILES.items():
        score = 0.0
        for trait_name, (direction, weight) in traits.items():
            val = tm.get(trait_name, 0.5)
            if direction == "high":
                score += val * weight
            else:
                score += (1 - val) * weight
        results.append(ArchetypeMatch(
            system="Jung", name=jtype, score=round(score, 3),
            description=f"Jungian Archetype: {jtype}",
        ))
    results.sort(key=lambda x: -x.score)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# DISC — 4 types from assertiveness × cooperation
# ═══════════════════════════════════════════════════════════════════════════

_DISC_PROFILES: dict[str, dict[str, tuple[str, float]]] = {
    "D — Dominance": {
        "assertiveness": ("high", 0.30), "social_dominance": ("high", 0.20),
        "compliance": ("low", 0.20), "conflict_assertiveness": ("high", 0.15),
        "warmth": ("low", 0.15),
    },
    "I — Influence": {
        "warmth": ("high", 0.25), "positive_emotions": ("high", 0.20),
        "gregariousness": ("high", 0.20), "charm_influence": ("high", 0.20),
        "assertiveness": ("high", 0.15),
    },
    "S — Steadiness": {
        "compliance": ("high", 0.20), "warmth": ("high", 0.20),
        "conflict_cooperativeness": ("high", 0.20), "emotional_regulation": ("high", 0.15),
        "loyalty_group": ("high", 0.15), "assertiveness": ("low", 0.10),
    },
    "C — Conscientiousness": {
        "order": ("high", 0.25), "deliberation": ("high", 0.20),
        "intuitive_vs_analytical": ("high", 0.20), "self_discipline": ("high", 0.15),
        "gregariousness": ("low", 0.10), "emotional_expressiveness": ("low", 0.10),
    },
}


def _disc_scores(tm: dict[str, float]) -> list[ArchetypeMatch]:
    """Compute DISC type match scores."""
    results = []
    for dtype, traits in _DISC_PROFILES.items():
        score = 0.0
        for trait_name, (direction, weight) in traits.items():
            val = tm.get(trait_name, 0.5)
            if direction == "high":
                score += val * weight
            else:
                score += (1 - val) * weight
        results.append(ArchetypeMatch(
            system="DISC", name=dtype, score=round(score, 3),
            description=dtype,
        ))
    results.sort(key=lambda x: -x.score)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# D&D Alignment — 9 types from morality × authority axes
# ═══════════════════════════════════════════════════════════════════════════

def _alignment_scores(tm: dict[str, float]) -> list[ArchetypeMatch]:
    """Compute D&D alignment from care/fairness (good-evil) and authority/compliance (lawful-chaotic)."""
    # Good-Evil axis: high = Good
    good_score = (
        tm.get("care_harm", 0.5) * 0.30
        + tm.get("altruism", 0.5) * 0.25
        + tm.get("empathy_affective", 0.5) * 0.25
        + (1 - tm.get("sadism", 0.5)) * 0.20
    )

    # Lawful-Chaotic axis: high = Lawful
    lawful_score = (
        tm.get("authority_respect", 0.5) * 0.25
        + tm.get("dutifulness", 0.5) * 0.25
        + tm.get("order", 0.5) * 0.20
        + tm.get("compliance", 0.5) * 0.15
        + (1 - tm.get("values_openness", 0.5)) * 0.15
    )

    alignments = {
        "Lawful Good": (lawful_score, good_score),
        "Neutral Good": (0.5 - abs(lawful_score - 0.5), good_score),
        "Chaotic Good": (1 - lawful_score, good_score),
        "Lawful Neutral": (lawful_score, 0.5 - abs(good_score - 0.5)),
        "True Neutral": (0.5 - abs(lawful_score - 0.5), 0.5 - abs(good_score - 0.5)),
        "Chaotic Neutral": (1 - lawful_score, 0.5 - abs(good_score - 0.5)),
        "Lawful Evil": (lawful_score, 1 - good_score),
        "Neutral Evil": (0.5 - abs(lawful_score - 0.5), 1 - good_score),
        "Chaotic Evil": (1 - lawful_score, 1 - good_score),
    }

    results = []
    for name, (law_fit, good_fit) in alignments.items():
        score = max(0.0, law_fit) * max(0.0, good_fit)
        results.append(ArchetypeMatch(
            system="Alignment", name=name, score=round(score, 3),
            description=f"D&D Alignment: {name}",
        ))
    results.sort(key=lambda x: -x.score)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def derive_archetypes(
    profile: PersonalityDNA,
    systems: list[str] | None = None,
) -> dict[str, list[ArchetypeMatch]]:
    """Derive archetype matches from a PersonalityDNA profile.

    Args:
        profile: The personality profile to analyze.
        systems: Optional list of systems to compute. If None, computes all.
            Valid values: "MBTI", "Enneagram", "Seduction", "Jung", "DISC", "Alignment"

    Returns:
        Dict mapping system name to sorted list of archetype matches.
    """
    tm = _trait_map(profile)
    all_systems = systems or ["MBTI", "Enneagram", "Seduction", "Jung", "DISC", "Alignment"]

    dispatch = {
        "MBTI": _mbti_scores,
        "Enneagram": _enneagram_scores,
        "Seduction": _seduction_scores,
        "Jung": _jung_scores,
        "DISC": _disc_scores,
        "Alignment": _alignment_scores,
    }

    results: dict[str, list[ArchetypeMatch]] = {}
    for system in all_systems:
        if system in dispatch:
            results[system] = dispatch[system](tm)

    return results


def top_archetypes(
    profile: PersonalityDNA,
    n: int = 1,
    systems: list[str] | None = None,
) -> dict[str, list[ArchetypeMatch]]:
    """Get top-N archetype matches per system."""
    all_results = derive_archetypes(profile, systems)
    return {system: matches[:n] for system, matches in all_results.items()}
