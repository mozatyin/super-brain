"""Behavioral feature extraction from conversation text (V3.2).

Extracts objective, non-LLM text signals (word counts, pronoun ratios,
hedging frequency, etc.) and maps them to small trait adjustments.
These provide a complementary signal source that doesn't depend on
LLM interpretation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from super_brain.models import PersonalityDNA, Trait


# --- Word lists ---

_SELF_REF = {"i", "me", "my", "mine", "myself"}
_OTHER_REF = {"you", "your", "yours", "we", "our", "ours", "they", "their", "them"}

_HEDGING = {
    "maybe", "perhaps", "probably", "possibly", "might", "sometimes",
    "somewhat", "fairly", "relatively", "apparently",
}
_HEDGING_PHRASES = [
    "sort of", "kind of", "i think", "i guess", "i suppose",
    "i feel like", "could be", "not sure",
]

_ABSOLUTIST = {
    "always", "never", "definitely", "absolutely", "certainly",
    "totally", "completely", "everyone", "nobody", "obviously",
    "clearly", "undoubtedly",
}

_POSITIVE_EMO = {
    "love", "great", "amazing", "awesome", "happy", "excited",
    "wonderful", "fantastic", "enjoy", "glad", "thrilled", "grateful",
    "beautiful", "excellent", "brilliant", "fun", "passionate",
}
_NEGATIVE_EMO = {
    "hate", "terrible", "awful", "angry", "frustrated", "worried",
    "scared", "sad", "annoyed", "anxious", "stressed", "miserable",
    "upset", "furious", "depressed", "overwhelmed", "dread",
}

_POLITENESS = {
    "please", "thanks", "thank", "sorry", "excuse", "pardon",
    "appreciate", "grateful", "kindly",
}
_POLITENESS_PHRASES = [
    "thank you", "excuse me", "if you don't mind", "i appreciate",
    "i'm sorry", "sorry about", "no worries",
]

_CURIOSITY_PHRASES = [
    "i wonder", "how does", "how do", "why is", "why do",
    "what if", "that's interesting", "tell me more",
    "how come", "what makes",
]

_DECISIVENESS = {
    "decided", "definitely", "absolutely", "certainly",
    "committed", "determined",
}
_DECISIVENESS_PHRASES = [
    "i will", "i've decided", "let's do", "let's go",
    "i'm going to", "no question", "for sure",
]


@dataclass
class BehavioralFeatures:
    """Objective text-based behavioral signals from speaker turns."""
    turn_count: int
    total_words: int
    avg_words_per_turn: float
    words_std: float
    self_ref_ratio: float       # self-reference words / total words
    other_ref_ratio: float      # other-reference words / total words
    hedging_ratio: float        # hedging words+phrases / total words
    absolutist_ratio: float     # absolutist words / total words
    question_ratio: float       # sentences ending in ? / total sentences
    exclamation_ratio: float    # sentences ending in ! / total sentences
    pos_emotion_ratio: float    # positive emotion words / total words
    neg_emotion_ratio: float    # negative emotion words / total words
    politeness_ratio: float     # politeness words+phrases / total words
    curiosity_ratio: float      # curiosity phrases / total words
    decisiveness_ratio: float   # decisiveness words+phrases / total words


def _tokenize(text: str) -> list[str]:
    """Simple lowercase word tokenization."""
    return re.findall(r"[a-z']+", text.lower())


def _count_sentences(text: str) -> tuple[int, int, int]:
    """Count total sentences, questions, exclamations."""
    # Split on sentence-ending punctuation
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    total = max(len(sentences), 1)
    questions = len(re.findall(r'\?', text))
    exclamations = len(re.findall(r'!', text))
    return total, questions, exclamations


def _count_phrases(text_lower: str, phrases: list[str]) -> int:
    """Count occurrences of multi-word phrases."""
    count = 0
    for phrase in phrases:
        count += text_lower.count(phrase)
    return count


def extract_features(
    conversation: list[dict],
    speaker_role: str = "speaker",
) -> BehavioralFeatures:
    """Extract behavioral features from speaker turns in a conversation.

    Args:
        conversation: List of dicts with 'role' and 'text' keys.
        speaker_role: Role label for the speaker (default "speaker").

    Returns:
        BehavioralFeatures with computed ratios.
    """
    turns = [msg["text"] for msg in conversation if msg["role"] == speaker_role]
    if not turns:
        return BehavioralFeatures(
            turn_count=0, total_words=0, avg_words_per_turn=0, words_std=0,
            self_ref_ratio=0, other_ref_ratio=0, hedging_ratio=0,
            absolutist_ratio=0, question_ratio=0, exclamation_ratio=0,
            pos_emotion_ratio=0, neg_emotion_ratio=0,
            politeness_ratio=0, curiosity_ratio=0, decisiveness_ratio=0,
        )

    # Per-turn word counts
    turn_word_counts = []
    all_words: list[str] = []
    all_text = ""

    total_sentences = 0
    total_questions = 0
    total_exclamations = 0

    for turn in turns:
        words = _tokenize(turn)
        turn_word_counts.append(len(words))
        all_words.extend(words)
        all_text += " " + turn

        sents, qs, excs = _count_sentences(turn)
        total_sentences += sents
        total_questions += qs
        total_exclamations += excs

    total_words = len(all_words)
    if total_words == 0:
        return BehavioralFeatures(
            turn_count=len(turns), total_words=0, avg_words_per_turn=0,
            words_std=0, self_ref_ratio=0, other_ref_ratio=0, hedging_ratio=0,
            absolutist_ratio=0, question_ratio=0, exclamation_ratio=0,
            pos_emotion_ratio=0, neg_emotion_ratio=0,
            politeness_ratio=0, curiosity_ratio=0, decisiveness_ratio=0,
        )

    avg_words = total_words / len(turns)
    if len(turns) > 1:
        variance = sum((c - avg_words) ** 2 for c in turn_word_counts) / len(turns)
        words_std = variance ** 0.5
    else:
        words_std = 0.0

    # Count word-level features
    word_set_counts = {
        "self": 0, "other": 0, "hedge": 0, "abs": 0, "pos": 0, "neg": 0,
        "polite": 0, "decisive": 0,
    }
    for w in all_words:
        if w in _SELF_REF:
            word_set_counts["self"] += 1
        if w in _OTHER_REF:
            word_set_counts["other"] += 1
        if w in _HEDGING:
            word_set_counts["hedge"] += 1
        if w in _ABSOLUTIST:
            word_set_counts["abs"] += 1
        if w in _POSITIVE_EMO:
            word_set_counts["pos"] += 1
        if w in _NEGATIVE_EMO:
            word_set_counts["neg"] += 1
        if w in _POLITENESS:
            word_set_counts["polite"] += 1
        if w in _DECISIVENESS:
            word_set_counts["decisive"] += 1

    # Add phrase-level hedging
    text_lower = all_text.lower()
    phrase_hedge_count = _count_phrases(text_lower, _HEDGING_PHRASES)

    # Phrase-level counts for new features
    phrase_polite_count = _count_phrases(text_lower, _POLITENESS_PHRASES)
    phrase_curiosity_count = _count_phrases(text_lower, _CURIOSITY_PHRASES)
    phrase_decisive_count = _count_phrases(text_lower, _DECISIVENESS_PHRASES)

    return BehavioralFeatures(
        turn_count=len(turns),
        total_words=total_words,
        avg_words_per_turn=round(avg_words, 1),
        words_std=round(words_std, 1),
        self_ref_ratio=round(word_set_counts["self"] / total_words, 4),
        other_ref_ratio=round(word_set_counts["other"] / total_words, 4),
        hedging_ratio=round(
            (word_set_counts["hedge"] + phrase_hedge_count) / total_words, 4
        ),
        absolutist_ratio=round(word_set_counts["abs"] / total_words, 4),
        question_ratio=round(total_questions / total_sentences, 4),
        exclamation_ratio=round(total_exclamations / total_sentences, 4),
        pos_emotion_ratio=round(word_set_counts["pos"] / total_words, 4),
        neg_emotion_ratio=round(word_set_counts["neg"] / total_words, 4),
        politeness_ratio=round(
            (word_set_counts["polite"] + phrase_polite_count) / total_words, 4
        ),
        curiosity_ratio=round(phrase_curiosity_count / total_words, 4),
        decisiveness_ratio=round(
            (word_set_counts["decisive"] + phrase_decisive_count) / total_words, 4
        ),
    )


# --- Trait adjustment mapping ---

# Each rule: (feature_name, threshold, direction, trait_name, delta)
# direction: "above" = apply when feature > threshold, "below" = feature < threshold
_ADJUSTMENT_RULES: list[tuple[str, float, str, str, float]] = [
    # Self-reference → narcissism
    ("self_ref_ratio", 0.08, "above", "narcissism", 0.06),
    ("self_ref_ratio", 0.03, "below", "narcissism", -0.06),

    # Hedging → assertiveness (inverse), modesty
    ("hedging_ratio", 0.020, "above", "assertiveness", -0.07),
    ("hedging_ratio", 0.020, "above", "modesty", 0.05),
    ("hedging_ratio", 0.005, "below", "assertiveness", 0.05),

    # Absolutist → assertiveness, deliberation (inverse)
    ("absolutist_ratio", 0.010, "above", "assertiveness", 0.07),
    ("absolutist_ratio", 0.010, "above", "deliberation", -0.05),

    # Response length → need_for_cognition, extraversion
    ("avg_words_per_turn", 200, "above", "need_for_cognition", 0.05),
    ("avg_words_per_turn", 80, "below", "need_for_cognition", -0.07),
    ("avg_words_per_turn", 200, "above", "warmth", 0.03),
    ("avg_words_per_turn", 60, "below", "warmth", -0.05),

    # Questions → intellectual_curiosity
    ("question_ratio", 0.20, "above", "intellectual_curiosity", 0.05),

    # Exclamations → positive_emotions, excitement_seeking
    ("exclamation_ratio", 0.25, "above", "positive_emotions", 0.05),
    ("exclamation_ratio", 0.25, "above", "excitement_seeking", 0.05),
    ("exclamation_ratio", 0.05, "below", "positive_emotions", -0.05),

    # Negative emotion words → anxiety, emotional_volatility
    ("neg_emotion_ratio", 0.015, "above", "anxiety", 0.06),
    ("neg_emotion_ratio", 0.015, "above", "emotional_volatility", 0.05),
    ("neg_emotion_ratio", 0.003, "below", "anxiety", -0.05),

    # Positive emotion words → positive_emotions
    ("pos_emotion_ratio", 0.020, "above", "positive_emotions", 0.05),

    # Self vs other reference balance → altruism, tender_mindedness
    ("other_ref_ratio", 0.04, "above", "altruism", 0.05),
    ("other_ref_ratio", 0.01, "below", "altruism", -0.05),

    # ── V3.2: Rules for difficult + new traits ──
    # competence signals
    ("avg_words_per_turn", 150, "above", "competence", 0.04),
    ("absolutist_ratio", 0.012, "above", "competence", 0.03),
    ("hedging_ratio", 0.025, "above", "competence", -0.05),
    # social_dominance signals
    ("question_ratio", 0.30, "above", "social_dominance", -0.05),
    ("avg_words_per_turn", 180, "above", "social_dominance", 0.04),
    ("self_ref_ratio", 0.07, "above", "social_dominance", 0.04),
    # self_consciousness
    ("hedging_ratio", 0.025, "above", "self_consciousness", 0.06),
    # intuitive_vs_analytical (toward analytical)
    ("avg_words_per_turn", 160, "above", "intuitive_vs_analytical", 0.05),
    # hot_cold_oscillation
    ("words_std", 80, "above", "hot_cold_oscillation", 0.06),
    ("words_std", 30, "below", "hot_cold_oscillation", -0.05),
    # decisiveness (new trait)
    ("hedging_ratio", 0.020, "above", "decisiveness", -0.08),
    ("absolutist_ratio", 0.010, "above", "decisiveness", 0.06),
    # curiosity (new trait)
    ("question_ratio", 0.25, "above", "curiosity", 0.06),
    # verbosity (new trait)
    ("avg_words_per_turn", 150, "above", "verbosity", 0.08),
    ("avg_words_per_turn", 60, "below", "verbosity", -0.08),
]


def compute_adjustments(features: BehavioralFeatures) -> dict[str, float]:
    """Map behavioral features to trait adjustment deltas.

    Returns:
        Dict of {trait_name: delta} where delta is typically ±0.03-0.10.
        Only includes traits where a rule fired.
    """
    adjustments: dict[str, float] = {}

    for feat_name, threshold, direction, trait_name, delta in _ADJUSTMENT_RULES:
        value = getattr(features, feat_name)
        if direction == "above" and value > threshold:
            adjustments[trait_name] = adjustments.get(trait_name, 0) + delta
        elif direction == "below" and value < threshold:
            adjustments[trait_name] = adjustments.get(trait_name, 0) + delta

    return adjustments


def apply_adjustments(
    profile: PersonalityDNA,
    adjustments: dict[str, float],
) -> PersonalityDNA:
    """Apply behavioral feature adjustments to a personality profile.

    Each adjustment is additive and clamped to [0.0, 1.0].
    """
    if not adjustments:
        return profile

    adjusted_traits = []
    for trait in profile.traits:
        delta = adjustments.get(trait.name, 0.0)
        if delta != 0.0:
            new_value = max(0.0, min(1.0, round(trait.value + delta, 3)))
            adjusted_traits.append(Trait(
                dimension=trait.dimension,
                name=trait.name,
                value=new_value,
                confidence=trait.confidence,
                evidence=trait.evidence,
            ))
        else:
            adjusted_traits.append(trait)

    return PersonalityDNA(
        id=profile.id,
        version=profile.version,
        created=profile.created,
        updated=profile.updated,
        sample_summary=profile.sample_summary,
        traits=adjusted_traits,
        trait_relations=profile.trait_relations,
    )
