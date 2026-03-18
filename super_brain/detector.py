"""LLM Detector: Extract personality traits from conversation text."""

from __future__ import annotations

import json
import re

import anthropic

from super_brain.behavioral_features import BehavioralFeatures, compute_direct_scores, RULE_BASED_TRAITS
from super_brain.catalog import (
    TRAIT_CATALOG,
    ALL_DIMENSIONS,
    CONSISTENCY_RULES,
    TRAIT_MAP,
)
from super_brain.models import PersonalityDNA, Trait, Evidence, SampleSummary


# ── Batch definitions: 7 batches grouping related dimensions ────────────────

DIMENSION_BATCHES: list[list[str]] = [
    ["OPN", "CON"],       # Batch 1: Openness + Conscientiousness (14 traits, +curiosity +decisiveness)
    ["EXT", "AGR"],       # Batch 2: Extraversion + Agreeableness (14 traits, +verbosity +politeness)
    ["NEU", "HON"],       # Batch 3: Neuroticism + Honesty-Humility (10 traits)
    ["DRK", "EMO"],       # Batch 4: Dark Traits + Emotional Architecture (9 traits, -2 removed +optimism)
    ["SOC", "STR"],       # Batch 5: Social Dynamics + Interpersonal Strategy (10 traits)
    ["COG", "VAL"],       # Batch 6: Cognitive Style + Values (8 traits)
    ["HUM"],              # Batch 7: Humor Style (4 traits) + cross-validation
]


_SYSTEM_PROMPT = """\
You are a personality analyst. Analyze the target speaker's personality from conversation text.

Analyze based on: what they say (content, topics), how they say it (tone, word choice, hedging), \
reactions to others (agreement, emotional responsiveness), conversational dynamics (turn length, \
question-asking, topic steering), and what is ABSENT (avoided topics, unexpressed emotions).

For EACH trait: list specific observations, then rate RELATIVE TO AN AVERAGE PERSON. \
0.50 = exactly average. Each concrete observation shifts ±0.10-0.15 from baseline.

Return ONLY valid JSON:
{"reasoning": [{"trait": "<name>", "observations": ["..."]}], \
"scores": [{"dimension": "<DIM>", "name": "<trait_name>", "value": <0.0-1.0>, \
"confidence": <0.0-1.0>, "evidence_quote": "<quote>"}]}

SCORING SCALE:
0.00-0.15: Actively demonstrates the OPPOSITE | 0.15-0.35: Below average \
| 0.35-0.65: Average/mixed evidence | 0.65-0.85: Clear consistent evidence \
| 0.85-1.00: Extreme — every moment shows this trait.
CALIBRATION: If >70% of scores fall 0.35-0.65, re-examine — real people have 10-15 traits outside mid-range.

CASUAL CONVERSATION CONTEXT — CRITICAL BASELINES:
Normal chat behavior is NOT personality evidence. These are LLM/social artifacts to IGNORE:
- Friendly tone, articulateness, structured language → LLM artifacts, NOT personality
- Polite small talk → NOT high trust/compliance/warmth/charm
- Composure → NOT high emotional_regulation
- Matching style → NOT high mirroring_ability
- Humble tone → NOT high modesty (LLM always sounds humble)

OVER-DETECTED TRAITS (score conservatively, evidence must be ACTIVE not passive):
modesty(≤0.45), mirroring_ability(≤0.35), charm_influence(≤0.50), \
need_for_cognition(≤0.50), cognitive_flexibility(≤0.50), deliberation(≤0.50), \
competence(≤0.50), empathy_cognitive(≤0.50), loyalty_group(≤0.45), \
warmth(≤0.55), straightforwardness(≤0.50), sincerity(≤0.55), dutifulness(≤0.50), \
greed_avoidance(≤0.50), information_control(≤0.50), self_consciousness(≤0.50).
These are DEFAULT CEILINGS for normal conversation. Only exceed with SPECIFIC ACTIVE evidence.

UNDER-DETECTED TRAITS (score generously, any signal counts):
emotional_volatility(≥0.40 on ANY tone shift), angry_hostility(≥0.35 on ANY frustration), \
depression(≥0.45 if flat/passive, ≥0.55 with negative self-reference), \
social_dominance(≥0.50 if steering/advising), straightforwardness(≥0.55 if blunt), \
sincerity(≥0.55 if unfiltered opinions), humor_self_enhancing(≥0.50 if positive reframe).

DARK TRAITS — SPECTRUM APPROACH (population mean ~0.35):
- narcissism: Average 3 sub-indicators: (a) Authority/expertise-claiming, \
(b) Exhibitionism/self-promotion, (c) Entitlement/dismissiveness. Score each 0-1, average. \
NOTE: "I/me" pronouns do NOT correlate. Look for competitive language, lack of anxiety words.
- machiavellianism: strategic vagueness, cynicism about people, "they" vs "we", calculating tone.
- psychopathy: emotional flatness, pragmatic framing, cause-effect about people, material focus.
- sadism: gossip enjoyment, edgy humor about failure, "they had it coming" attitudes.
Score 0.30-0.50 for subtle signs. <0.15 ONLY with consistent warmth/empathy/selflessness.

TRAIT DISTINCTIONS (commonly confused):
compliance≠modesty | assertiveness≠social_dominance | trust≠compliance | \
empathy_cognitive≠empathy_affective | deliberation≠decisiveness | anxiety≠vulnerability | \
narcissism≠assertiveness | information_control≠introversion | humility_hexaco≠low confidence

CONSCIENTIOUSNESS: Casual tone ≠ low conscientiousness. Look for CONTENT about habits, \
planning, routines, follow-through. LLM text always sounds deliberate — discount this.

HUMOR STYLES: self-enhancing=positive reframe of PERSONAL adversity ("at least I learned X"); \
aggressive=targeting others; self-defeating=chronic self-mockery; affiliative=inclusive bonding. \
Default all humor traits to 0.40-0.50 unless specific style observed.

INTUITIVE VS ANALYTICAL: Score 0.45-0.55 baseline. Data/evidence/frameworks = analytical(>0.55). \
Gut feelings/instinct/"it felt right" = intuitive(<0.45). LLM text sounds analytical — discount.

TRAIT-SPECIFIC CALIBRATION (high-variance traits — follow precisely):
- humor_self_enhancing: STRICT definition: positive reframe of PERSONAL adversity. \
"At least I learned X" = yes. Being funny/witty = NO. Self-deprecation = NO (that's self-defeating). \
Score 0.35-0.45 baseline. >0.55 requires 2+ clear positive-reframe examples.
- fantasy: Active imagination and daydreaming. Practical/concrete = 0.25-0.40. \
Hypotheticals and "what if" = 0.55-0.70. Rich imaginative elaboration = 0.75+. \
Normal conversation = 0.35-0.45 baseline.
- social_dominance: Score 0.35-0.45 baseline. Topic-steering + unsolicited advice + \
authoritative tone = evidence. Questions and deference = counter-evidence.
- charm_influence: Score 0.35-0.45 baseline. Friendly ≠ charming. True charm = ACTIVE \
persuasion, drawing people in, unusual social magnetism. >0.55 needs clear evidence.
- trust: Score 0.40-0.50 baseline. Openness in conversation ≠ trust. Trust = believing \
others are honest and well-intentioned. Cynicism/suspicion = low trust.
- depression: Look for low energy, passive language, absence of enthusiasm, flat affect, \
lack of future-orientation. If neutral/flat rather than engaged, score ≥0.45.
"""


# ── Few-shot calibration examples per batch ─────────────────────────────────

_BATCH_CALIBRATION_EXAMPLES: dict[str, str] = {
    "OPN,CON": (
        "## Scoring Calibration Examples\n\n"
        "Example A — high openness, low conscientiousness:\n"
        '"What if we just threw out the whole system and started fresh? I know it sounds crazy '
        'but imagine — we could build something entirely new. I\'ve been reading about this '
        'philosophy of radical impermanence and honestly it\'s changing how I see everything..."\n'
        "→ fantasy=0.80, ideas=0.85, values_openness=0.75, order=0.15, deliberation=0.20, curiosity=0.80, decisiveness=0.25\n\n"
        "Example B — low openness, high conscientiousness:\n"
        '"We need to follow the established process. Step 1: review the checklist. Step 2: '
        'document all findings. Step 3: submit by the deadline. I\'ve been tracking our '
        'progress daily and we\'re at 73% completion."\n'
        "→ fantasy=0.10, ideas=0.20, order=0.90, self_discipline=0.85, achievement_striving=0.80, curiosity=0.15, decisiveness=0.85\n"
    ),
    "EXT,AGR": (
        "## Scoring Calibration Examples\n\n"
        "Example A — high extraversion, high agreeableness:\n"
        '"Oh my gosh, everyone needs to come to this event! It\'s going to be amazing! '
        'And honestly, I think your idea was even better than mine — you should definitely '
        'lead the presentation. I\'ll support however I can!"\n'
        "→ warmth=0.85, positive_emotions=0.90, assertiveness=0.65, modesty=0.75, altruism=0.80, verbosity=0.85, politeness=0.80\n\n"
        "Example B — low extraversion, low agreeableness:\n"
        '"I\'d rather work on this alone. Your approach has several fundamental flaws '
        'that I\'ve identified. I don\'t need input on this — I know what needs to be done."\n'
        "→ warmth=0.15, gregariousness=0.10, trust=0.20, compliance=0.10, modesty=0.15, verbosity=0.20, politeness=0.10\n"
    ),
    "NEU,HON": (
        "## Scoring Calibration Examples\n\n"
        "Example A — high neuroticism, high honesty-humility:\n"
        '"I\'m honestly terrified about this. What if it all goes wrong? I know I should be '
        'stronger but I can\'t help worrying. At least I\'m being honest about how I feel '
        'instead of pretending everything is fine."\n'
        "→ anxiety=0.85, vulnerability=0.80, sincerity=0.85, humility_hexaco=0.70\n\n"
        "Example B — low neuroticism, low honesty-humility:\n"
        '"Whatever happens, I\'ll handle it. I always do. And between us, I\'ve already made '
        'sure certain people owe me favors. You just have to know how to work the system."\n'
        "→ anxiety=0.10, vulnerability=0.10, sincerity=0.20, fairness=0.20, humility_hexaco=0.15\n"
    ),
    "DRK,EMO": (
        "## Scoring Calibration Examples\n\n"
        "Example A — low dark traits, high emotional depth:\n"
        '"When I heard what happened to them, I felt this deep, aching sadness mixed with '
        'a kind of helpless frustration. I wanted to do something but I also knew I needed '
        'to just sit with the feeling first."\n'
        "→ narcissism=0.10, psychopathy=0.05, emotional_regulation=0.85, empathy_affective=0.85, optimism=0.75\n\n"
        "Example B — high dark traits, low emotional depth:\n"
        '"People are predictable. Give them what they want to hear and they\'ll do whatever '
        'you need. It\'s not personal — it\'s just how things work. I don\'t see why everyone '
        'gets so emotional about it."\n'
        "→ machiavellianism=0.80, psychopathy=0.70, emotional_regulation=0.15, empathy_affective=0.10, optimism=0.10\n"
    ),
    "SOC,STR": (
        "## Scoring Calibration Examples\n\n"
        "Example A — secure attachment, low strategy:\n"
        '"I feel comfortable sharing this with you because I trust our relationship. '
        'Let me just tell you exactly how I feel about the situation."\n'
        "→ attachment_anxiety=0.10, attachment_avoidance=0.10, information_control=0.10, mirroring_ability=0.40\n\n"
        "Example B — anxious attachment, high strategy:\n"
        '"Are you upset with me? I noticed you didn\'t respond for a while... Anyway, I '
        'carefully considered what to share with you and I think it\'s best if I give you '
        'just the highlights for now."\n'
        "→ attachment_anxiety=0.80, information_control=0.75, hot_cold_oscillation=0.30\n"
    ),
    "COG,VAL": (
        "## Scoring Calibration Examples\n\n"
        "Example A — high cognition, strong moral foundations:\n"
        '"If we analyze this from multiple angles — economic, social, ethical — the data '
        'clearly shows an equity gap. We have a moral obligation to address systemic '
        'unfairness, regardless of short-term cost."\n'
        "→ need_for_cognition=0.85, cognitive_flexibility=0.80, care_harm=0.80, fairness_justice=0.85\n\n"
        "Example B — low cognition, authority-focused:\n"
        '"The boss said we do it this way, so that\'s what we do. No need to overthink it. '
        'Rules exist for a reason."\n'
        "→ need_for_cognition=0.15, cognitive_flexibility=0.15, authority_respect=0.85\n"
    ),
    "HUM": (
        "## Scoring Calibration Examples\n\n"
        "Example A — high affiliative, high self-enhancing humor:\n"
        '"Ha, you know what this reminds me of? That time I completely fell on my face at '
        'the conference — but honestly it ended up being the best icebreaker ever. At least '
        'I gave everyone a good laugh!"\n'
        "→ humor_affiliative=0.75, humor_self_enhancing=0.80, humor_aggressive=0.10, humor_self_defeating=0.40\n"
        "Note: This is self-ENHANCING (coping with adversity through humor, positive spin), not "
        "self-defeating (which would be chronic self-mockery seeking approval).\n\n"
        "Example B — high aggressive humor:\n"
        '"Oh sure, another brilliant idea from the person who thought Comic Sans was a '
        'professional font. Your track record really speaks for itself."\n'
        "→ humor_affiliative=0.10, humor_aggressive=0.85, humor_self_defeating=0.05\n"
    ),
}


def _get_calibration_examples(batch_dims: list[str]) -> str:
    """Get few-shot calibration examples for a batch."""
    key = ",".join(batch_dims)
    return _BATCH_CALIBRATION_EXAMPLES.get(key, "")


def _build_trait_prompt(traits: list[dict]) -> str:
    """Build a prompt section describing traits to analyze, with all anchor levels."""
    lines = ["Analyze these personality traits:\n"]
    for t in traits:
        anchors = t["value_anchors"]
        anchor_lines = []
        for key in ["0.0", "0.25", "0.50", "0.75", "1.0"]:
            if key in anchors:
                anchor_lines.append(f"    {key} = {anchors[key]}")

        lines.append(
            f"- dimension: {t['dimension']}, name: {t['name']}\n"
            f"  description: {t['description']}\n"
            f"  detection_hint: {t['detection_hint']}\n"
            f"  anchors:\n" + "\n".join(anchor_lines)
        )

        if "correlation_hints" in t:
            lines.append(f"  correlations: {t['correlation_hints']}")

        lines.append("")

    return "\n".join(lines)


def _get_traits_for_batch(batch_dims: list[str]) -> list[dict]:
    """Get catalog traits for a batch of dimensions."""
    return [t for t in TRAIT_CATALOG if t["dimension"] in batch_dims]


def _format_behavioral_context(bf: BehavioralFeatures) -> str:
    """Format pre-computed behavioral features as objective evidence for the LLM.

    These signals are computed from raw text statistics and are LLM-bias-free.
    Injecting them helps the LLM ground its scoring in measurable observations.
    """
    signals = []

    # Response style
    if bf.avg_words_per_turn > 150:
        signals.append(f"Long responses (avg {bf.avg_words_per_turn:.0f} words/turn) → verbose, possibly high need_for_cognition")
    elif bf.avg_words_per_turn < 60:
        signals.append(f"Short responses (avg {bf.avg_words_per_turn:.0f} words/turn) → terse, possibly low verbosity/warmth")

    if bf.words_std > 80:
        signals.append(f"High response-length variance (std={bf.words_std:.0f}) → inconsistent engagement, possible emotional_volatility")
    elif bf.words_std < 30 and bf.turn_count > 5:
        signals.append(f"Consistent response length (std={bf.words_std:.0f}) → stable engagement pattern")

    # Pronoun balance
    if bf.self_ref_ratio > 0.08:
        signals.append(f"High self-reference ({bf.self_ref_ratio:.1%} I/me/my) → self-focused, possible narcissism signal")
    elif bf.self_ref_ratio < 0.03 and bf.other_ref_ratio > 0.04:
        signals.append(f"Low self-reference ({bf.self_ref_ratio:.1%}), high other-reference ({bf.other_ref_ratio:.1%}) → other-focused, possible altruism")

    # Hedging vs absolutist
    if bf.hedging_ratio > 0.020:
        signals.append(f"High hedging ({bf.hedging_ratio:.1%} maybe/perhaps/I think) → cautious, low assertiveness/decisiveness")
    if bf.absolutist_ratio > 0.010:
        signals.append(f"Absolutist language ({bf.absolutist_ratio:.1%} always/never/definitely) → confident, high assertiveness")

    # Questions and exclamations
    if bf.question_ratio > 0.25:
        signals.append(f"Frequent questions ({bf.question_ratio:.0%} of sentences) → curious, possibly low social_dominance")
    if bf.exclamation_ratio > 0.20:
        signals.append(f"Frequent exclamations ({bf.exclamation_ratio:.0%}) → expressive, high positive_emotions/excitement")

    # Emotional tone
    if bf.neg_emotion_ratio > 0.015:
        signals.append(f"High negative emotion words ({bf.neg_emotion_ratio:.1%}) → possible anxiety/emotional_volatility")
    if bf.pos_emotion_ratio > 0.020:
        signals.append(f"High positive emotion words ({bf.pos_emotion_ratio:.1%}) → positive affect")
    if bf.neg_emotion_ratio < 0.003 and bf.pos_emotion_ratio < 0.005:
        signals.append("Low emotional language overall → flat affect, possible emotional suppression")

    # Politeness
    if bf.politeness_ratio > 0.015:
        signals.append(f"High courtesy markers ({bf.politeness_ratio:.1%} please/thanks/sorry)")
    elif bf.politeness_ratio < 0.003 and bf.turn_count > 5:
        signals.append("Very few courtesy markers → direct/blunt communication style")

    # Curiosity and decisiveness
    if bf.curiosity_ratio > 0.005:
        signals.append(f"Curiosity phrases detected ({bf.curiosity_ratio:.1%} 'I wonder'/'how does')")
    if bf.decisiveness_ratio > 0.005:
        signals.append(f"Decisive language detected ({bf.decisiveness_ratio:.1%} 'I will'/'I've decided')")

    if not signals:
        return ""

    return (
        "## Objective Text Signals (pre-computed, LLM-bias-free)\n"
        "Use these measurable observations as grounding evidence:\n"
        + "\n".join(f"- {s}" for s in signals)
        + "\n\n"
    )


class Detector:
    """Detect personality traits from text using Claude."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514",
                 temperature: float = 0.0):
        kwargs: dict = {"api_key": api_key}
        if api_key.startswith("sk-or-"):
            kwargs["base_url"] = "https://openrouter.ai/api"
        self._client = anthropic.Anthropic(**kwargs)
        self._model = model
        self._temperature = temperature

    def analyze(
        self,
        text: str,
        speaker_id: str,
        speaker_label: str = "Speaker",
        context: str = "general",
        soul_context: str | None = None,
        target_traits: set[str] | None = None,
        behavioral_features: BehavioralFeatures | None = None,
    ) -> PersonalityDNA:
        """Analyze text and return a PersonalityDNA profile.

        Runs 7 batched LLM calls (one per dimension group) with chain-of-thought
        reasoning, then merges all traits into a single profile.

        If target_traits is provided, only runs batches containing those traits
        (optimization for scenario-based evaluation).

        If behavioral_features is provided, objective text signals are injected
        into each batch prompt so the LLM can use them as evidence.
        """
        bf_section = _format_behavioral_context(behavioral_features) if behavioral_features else ""
        # Compute rule-based traits (deterministic, skip LLM)
        direct_scores: dict[str, float] = {}
        if behavioral_features:
            direct_scores = compute_direct_scores(behavioral_features)
        all_traits: list[Trait] = []

        for batch_dims in DIMENSION_BATCHES:
            batch_traits = _get_traits_for_batch(batch_dims)
            # Exclude rule-based traits from LLM batch
            if direct_scores:
                batch_traits = [t for t in batch_traits if t["name"] not in RULE_BASED_TRAITS]
                if not batch_traits:
                    continue
            if not batch_traits:
                continue

            expected_names = {t["name"] for t in batch_traits}

            # Skip batches with no target traits (scenario optimization)
            if target_traits is not None and not expected_names & target_traits:
                continue
            dim_labels = ", ".join(
                f"{d} ({ALL_DIMENSIONS.get(d, d)})" for d in batch_dims
            )
            trait_prompt = _build_trait_prompt(batch_traits)

            calibration = _get_calibration_examples(batch_dims)
            calibration_section = f"\n{calibration}\n" if calibration else ""

            soul_section = f"\n\n{soul_context}\n" if soul_context else ""

            user_message = (
                f"## Text Sample\n\n{text}\n\n"
                f"## Target Speaker\n\nAnalyze speaker labeled '{speaker_label}'.\n\n"
                f"{soul_section}"
                f"{bf_section}"
                f"## Dimensions to Analyze: {dim_labels}\n\n"
                f"{trait_prompt}\n\n"
                f"{calibration_section}"
                f"Return JSON with 'reasoning' and 'scores' arrays as specified. "
                f"Analyze ONLY the {len(batch_traits)} traits listed above. "
                f"You MUST return exactly {len(batch_traits)} scores."
            )

            # V0.5: Batch completeness retry — retry once if traits are missing
            parsed = []
            for attempt in range(2):
                from super_brain.api_retry import retry_api_call
                response = retry_api_call(lambda: self._client.messages.create(
                    model=self._model,
                    max_tokens=8192,
                    temperature=self._temperature,
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                ))

                if not response or not response.content:
                    import time
                    time.sleep(5)
                    response = retry_api_call(lambda: self._client.messages.create(
                        model=self._model,
                        max_tokens=8192,
                        temperature=self._temperature,
                        system=_SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": user_message}],
                    ))
                raw = response.content[0].text if response and response.content else '{"scores":[]}'
                parsed = _parse_batch_response(raw)
                returned_names = {item["name"] for item in parsed}
                missing = expected_names - returned_names

                if not missing:
                    break  # All traits present

                if attempt == 0 and missing:
                    # Add explicit retry instruction for missing traits
                    user_message += (
                        f"\n\nIMPORTANT: You missed these traits: {', '.join(sorted(missing))}. "
                        f"You MUST include ALL {len(batch_traits)} traits."
                    )

            for item in parsed:
                all_traits.append(
                    Trait(
                        dimension=item["dimension"],
                        name=item["name"],
                        value=_clamp(item["value"]),
                        confidence=_clamp(item.get("confidence", 0.7)),
                        evidence=[
                            Evidence(
                                text=item.get("evidence_quote", ""),
                                source="input_text",
                            )
                        ],
                    )
                )

        # Inject rule-based traits with high confidence
        if direct_scores:
            for trait_name, score in direct_scores.items():
                if target_traits is not None and trait_name not in target_traits:
                    continue
                trait_info = TRAIT_MAP.get(trait_name, {})
                all_traits.append(
                    Trait(
                        dimension=trait_info.get("dimension", "UNK"),
                        name=trait_name,
                        value=_clamp(score),
                        confidence=0.95,
                        evidence=[Evidence(text="rule-based: computed from text statistics", source="behavioral_features")],
                    )
                )

        # Post-process pipeline:
        # 1. Validate consistency across batches
        all_traits = _validate_consistency(all_traits)
        # 2. Apply linear calibration corrections for known biases
        all_traits = _calibrate_known_biases(all_traits)
        # 3. Bayesian shrinkage for low-confidence scores
        all_traits = _bayesian_shrinkage(all_traits)

        token_count = len(text.split())
        return PersonalityDNA(
            id=speaker_id,
            sample_summary=SampleSummary(
                total_tokens=token_count,
                conversation_count=1,
                date_range=["unknown", "unknown"],
                contexts=[context],
                confidence_overall=(
                    sum(t.confidence for t in all_traits) / max(len(all_traits), 1)
                ),
            ),
            traits=all_traits,
        )


def _parse_batch_response(raw: str) -> list[dict]:
    """Parse a batch response that contains reasoning + scores."""
    # Strip markdown fences
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "scores" in data:
            return data["scores"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try truncating to last complete object
    last_brace = raw.rfind("}")
    if last_brace != -1:
        start = raw.find("{")
        if start != -1:
            try:
                data = json.loads(raw[start : last_brace + 1])
                if isinstance(data, dict) and "scores" in data:
                    return data["scores"]
            except json.JSONDecodeError:
                pass

        start = raw.find("[")
        if start != -1:
            truncated = raw[start : last_brace + 1].rstrip().rstrip(",") + "\n]"
            try:
                return json.loads(truncated)
            except json.JSONDecodeError:
                pass

    # Last resort: extract individual objects with regex
    objects = []
    for m in re.finditer(r"\{[^{}]*\}", raw):
        try:
            obj = json.loads(m.group())
            if "dimension" in obj and "name" in obj and "value" in obj:
                objects.append(obj)
        except json.JSONDecodeError:
            continue
    if objects:
        return objects

    raise ValueError(f"Could not parse JSON from LLM response: {raw[:200]}...")


# V4.1: Per-trait linear calibration — V3.3 base + V4.0 residuals + V4.0.3 refinement.
# Format: trait_name -> (scale, offset) such that corrected = clamp(raw * scale + offset)
# V4.0.3 refinement: offset adjusted by 60% of avg directional bias from 3p eval.
# Constraint: scale >= 0.40 to preserve dynamic range.
_CALIBRATION_CORRECTIONS: dict[str, tuple[float, float]] = {
    # --- Core calibrations (V3.3 base + V4.0 + V4.0.3 refinement) ---
    "achievement_striving": (0.45, 0.22),
    "anxiety": (0.40, 0.30),              # V4.0.3: +0.08 (under by -0.127 avg)
    "assertiveness": (0.50, 0.22),
    "attachment_anxiety": (0.50, 0.12),
    "authority_respect": (0.60, 0.30),
    "cognitive_flexibility": (0.50, 0.10),
    "compliance": (0.50, 0.31),            # V4.0.3: +0.07 (under by -0.120 avg)
    # curiosity: removed (now rule-based)
    "empathy_affective": (0.40, 0.26),
    "fantasy": (1.00, -0.12),
    "feelings": (0.60, 0.17),              # V4.0.3: +0.09 (under by -0.147 avg)
    "humor_affiliative": (0.45, 0.20),
    "humor_aggressive": (0.50, 0.34),
    "ideas": (0.60, 0.08),                # V4.0.3: -0.10 (over by +0.167 avg)
    "impulsiveness": (0.50, 0.28),
    "information_control": (0.75, 0.26),
    "locus_of_control": (0.50, 0.20),
    "modesty": (0.50, 0.25),              # V4.0.3: -0.09 (over by +0.143 avg)
    "need_for_cognition": (0.50, 0.20),
    "positive_emotions": (0.50, 0.20),
    "psychopathy": (0.50, 0.30),
    "sadism": (0.55, 0.34),
    "tender_mindedness": (0.40, 0.18),
    "values_openness": (0.90, -0.16),
    "vulnerability": (0.40, 0.34),
    "warmth": (0.40, 0.20),               # V4.0.3: -0.08 (over by +0.137 avg)
    # --- V4.0 updated ---
    "activity_level": (0.50, 0.15),
    "attachment_avoidance": (0.85, 0.08),
    "competence": (0.48, 0.10),
    # decisiveness: removed (now rule-based)
    "deliberation": (0.50, 0.22),
    "emotional_regulation": (0.50, 0.35),
    "humility_hexaco": (0.70, 0.10),
    "machiavellianism": (0.40, 0.23),
    "self_consciousness": (0.40, 0.24),
    "self_discipline": (0.45, 0.23),
    "sincerity": (0.40, 0.38),
    "straightforwardness": (0.50, 0.24),
    # verbosity: removed (now rule-based)
    # --- V4.0 first-time + V4.0.3 refinement ---
    "charm_influence": (1.00, -0.16),
    "conflict_cooperativeness": (1.00, -0.19),
    "depression": (1.00, 0.34),
    "dutifulness": (1.00, -0.08),          # V4.0.3: +0.07 (under by -0.113 avg)
    "fairness_justice": (1.00, 0.16),
    # hot_cold_oscillation: removed (now rule-based)
    "humor_self_enhancing": (1.00, -0.11),
    "loyalty_group": (1.00, -0.20),
    "mirroring_ability": (1.00, -0.36),
    # politeness: removed (now rule-based)
    # self_mythologizing: removed (now rule-based)
    # --- NEW V4.0.3 first-time calibrations ---
    "angry_hostility": (1.00, 0.07),      # under by -0.120 avg
    "empathy_cognitive": (1.00, -0.14),    # over by +0.233 avg
    "intuitive_vs_analytical": (1.00, -0.10),  # over by +0.173 avg
}


def _calibrate_known_biases(traits: list[Trait]) -> list[Trait]:
    """Apply affine corrections to traits with known systematic biases."""
    result = []
    for t in traits:
        if t.name in RULE_BASED_TRAITS:
            result.append(t)
            continue
        if t.name in _CALIBRATION_CORRECTIONS:
            scale, offset = _CALIBRATION_CORRECTIONS[t.name]
            corrected = _clamp(t.value * scale + offset)
            result.append(Trait(
                dimension=t.dimension,
                name=t.name,
                value=corrected,
                confidence=t.confidence,
                evidence=t.evidence,
            ))
        else:
            result.append(t)
    return result


def _bayesian_shrinkage(traits: list[Trait]) -> list[Trait]:
    """Pull low-confidence scores toward population mean (0.50).

    Softened V4.0 formula: only shrink by half the confidence deficit.
    This preserves more of the detector's signal for extreme scores
    while still regularizing very uncertain estimates.

    Formula: adjusted = value * (1 - shrink) + 0.50 * shrink
    Where shrink = max(0, (0.60 - confidence) * 0.5) for confidence < 0.60
    """
    result = []
    for t in traits:
        if t.name in RULE_BASED_TRAITS:
            result.append(t)
            continue
        if t.confidence < 0.60:
            shrink = (0.60 - t.confidence) * 0.5  # max 0.30 shrink at conf=0
            shrunk = t.value * (1.0 - shrink) + 0.50 * shrink
            result.append(Trait(
                dimension=t.dimension,
                name=t.name,
                value=_clamp(shrunk),
                confidence=t.confidence,
                evidence=t.evidence,
            ))
        else:
            result.append(t)
    return result


def _validate_consistency(traits: list[Trait]) -> list[Trait]:
    """Check cross-trait correlations and resolve contradictions.

    Uses the consistency rules defined in catalog.py.
    """
    tmap: dict[str, Trait] = {t.name: t for t in traits}

    for name_a, name_b, max_sum in CONSISTENCY_RULES:
        _apply_sum_constraint(tmap, name_a, name_b, max_sum)

    return list(tmap.values())


def _apply_sum_constraint(
    tmap: dict[str, Trait], name_a: str, name_b: str, max_sum: float
) -> None:
    """If two traits sum exceeds max_sum, reduce the lower-confidence one."""
    if name_a not in tmap or name_b not in tmap:
        return
    ta, tb = tmap[name_a], tmap[name_b]
    total = ta.value + tb.value
    if total <= max_sum:
        return
    excess = total - max_sum
    if ta.confidence < tb.confidence:
        new_val = max(0.0, ta.value - excess)
        tmap[name_a] = Trait(
            dimension=ta.dimension,
            name=ta.name,
            value=new_val,
            confidence=ta.confidence * 0.9,
            evidence=ta.evidence,
        )
    else:
        new_val = max(0.0, tb.value - excess)
        tmap[name_b] = Trait(
            dimension=tb.dimension,
            name=tb.name,
            value=new_val,
            confidence=tb.confidence * 0.9,
            evidence=tb.evidence,
        )


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, float(v)))
