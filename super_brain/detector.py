"""LLM Detector: Extract personality traits from conversation text."""

from __future__ import annotations

import json
import re

import anthropic

from super_brain.catalog import (
    TRAIT_CATALOG,
    ALL_DIMENSIONS,
    CONSISTENCY_RULES,
    TRAIT_MAP,
)
from super_brain.models import PersonalityDNA, Trait, Evidence, SampleSummary


# ── Batch definitions: 7 batches grouping related dimensions ────────────────

DIMENSION_BATCHES: list[list[str]] = [
    ["OPN", "CON"],       # Batch 1: Openness + Conscientiousness (12 traits)
    ["EXT", "AGR"],       # Batch 2: Extraversion + Agreeableness (12 traits)
    ["NEU", "HON"],       # Batch 3: Neuroticism + Honesty-Humility (10 traits)
    ["DRK", "EMO"],       # Batch 4: Dark Traits + Emotional Architecture (10 traits)
    ["SOC", "STR"],       # Batch 5: Social Dynamics + Interpersonal Strategy (10 traits)
    ["COG", "VAL"],       # Batch 6: Cognitive Style + Values (8 traits)
    ["HUM"],              # Batch 7: Humor Style (4 traits) + cross-validation
]


_SYSTEM_PROMPT = """\
You are a personality analyst specializing in detecting personality traits from \
natural conversation. You will be given a CONVERSATION between two people. \
Focus on the target speaker and analyze their personality based on:
- What they say (content, topics they gravitate toward or avoid)
- How they say it (tone, word choice, sentence structure, hedging, directness)
- How they REACT to the other person (agreement/disagreement, emotional responsiveness)
- Conversational dynamics (turn length, question-asking, topic steering)
- What is ABSENT (topics avoided, emotions not expressed, perspectives not considered)

For EACH trait:
1. List specific text observations (quotes, patterns, behavioral indicators).
2. Rate this person RELATIVE TO AN AVERAGE PERSON. Ask: "Compared to a typical person \
in casual conversation, does this speaker show MORE or LESS of this trait?" \
Score 0.50 = exactly average. Above 0.50 = more than average. Below 0.50 = less than average.

Return ONLY valid JSON with this structure:
{
  "reasoning": [
    {"trait": "<name>", "observations": ["observation 1", "observation 2", ...]}
  ],
  "scores": [
    {
      "dimension": "<DIM>",
      "name": "<trait_name>",
      "value": <float 0.0-1.0>,
      "confidence": <float 0.0-1.0>,
      "evidence_quote": "<short quote from text>"
    }
  ]
}

Guidelines:
- value: your assessment on the 0.0-1.0 scale, using the provided anchors as calibration points
- confidence: how certain you are (lower if insufficient evidence in the text)
- evidence_quote: a short direct quote from the text supporting your score

UNIVERSAL BASELINE CALIBRATION (CRITICAL — READ CAREFULLY):
- For ANY trait where you lack strong evidence, your default score MUST be 0.40-0.55. \
Most people cluster in the 0.30-0.70 range.
- HARD RULES for extreme scores:
  * Scores of 0.05-0.15: FORBIDDEN unless you have 3+ clear counter-observations. \
    If you are tempted to score 0.10 or 0.15, you MUST ask: "Did this person explicitly \
    demonstrate the OPPOSITE of this trait?" If not, score 0.35-0.45 instead.
  * Scores of 0.85-0.95: FORBIDDEN unless you have 3+ strong supporting observations. \
    If you are tempted to score 0.85+, you MUST ask: "Did this person demonstrate this \
    trait in multiple distinct ways?" If not, score 0.55-0.65 instead.
- A single observation moves a trait by +/-0.10-0.15 from baseline, not to an extreme.
- The range 0.25-0.75 should cover 90%+ of your scores. Extreme scores (<0.20 or >0.80) \
should appear for fewer than 5 traits out of the full set.

CONVERSATION CONTEXT AWARENESS:
- This text comes from CASUAL CONVERSATION. Critical implications:
  * Friendly tone in casual chat is the SOCIAL NORM, not evidence of high warmth/charm. \
Score warmth and charm_influence at 0.45-0.55 unless the person is DISTINCTLY warmer or colder.
  * Composure in casual chat is EXPECTED, not evidence of high emotional_regulation. \
Score emotional_regulation at 0.45-0.55 unless you see specific regulation effort or dysregulation.
  * Polite small talk is NOT evidence of EXTREME trust (0.75+), compliance (0.75+), or \
cooperativeness (0.75+). However, it IS consistent with MODERATE levels (0.40-0.55). \
Score trust/compliance/modesty at 0.40-0.55 as baseline. Only score BELOW 0.30 when you \
see ACTIVE distrust, confrontation, or immodesty. Only score ABOVE 0.65 when you see \
DISTINCTIVE prosocial signals beyond normal social politeness.

LLM-GENERATED TEXT BIAS CORRECTION (CRITICAL):
- ALL text in this conversation will sound articulate and well-structured because it's \
generated text. You MUST heavily discount:
  * Articulateness → NOT evidence of high need_for_cognition or analytical thinking
  * Warmth → NOT evidence of high charm_influence or warmth trait
  * Self-deprecating humor → NOT evidence of high humor_self_enhancing (that's self-DEFEATING)
  * Finding humor in stories → NOT evidence of high humor_self_enhancing unless they \
    EXPLICITLY reframe adversity positively ("at least I learned something")
  * Matching conversational style → NOT evidence of high mirroring_ability
- need_for_cognition: Default to 0.40-0.50. LLM text sounds articulate regardless — discount this. \
Only score >0.60 if the person UNPROMPTED initiates intellectual analysis or says they enjoy thinking. \
Only score <0.30 if they actively avoid complexity ("I don't overthink things").
- intuitive_vs_analytical: Score at 0.45-0.55 unless you see clear "I feel like" (intuitive) \
vs "the data shows" (analytical) patterns. Structured sentences = LLM artifact, not analytical style.
- humor_self_enhancing: This is about using humor to COPE with PERSONAL adversity with a POSITIVE \
reframe ("at least I learned X", "best mistake I ever made"). NOT about telling funny stories, \
being witty, or self-deprecation. Default to 0.40-0.50 unless you see specific positive-reframe coping.

HUMILITY CALIBRATION:
- humility_hexaco is about feeling entitled vs. feeling ordinary. It is NOT the opposite \
of confidence or assertiveness. A person can be confident AND humble. Score at 0.45-0.55 \
as baseline unless you see EXPLICIT entitlement ("I deserve...", demanding special treatment) \
or explicit humility ("I'm nothing special", refusing praise).

DARK TRAIT CALIBRATION (CRITICAL):
- Dark traits exist on a SPECTRUM. Most people have non-zero levels (population mean ~0.35).
- In casual conversation, moderate dark traits (0.3-0.6) manifest as SUBTLE patterns:
  * Moderate narcissism: steering conversation back to self, implicit self-superiority, \
    competitive framing, lack of anxiety/tentative words, not asking follow-up questions
  * Moderate machiavellianism: strategic vagueness, cynical observations about people, \
    reading the room before sharing, higher use of "they" vs "we", calculating tone
  * Moderate psychopathy: emotional flatness relative to topic, pragmatic/transactional \
    framing, cause-and-effect language about people ("because..."), material focus over \
    emotional focus, matter-of-fact about others' problems
  * Moderate sadism: enjoying gossip about others' misfortune, edgy humor about failure, \
    "they had it coming" attitudes, amusement at discomfort
- Score 0.30-0.50 when you see subtle signs. Score <0.15 ONLY when the person actively \
demonstrates consistent warmth, empathy, and selflessness throughout.
- NOTE: First-person pronoun use ("I", "me") does NOT correlate with narcissism. \
Narcissism correlates with competitive language, lack of anxiety words, and second-person \
challenges.

HUMOR STYLE CALIBRATION:
- Distinguish: self-enhancing = positive spin on adversity; aggressive = targeting others; \
self-defeating = chronic self-mockery; affiliative = inclusive bonding humor.
- humor_affiliative: Score 0.35-0.45 baseline. LLM text sounds humorous naturally — discount. \
Only >0.60 if humor is clearly a PRIMARY social strategy. <0.30 if serious and dry.
- humor_self_enhancing: Score 0.40-0.50 baseline. Requires POSITIVE REFRAME of PERSONAL \
adversity ("at least I learned X"). NOT just being witty. >0.60 needs 2+ examples. \
If person complains WITHOUT silver lining, score 0.25-0.35.

CONSCIENTIOUSNESS CALIBRATION (CRITICAL):
- In casual conversation, conscientiousness shows through HOW someone talks about tasks, plans, \
and responsibilities — not through the conversational tone itself.
- self_discipline: Look for mentions of completing tasks, sticking to routines, following through \
on commitments vs. mentioning procrastination, unfinished projects, getting distracted.
- order: Look for structured thinking (listing things, step-by-step reasoning) vs. jumping \
between topics randomly, mentioning messy spaces or lost items.
- achievement_striving: Look for goal-talk, ambition mentions, competitive drive vs. \
contentment with status quo, lack of drive.
- deliberation: Look for careful reasoning ("let me think...", weighing options) vs. \
impulsive decisions ("let's just do it", "I didn't think about it").
- Casual, relaxed tone does NOT mean low conscientiousness. Many conscientious people chat \
casually. Score at 0.45-0.55 baseline unless you hear specific CONTENT about habits and work ethic.

ADDITIONAL TRAIT-SPECIFIC CALIBRATION:
- mirroring_ability: Score at 0.30-0.40 baseline. In casual conversation, everyone naturally \
matches their partner's tone somewhat — this is BASIC social behavior, NOT mirroring ability. \
Only score >0.55 if you see DELIBERATE, dramatic style shifts to match the other person \
(e.g., adopting their slang, matching their energy level shift). \
Score <0.25 only if the person maintains a rigidly different style regardless of their partner.
- self_consciousness: Score at 0.40-0.50 baseline. Social ease in casual chat is NORMAL. \
Only score <0.25 if person is truly socially unaware. Score >0.55 if person explicitly worries \
about how they're perceived.
- tender_mindedness: High tender_mindedness means EMOTIONAL compassion reactions to suffering, \
not just being polite. Look for "that's so sad", visceral sympathy, protective instincts.
- emotional_volatility: Look for ACTUAL tone shifts between different messages in the conversation. \
If tone is consistent throughout, score 0.35-0.45. If tone varies significantly, score 0.55-0.70.
- information_control: Look for ACTIVE deflection of personal questions, strategic vagueness, \
redirecting topics. Normal conversational flow is NOT evidence of low info control. \
Score 0.40-0.50 baseline.
- charm_influence: Score 0.40-0.50 baseline. Friendly chat = normal, NOT high charm. \
>0.60 only with ACTIVE persuasion or unusual magnetism.
- modesty: Score 0.35-0.45 baseline. Self-deprecating humor and casual tone = LLM artifact, \
NOT modesty. >0.60 only with EXPLICIT self-minimizing. <0.30 if boasting.
- straightforwardness: Score 0.40-0.50 baseline. Clear writing = LLM artifact. \
>0.65 only with BLUNT confrontation or uncomfortable truths. <0.35 if diplomatic/indirect.
- greed_avoidance: Score 0.40-0.50 baseline. Not mentioning money = neutral. \
>0.60 only with explicit anti-materialism. <0.35 if status-conscious.
- sincerity: Score 0.45-0.55 baseline. Being open ≠ sincere. >0.70 only with refusing \
to manipulate. <0.35 if using strategic flattery.
- empathy_cognitive: Score 0.40-0.50 baseline. Acknowledging stories = basic conversation. \
>0.60 only with precise emotional labeling.
- fairness: Score 0.40-0.50 baseline. >0.60 only with explicit fairness principles.

GENERAL CALIBRATION:
- Be precise. Use the anchor descriptions to calibrate your scores.
- A score of 0.25 should match the 0.25 anchor, 0.50 the 0.50 anchor, etc.
- Mid-range scores (0.35-0.65) are valid and often correct. Do not default to extremes.
- Personality traits are about PATTERNS, not single instances.
- Distinguish between the TRAIT and its expression context.
- Trust your observations over stereotypes.

FINAL CHECK: Before submitting, scan your scores. If more than 5 traits are <0.20 or >0.80, \
you are likely over-confident. Redistribute extreme scores toward the 0.30-0.70 range.
"""


# ── Few-shot calibration examples per batch ─────────────────────────────────

_BATCH_CALIBRATION_EXAMPLES: dict[str, str] = {
    "OPN,CON": (
        "## Scoring Calibration Examples\n\n"
        "Example A — high openness, low conscientiousness:\n"
        '"What if we just threw out the whole system and started fresh? I know it sounds crazy '
        'but imagine — we could build something entirely new. I\'ve been reading about this '
        'philosophy of radical impermanence and honestly it\'s changing how I see everything..."\n'
        "→ fantasy=0.80, ideas=0.85, values_openness=0.75, order=0.15, deliberation=0.20\n\n"
        "Example B — low openness, high conscientiousness:\n"
        '"We need to follow the established process. Step 1: review the checklist. Step 2: '
        'document all findings. Step 3: submit by the deadline. I\'ve been tracking our '
        'progress daily and we\'re at 73% completion."\n'
        "→ fantasy=0.10, ideas=0.20, order=0.90, self_discipline=0.85, achievement_striving=0.80\n"
    ),
    "EXT,AGR": (
        "## Scoring Calibration Examples\n\n"
        "Example A — high extraversion, high agreeableness:\n"
        '"Oh my gosh, everyone needs to come to this event! It\'s going to be amazing! '
        'And honestly, I think your idea was even better than mine — you should definitely '
        'lead the presentation. I\'ll support however I can!"\n'
        "→ warmth=0.85, positive_emotions=0.90, assertiveness=0.65, modesty=0.75, altruism=0.80\n\n"
        "Example B — low extraversion, low agreeableness:\n"
        '"I\'d rather work on this alone. Your approach has several fundamental flaws '
        'that I\'ve identified. I don\'t need input on this — I know what needs to be done."\n'
        "→ warmth=0.15, gregariousness=0.10, trust=0.20, compliance=0.10, modesty=0.15\n"
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
        "→ narcissism=0.10, psychopathy=0.05, emotional_granularity=0.85, empathy_affective=0.85\n\n"
        "Example B — high dark traits, low emotional depth:\n"
        '"People are predictable. Give them what they want to hear and they\'ll do whatever '
        'you need. It\'s not personal — it\'s just how things work. I don\'t see why everyone '
        'gets so emotional about it."\n'
        "→ machiavellianism=0.80, psychopathy=0.70, emotional_granularity=0.15, empathy_affective=0.10\n"
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


class Detector:
    """Detect personality traits from text using Claude."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        kwargs: dict = {"api_key": api_key}
        if api_key.startswith("sk-or-"):
            kwargs["base_url"] = "https://openrouter.ai/api"
        self._client = anthropic.Anthropic(**kwargs)
        self._model = model

    def analyze(
        self,
        text: str,
        speaker_id: str,
        speaker_label: str = "Speaker",
        context: str = "general",
        soul_context: str | None = None,
    ) -> PersonalityDNA:
        """Analyze text and return a PersonalityDNA profile.

        Runs 7 batched LLM calls (one per dimension group) with chain-of-thought
        reasoning, then merges all traits into a single profile.
        """
        all_traits: list[Trait] = []

        for batch_dims in DIMENSION_BATCHES:
            batch_traits = _get_traits_for_batch(batch_dims)
            if not batch_traits:
                continue

            expected_names = {t["name"] for t in batch_traits}
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
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=8192,
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                )

                raw = response.content[0].text
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

        # Post-process: validate consistency across batches
        all_traits = _validate_consistency(all_traits)

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


# Known detector biases from V0.9-V1.3 eval data.
# Format: trait_name -> (scale, offset) such that corrected = clamp(raw * scale + offset)
# These compress over-detected traits toward baseline and expand under-detected ones.
_CALIBRATION_CORRECTIONS: dict[str, tuple[float, float]] = {
    # Over-detected (detector gives too-high scores for low true values)
    "humor_self_enhancing": (0.65, 0.10),     # compress: 0.70→0.56, 0.40→0.36
    "charm_influence": (0.70, 0.10),           # compress: 0.70→0.59, 0.40→0.38
    "mirroring_ability": (0.60, 0.12),         # compress: 0.65→0.51, 0.35→0.33
    "humor_affiliative": (0.70, 0.08),         # compress: 0.70→0.57, 0.40→0.36
    "cognitive_flexibility": (0.75, 0.10),     # compress: 0.65→0.59, 0.25→0.29
    "fairness": (0.80, 0.05),                  # compress: 0.70→0.61, 0.40→0.37
}


def _calibrate_known_biases(traits: list[Trait]) -> list[Trait]:
    """Apply affine corrections to traits with known systematic biases."""
    result = []
    for t in traits:
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

    When the detector is uncertain (low confidence), extreme scores are
    unreliable. This shrinks them toward the prior (population mean = 0.50):
        adjusted = raw * confidence + 0.50 * (1 - confidence)

    High-confidence scores remain nearly unchanged. Low-confidence scores
    are pulled toward baseline, reducing the impact of incorrect extremes.
    """
    result = []
    for t in traits:
        # Only apply shrinkage when confidence < 0.70 (i.e., detector is unsure)
        if t.confidence < 0.70:
            shrunk = t.value * t.confidence + 0.50 * (1.0 - t.confidence)
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
