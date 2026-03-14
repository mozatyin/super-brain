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
    ["OPN", "CON"],       # Batch 1: Openness + Conscientiousness (14 traits, +curiosity +decisiveness)
    ["EXT", "AGR"],       # Batch 2: Extraversion + Agreeableness (14 traits, +verbosity +politeness)
    ["NEU", "HON"],       # Batch 3: Neuroticism + Honesty-Humility (10 traits)
    ["DRK", "EMO"],       # Batch 4: Dark Traits + Emotional Architecture (9 traits, -2 removed +optimism)
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
should appear for fewer than 5 traits out of the full 69-trait set.

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
- intuitive_vs_analytical: Score at 0.45-0.55 baseline. LLM text always sounds structured/analytical — \
HEAVILY discount this. Only score >0.60 (analytical) if person explicitly uses data/evidence/frameworks \
to reason ("after weighing the options", "the numbers show"). Only score <0.40 (intuitive) if person \
explicitly relies on gut feelings ("I just know", "it felt right", "my instinct says"). Story-based \
explanations with feelings = intuitive (0.35-0.45). Criteria-based explanations with evidence = analytical (0.55-0.65).
- cognitive_flexibility: Score at 0.45-0.55 baseline. In casual chat, people rarely demonstrate \
perspective-switching. Only score >0.60 if person EXPLICITLY considers opposing viewpoints or changes \
position when challenged ("that's a good point, I hadn't thought of it that way"). Only score <0.35 \
if person shows rigid black-and-white thinking ("it's always like this", refuses to consider alternatives). \
Normal agreeableness in conversation is NOT cognitive flexibility.
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
contentment with status quo, lack of drive. Score 0.40-0.50 baseline.
- deliberation: Score 0.40-0.50 baseline. LLM text ALWAYS sounds deliberate and thoughtful — \
HEAVILY discount this. Only score >0.60 if person EXPLICITLY mentions weighing options or careful \
planning ("I thought long and hard about", "after considering all the factors"). Only score <0.35 \
if person mentions impulsive decisions ("I just went for it without thinking").
- dutifulness: Score 0.40-0.50 baseline. Being cooperative in casual conversation is NOT evidence \
of dutifulness. Only score >0.60 if person mentions obligation, duty, following rules, or keeping \
promises as important values. Only score <0.30 if person explicitly dismisses obligations or rules.
- competence: Score 0.40-0.50 baseline. Sounding articulate = LLM artifact, NOT competence. \
Only score >0.60 if person describes executing tasks skillfully or mentions track record of success. \
Only score <0.35 if person expresses self-doubt about abilities or mentions failures.
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
- verbosity: Directly measurable from response length. High = consistently long responses with \
tangents and examples. Low = terse, direct answers. One of the most objective traits — trust \
the text length. Score 0.40-0.50 baseline for moderate-length responses.
- curiosity: Look for question-asking frequency AND topic exploration. Asking questions because \
the conversation requires it ≠ curiosity. Score >0.60 only if person UNPROMPTED explores new \
topics or asks 'I wonder' type questions. Score 0.40-0.50 baseline.
- politeness: Directly countable: please/thank you/sorry frequency. Score based on courtesy \
marker density. Some cultural contexts use more courtesy markers — don't over-interpret. \
Score 0.40-0.50 baseline for normal conversational courtesy.
- optimism: Ratio of positive to negative framing. Solution-focus vs problem-focus. 'At least' \
and 'on the bright side' = high optimism signals. Persistent dwelling on problems without \
positive reframing = low. Score 0.40-0.50 baseline for neutral framing.
- decisiveness: Inverse of hedging. 'I will' and 'let's do it' = decisive. 'Maybe' and 'I'm \
not sure' = indecisive. Score 0.40-0.50 baseline. Note: deliberation (thinking carefully) \
is NOT the same as indecisiveness (unable to choose).

GENERAL CALIBRATION:
- Be precise. Use the anchor descriptions to calibrate your scores.
- A score of 0.25 should match the 0.25 anchor, 0.50 the 0.50 anchor, etc.
- Mid-range scores (0.35-0.65) are valid and often correct. Do not default to extremes.
- Personality traits are about PATTERNS, not single instances.
- Distinguish between the TRAIT and its expression context.
- Trust your observations over stereotypes.

BIAS ALERT — UNDER-DETECTED TRAITS (adjust upward if ANY signal present):
Research shows personality raters systematically UNDER-RATE these traits in text:
- emotional_volatility: If you see ANY tone shift between messages, score ≥0.40
- angry_hostility: If ANY irritation, frustration, or blame language appears, score ≥0.35
- modesty: If person deflects praise or minimizes achievements, score ≥0.55
- social_dominance: If person steers topics, gives unsolicited advice, or speaks authoritatively, score ≥0.50
- humor_self_enhancing: If person reframes adversity positively ("at least..."), score ≥0.50

CONTRASTIVE NOTES — COMMONLY CONFUSED TRAIT PAIRS:
- compliance ≠ modesty: Compliance = yielding in conflict; modesty = not boasting. Different.
- assertiveness ≠ social_dominance: Assertiveness = speaking up; social_dominance = seeking hierarchical control.
- trust ≠ compliance: Trust = believing others are honest; compliance = avoiding confrontation.
- empathy_cognitive ≠ empathy_affective: Cognitive = understanding emotions; affective = FEELING them yourself.
- deliberation ≠ decisiveness: Deliberation = thinking carefully; decisiveness = choosing quickly. They CAN co-exist.
- anxiety ≠ vulnerability: Anxiety = anticipatory worry; vulnerability = crumbling under actual pressure.
- narcissism ≠ assertiveness: Narcissism = entitlement and grandiosity; assertiveness = simply speaking up.
- information_control ≠ introversion: Info control = strategic concealment; introversion = less talking.

USE THE FULL 0.0-1.0 RANGE:
People truly vary. Some people ARE extremely low (0.10) on traits and some ARE extremely high (0.90).
- 0.00-0.10: ABSENT — the person actively demonstrates the OPPOSITE of this trait
- 0.15-0.25: Very low — clear counter-evidence
- 0.30-0.40: Below average — some counter-evidence
- 0.45-0.55: Average — no strong evidence either way
- 0.60-0.70: Above average — clear positive evidence
- 0.75-0.85: High — strong, consistent evidence from multiple observations
- 0.90-1.00: Extreme — every relevant moment shows this trait

FINAL CHECK: Before submitting, scan your scores. If ALL scores cluster in 0.35-0.65, you are being \
too conservative. At least 30% of scores should be outside this range for a distinctive personality.
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
        target_traits: set[str] | None = None,
    ) -> PersonalityDNA:
        """Analyze text and return a PersonalityDNA profile.

        Runs 7 batched LLM calls (one per dimension group) with chain-of-thought
        reasoning, then merges all traits into a single profile.

        If target_traits is provided, only runs batches containing those traits
        (optimization for scenario-based evaluation).
        """
        all_traits: list[Trait] = []

        for batch_dims in DIMENSION_BATCHES:
            batch_traits = _get_traits_for_batch(batch_dims)
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
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                ))

                if not response or not response.content:
                    import time
                    time.sleep(5)
                    response = retry_api_call(lambda: self._client.messages.create(
                        model=self._model,
                        max_tokens=8192,
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


# V3.3: Per-trait linear calibration via cross-validated grid search on 10-profile eval.
# Format: trait_name -> (scale, offset) such that corrected = clamp(raw * scale + offset)
# Traits from V3.2.1 adjusted with 10p data; new traits added where 7+/10 consistent bias.
# Constraint: a >= 0.40 to preserve dynamic range.
_CALIBRATION_CORRECTIONS: dict[str, tuple[float, float]] = {
    # --- V3.2 new traits (kept from V3.2.1, minor adjustments) ---
    "verbosity": (1.00, -0.40),          # over-detected (kept — 10p confirms)
    "curiosity": (0.80, 0.04),           # slight over-detection (kept)
    "decisiveness": (1.10, -0.28),        # V3.3: now over-detected (+0.144 bias, 8/10)
    # --- Difficult traits (adjusted with 10p data) ---
    "self_consciousness": (0.40, 0.12),  # over-detected (kept — 10p confirms)
    "information_control": (0.75, 0.26), # under-detected (kept)
    "competence": (0.48, 0.26),          # composed: still over-detected +0.26 at 10p
    # --- V3.2.1 calibrations adjusted for 10p ---
    "feelings": (0.60, 0.08),            # composed: was over-correcting, 10p bias -0.09
    "anxiety": (0.40, 0.22),             # kept — 10p confirms near-zero residual
    "attachment_avoidance": (0.85, 0.28),# kept — still some variance but direction ok
    "straightforwardness": (0.50, -0.02),# kept — mixed results across profiles
    "sadism": (0.55, 0.34),              # kept
    "sincerity": (0.40, 0.14),           # kept
    "modesty": (0.50, 0.34),             # V3.3: under-detected (-0.142 bias, 7/10)
    "humility_hexaco": (0.70, -0.02),    # kept — 10p variance too high for reliable adjustment
    "self_discipline": (0.45, 0.36),     # kept — 10p confirms +0.10 residual
    "empathy_affective": (0.40, 0.26),   # kept
    "tender_mindedness": (0.40, 0.18),   # kept — 10p shows -0.056, adjustment marginal
    "fantasy": (1.00, -0.12),            # kept
    "ideas": (0.60, 0.18),               # kept
    "machiavellianism": (0.40, 0.36),    # kept
    "warmth": (0.40, 0.28),              # kept
    "humor_affiliative": (0.45, 0.20),   # kept
    "vulnerability": (0.40, 0.34),       # kept
    # --- NEW calibrations from 10-profile eval (7+/10 consistent bias) ---
    "deliberation": (0.50, 0.10),        # 10/10 over-detected (+0.207 avg bias)
    "positive_emotions": (0.50, 0.20),   # 8/10 over-detected (+0.096 avg bias)
    "compliance": (0.50, 0.24),          # 8/10 over-detected (+0.062 avg bias)
    "humor_aggressive": (0.50, 0.34),    # 8/10 under-detected (-0.185 avg bias)
    "authority_respect": (0.60, 0.30),   # 10p under-detected, 3p over-detected — moderate
    "locus_of_control": (0.50, 0.20),    # 8/10 over-detected (+0.158 avg bias)
    "need_for_cognition": (0.50, 0.20),  # 8/10 over-detected (+0.155 avg bias)
    "emotional_regulation": (0.50, 0.16),# 8/10 over-detected (+0.115 avg bias)
    "assertiveness": (0.50, 0.22),       # 8/10 over-detected (+0.019 avg bias)
    "impulsiveness": (0.50, 0.28),       # 7/10 under-detected (-0.087 avg bias)
    "psychopathy": (0.50, 0.30),         # 7/10 under-detected (-0.135 avg bias)
    "values_openness": (0.90, -0.16),    # 8/10 over-detected (+0.160 avg bias)
    "activity_level": (0.50, 0.30),      # 7/10 over-detected (+0.084 avg bias)
    "attachment_anxiety": (0.50, 0.12),  # 7/10 over-detected (+0.091 avg bias)
    "achievement_striving": (0.45, 0.22),# 8/10 over-detected (+0.112 avg bias)
    "cognitive_flexibility": (0.50, 0.10),# 7/10 over-detected (+0.202 avg bias), high variance
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
    """Pull scores toward population mean (0.50) based on confidence.

    Two-tier shrinkage:
    1. Universal mild shrinkage (10%) — counteracts LLM tendency to over-commit
    2. Additional shrinkage for low-confidence scores (< 0.70)

    Formula: adjusted = value * (1 - shrink) + 0.50 * shrink
    Where shrink = 0.10 (universal) + confidence-based (for low-conf)
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
