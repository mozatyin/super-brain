"""Soul-Aware Diagnostic Questions (V3.0).

Uses LLM to generate contextual, personalized diagnostic questions based on:
- Already-exposed personality traits (Soul context)
- Low-confidence traits that need more evidence
- Conversation history for natural flow
- Psychology-informed question types for maximum diagnostic value

Each generated question is significantly more diagnostic than generic static
questions because it leverages what we already know about the person.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic

from super_brain.catalog import TRAIT_CATALOG
from super_brain.models import IncisiveQuestion


_DIAGNOSTIC_SYSTEM = """\
You are a personality assessment expert generating diagnostic conversation questions.

Your goal: Generate natural-sounding questions that efficiently reveal specific \
personality traits, based on what we ALREADY KNOW about this person.

## Diagnostic Question Types (ordered by diagnostic power):

1. **Situational Dilemma**: Present a realistic scenario with no clear right answer \
that forces trait-revealing choices.
   Example: "If a friend asked you to cover for them about something minor, how would you handle that?"

2. **Forced-Choice Preference**: Two appealing options that differentiate on the target trait.
   Example: "Would you rather have a job that's stable but boring, or exciting but uncertain?"

3. **Attribution Question**: Ask how they explain events, revealing internal models.
   Example: "When a project doesn't go as planned, what do you think is usually the reason?"

4. **Counterfactual**: "What if" scenarios that reveal values and priorities.
   Example: "If you could change one thing about how you handle conflict, what would it be?"

5. **Value Ranking**: Ask them to prioritize competing values.
   Example: "What matters more to you — being liked or being respected?"

## Rules:
- Questions MUST feel natural in casual conversation (not like a psych test)
- Use what you know about the person to make questions contextually relevant
- Each question should target 1-2 specific traits
- Avoid yes/no questions — open-ended responses reveal more
- Don't repeat topics already discussed in the conversation
- Generate questions in order of diagnostic priority (most important first)
- Keep questions concise (1-2 sentences max)

Return ONLY a JSON array:
[
  {
    "question": "natural conversation question",
    "target_traits": ["trait_name_1"],
    "question_type": "situational_dilemma",
    "rationale": "what the answer reveals"
  }
]
"""

# Trait catalog lookup for detection hints
_TRAIT_LOOKUP: dict[str, dict] = {t["name"]: t for t in TRAIT_CATALOG}


def _build_soul_context(
    confidence_map: dict[str, float],
    trait_values: dict[str, float] | None = None,
    conversation: list[dict] | None = None,
    known_facts: list[str] | None = None,
    reality_summary: str | None = None,
) -> str:
    """Build a Soul context section for the diagnostic prompt.

    Args:
        confidence_map: trait_name → confidence score.
        trait_values: trait_name → value (0-1) for high-confidence traits.
        conversation: Full conversation history.
        known_facts: Known facts about the person.
        reality_summary: Brief life situation description.

    Returns:
        Formatted string for inclusion in the LLM prompt.
    """
    sections = []

    # High-confidence traits (what we already know)
    high_conf = sorted(
        [(k, v) for k, v in confidence_map.items() if v >= 0.5],
        key=lambda x: -x[1],
    )
    if high_conf and trait_values:
        trait_lines = []
        for name, conf in high_conf[:12]:
            val = trait_values.get(name)
            if val is not None:
                level = "high" if val > 0.65 else "low" if val < 0.35 else "moderate"
                trait_lines.append(f"- {name}: {level} ({val:.2f}, confidence={conf:.2f})")
            else:
                trait_lines.append(f"- {name}: confidence={conf:.2f}")
        sections.append(
            "## What We Already Know (high-confidence traits):\n" + "\n".join(trait_lines)
        )
    elif high_conf:
        trait_lines = [f"- {name}: confidence={conf:.2f}" for name, conf in high_conf[:12]]
        sections.append(
            "## What We Already Know (high-confidence traits):\n" + "\n".join(trait_lines)
        )

    # Known facts
    if known_facts:
        sections.append(
            "## Known Facts About This Person:\n"
            + "\n".join(f"- {f}" for f in known_facts[:10])
        )

    # Reality summary
    if reality_summary:
        sections.append(f"## Life Situation:\n{reality_summary}")

    # Recent conversation (last 4 exchanges for context)
    if conversation:
        recent = conversation[-8:]
        conv_lines = []
        for msg in recent:
            label = "Person A" if msg["role"] == "chatter" else "Person B (target)"
            conv_lines.append(f"{label}: {msg['text']}")
        sections.append("## Recent Conversation:\n" + "\n".join(conv_lines))

    return "\n\n".join(sections) if sections else "No prior context available."


def _build_target_section(
    low_confidence_traits: list[str],
    confidence_map: dict[str, float],
    max_targets: int = 8,
) -> str:
    """Build target trait section with detection hints.

    Args:
        low_confidence_traits: Traits needing more evidence.
        confidence_map: Current confidence scores.
        max_targets: Maximum traits to include.

    Returns:
        Formatted string describing target traits.
    """
    sorted_traits = sorted(
        low_confidence_traits,
        key=lambda t: confidence_map.get(t, 0.0),
    )[:max_targets]

    lines = []
    for name in sorted_traits:
        conf = confidence_map.get(name, 0.0)
        info = _TRAIT_LOOKUP.get(name)
        if info:
            lines.append(
                f"- **{name}** (confidence={conf:.2f}): {info['description']}\n"
                f"  Detection hint: {info['detection_hint']}"
            )
        else:
            lines.append(f"- **{name}** (confidence={conf:.2f})")

    return "## Target Traits (need more evidence):\n" + "\n".join(lines)


def _parse_diagnostic_response(raw: str) -> list[dict]:
    """Parse LLM response into list of question dicts.

    Handles markdown code blocks, direct JSON arrays, and wrapped objects.

    Returns:
        List of question dicts, or empty list on parse failure.
    """
    text = raw.strip()
    # Strip markdown code blocks
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "questions" in result:
            return result["questions"]
    except json.JSONDecodeError:
        pass

    # Try to find JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return []


def generate_diagnostic_questions(
    low_confidence_traits: list[str],
    confidence_map: dict[str, float],
    conversation: list[dict] | None = None,
    trait_values: dict[str, float] | None = None,
    known_facts: list[str] | None = None,
    reality_summary: str | None = None,
    api_key: str = "",
    model: str = "claude-sonnet-4-20250514",
    max_questions: int = 5,
) -> list[IncisiveQuestion]:
    """Generate Soul-aware diagnostic questions using LLM.

    Uses what we already know about the person to generate contextual,
    psychology-informed questions that efficiently reveal target traits.

    Args:
        low_confidence_traits: Trait names that need more evidence.
        confidence_map: Current trait → confidence mapping.
        conversation: Full conversation history.
        trait_values: trait_name → value for known traits.
        known_facts: List of known facts about the person.
        reality_summary: Brief description of person's life situation.
        api_key: Anthropic/OpenRouter API key.
        model: Model to use for question generation.
        max_questions: Maximum questions to generate.

    Returns:
        List of IncisiveQuestion objects, sorted by priority.
    """
    if not low_confidence_traits or not api_key:
        return []

    soul_context = _build_soul_context(
        confidence_map=confidence_map,
        trait_values=trait_values,
        conversation=conversation,
        known_facts=known_facts,
        reality_summary=reality_summary,
    )

    target_section = _build_target_section(
        low_confidence_traits=low_confidence_traits,
        confidence_map=confidence_map,
    )

    user_message = (
        f"{soul_context}\n\n{target_section}\n\n"
        f"Generate {max_questions} diagnostic questions targeting these traits. "
        f"Make them natural and contextually relevant to what you know about this person.\n\n"
        f"Return ONLY the JSON array."
    )

    kwargs: dict[str, Any] = {"api_key": api_key}
    if api_key.startswith("sk-or-"):
        kwargs["base_url"] = "https://openrouter.ai/api"
    client = anthropic.Anthropic(**kwargs)

    from super_brain.api_retry import retry_api_call

    response = retry_api_call(
        lambda: client.messages.create(
            model=model,
            max_tokens=2048,
            system=_DIAGNOSTIC_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
        )
    )

    raw = response.content[0].text
    parsed = _parse_diagnostic_response(raw)

    questions = []
    for item in parsed[:max_questions]:
        if not isinstance(item, dict) or "question" not in item:
            continue
        targets = item.get("target_traits", [])
        primary_target = targets[0] if targets else "unknown"
        conf = confidence_map.get(primary_target, 0.0)

        questions.append(
            IncisiveQuestion(
                question=item["question"],
                target=primary_target,
                priority=1.0 - conf,
                source="soul_aware_diagnostic",
            )
        )

    return questions
