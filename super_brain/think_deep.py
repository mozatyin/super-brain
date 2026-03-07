"""ThinkDeep: Triggered strategic analysis of intentions and gaps (V2.5).

Unlike ThinkSlow (periodic personality extraction) and FactExtractor (periodic
fact extraction), ThinkDeep is triggered by specific conditions — when enough
facts are accumulated, contradictions emerge, or information goes stale.

It receives the FULL Soul state and produces a strategic analysis:
intentions, reality-intention gaps, bridge questions, and a critical question.
"""

from __future__ import annotations

import json

import anthropic

from super_brain.models import (
    Soul, Intention, Gap, ThinkDeepResult,
)


_THINK_DEEP_SYSTEM = """\
You are a strategic conversation analyst. You have been given a Soul profile — \
a comprehensive understanding of a person built from conversation analysis. Your \
job is to identify their INTENTIONS (what they want), the GAPS between their \
intentions and reality, and generate BRIDGE QUESTIONS that explore those gaps.

Think like a skilled therapist or coach: what is this person really after? What's \
holding them back? What's the ONE question that would unlock the deepest insight?

Return ONLY valid JSON:
{
  "soul_narrative": "<1-2 sentence narrative of who this person is and where they're at>",
  "intentions": [
    {
      "description": "<what they want>",
      "domain": "<career|relationship|personal_growth|health|creative|financial>",
      "strength": <0.0-1.0>,
      "blockers": ["<what's in the way>", ...]
    }
  ],
  "gaps": [
    {
      "intention": "<what they want>",
      "reality": "<where they actually are>",
      "bridge_question": "<question that explores this gap>",
      "priority": <0.0-1.0>
    }
  ],
  "critical_question": "<THE single most important question to ask next>",
  "conversation_strategy": "<1 sentence: how should the conversation shift?>"
}

IMPORTANT:
- Intentions should be INFERRED from facts, behavior, and stated desires — not just what they explicitly say.
- Gaps require both an intention AND a contradicting reality. No gap without both sides.
- The critical_question should be the question that, if answered honestly, would reveal the most about this person.
- Bridge questions should be natural conversation questions, NOT therapy-speak.
- Strength: 0.3-0.5 for hinted, 0.6-0.8 for discussed, 0.9-1.0 for central life theme.
- If insufficient information, return fewer intentions/gaps. Don't fabricate.
"""


def _format_conversation(conversation: list[dict]) -> str:
    """Format conversation for ThinkDeep input."""
    lines = []
    for msg in conversation:
        label = "Person A" if msg["role"] == "chatter" else "Person B"
        lines.append(f"{label}: {msg['text']}")
    return "\n\n".join(lines)


def _build_soul_context(soul: Soul) -> str:
    """Build a text summary of the Soul state for the LLM prompt."""
    sections = []

    # Character traits
    if soul.character.traits:
        trait_lines = [
            f"  {t.name} ({t.dimension}): {t.value:.2f} (conf={t.confidence:.2f})"
            for t in soul.character.traits
        ]
        sections.append("CHARACTER TRAITS:\n" + "\n".join(trait_lines))

    # Facts
    if soul.facts:
        fact_lines = [f"  [{f.category}] {f.content} (conf={f.confidence:.1f})" for f in soul.facts]
        sections.append("KNOWN FACTS:\n" + "\n".join(fact_lines))
    else:
        sections.append("KNOWN FACTS:\n  No facts extracted yet.")

    # Reality
    if soul.reality:
        sections.append(
            f"REALITY SNAPSHOT:\n  {soul.reality.summary}\n"
            f"  Domains: {soul.reality.domains}\n"
            f"  Constraints: {soul.reality.constraints}\n"
            f"  Resources: {soul.reality.resources}"
        )

    # Existing intentions
    if soul.intentions:
        int_lines = [
            f"  {i.description} ({i.domain}, strength={i.strength:.1f})"
            for i in soul.intentions
        ]
        sections.append("PREVIOUSLY DETECTED INTENTIONS:\n" + "\n".join(int_lines))

    # Secrets
    if soul.secrets:
        sections.append("SECRETS/PATTERNS:\n" + "\n".join(f"  - {s}" for s in soul.secrets))

    # Contradictions
    if soul.contradictions:
        sections.append("CONTRADICTIONS:\n" + "\n".join(f"  - {c}" for c in soul.contradictions))

    return "\n\n".join(sections)


def _parse_think_deep_response(raw: str) -> dict:
    """Parse ThinkDeep JSON response with fallback."""
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {
        "soul_narrative": "",
        "intentions": [],
        "gaps": [],
        "critical_question": "",
        "conversation_strategy": "",
    }


class ThinkDeep:
    """Triggered strategic analysis of intentions and gaps."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        kwargs: dict = {"api_key": api_key}
        if api_key.startswith("sk-or-"):
            kwargs["base_url"] = "https://openrouter.ai/api"
        self._client = anthropic.Anthropic(**kwargs)
        self._model = model

    def analyze(
        self,
        soul: Soul,
        conversation: list[dict],
    ) -> ThinkDeepResult:
        """Run strategic analysis on the full Soul state."""
        soul_context = _build_soul_context(soul)
        conv_text = _format_conversation(conversation)

        user_message = (
            f"## Soul Profile\n\n{soul_context}\n\n"
            f"## Recent Conversation\n\n{conv_text}\n\n"
            f"## Task\n\nAnalyze Person B's intentions, reality-intention gaps, "
            f"and generate bridge questions. Return JSON."
        )

        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=_THINK_DEEP_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
        )

        raw = response.content[0].text
        data = _parse_think_deep_response(raw)

        # Build typed models
        intentions = []
        for raw_int in data.get("intentions", []):
            intentions.append(Intention(
                description=raw_int.get("description", ""),
                domain=raw_int.get("domain", "personal_growth"),
                strength=max(0.0, min(1.0, float(raw_int.get("strength", 0.5)))),
                blockers=raw_int.get("blockers", []),
            ))

        gaps = []
        for raw_gap in data.get("gaps", []):
            gaps.append(Gap(
                intention=raw_gap.get("intention", ""),
                reality=raw_gap.get("reality", ""),
                bridge_question=raw_gap.get("bridge_question", ""),
                priority=max(0.0, min(1.0, float(raw_gap.get("priority", 0.5)))),
            ))

        return ThinkDeepResult(
            soul_narrative=data.get("soul_narrative", ""),
            intentions=intentions,
            gaps=gaps,
            critical_question=data.get("critical_question", ""),
            conversation_strategy=data.get("conversation_strategy", ""),
        )
