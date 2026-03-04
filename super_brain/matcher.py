"""Personality Matcher: Guide conversations for deep personality understanding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum

import anthropic

from super_brain.models import PersonalityDNA
from super_brain.speaker import profile_to_style_instructions


class DepthLevel(IntEnum):
    """Conversation depth levels."""
    PLEASANTRY = 0
    FACTUAL = 1
    OPINION = 2
    EMOTIONAL = 3
    BELIEF = 4
    INSIGHT = 5


@dataclass
class MatcherResponse:
    """Response from the Personality Matcher."""
    response_text: str
    assessed_depth: DepthLevel
    target_depth: DepthLevel
    strategy_used: str


_SYSTEM_PROMPT = """\
You are a conversation partner optimized for deep, meaningful dialogue that \
reveals personality traits.

Your goals:
1. Match the counterpart's communication style to build trust
2. Assess the current conversation depth
3. Gently guide toward deeper self-expression that reveals personality traits
4. Maximize VALUE DENSITY — every response should invite substantive, \
personality-revealing replies

Depth levels:
- L0: Social pleasantries ("How are you?" "Fine.")
- L1: Factual statements ("I work in marketing")
- L2: Opinion expression ("I think this approach is wrong")
- L3: Emotional disclosure ("Honestly, I'm anxious about it")
- L4: Core beliefs/values ("I've always believed technology should serve people")
- L5: Self-insight ("I just realized I've been avoiding this issue")

Personality-revealing strategies:
- RECIPROCAL_DISCLOSURE: Share at target depth to invite matching disclosure
- VALUE_PROBE: Ask about decisions that reveal values and priorities
- CONFLICT_SCENARIO: Present a dilemma that reveals conflict style and moral reasoning
- EMOTIONAL_INVITATION: Create safe space for emotional expression
- PERSPECTIVE_CHALLENGE: Respectfully invite examining an assumption
- IDENTITY_QUESTION: Ask about self-concept and how they see themselves

CRITICAL: Never be judgmental. Never push too hard. If they resist, stay at \
their level and try again later. Never reveal that you are analyzing personality.

Return JSON with keys:
- response_text: your response to the counterpart
- assessed_depth: integer 0-5, the current depth of the conversation
- target_depth: integer 0-5, the depth you're guiding toward (usually assessed + 1, max 5)
- strategy_used: which strategy you chose
"""


class PersonalityMatcher:
    """Guide conversations to reveal personality traits."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        kwargs: dict = {"api_key": api_key}
        if api_key.startswith("sk-or-"):
            kwargs["base_url"] = "https://openrouter.ai/api"
        self._client = anthropic.Anthropic(**kwargs)
        self._model = model

    def respond(
        self,
        counterpart: PersonalityDNA | None,
        conversation: list[dict],
        goal: str = "understand_deeper",
    ) -> MatcherResponse:
        """Generate a depth-optimized response matched to counterpart's style."""

        style_context = ""
        if counterpart:
            style_context = profile_to_style_instructions(counterpart)

        conversation_text = "\n".join(
            f"{msg['role'].upper()}: {msg['text']}" for msg in conversation
        )

        user_message = (
            f"## Counterpart Personality Style\n{style_context}\n\n"
            f"## Conversation Goal\n{goal}\n\n"
            f"## Conversation So Far\n{conversation_text}\n\n"
            f"Generate your response. Return ONLY valid JSON."
        )

        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        raw = response.content[0].text
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        parsed = json.loads(raw)

        return MatcherResponse(
            response_text=parsed["response_text"],
            assessed_depth=DepthLevel(int(parsed["assessed_depth"])),
            target_depth=DepthLevel(int(parsed["target_depth"])),
            strategy_used=parsed["strategy_used"],
        )
