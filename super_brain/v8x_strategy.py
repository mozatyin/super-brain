"""V8x behavioral→strategy mapping engine.

Replaces Big Five classification with real-time reply adaptation.
Behavioral features drive strategy selection — no personality labels needed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReplyStrategy:
    """Reply strategy parameters for the conversation composer."""
    tone: str           # "warm" | "direct" | "playful" | "neutral" | "firm"
    depth: str          # "surface" | "medium" | "deep"
    question_type: str  # "specific" | "open" | "reflective" | "none"
    empathy_level: float   # 0-1
    mirror_ratio: float    # 0-1, how much to reference user's exact words


class V8xStrategyEngine:
    """Maps behavioral features to reply strategies and D1 hooks."""

    def compute_strategy(
        self,
        behavioral_features: dict,
        detector_results: list,
        turn_number: int,
    ) -> ReplyStrategy:
        """Map behavioral features to a ReplyStrategy.

        Args:
            behavioral_features: Dict with keys like neg_emotion_ratio,
                word_count, question_ratio, etc.
            detector_results: List of detector output dicts (reserved for future use).
            turn_number: Current conversation turn (1-indexed).

        Returns:
            ReplyStrategy with all fields populated.
        """
        bf = behavioral_features

        # Defaults
        tone = "neutral"
        depth = "medium"
        question_type = "open"
        empathy_level = 0.5
        mirror_ratio = 0.4

        # --- Signal-based overrides (order matters: later wins for same field) ---

        # Low word count: draw them out
        word_count = bf.get("word_count", 50)
        if word_count < 20:
            tone = "playful"
            question_type = "specific"

        # High question ratio: match their questioning style
        question_ratio = bf.get("question_ratio", 0.0)
        if question_ratio > 0.3:
            question_type = "specific"
            mirror_ratio = 0.6

        # High exclamation ratio: match energy
        exclamation_ratio = bf.get("exclamation_ratio", 0.0)
        if exclamation_ratio > 0.2:
            tone = "direct"

        # High hedging: gentle approach
        hedging_ratio = bf.get("hedging_ratio", 0.0)
        if hedging_ratio > 0.1:
            tone = "warm"
            question_type = "reflective"

        # High absolutist: be firm, don't hedge
        absolutist_ratio = bf.get("absolutist_ratio", 0.0)
        if absolutist_ratio > 0.1:
            tone = "firm"

        # High self-reference: they're ready to go inward
        self_ref_ratio = bf.get("self_ref_ratio", 0.0)
        if self_ref_ratio > 0.2:
            empathy_level = max(empathy_level, 0.7)
            if turn_number >= 5:
                depth = "deep"

        # High negative emotion: warmth + empathy (highest priority for tone)
        neg_emotion_ratio = bf.get("neg_emotion_ratio", 0.0)
        if neg_emotion_ratio > 0.1:
            tone = "warm"
            empathy_level = max(empathy_level, 0.8)
            depth = "medium" if depth == "surface" else depth

        # --- Turn-number overrides ---

        if turn_number == 1:
            depth = "surface"
            mirror_ratio = 0.8

        if turn_number >= 5 and self_ref_ratio > 0.2:
            depth = "deep"

        return ReplyStrategy(
            tone=tone,
            depth=depth,
            question_type=question_type,
            empathy_level=round(empathy_level, 2),
            mirror_ratio=round(mirror_ratio, 2),
        )

    def generate_composer_directive(
        self,
        strategy: ReplyStrategy,
        turn_number: int,
    ) -> str:
        """Convert a ReplyStrategy to a natural-language composer instruction (<50 tokens).

        Args:
            strategy: The computed ReplyStrategy.
            turn_number: Current conversation turn.

        Returns:
            A concise instruction string for the conversation composer.
        """
        parts: list[str] = []

        # Tone
        if strategy.tone == "warm":
            parts.append("Use warm tone.")
        elif strategy.tone == "direct":
            parts.append("Be direct and concise.")
        elif strategy.tone == "playful":
            parts.append("Keep it light and playful.")
        elif strategy.tone == "firm":
            parts.append("Be firm and clear.")
        else:
            parts.append("Use neutral tone.")

        # Mirror
        if strategy.mirror_ratio >= 0.6:
            parts.append("Reference their exact words.")

        # Question type
        if strategy.question_type == "specific":
            parts.append("Ask one specific question.")
        elif strategy.question_type == "reflective":
            parts.append("Ask one reflective question about what they haven't said.")
        elif strategy.question_type == "open":
            parts.append("Ask one open question.")

        # Depth
        if strategy.depth == "deep":
            parts.append("Go deep.")
        elif strategy.depth == "surface":
            parts.append("Stay on the surface.")
        else:
            parts.append("Medium depth.")

        # Empathy guidance
        if strategy.empathy_level >= 0.7:
            parts.append("Empathize genuinely.")
        elif strategy.empathy_level <= 0.4:
            parts.append("Don't over-empathize.")

        return " ".join(parts)

    def generate_d1_hook(
        self,
        behavioral: dict,
        detector_results: list,
        conversation_summary: str = "",
    ) -> dict:
        """Generate D1 retention hook based on user behavior pattern.

        Args:
            behavioral: Dict of behavioral features.
            detector_results: List of detector output dicts.
            conversation_summary: Optional summary of conversation so far.

        Returns:
            Dict with hook_type, hook_text, push_text.
        """
        hook_type = self._classify_hook_type(behavioral)

        hook_text, push_text = _HOOK_TEMPLATES[hook_type]

        return {
            "hook_type": hook_type,
            "hook_text": hook_text,
            "push_text": push_text,
        }

    def _classify_hook_type(self, bf: dict) -> str:
        """Determine hook type from behavioral features."""
        word_count = bf.get("word_count", 50)
        question_ratio = bf.get("question_ratio", 0.0)
        topic_count = bf.get("topic_count", 1)
        action_intent = bf.get("action_intent", False)
        absolutist_ratio = bf.get("absolutist_ratio", 0.0)
        self_ref_ratio = bf.get("self_ref_ratio", 0.1)

        # Action user: explicit intent or very direct language
        if action_intent:
            return "action"

        # Defensive: high absolutist + low self-reference
        if absolutist_ratio > 0.1 and self_ref_ratio < 0.05:
            return "defensive"

        # Curious: lots of questions + multiple topics
        if question_ratio > 0.3 and topic_count >= 3:
            return "curious"

        # Passive: short messages + no questions
        if word_count < 20 and question_ratio < 0.1:
            return "passive"

        # Default fallback
        return "passive"


# --- Hook templates by type ---
_HOOK_TEMPLATES: dict[str, tuple[str, str]] = {
    "passive": (
        "I noticed something interesting about how you communicate. Want to see what I picked up?",
        "Something caught my eye about you yesterday...",
    ),
    "curious": (
        "You had some great questions last time. I've been thinking about one in particular.",
        "I found an answer to something you asked yesterday.",
    ),
    "action": (
        "Ready to take the next step? I have a suggestion based on our conversation.",
        "Here's that next step we talked about.",
    ),
    "defensive": (
        "I respect how direct you are. There's one thing I think you'd actually find useful.",
        "Something straightforward I think you'll appreciate.",
    ),
}
