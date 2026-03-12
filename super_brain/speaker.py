"""LLM Speaker: Generate text embodying a specific personality profile."""

from __future__ import annotations

import anthropic

from super_brain.models import PersonalityDNA, Trait
from super_brain.catalog import ALL_DIMENSIONS, TRAIT_MAP


def _value_to_instruction(trait: Trait) -> str:
    """Convert a trait value to a level description with nearest anchor text."""
    v = trait.value
    if v < 0.05:
        level = "negligible (essentially absent)"
    elif v < 0.15:
        level = "very low"
    elif v < 0.25:
        level = "low"
    elif v < 0.35:
        level = "low-moderate"
    elif v < 0.45:
        level = "moderate-low"
    elif v < 0.55:
        level = "moderate"
    elif v < 0.65:
        level = "moderate-high"
    elif v < 0.75:
        level = "high"
    elif v < 0.85:
        level = "high-very high"
    elif v < 0.95:
        level = "very high"
    else:
        level = "extreme (maximum)"

    anchor_text = ""
    catalog_entry = TRAIT_MAP.get((trait.dimension, trait.name))
    if catalog_entry and "value_anchors" in catalog_entry:
        anchors = catalog_entry["value_anchors"]
        anchor_keys = sorted(float(k) for k in anchors)
        nearest = min(anchor_keys, key=lambda a: abs(a - v))
        nearest_key = (
            f"{nearest:.2f}"
            if nearest not in (0.0, 1.0)
            else ("0.0" if nearest == 0.0 else "1.0")
        )
        anchor_text = anchors.get(nearest_key, "")

    desc = f"{level} ({v:.2f})"
    if anchor_text:
        desc += f" — {anchor_text}"
    return desc


def _generate_boundary_constraints(profile: PersonalityDNA) -> str:
    """For mid-range traits, show both extremes as boundaries to avoid."""
    constraints: list[str] = []

    for t in profile.traits:
        catalog_entry = TRAIT_MAP.get((t.dimension, t.name))
        if not catalog_entry:
            continue
        anchors = catalog_entry.get("value_anchors", {})

        if 0.25 <= t.value <= 0.75:
            low_desc = anchors.get("0.0", "")
            high_desc = anchors.get("1.0", "")
            anchor_keys = sorted(float(k) for k in anchors)
            nearest = min(anchor_keys, key=lambda a: abs(a - t.value))
            nearest_key = (
                f"{nearest:.2f}"
                if nearest not in (0.0, 1.0)
                else ("0.0" if nearest == 0.0 else "1.0")
            )
            target_desc = anchors.get(nearest_key, "")

            direction = (
                "slightly lower"
                if t.value < 0.50
                else "slightly higher" if t.value > 0.50 else "dead center"
            )
            constraints.append(
                f"- {t.name} ({t.value:.2f}): AIM FOR '{target_desc}'. "
                f"TOO LOW: '{low_desc}'. TOO HIGH: '{high_desc}'. "
                f"If unsure, err {direction}."
            )
        elif t.value < 0.25:
            high_desc = anchors.get("1.0", anchors.get("0.75", ""))
            if high_desc:
                constraints.append(
                    f"- {t.name} ({t.value:.2f}): Keep very low. "
                    f"NEVER drift toward '{high_desc}'."
                )
        else:  # > 0.75
            low_desc = anchors.get("0.0", anchors.get("0.25", ""))
            if low_desc:
                constraints.append(
                    f"- {t.name} ({t.value:.2f}): Keep very high. "
                    f"DO NOT fall to '{low_desc}'."
                )

    return "\n".join(constraints)


def _generate_interaction_warnings(profile: PersonalityDNA) -> str:
    """Detect trait combinations that cause systematic drift and add warnings."""
    tmap = {t.name: t.value for t in profile.traits}
    warnings: list[str] = []

    # High narcissism + low modesty — don't go full villain
    if tmap.get("narcissism", 0) > 0.6 and tmap.get("modesty", 1) < 0.3:
        warnings.append(
            "- WARNING: High narcissism + low modesty. Express confidence and self-focus "
            "but avoid cartoonish villain speech. Make the narcissism feel natural and subtle."
        )

    # High empathy + high dark traits — the 'charming manipulator'
    if tmap.get("empathy_cognitive", 0) > 0.6 and tmap.get("machiavellianism", 0) > 0.5:
        warnings.append(
            "- WARNING: High cognitive empathy + Machiavellianism. This person understands "
            "others well AND uses that strategically. Show perceptiveness about others' feelings "
            "alongside calculated behavior."
        )

    # High anxiety + high assertiveness — paradoxical but valid
    if tmap.get("anxiety", 0) > 0.6 and tmap.get("assertiveness", 0) > 0.6:
        warnings.append(
            "- NOTE: High anxiety + high assertiveness. This person pushes through despite "
            "worry. Show assertive statements with underlying nervous energy."
        )


    return "\n".join(warnings)


def profile_to_style_instructions(
    profile: PersonalityDNA, intensity_scale: float = 1.0
) -> str:
    """Convert a PersonalityDNA profile into natural-language style instructions."""
    lines = ["<personality_profile>"]

    by_dim: dict[str, list[Trait]] = {}
    for t in profile.traits:
        by_dim.setdefault(t.dimension, []).append(t)

    for dim_code, traits in by_dim.items():
        dim_name = ALL_DIMENSIONS.get(dim_code, dim_code)
        lines.append(f"\n## {dim_name}")
        for t in traits:
            scaled_value = min(1.0, t.value * intensity_scale)
            scaled = Trait(
                dimension=t.dimension,
                name=t.name,
                value=scaled_value,
                confidence=t.confidence,
            )
            instruction = _value_to_instruction(scaled)
            lines.append(f"- {t.name}: {instruction}")

    lines.append("</personality_profile>")

    # Boundary constraints
    boundaries = _generate_boundary_constraints(profile)
    if boundaries:
        lines.append("\n<boundaries>")
        lines.append("BOUNDARY AWARENESS — stay between these extremes:")
        lines.append(boundaries)
        lines.append("</boundaries>")

    # Interaction warnings
    interaction_warnings = _generate_interaction_warnings(profile)
    if interaction_warnings:
        lines.append("\n<interaction_warnings>")
        lines.append("IMPORTANT — trait interactions to watch for:")
        lines.append(interaction_warnings)
        lines.append("</interaction_warnings>")

    return "\n".join(lines)


class Speaker:
    """Generate text embodying a PersonalityDNA profile."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        kwargs: dict = {"api_key": api_key}
        if api_key.startswith("sk-or-"):
            kwargs["base_url"] = "https://openrouter.ai/api"
        self._client = anthropic.Anthropic(**kwargs)
        self._model = model

    def generate(
        self,
        profile: PersonalityDNA,
        content: str,
        intensity: float = 1.0,
        context: str | None = None,
    ) -> str:
        """Generate text expressing the given content in the profile's personality style."""
        style_instructions = profile_to_style_instructions(
            profile, intensity_scale=intensity
        )

        system_prompt = (
            "<role>\n"
            "You are a personality actor. You must respond to the user's prompt AS IF you were "
            "a person with EXACTLY the personality profile described below. Your word choice, "
            "emotional tone, reasoning style, values, humor, and social behavior should all "
            "reflect these trait scores.\n\n"
            "Key rules:\n"
            "- DO NOT mention that you are acting or that you have a personality profile\n"
            "- DO NOT use clinical psychology terms (narcissism, extraversion, etc.)\n"
            "- Just BE this person naturally\n"
            "- Let the personality traits influence HOW you express yourself, not WHAT you say\n"
            "- Pay special attention to extreme scores (>0.8 or <0.2) — these define the character\n"
            "</role>\n\n"
            f"{style_instructions}"
        )

        response = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"Respond to this as the person described:\n\n{content}",
                }
            ],
        )

        return response.content[0].text
