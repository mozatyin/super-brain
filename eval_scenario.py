"""Scenario-driven personality evaluation (V4.0).

Instead of one 20-turn generic conversation, runs 16 focused scenarios.
Each scenario targets specific traits through carefully designed situations.

Based on Trait Activation Theory (Tett & Burnett 2003):
- Every trait gets a scenario designed to elicit it
- Chatter has a "script" to guide the conversation into trait-relevant territory
- Detector scores only the traits relevant to each scenario
- Coverage tracking ensures 100% trait activation

Usage:
    ANTHROPIC_API_KEY=... python eval_scenario.py [n_profiles] [max_scenarios]
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

import anthropic

from super_brain.catalog import ALL_DIMENSIONS, TRAIT_CATALOG
from super_brain.models import PersonalityDNA
from super_brain.detector import Detector
from super_brain.profile_gen import generate_profile
from super_brain.scenarios import SCENARIOS, Scenario, get_scenario_sequence


def _retry_api_call(fn, max_retries=4, base_delay=5):
    """Retry API calls on transient errors including timeouts."""
    for attempt in range(max_retries):
        try:
            return fn()
        except (anthropic.PermissionDeniedError, anthropic.RateLimitError,
                anthropic.InternalServerError, anthropic.APITimeoutError,
                anthropic.APIConnectionError) as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"    [retry {attempt+1}/{max_retries}] {type(e).__name__}, waiting {delay}s...")
            time.sleep(delay)
    raise RuntimeError("Unreachable")


# ── Speaker: acts in character during scenarios ────────────────────────────

def _build_scenario_speaker_system(profile: PersonalityDNA, scenario: Scenario) -> str:
    """Build Speaker system prompt tailored to the current scenario."""
    tmap = {t.name: t.value for t in profile.traits}

    # Build trait hints only for target traits (focused)
    trait_lines = []
    for tname in scenario.target_traits:
        val = tmap.get(tname, 0.5)
        level = "HIGH" if val > 0.60 else "LOW" if val < 0.40 else "MODERATE"
        trait_lines.append(f"- {tname}: {val:.2f} ({level})")

    # Get full style from existing speaker module
    from super_brain.speaker import profile_to_style_instructions
    style = profile_to_style_instructions(profile)

    # Focused behavioral hints for ALL target traits with specific guidance
    behavioral_hints = []
    for tname in scenario.target_traits:
        val = tmap.get(tname, 0.5)
        if val > 0.75:
            behavioral_hints.append(
                f"- {tname}={val:.2f} (VERY HIGH): Express this STRONGLY. "
                f"This should be one of your most obvious traits in this conversation."
            )
        elif val > 0.60:
            behavioral_hints.append(
                f"- {tname}={val:.2f} (HIGH): Show this clearly — lean into it. "
                f"A perceptive observer should notice this about you."
            )
        elif val < 0.20:
            behavioral_hints.append(
                f"- {tname}={val:.2f} (VERY LOW): Show the OPPOSITE of this trait strongly. "
                f"This should be noticeably absent or reversed in your responses."
            )
        elif val < 0.35:
            behavioral_hints.append(
                f"- {tname}={val:.2f} (LOW): Show less of this than average. "
                f"A perceptive observer should notice you're below average on this."
            )
        else:
            behavioral_hints.append(
                f"- {tname}={val:.2f} (MODERATE): Express this at an average level. "
                f"Neither notably high nor low."
            )

    hints_section = (
        "\n<scenario_focus>\n"
        "The current conversation is designed to reveal these traits. "
        "Express each one at the indicated level:\n"
        + "\n".join(behavioral_hints) +
        "\n</scenario_focus>\n"
    )

    return (
        "You are a person with the personality described below. You are having a "
        "casual conversation. Stay true to your personality throughout.\n\n"
        f"YOUR PERSONALITY (key traits for this topic):\n"
        + "\n".join(trait_lines) +
        f"\n\nFULL PERSONALITY STYLE:\n{style}\n"
        f"{hints_section}\n"
        "Rules:\n"
        "- Respond naturally to what the other person says\n"
        "- NEVER mention personality traits or that you have a profile\n"
        "- NEVER use *asterisk actions*. Just speak naturally.\n"
        "- Let your personality show through word choice, tone, and emotional reactions\n"
        "- Keep responses 2-5 sentences. Be natural, not performative.\n"
    )


# ── Chatter: guides conversation into trait-relevant territory ─────────────

def _build_scenario_chatter_system(scenario: Scenario, turn: int, total_turns: int) -> str:
    """Build Chatter system prompt for scenario-guided conversation."""
    return (
        "You are having a natural, casual conversation with someone. You have a specific "
        "topic to explore but must introduce it NATURALLY, not like an interview.\n\n"
        f"YOUR SCENARIO GUIDE:\n{scenario.chatter_setup}\n\n"
        f"Turn {turn}/{total_turns} of this topic.\n\n"
        "Rules:\n"
        "- Keep messages SHORT (1-2 sentences). Get THEM talking.\n"
        "- Ask ONE question at a time.\n"
        "- Be warm and genuinely curious.\n"
        "- Let the conversation flow naturally — don't rush through your script.\n"
        "- Reflect back what they said before asking the next question.\n"
        "- Do NOT probe their personality directly.\n"
    )


class ScenarioChatter:
    """Chatter that follows scenario scripts."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        kwargs: dict = {"api_key": api_key, "timeout": 120.0}
        if api_key.startswith("sk-or-"):
            kwargs["base_url"] = "https://openrouter.ai/api"
        self._client = anthropic.Anthropic(**kwargs)
        self._model = model

    def next_message(self, conversation: list[dict], scenario: Scenario,
                     turn: int, total_turns: int) -> str:
        system = _build_scenario_chatter_system(scenario, turn, total_turns)
        messages = []
        for msg in conversation:
            role = "assistant" if msg["role"] == "chatter" else "user"
            messages.append({"role": role, "content": msg["text"]})

        if not messages:
            messages = [{"role": "user", "content": "Start the conversation."}]

        response = _retry_api_call(lambda: self._client.messages.create(
            model=self._model,
            max_tokens=150,
            system=system,
            messages=messages,
        ))
        if response and response.content:
            return response.content[0].text
        time.sleep(5)
        response = _retry_api_call(lambda: self._client.messages.create(
            model=self._model,
            max_tokens=150,
            system=system,
            messages=messages,
        ))
        return response.content[0].text if response and response.content else "That's interesting, tell me more."


class ScenarioSpeaker:
    """Speaker that responds in character with scenario-focused traits."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        kwargs: dict = {"api_key": api_key, "timeout": 120.0}
        if api_key.startswith("sk-or-"):
            kwargs["base_url"] = "https://openrouter.ai/api"
        self._client = anthropic.Anthropic(**kwargs)
        self._model = model

    def respond(self, profile: PersonalityDNA, conversation: list[dict],
                scenario: Scenario) -> str:
        system = _build_scenario_speaker_system(profile, scenario)
        messages = []
        for msg in conversation:
            role = "user" if msg["role"] == "chatter" else "assistant"
            messages.append({"role": role, "content": msg["text"]})

        response = _retry_api_call(lambda: self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=system,
            messages=messages,
        ))
        if response and response.content:
            return response.content[0].text
        time.sleep(5)
        response = _retry_api_call(lambda: self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=system,
            messages=messages,
        ))
        return response.content[0].text if response and response.content else "..."


# ── Run one scenario ───────────────────────────────────────────────────────

def run_scenario(
    chatter: ScenarioChatter,
    speaker: ScenarioSpeaker,
    profile: PersonalityDNA,
    scenario: Scenario,
    seed: int = 0,
) -> list[dict]:
    """Run a single scenario conversation."""
    conversation: list[dict] = []

    for turn in range(scenario.turns):
        # Chatter introduces/continues the scenario
        chatter_msg = chatter.next_message(
            conversation, scenario, turn + 1, scenario.turns,
        )
        conversation.append({"role": "chatter", "text": chatter_msg})

        # Speaker responds in character
        speaker_msg = speaker.respond(profile, conversation, scenario)
        conversation.append({"role": "speaker", "text": speaker_msg})

    return conversation


# ── Detect traits from scenario conversation ───────────────────────────────

def detect_scenario_traits(
    detector: Detector,
    conversation: list[dict],
    scenario: Scenario,
) -> dict[str, tuple[float, float]]:
    """Detect target traits from a scenario conversation.

    Returns {trait_name: (value, confidence)} for each target trait.
    """
    # Format conversation
    lines = []
    for msg in conversation:
        label = "Person A" if msg["role"] == "chatter" else "Person B"
        lines.append(f"{label}: {msg['text']}")
    full_text = "\n\n".join(lines)

    # Use detector's analyze method — only run batches with target traits
    detected = detector.analyze(
        text=full_text,
        speaker_id="eval_scenario",
        speaker_label="Person B",
        target_traits=set(scenario.target_traits),
    )

    # Apply behavioral adjustments
    from super_brain.behavioral_features import (
        extract_features, compute_adjustments, apply_adjustments,
    )
    bf = extract_features(conversation, speaker_role="speaker")
    bf_adj = compute_adjustments(bf)
    if bf_adj:
        detected = apply_adjustments(detected, bf_adj)

    # Extract only target traits
    detected_map = {t.name: (t.value, t.confidence) for t in detected.traits}
    results = {}
    for tname in scenario.target_traits:
        if tname in detected_map:
            results[tname] = detected_map[tname]
    return results


# ── Full evaluation ────────────────────────────────────────────────────────

def evaluate_profile(
    chatter: ScenarioChatter,
    speaker: ScenarioSpeaker,
    detector: Detector,
    profile: PersonalityDNA,
    scenarios: list[Scenario],
    seed: int = 0,
) -> dict:
    """Run all scenarios for one profile and compare with ground truth."""
    gt = {t.name: t.value for t in profile.traits}

    # Accumulate per-trait detections (multiple scenarios may detect same trait)
    trait_detections: dict[str, list[tuple[float, float]]] = {}  # {name: [(value, conf), ...]}
    scenario_results = []

    for i, scenario in enumerate(scenarios):
        print(f"    Scenario {i+1}/{len(scenarios)}: {scenario.id} ({scenario.turns} turns)")
        start = time.time()

        conversation = run_scenario(chatter, speaker, profile, scenario, seed=seed + i)
        detected = detect_scenario_traits(detector, conversation, scenario)

        elapsed = time.time() - start
        sc_result = {
            "scenario_id": scenario.id,
            "target_traits": scenario.target_traits,
            "conversation": conversation,
            "detected": {k: {"value": v[0], "confidence": v[1]} for k, v in detected.items()},
            "elapsed": round(elapsed, 1),
        }
        scenario_results.append(sc_result)

        for tname, (val, conf) in detected.items():
            trait_detections.setdefault(tname, []).append((val, conf))

        print(f"      → detected {len(detected)} traits in {elapsed:.0f}s")

    # Aggregate: confidence-weighted mean per trait
    final_scores: dict[str, float] = {}
    for tname, detections in trait_detections.items():
        if not detections:
            continue
        total_conf = sum(c for _, c in detections)
        if total_conf > 0:
            final_scores[tname] = sum(v * c for v, c in detections) / total_conf
        else:
            final_scores[tname] = sum(v for v, _ in detections) / len(detections)

    # Compare with ground truth
    trait_results = []
    all_errors = []
    dim_errors: dict[str, list[float]] = {}

    for t in TRAIT_CATALOG:
        name = t["name"]
        dim = t["dimension"]
        original = gt.get(name)
        detected_val = final_scores.get(name)

        if original is None or detected_val is None:
            continue

        error = abs(original - detected_val)
        all_errors.append(error)
        dim_errors.setdefault(dim, []).append(error)

        status = "GOOD" if error <= 0.15 else "OK" if error <= 0.25 else "BAD"
        trait_results.append({
            "trait": f"{dim}:{name}",
            "original": round(original, 2),
            "detected": round(detected_val, 2),
            "error": round(error, 2),
            "n_observations": len(trait_detections.get(name, [])),
            "status": status,
        })

    mae = statistics.mean(all_errors) if all_errors else 1.0
    within_025 = sum(1 for e in all_errors if e <= 0.25)
    within_040 = sum(1 for e in all_errors if e <= 0.40)

    # Coverage
    covered = len(final_scores)
    total = len(TRAIT_CATALOG)

    dim_mae = {}
    for dim, errs in sorted(dim_errors.items()):
        dim_mae[dim] = round(statistics.mean(errs), 3)

    return {
        "mae": round(mae, 3),
        "within_025": within_025,
        "within_040": within_040,
        "total": len(all_errors),
        "coverage": f"{covered}/{total}",
        "traits": sorted(trait_results, key=lambda x: -x["error"]),
        "dim_mae": dim_mae,
        "scenarios": scenario_results,
    }


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    n_profiles = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    max_scenarios = int(sys.argv[2]) if len(sys.argv) > 2 else len(SCENARIOS)

    print(f"=== Scenario-Driven Eval V4.0 ===")
    print(f"Profiles: {n_profiles}, Scenarios per profile: {max_scenarios}")
    print(f"Total scenarios available: {len(SCENARIOS)}")
    print()

    chatter = ScenarioChatter(api_key=api_key)
    speaker = ScenarioSpeaker(api_key=api_key)
    detector = Detector(api_key=api_key)

    all_results = {}
    all_maes = []

    for p in range(n_profiles):
        seed = p * 100
        profile = generate_profile(f"profile_{p}", seed=seed)
        scenarios = get_scenario_sequence(seed)[:max_scenarios]

        print(f"Profile {p}: {profile.id} ({len(scenarios)} scenarios)")

        result = evaluate_profile(
            chatter, speaker, detector, profile, scenarios, seed=seed,
        )
        all_results[f"profile_{p}"] = result
        all_maes.append(result["mae"])

        print(f"  → MAE: {result['mae']:.3f}, Coverage: {result['coverage']}, "
              f"Within 0.25: {result['within_025']}/{result['total']}")
        print()

        # Save progress after each profile
        outfile = f"eval_scenario_results_v40_{n_profiles}p.json"
        with open(outfile, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  [saved progress to {outfile}]")

    overall_mae = statistics.mean(all_maes)
    print(f"=== Overall MAE: {overall_mae:.3f} ===")
    print(f"Per-profile MAEs: {[f'{m:.3f}' for m in all_maes]}")

    # Per-dimension summary
    dim_totals: dict[str, list[float]] = {}
    for pkey, pdata in all_results.items():
        for dim, mae in pdata["dim_mae"].items():
            dim_totals.setdefault(dim, []).append(mae)

    print("\nPer-dimension MAE:")
    for dim in sorted(dim_totals.keys()):
        avg = statistics.mean(dim_totals[dim])
        print(f"  {dim}: {avg:.3f}")

    # Save results
    outfile = f"eval_scenario_results_v40_{n_profiles}p.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
