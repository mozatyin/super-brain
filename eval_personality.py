"""Evaluate Detector accuracy by generating text from known profiles and re-detecting."""

import json
import os
import statistics
import sys
from pathlib import Path

from super_brain.catalog import ALL_DIMENSIONS
from super_brain.models import PersonalityDNA, Trait, SampleSummary
from super_brain.speaker import Speaker
from super_brain.detector import Detector


def make_profile(profile_id: str, traits: list[dict]) -> PersonalityDNA:
    """Build a PersonalityDNA from a shorthand trait list."""
    return PersonalityDNA(
        id=profile_id,
        sample_summary=SampleSummary(
            total_tokens=0,
            conversation_count=0,
            date_range=["unknown", "unknown"],
            contexts=["evaluation"],
            confidence_overall=1.0,
        ),
        traits=[
            Trait(
                dimension=t["dim"],
                name=t["name"],
                value=t["value"],
                confidence=1.0,
            )
            for t in traits
        ],
    )


# ── 6 distinct personality personas ─────────────────────────────────────────

PROFILES = {
    "neurotic_introvert": make_profile("neurotic_introvert", [
        # High neuroticism
        {"dim": "NEU", "name": "anxiety", "value": 0.85},
        {"dim": "NEU", "name": "depression", "value": 0.70},
        {"dim": "NEU", "name": "self_consciousness", "value": 0.80},
        {"dim": "NEU", "name": "vulnerability", "value": 0.75},
        # Low extraversion
        {"dim": "EXT", "name": "gregariousness", "value": 0.10},
        {"dim": "EXT", "name": "assertiveness", "value": 0.15},
        {"dim": "EXT", "name": "positive_emotions", "value": 0.20},
        # High emotional depth
        {"dim": "EMO", "name": "emotional_granularity", "value": 0.85},
        {"dim": "EMO", "name": "emotional_expressiveness", "value": 0.70},
        {"dim": "EMO", "name": "empathy_affective", "value": 0.75},
        # Avoidant attachment
        {"dim": "SOC", "name": "attachment_avoidance", "value": 0.65},
        {"dim": "SOC", "name": "attachment_anxiety", "value": 0.70},
    ]),

    "charismatic_leader": make_profile("charismatic_leader", [
        # High extraversion
        {"dim": "EXT", "name": "assertiveness", "value": 0.90},
        {"dim": "EXT", "name": "warmth", "value": 0.80},
        {"dim": "EXT", "name": "activity_level", "value": 0.85},
        {"dim": "EXT", "name": "positive_emotions", "value": 0.85},
        # Low neuroticism
        {"dim": "NEU", "name": "anxiety", "value": 0.10},
        {"dim": "NEU", "name": "vulnerability", "value": 0.10},
        # Strategy / presence
        {"dim": "STR", "name": "self_mythologizing", "value": 0.75},
        {"dim": "SOC", "name": "charm_influence", "value": 0.85},
        {"dim": "SOC", "name": "social_dominance", "value": 0.70},
        # Strong values
        {"dim": "COG", "name": "locus_of_control", "value": 0.90},
        {"dim": "VAL", "name": "loyalty_group", "value": 0.75},
        {"dim": "HUM", "name": "humor_affiliative", "value": 0.70},
    ]),

    "empathetic_healer": make_profile("empathetic_healer", [
        # Extreme empathy / agreeableness
        {"dim": "EMO", "name": "empathy_cognitive", "value": 0.90},
        {"dim": "EMO", "name": "empathy_affective", "value": 0.90},
        {"dim": "AGR", "name": "altruism", "value": 0.90},
        {"dim": "AGR", "name": "tender_mindedness", "value": 0.90},
        {"dim": "EXT", "name": "warmth", "value": 0.90},
        # High emotional awareness
        {"dim": "EMO", "name": "emotional_granularity", "value": 0.80},
        {"dim": "EMO", "name": "emotional_expressiveness", "value": 0.80},
        # Values
        {"dim": "VAL", "name": "care_harm", "value": 0.95},
        {"dim": "VAL", "name": "fairness_justice", "value": 0.80},
        # Affiliative humor
        {"dim": "HUM", "name": "humor_affiliative", "value": 0.70},
        # Low dark traits
        {"dim": "DRK", "name": "narcissism", "value": 0.05},
        {"dim": "DRK", "name": "psychopathy", "value": 0.05},
    ]),

    "cold_strategist": make_profile("cold_strategist", [
        # High dark traits
        {"dim": "DRK", "name": "machiavellianism", "value": 0.80},
        {"dim": "DRK", "name": "psychopathy", "value": 0.60},
        # Low empathy / warmth
        {"dim": "EMO", "name": "empathy_affective", "value": 0.10},
        {"dim": "EXT", "name": "warmth", "value": 0.15},
        {"dim": "EMO", "name": "emotional_expressiveness", "value": 0.15},
        # High cognition
        {"dim": "COG", "name": "need_for_cognition", "value": 0.85},
        {"dim": "COG", "name": "intuitive_vs_analytical", "value": 0.90},
        # Strategy
        {"dim": "STR", "name": "information_control", "value": 0.80},
        {"dim": "STR", "name": "mirroring_ability", "value": 0.65},
        # Low honesty-humility
        {"dim": "HON", "name": "sincerity", "value": 0.15},
        {"dim": "HON", "name": "fairness", "value": 0.20},
        # High deliberation
        {"dim": "CON", "name": "deliberation", "value": 0.80},
    ]),

    "creative_rebel": make_profile("creative_rebel", [
        # High openness
        {"dim": "OPN", "name": "fantasy", "value": 0.90},
        {"dim": "OPN", "name": "aesthetics", "value": 0.85},
        {"dim": "OPN", "name": "ideas", "value": 0.85},
        {"dim": "OPN", "name": "values_openness", "value": 0.90},
        # Low compliance / authority
        {"dim": "AGR", "name": "compliance", "value": 0.10},
        {"dim": "VAL", "name": "authority_respect", "value": 0.10},
        # High cognitive flexibility
        {"dim": "COG", "name": "cognitive_flexibility", "value": 0.85},
        # Moderate extraversion
        {"dim": "EXT", "name": "assertiveness", "value": 0.70},
        {"dim": "EXT", "name": "excitement_seeking", "value": 0.80},
        # Emotional depth
        {"dim": "EMO", "name": "emotional_expressiveness", "value": 0.75},
        {"dim": "OPN", "name": "feelings", "value": 0.80},
        # Humor
        {"dim": "HUM", "name": "humor_aggressive", "value": 0.55},
    ]),

    "disciplined_achiever": make_profile("disciplined_achiever", [
        # Extreme conscientiousness
        {"dim": "CON", "name": "order", "value": 0.90},
        {"dim": "CON", "name": "achievement_striving", "value": 0.90},
        {"dim": "CON", "name": "self_discipline", "value": 0.90},
        {"dim": "CON", "name": "deliberation", "value": 0.80},
        {"dim": "CON", "name": "competence", "value": 0.85},
        # High internal locus
        {"dim": "COG", "name": "locus_of_control", "value": 0.90},
        # Low impulsiveness
        {"dim": "NEU", "name": "impulsiveness", "value": 0.05},
        # Analytical
        {"dim": "COG", "name": "intuitive_vs_analytical", "value": 0.80},
        # Moderate assertiveness
        {"dim": "EXT", "name": "assertiveness", "value": 0.65},
        # Low emotional volatility
        {"dim": "EMO", "name": "emotional_regulation", "value": 0.85},
        {"dim": "EMO", "name": "emotional_volatility", "value": 0.10},
        # Values
        {"dim": "VAL", "name": "authority_respect", "value": 0.60},
    ]),
}


# ── 10 prompts designed to elicit personality-revealing text ─────────────────

PROMPTS = [
    "Describe how you handle a situation when someone close to you disappoints you deeply",
    "What do you think about people who always play it safe vs. those who take big risks?",
    "Tell me about a time you had to make a difficult decision with no clear right answer",
    "How do you react when someone disagrees with you on something you feel strongly about?",
    "Describe your ideal way to spend a weekend — and what it says about who you are",
    "What's your relationship with failure? How do you process setbacks?",
    "If you could change one thing about how people interact with each other, what would it be?",
    "Describe a conflict you had with someone and how you handled it",
    "What matters most to you in life, and how does that show up in your daily choices?",
    "React to this: your coworker takes credit for your idea in a meeting",
]


def _detect_with_averaging(
    detector: Detector,
    conversation: str,
    profile_name: str,
    n_samples: int = 2,
) -> dict[str, float]:
    """Run detection n_samples times and average the results."""
    accumulated: dict[str, list[float]] = {}

    for i in range(n_samples):
        detected = detector.analyze(
            text=conversation,
            speaker_id=f"eval_{profile_name}_s{i}",
            speaker_label="Speaker",
        )
        for t in detected.traits:
            key = f"{t.dimension}:{t.name}"
            accumulated.setdefault(key, []).append(t.value)

    return {key: statistics.mean(vals) for key, vals in accumulated.items()}


def run_eval(api_key: str, baseline_path: str | None = None, n_samples: int = 2):
    speaker = Speaker(api_key=api_key)
    detector = Detector(api_key=api_key)

    all_errors: list[float] = []
    dim_errors: dict[str, list[float]] = {}
    results: dict[str, dict] = {}

    # Load baseline if provided
    baseline: dict | None = None
    if baseline_path and Path(baseline_path).exists():
        baseline = json.loads(Path(baseline_path).read_text())
        print(f"  Loaded baseline from {baseline_path}")

    for profile_name, profile in PROFILES.items():
        print(f"\n{'='*60}")
        print(f"  Profile: {profile_name}")
        print(f"{'='*60}")

        # Generate text from personality profile
        print("  Generating text...", end=" ", flush=True)
        lines = []
        for prompt in PROMPTS:
            text = speaker.generate(profile=profile, content=prompt)
            lines.append(f"Speaker: {text}")
        conversation = "\n\n".join(lines)
        word_count = len(conversation.split())
        print(f"done ({word_count} words)")

        # Detect with multi-sample averaging
        print(f"  Detecting traits ({n_samples} samples)...", end=" ", flush=True)
        detected_map = _detect_with_averaging(
            detector, conversation, profile_name, n_samples
        )
        print("done")

        # Compare
        profile_errors: list[float] = []
        trait_results: list[dict] = []

        original_map: dict[str, float] = {}
        for t in profile.traits:
            original_map[f"{t.dimension}:{t.name}"] = t.value

        for key, original_val in original_map.items():
            detected_val = detected_map.get(key)
            if detected_val is None:
                print(f"    WARNING: {key} not found in detection output")
                continue
            error = abs(original_val - detected_val)
            profile_errors.append(error)
            all_errors.append(error)

            dim = key.split(":")[0]
            dim_errors.setdefault(dim, []).append(error)

            status = "OK" if error <= 0.25 else "MISS" if error <= 0.4 else "BAD"

            delta_str = ""
            if baseline and profile_name in baseline:
                bl_traits = baseline[profile_name].get("traits", [])
                bl_match = next((t for t in bl_traits if t["trait"] == key), None)
                if bl_match:
                    delta = error - bl_match["error"]
                    delta_str = f"  Δ{delta:+.2f}"

            trait_results.append({
                "trait": key,
                "original": original_val,
                "detected": round(detected_val, 3),
                "error": round(error, 3),
                "status": status,
                "delta": delta_str,
            })

        trait_results.sort(key=lambda x: -x["error"])

        print(
            f"\n  {'Trait':<35} {'Orig':>6} {'Det':>6} {'Err':>6}  Status"
            f"{'  Δbase' if baseline else ''}"
        )
        print(
            f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6}  {'-'*6}"
            f"{'------' if baseline else ''}"
        )
        for r in trait_results:
            print(
                f"  {r['trait']:<35} {r['original']:>6.2f} "
                f"{r['detected']:>6.2f} {r['error']:>6.3f}  {r['status']}{r['delta']}"
            )

        mae = statistics.mean(profile_errors) if profile_errors else float("nan")
        within_025 = sum(1 for e in profile_errors if e <= 0.25)
        within_04 = sum(1 for e in profile_errors if e <= 0.4)
        total = len(profile_errors)

        print(
            f"\n  MAE: {mae:.3f} | Within 0.25: {within_025}/{total} | "
            f"Within 0.40: {within_04}/{total}"
        )

        results[profile_name] = {
            "mae": mae,
            "within_025": within_025,
            "within_04": within_04,
            "total": total,
            "traits": trait_results,
        }

    # ── Per-dimension MAE ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PER-DIMENSION MAE")
    print(f"{'='*60}")
    print(f"\n  {'Dimension':<35} {'MAE':>6} {'Count':>6}")
    print(f"  {'-'*35} {'-'*6} {'-'*6}")
    for dim_code in sorted(dim_errors.keys()):
        errors = dim_errors[dim_code]
        dim_label = f"{dim_code} ({ALL_DIMENSIONS.get(dim_code, '?')})"
        print(f"  {dim_label:<35} {statistics.mean(errors):>6.3f} {len(errors):>6}")

    # ── Overall summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*60}")
    print(
        f"\n  {'Profile':<25} {'MAE':>6} {'<=0.25':>8} {'<=0.40':>8} {'Total':>6}"
    )
    print(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*8} {'-'*6}")
    for name, r in results.items():
        print(
            f"  {name:<25} {r['mae']:>6.3f} "
            f"{r['within_025']:>8}/{r['total']:<4} "
            f"{r['within_04']:>6}/{r['total']:<4}"
        )

    overall_mae = statistics.mean(all_errors)
    overall_025 = sum(1 for e in all_errors if e <= 0.25)
    overall_04 = sum(1 for e in all_errors if e <= 0.4)
    total_traits = len(all_errors)

    print(f"\n  Overall MAE: {overall_mae:.3f}")
    print(
        f"  Traits within 0.25: {overall_025}/{total_traits} "
        f"({100*overall_025/total_traits:.1f}%)"
    )
    print(
        f"  Traits within 0.40: {overall_04}/{total_traits} "
        f"({100*overall_04/total_traits:.1f}%)"
    )

    # ── Save results ─────────────────────────────────────────────────────
    version_tag = "v0.1"
    output_path = Path(f"eval_results_{version_tag}.json")
    save_data = dict(results)
    save_data["_overall"] = {
        "mae": overall_mae,
        "within_025": overall_025,
        "within_04": overall_04,
        "total": total_traits,
    }
    save_data["_dim_mae"] = {
        dim: statistics.mean(errs) for dim, errs in dim_errors.items()
    }
    output_path.write_text(json.dumps(save_data, indent=2, default=str))
    print(f"\n  Results saved to {output_path}")

    return results


if __name__ == "__main__":
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY env var to run evaluation.")
        sys.exit(1)

    baseline_path = sys.argv[1] if len(sys.argv) > 1 else None
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    run_eval(api_key, baseline_path=baseline_path, n_samples=n_samples)
