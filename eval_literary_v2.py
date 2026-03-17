"""Literary Character Progressive Detection V2 — Cumulative Input.

Instead of detecting each segment independently, feeds ALL dialogue up to
the current point through the full V4.1 pipeline (calibration + behavioral
features + Bayesian shrinkage). Measures how quickly the profile converges
to the final full-data result.

GT = weighted average of:
  - V4.1 full-data detection (weight 0.4)
  - LLM omniscient scoring (weight 0.3)
  - OpenPsychometrics crowd data (weight 0.3)

Usage:
    ANTHROPIC_API_KEY=... python eval_literary_v2.py [character] [segment_size]
    ANTHROPIC_API_KEY=... python eval_literary_v2.py scarlett 25
    ANTHROPIC_API_KEY=... python eval_literary_v2.py all 25
"""

import json
import os
import re
import sys
import time
from pathlib import Path

from super_brain.detector import Detector
from super_brain.behavioral_features import (
    extract_features, compute_adjustments, apply_adjustments,
)
from super_brain.literary import segment_dialogue, compute_mae


CHARACTERS = [
    "scarlett", "sherlock", "elizabeth",
    "darcy", "watson", "hamlet", "jane_eyre", "scrooge", "ahab", "huck",
]

CHARACTER_NAMES = {
    "scarlett": "Scarlett",
    "sherlock": "Holmes",
    "elizabeth": "Elizabeth",
    "darcy": "Darcy",
    "watson": "Watson",
    "hamlet": "Hamlet",
    "jane_eyre": "Jane",
    "scrooge": "Scrooge",
    "ahab": "Ahab",
    "huck": "Huck",
}

# GT source weights
GT_WEIGHTS = {
    "gt_v41": 0.4,   # V4.1 full-data detection
    "gt_llm": 0.3,   # LLM omniscient
    "gt_crowd": 0.3,  # OpenPsychometrics crowd
}


def load_dialogue(character: str) -> list[str]:
    """Load dialogue quotes from file."""
    path = Path(f"data/literary/{character}/dialogue.txt")
    quotes = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            m = re.match(r"\[\d+\]\s*(.*)", line)
            if m:
                quotes.append(m.group(1))
            elif line:
                quotes.append(line)
    return [q for q in quotes if len(q) > 3]


def load_ground_truths(character: str) -> dict[str, dict[str, float]]:
    """Load external GT sources (LLM omniscient + crowd)."""
    gt = {}
    for source in ["gt_llm", "gt_crowd"]:
        path = Path(f"data/literary/{character}/{source}.json")
        if path.exists():
            gt[source] = json.loads(path.read_text())
    return gt


def format_as_text(quotes: list[str], char_name: str) -> str:
    """Format quotes as labeled dialogue for the detector."""
    return "\n\n".join(f"{char_name}: {q}" for q in quotes)


def format_as_conversation(quotes: list[str]) -> list[dict]:
    """Format quotes as conversation dicts for behavioral feature extraction."""
    return [{"role": "speaker", "text": q} for q in quotes]


def detect_full_pipeline(
    detector: Detector,
    quotes: list[str],
    char_name: str,
    speaker_id: str,
) -> dict[str, float]:
    """Run V4.1 full pipeline: detect + behavioral features.

    Returns {trait_name: value} dict.
    """
    text = format_as_text(quotes, char_name)

    # Stage 1: V4.1 detector (includes calibration + Bayesian shrinkage)
    result = detector.analyze(
        text=text,
        speaker_id=speaker_id,
        speaker_label=char_name,
    )

    # Stage 2: Behavioral feature adjustments
    conversation = format_as_conversation(quotes)
    bf = extract_features(conversation, speaker_role="speaker")
    bf_adj = compute_adjustments(bf)
    if bf_adj:
        result = apply_adjustments(result, bf_adj)

    return {t.name: t.value for t in result.traits}


def build_composite_gt(
    gt_v41: dict[str, float],
    gt_external: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Build weighted composite GT from all sources."""
    all_traits = set(gt_v41.keys())
    for gt in gt_external.values():
        all_traits |= set(gt.keys())

    composite = {}
    for trait in all_traits:
        total_weight = 0.0
        weighted_sum = 0.0

        if trait in gt_v41:
            w = GT_WEIGHTS["gt_v41"]
            weighted_sum += gt_v41[trait] * w
            total_weight += w

        for source, gt_data in gt_external.items():
            if trait in gt_data:
                w = GT_WEIGHTS.get(source, 0.3)
                weighted_sum += gt_data[trait] * w
                total_weight += w

        if total_weight > 0:
            composite[trait] = weighted_sum / total_weight

    return composite


def run_experiment(character: str, segment_size: int, api_key: str) -> dict:
    """Run cumulative progressive detection for one character."""
    print(f"\n{'='*60}")
    print(f"Character: {character} (V2 — Cumulative Input)")
    print(f"{'='*60}")

    quotes = load_dialogue(character)
    gt_external = load_ground_truths(character)
    char_name = CHARACTER_NAMES[character]

    # Create cumulative checkpoints
    # Use segment_size to determine checkpoint intervals
    checkpoints = list(range(segment_size, len(quotes), segment_size))
    if checkpoints and checkpoints[-1] != len(quotes):
        checkpoints.append(len(quotes))
    if not checkpoints:
        checkpoints = [len(quotes)]

    print(f"Quotes: {len(quotes)}, Checkpoints: {len(checkpoints)}, Segment size: {segment_size}")
    print(f"External GTs: {list(gt_external.keys())}")

    detector = Detector(api_key=api_key)

    # Step 1: Generate V4.1 full-data GT
    print(f"\n  [GT] Running V4.1 on ALL {len(quotes)} quotes...")
    start = time.time()
    gt_v41 = detect_full_pipeline(
        detector, quotes, char_name, f"{character}_full",
    )
    gt_elapsed = time.time() - start
    print(f"  [GT] Done in {gt_elapsed:.0f}s, detected {len(gt_v41)} traits")

    # Save V4.1 GT
    outdir = Path(f"data/literary/{character}")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "gt_v41.json", "w") as f:
        json.dump(gt_v41, f, indent=2)

    # Build composite GT
    composite_gt = build_composite_gt(gt_v41, gt_external)
    with open(outdir / "gt_composite.json", "w") as f:
        json.dump(composite_gt, f, indent=2)

    print(f"  [GT] Composite GT: {len(composite_gt)} traits from {1 + len(gt_external)} sources")

    # Step 2: Cumulative detection at each checkpoint
    convergence = []
    prev_profile = None

    for i, cp in enumerate(checkpoints):
        cumulative_quotes = quotes[:cp]
        print(f"  Checkpoint {i+1}/{len(checkpoints)}: {cp} quotes...", end=" ", flush=True)
        start = time.time()

        profile = detect_full_pipeline(
            detector, cumulative_quotes, char_name,
            f"{character}_cp{i}",
        )

        elapsed = time.time() - start

        # MAE vs each GT source
        profile_dict = {k: {"value": v, "confidence": 0.8} for k, v in profile.items()}
        mae_composite = compute_mae(profile_dict, composite_gt)
        mae_v41 = compute_mae(profile_dict, gt_v41)
        mae_llm = compute_mae(profile_dict, gt_external.get("gt_llm", {})) if "gt_llm" in gt_external else None
        mae_crowd = compute_mae(profile_dict, gt_external.get("gt_crowd", {})) if "gt_crowd" in gt_external else None

        # Profile delta from previous checkpoint
        delta = 0.0
        if prev_profile:
            shared = set(profile.keys()) & set(prev_profile.keys())
            if shared:
                delta = sum(abs(profile[k] - prev_profile[k]) for k in shared) / len(shared)

        entry = {
            "checkpoint": i,
            "cumulative_quotes": cp,
            "pct_dialogue": round(cp / len(quotes) * 100, 1),
            "mae_vs_composite": round(mae_composite, 3),
            "mae_vs_v41_full": round(mae_v41, 3),
            "mae_vs_gt_llm": round(mae_llm, 3) if mae_llm is not None else None,
            "mae_vs_gt_crowd": round(mae_crowd, 3) if mae_crowd is not None else None,
            "profile_delta": round(delta, 4),
            "elapsed": round(elapsed, 1),
        }
        convergence.append(entry)
        prev_profile = profile

        print(f"MAE(comp)={mae_composite:.3f} MAE(v41)={mae_v41:.3f} delta={delta:.4f} | {elapsed:.0f}s")

    # Find convergence point — noise-aware threshold
    # LLM randomness floor is ~0.05, so use 0.08 (1.5x noise) as practical threshold.
    # Require 2 consecutive checkpoints below threshold (not 3, since even stable
    # profiles occasionally spike due to LLM sampling).
    CONVERGENCE_THRESHOLD = 0.08
    convergence_point = None
    for i in range(1, len(convergence)):
        if all(convergence[j]["profile_delta"] < CONVERGENCE_THRESHOLD for j in range(i-1, i+1)):
            convergence_point = convergence[i-1]
            break

    # Save results
    result = {
        "character": character,
        "total_quotes": len(quotes),
        "total_checkpoints": len(checkpoints),
        "segment_size": segment_size,
        "gt_sources": ["gt_v41"] + list(gt_external.keys()),
        "convergence": convergence,
        "convergence_point": convergence_point,
        "final": convergence[-1] if convergence else None,
    }

    with open(outdir / "convergence_v2.json", "w") as f:
        json.dump(result, f, indent=2)

    # Save per-checkpoint profiles for trajectory analysis
    # (re-run would be needed for full trajectories, but convergence is the key output)

    print(f"\n  Convergence point: ", end="")
    if convergence_point:
        print(f"at {convergence_point['cumulative_quotes']} quotes "
              f"({convergence_point['pct_dialogue']}% of dialogue)")
    else:
        print("not reached (delta never stabilized)")

    print(f"  Final MAE(composite): {convergence[-1]['mae_vs_composite']:.3f}")
    print(f"  Final MAE(v41): {convergence[-1]['mae_vs_v41_full']:.3f}")

    return result


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    character = sys.argv[1] if len(sys.argv) > 1 else "scarlett"
    segment_size = int(sys.argv[2]) if len(sys.argv) > 2 else 25

    characters = CHARACTERS if character == "all" else [character]

    all_results = {}
    for char in characters:
        result = run_experiment(char, segment_size, api_key)
        all_results[char] = result

    if len(characters) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for char, result in all_results.items():
            final = result["final"]
            cp = result["convergence_point"]
            cp_str = f"{cp['cumulative_quotes']}q ({cp['pct_dialogue']}%)" if cp else "N/A"
            print(f"  {char}: MAE(comp)={final['mae_vs_composite']:.3f} "
                  f"MAE(v41)={final['mae_vs_v41_full']:.3f} "
                  f"converge={cp_str}")


if __name__ == "__main__":
    main()
