# eval_literary.py
"""Literary Character Progressive Personality Detection Experiment.

Feeds character dialogue segment-by-segment through ProgressiveDetector,
recording convergence curves against dual ground truth.

Usage:
    ANTHROPIC_API_KEY=... python eval_literary.py [character] [segment_size]
    ANTHROPIC_API_KEY=... python eval_literary.py scarlett 12
    ANTHROPIC_API_KEY=... python eval_literary.py all 12
"""

import json
import os
import sys
import re
import time
from pathlib import Path

from super_brain.progressive import ProgressiveDetector
from super_brain.literary import segment_dialogue, compute_mae


CHARACTERS = ["scarlett", "sherlock", "elizabeth"]


def load_dialogue(character: str) -> list[str]:
    """Load dialogue quotes from file."""
    path = Path(f"data/literary/{character}/dialogue.txt")
    quotes = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            # Strip leading [N] numbering
            m = re.match(r"\[\d+\]\s*(.*)", line)
            if m:
                quotes.append(m.group(1))
            elif line:
                quotes.append(line)
    return [q for q in quotes if len(q) > 3]


def load_ground_truth(character: str) -> dict[str, dict[str, float]]:
    """Load both ground truth sources."""
    gt = {}
    for source in ["gt_llm", "gt_crowd"]:
        path = Path(f"data/literary/{character}/{source}.json")
        if path.exists():
            gt[source] = json.loads(path.read_text())
    return gt


def format_segment_text(quotes: list[str], character_name: str) -> str:
    """Format quotes into a conversation-like text for the detector."""
    lines = []
    for q in quotes:
        lines.append(f"{character_name}: {q}")
    return "\n\n".join(lines)


CHARACTER_NAMES = {
    "scarlett": "Scarlett",
    "sherlock": "Holmes",
    "elizabeth": "Elizabeth",
}


def run_experiment(character: str, segment_size: int, api_key: str) -> dict:
    """Run progressive detection experiment for one character."""
    print(f"\n{'='*60}")
    print(f"Character: {character}")
    print(f"{'='*60}")

    # Load data
    quotes = load_dialogue(character)
    gt_sources = load_ground_truth(character)
    segments = segment_dialogue(quotes, segment_size)
    char_name = CHARACTER_NAMES[character]

    print(f"Quotes: {len(quotes)}, Segments: {len(segments)}, GTs: {list(gt_sources.keys())}")

    # Run progressive detection
    detector = ProgressiveDetector(api_key=api_key)
    convergence = []

    for i, seg in enumerate(segments):
        print(f"  Segment {i+1}/{len(segments)} ({len(seg)} quotes)...", end=" ", flush=True)
        start = time.time()

        text = format_segment_text(seg, char_name)
        snapshot = detector.update(text, speaker_label=char_name)

        # Compute MAE vs each ground truth
        profile = detector.get_profile_dict()
        mae_results = {}
        for gt_name, gt_data in gt_sources.items():
            mae_results[f"mae_vs_{gt_name}"] = round(compute_mae(profile, gt_data), 3)

        # Count converged traits
        n_converged = sum(
            1 for t in profile.values()
            if t["confidence"] > 0.70
        )
        # Check stability (last 3 snapshots)
        if len(convergence) >= 3:
            history = detector.get_history()
            stable_count = 0
            for trait_name in profile:
                recent_vals = []
                for h in history[-3:]:
                    if trait_name in h["traits"]:
                        recent_vals.append(h["traits"][trait_name]["value"])
                if len(recent_vals) == 3 and max(recent_vals) - min(recent_vals) < 0.03:
                    stable_count += 1
            n_converged = min(n_converged, stable_count)

        elapsed = time.time() - start
        entry = {
            "segment_id": i,
            "cumulative_quotes": sum(len(s) for s in segments[:i+1]),
            **mae_results,
            "traits_converged": n_converged,
            "avg_confidence": round(
                sum(t["confidence"] for t in profile.values()) / max(len(profile), 1), 3
            ),
            "elapsed": round(elapsed, 1),
        }
        convergence.append(entry)

        mae_str = " | ".join(f"{k}={v:.3f}" for k, v in mae_results.items())
        print(f"{mae_str} | converged={n_converged} | {elapsed:.0f}s")

    # Save results
    outdir = Path(f"data/literary/{character}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Convergence curve
    with open(outdir / "convergence.json", "w") as f:
        json.dump({"character": character, "segments": convergence}, f, indent=2)

    # Trait trajectories
    trajectories = {}
    history = detector.get_history()
    profile = detector.get_profile_dict()
    for trait_name in profile:
        traj = {
            "gt_llm": gt_sources.get("gt_llm", {}).get(trait_name),
            "gt_crowd": gt_sources.get("gt_crowd", {}).get(trait_name),
            "trajectory": [],
        }
        for h in history:
            if trait_name in h["traits"]:
                traj["trajectory"].append({
                    "seg": h["segment_id"],
                    "value": h["traits"][trait_name]["value"],
                    "confidence": h["traits"][trait_name]["confidence"],
                })
        trajectories[trait_name] = traj

    with open(outdir / "trajectories.json", "w") as f:
        json.dump(trajectories, f, indent=2)

    # Final comparison
    final = {
        "character": character,
        "total_quotes": len(quotes),
        "total_segments": len(segments),
    }
    for gt_name, gt_data in gt_sources.items():
        final[f"final_mae_{gt_name}"] = round(compute_mae(profile, gt_data), 3)

    # GT consistency
    if "gt_llm" in gt_sources and "gt_crowd" in gt_sources:
        gt_errors = []
        for name, llm_val in gt_sources["gt_llm"].items():
            if name in gt_sources["gt_crowd"]:
                gt_errors.append(abs(llm_val - gt_sources["gt_crowd"][name]))
        if gt_errors:
            final["gt_consistency_mae"] = round(sum(gt_errors) / len(gt_errors), 3)

    with open(outdir / "final_comparison.json", "w") as f:
        json.dump(final, f, indent=2)

    print(f"\n  Final: {json.dumps({k: v for k, v in final.items() if 'mae' in k}, indent=2)}")
    return final


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    character = sys.argv[1] if len(sys.argv) > 1 else "scarlett"
    segment_size = int(sys.argv[2]) if len(sys.argv) > 2 else 12

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
            mae_str = " | ".join(
                f"{k}={v}" for k, v in result.items() if "mae" in k
            )
            print(f"  {char}: {mae_str}")


if __name__ == "__main__":
    main()
