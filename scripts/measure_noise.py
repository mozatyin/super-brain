"""Noise Floor Measurement for Super Brain V4.1.

Runs the EXACT same detection N times on the EXACT same input to measure
how much LLM output varies between runs (the noise floor).

Usage:
    source .env && export ANTHROPIC_API_KEY
    .venv/bin/python scripts/measure_noise.py
"""

import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from super_brain.detector import Detector
from super_brain.behavioral_features import (
    extract_features, compute_adjustments, apply_adjustments,
)

# Re-use the loader from eval_real_users
import re


def load_user_quotes(filepath: Path) -> list[str]:
    """Load user messages from file."""
    quotes = []
    for line in filepath.read_text().splitlines():
        line = line.strip()
        if line and re.match(r"\[\d+\]", line):
            m = re.match(r"\[\d+\]\s*(.*)", line)
            if m and len(m.group(1)) > 3:
                quotes.append(m.group(1))
    return quotes


def detect_full_pipeline(
    detector: Detector,
    quotes: list[str],
    speaker_label: str,
    speaker_id: str,
) -> dict[str, float]:
    """Run V4.1 full pipeline on quotes."""
    text = "\n\n".join(f"{speaker_label}: {q}" for q in quotes)

    result = detector.analyze(
        text=text,
        speaker_id=speaker_id,
        speaker_label=speaker_label,
    )

    conversation = [{"role": "speaker", "text": q} for q in quotes]
    bf = extract_features(conversation, speaker_role="speaker")
    bf_adj = compute_adjustments(bf)
    if bf_adj:
        result = apply_adjustments(result, bf_adj)

    return {t.name: t.value for t in result.traits}


def pairwise_mae(profile_a: dict[str, float], profile_b: dict[str, float]) -> float:
    """Compute MAE between two profiles on shared traits."""
    shared = set(profile_a.keys()) & set(profile_b.keys())
    if not shared:
        return 0.0
    return sum(abs(profile_a[k] - profile_b[k]) for k in shared) / len(shared)


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    # Config
    NUM_RUNS = 5
    NUM_MSGS = 50
    temperature = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    user_file = Path("data/real_users/real_user_E_122msgs.txt")

    # Load data
    all_quotes = load_user_quotes(user_file)
    quotes = all_quotes[:NUM_MSGS]
    print(f"Loaded {len(all_quotes)} messages from {user_file}, using first {NUM_MSGS}")
    print(f"Temperature: {temperature}")
    print(f"Will run {NUM_RUNS} identical detections to measure noise floor\n")

    detector = Detector(api_key=api_key, temperature=temperature)

    # Run N times
    profiles: list[dict[str, float]] = []
    run_times: list[float] = []

    for i in range(NUM_RUNS):
        print(f"  Run {i+1}/{NUM_RUNS}...", end=" ", flush=True)
        start = time.time()
        profile = detect_full_pipeline(
            detector, quotes, "User", f"noise_run_{i}"
        )
        elapsed = time.time() - start
        run_times.append(elapsed)
        profiles.append(profile)
        print(f"Done in {elapsed:.1f}s — {len(profile)} traits")

    print(f"\nAll {NUM_RUNS} runs complete. Total time: {sum(run_times):.0f}s\n")

    # Compute pairwise MAE for all C(N,2) pairs
    pairs = list(combinations(range(NUM_RUNS), 2))
    pairwise_maes = {}
    for i, j in pairs:
        mae = pairwise_mae(profiles[i], profiles[j])
        pairwise_maes[f"run{i+1}_vs_run{j+1}"] = round(mae, 4)

    mae_values = list(pairwise_maes.values())
    mean_mae = mean(mae_values)
    max_mae = max(mae_values)

    # Per-trait standard deviation
    all_traits = sorted(set().union(*(p.keys() for p in profiles)))
    trait_stats = {}
    for trait in all_traits:
        values = [p[trait] for p in profiles if trait in p]
        if len(values) >= 2:
            sd = stdev(values)
            avg = mean(values)
            trait_stats[trait] = {
                "mean": round(avg, 4),
                "stdev": round(sd, 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "range": round(max(values) - min(values), 4),
                "values": [round(v, 4) for v in values],
            }
        else:
            trait_stats[trait] = {
                "mean": round(values[0], 4) if values else 0,
                "stdev": 0,
                "min": round(values[0], 4) if values else 0,
                "max": round(values[0], 4) if values else 0,
                "range": 0,
                "values": [round(v, 4) for v in values],
            }

    # Sort by stdev
    sorted_by_var = sorted(trait_stats.items(), key=lambda x: x[1]["stdev"], reverse=True)
    top_10_variable = sorted_by_var[:10]
    top_10_stable = sorted_by_var[-10:]

    # Overall stdev stats
    all_stdevs = [v["stdev"] for v in trait_stats.values()]
    mean_stdev = mean(all_stdevs) if all_stdevs else 0
    max_stdev = max(all_stdevs) if all_stdevs else 0

    # Print summary
    print("=" * 60)
    print("NOISE FLOOR ANALYSIS")
    print("=" * 60)
    print(f"\nPairwise MAE across {len(pairs)} pairs:")
    for pair_label, mae in pairwise_maes.items():
        print(f"  {pair_label}: {mae:.4f}")
    print(f"\n  Mean pairwise MAE: {mean_mae:.4f}")
    print(f"  Max  pairwise MAE: {max_mae:.4f}")

    print(f"\nPer-trait stdev:")
    print(f"  Mean stdev: {mean_stdev:.4f}")
    print(f"  Max  stdev: {max_stdev:.4f}")

    print(f"\nTop 10 MOST VARIABLE traits:")
    for trait, stats in top_10_variable:
        print(f"  {trait:40s} stdev={stats['stdev']:.4f}  range={stats['range']:.4f}  mean={stats['mean']:.4f}")

    print(f"\nTop 10 MOST STABLE traits:")
    for trait, stats in top_10_stable:
        print(f"  {trait:40s} stdev={stats['stdev']:.4f}  range={stats['range']:.4f}  mean={stats['mean']:.4f}")

    # Save results
    outdir = Path("data/noise_analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    result = {
        "config": {
            "num_runs": NUM_RUNS,
            "num_msgs": NUM_MSGS,
            "temperature": temperature,
            "user_file": str(user_file),
            "total_time_seconds": round(sum(run_times), 1),
        },
        "summary": {
            "mean_pairwise_mae": round(mean_mae, 4),
            "max_pairwise_mae": round(max_mae, 4),
            "mean_trait_stdev": round(mean_stdev, 4),
            "max_trait_stdev": round(max_stdev, 4),
            "num_traits": len(all_traits),
        },
        "pairwise_maes": pairwise_maes,
        "top_10_variable": {t: s for t, s in top_10_variable},
        "top_10_stable": {t: s for t, s in top_10_stable},
        "all_trait_stats": trait_stats,
        "raw_profiles": [
            {k: round(v, 4) for k, v in p.items()} for p in profiles
        ],
        "run_times_seconds": [round(t, 1) for t in run_times],
    }

    temp_label = f"temp{temperature:.1f}".replace(".", "")
    outpath = outdir / f"noise_{temp_label}_5runs.json"
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
