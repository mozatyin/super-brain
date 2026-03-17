"""Compare old vs new pipeline on the same user data.

Runs the V4.1 full pipeline with the new optimizations:
- Compressed system prompt (43 lines vs 282)
- temperature=0.0 (was 1.0 default)
- Behavioral features injected into LLM prompt

Usage:
    source .env && export ANTHROPIC_API_KEY
    .venv/bin/python scripts/compare_pipeline.py
"""

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from super_brain.detector import Detector
from super_brain.behavioral_features import (
    extract_features, compute_adjustments, apply_adjustments,
)
from super_brain.literary import compute_mae


def load_user_quotes(filepath: Path) -> list[str]:
    quotes = []
    for line in filepath.read_text().splitlines():
        line = line.strip()
        if line and re.match(r"\[\d+\]", line):
            m = re.match(r"\[\d+\]\s*(.*)", line)
            if m and len(m.group(1)) > 3:
                quotes.append(m.group(1))
    return quotes


def detect_full_pipeline_v2(
    detector: Detector,
    quotes: list[str],
    speaker_label: str,
    speaker_id: str,
) -> dict[str, float]:
    """New pipeline: behavioral features injected INTO the LLM prompt."""
    text = "\n\n".join(f"{speaker_label}: {q}" for q in quotes)
    conversation = [{"role": "speaker", "text": q} for q in quotes]

    # Pre-compute behavioral features
    bf = extract_features(conversation, speaker_role="speaker")

    # Pass behavioral features to detector (injected into prompt)
    result = detector.analyze(
        text=text,
        speaker_id=speaker_id,
        speaker_label=speaker_label,
        behavioral_features=bf,
    )

    # Still apply post-hoc adjustments for traits the LLM might miss
    bf_adj = compute_adjustments(bf)
    if bf_adj:
        result = apply_adjustments(result, bf_adj)

    return {t.name: t.value for t in result.traits}


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    user_file = Path("data/real_users/real_user_E_122msgs.txt")
    segment_size = 25

    quotes = load_user_quotes(user_file)
    print(f"User E: {len(quotes)} messages, segment_size={segment_size}")

    checkpoints = list(range(segment_size, len(quotes), segment_size))
    if checkpoints and checkpoints[-1] != len(quotes):
        checkpoints.append(len(quotes))
    if not checkpoints:
        checkpoints = [len(quotes)]

    print(f"Checkpoints: {checkpoints}")
    print(f"Pipeline: compressed prompt + temp=0 + BF injection\n")

    detector = Detector(api_key=api_key, temperature=0.0)

    # Step 1: Full-data GT (with new pipeline)
    print(f"[GT] Running on ALL {len(quotes)} messages...")
    start = time.time()
    gt = detect_full_pipeline_v2(detector, quotes, "User", "userE_v2_full")
    gt_time = time.time() - start
    print(f"[GT] Done in {gt_time:.0f}s, {len(gt)} traits\n")

    # Step 2: Progressive detection
    convergence = []
    prev_profile = None

    for i, cp in enumerate(checkpoints):
        cumulative = quotes[:cp]
        print(f"  CP{i}: {cp} msgs...", end=" ", flush=True)
        start = time.time()

        profile = detect_full_pipeline_v2(
            detector, cumulative, "User", f"userE_v2_cp{i}"
        )
        elapsed = time.time() - start

        # MAE vs full-data GT
        profile_dict = {k: {"value": v, "confidence": 0.8} for k, v in profile.items()}
        mae = compute_mae(profile_dict, gt)

        # Delta from previous
        delta = 0.0
        if prev_profile:
            shared = set(profile.keys()) & set(prev_profile.keys())
            if shared:
                delta = sum(abs(profile[k] - prev_profile[k]) for k in shared) / len(shared)

        entry = {
            "checkpoint": i,
            "cumulative_msgs": cp,
            "pct": round(cp / len(quotes) * 100, 1),
            "mae_vs_full": round(mae, 3),
            "profile_delta": round(delta, 4),
            "elapsed": round(elapsed, 1),
        }
        convergence.append(entry)
        prev_profile = profile

        print(f"MAE={mae:.3f} delta={delta:.4f} | {elapsed:.0f}s")

    # Load old results for comparison
    old_results = json.loads(
        Path("data/real_users/results/real_user_E_122msgs_convergence.json").read_text()
    )

    # Print comparison
    print(f"\n{'='*70}")
    print("COMPARISON: Old Pipeline vs New Pipeline")
    print(f"{'='*70}")
    print(f"{'Msgs':>5} | {'Old MAE':>8} {'Old Δ':>8} | {'New MAE':>8} {'New Δ':>8} | {'MAE Δ':>7}")
    print(f"{'-'*5}-+-{'-'*8}-{'-'*8}-+-{'-'*8}-{'-'*8}-+-{'-'*7}")

    for new_cp in convergence:
        msgs = new_cp["cumulative_msgs"]
        old_cp = next(
            (c for c in old_results["convergence"] if c["cumulative_msgs"] == msgs),
            None
        )
        if old_cp:
            old_mae = old_cp["mae_vs_v41_full"]
            old_delta = old_cp["profile_delta"]
            improvement = old_mae - new_cp["mae_vs_full"]
            print(f"{msgs:5d} | {old_mae:8.3f} {old_delta:8.4f} | "
                  f"{new_cp['mae_vs_full']:8.3f} {new_cp['profile_delta']:8.4f} | "
                  f"{improvement:+7.3f}")
        else:
            print(f"{msgs:5d} | {'N/A':>8} {'N/A':>8} | "
                  f"{new_cp['mae_vs_full']:8.3f} {new_cp['profile_delta']:8.4f} |")

    # Save new results
    outdir = Path("data/real_users/results")
    outdir.mkdir(parents=True, exist_ok=True)
    result = {
        "user": "real_user_E_122msgs",
        "pipeline": "v2_compressed_temp0_bf",
        "total_msgs": len(quotes),
        "segment_size": segment_size,
        "convergence": convergence,
        "final_mae": convergence[-1]["mae_vs_full"] if convergence else None,
    }
    with open(outdir / "real_user_E_122msgs_convergence_v2.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nNew final MAE: {convergence[-1]['mae_vs_full']:.3f}")
    print(f"Old final MAE: {old_results['final_mae']}")

    # Summary stats
    old_maes = [c["mae_vs_v41_full"] for c in old_results["convergence"]]
    new_maes = [c["mae_vs_full"] for c in convergence]
    old_deltas = [c["profile_delta"] for c in old_results["convergence"]]
    new_deltas = [c["profile_delta"] for c in convergence]

    print(f"\nAvg MAE: old={sum(old_maes)/len(old_maes):.3f} new={sum(new_maes)/len(new_maes):.3f}")
    print(f"Avg delta: old={sum(old_deltas)/len(old_deltas):.4f} new={sum(new_deltas)/len(new_deltas):.4f}")


if __name__ == "__main__":
    main()
