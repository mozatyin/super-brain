"""Real User Progressive Personality Detection.

Tests how quickly V4.1 converges on real user chat data.
No external GT — uses V4.1 full-data detection as the reference.

Usage:
    ANTHROPIC_API_KEY=... python eval_real_users.py [user_file] [segment_size]
    ANTHROPIC_API_KEY=... python eval_real_users.py all 50
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


def run_user_experiment(filepath: Path, segment_size: int, api_key: str) -> dict:
    """Run cumulative detection on one real user."""
    user_label = filepath.stem
    quotes = load_user_quotes(filepath)

    print(f"\n{'='*60}")
    print(f"User: {user_label} ({len(quotes)} messages)")
    print(f"{'='*60}")

    # Checkpoints
    checkpoints = list(range(segment_size, len(quotes), segment_size))
    if checkpoints and checkpoints[-1] != len(quotes):
        checkpoints.append(len(quotes))
    if not checkpoints:
        checkpoints = [len(quotes)]

    print(f"Checkpoints: {len(checkpoints)}, Segment size: {segment_size}")

    detector = Detector(api_key=api_key)

    # Step 1: V4.1 full-data GT
    print(f"\n  [GT] Running V4.1 on ALL {len(quotes)} messages...")
    start = time.time()
    gt_v41 = detect_full_pipeline(detector, quotes, "User", f"{user_label}_full")
    gt_elapsed = time.time() - start
    print(f"  [GT] Done in {gt_elapsed:.0f}s, {len(gt_v41)} traits")

    # Step 2: Cumulative detection
    convergence = []
    prev_profile = None

    for i, cp in enumerate(checkpoints):
        cumulative = quotes[:cp]
        print(f"  CP {i+1}/{len(checkpoints)}: {cp} msgs...", end=" ", flush=True)
        start = time.time()

        profile = detect_full_pipeline(detector, cumulative, "User", f"{user_label}_cp{i}")
        elapsed = time.time() - start

        # MAE vs full-data GT
        profile_dict = {k: {"value": v, "confidence": 0.8} for k, v in profile.items()}
        mae_v41 = compute_mae(profile_dict, gt_v41)

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
            "mae_vs_v41_full": round(mae_v41, 3),
            "profile_delta": round(delta, 4),
            "elapsed": round(elapsed, 1),
        }
        convergence.append(entry)
        prev_profile = profile

        print(f"MAE(v41)={mae_v41:.3f} delta={delta:.4f} | {elapsed:.0f}s")

    # Save
    outdir = Path("data/real_users/results")
    outdir.mkdir(parents=True, exist_ok=True)

    result = {
        "user": user_label,
        "total_msgs": len(quotes),
        "segment_size": segment_size,
        "convergence": convergence,
        "final_mae": convergence[-1]["mae_vs_v41_full"] if convergence else None,
    }

    with open(outdir / f"{user_label}_convergence.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Final MAE(v41): {convergence[-1]['mae_vs_v41_full']:.3f}")
    return result


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    segment_size = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    data_dir = Path("data/real_users")

    if target == "all":
        files = sorted(data_dir.glob("real_user_*.txt"))
    else:
        files = [data_dir / target]

    all_results = {}
    for f in files:
        if not f.exists():
            print(f"File not found: {f}")
            continue
        result = run_user_experiment(f, segment_size, api_key)
        all_results[f.stem] = result

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for name, r in all_results.items():
            segs = r["convergence"]
            # Find MAE at 50 msgs
            m50 = next((s["mae_vs_v41_full"] for s in segs if s["cumulative_msgs"] >= 50), "-")
            print(f"  {name}: {r['total_msgs']} msgs | 50msg MAE={m50} | final MAE={r['final_mae']}")


if __name__ == "__main__":
    main()
