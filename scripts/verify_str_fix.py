"""Quick verification: run new pipeline on 3 characters most affected by STR divergence.

Compares against old gt_v41.json to check if trait-specific calibration fixed the drift.
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from super_brain.detector import Detector
from super_brain.behavioral_features import extract_features, compute_adjustments, apply_adjustments
from super_brain.catalog import TRAIT_CATALOG

# Characters with highest divergence
TEST_CHARS = {
    "scarlett": "Scarlett",
    "huck": "Huck",
    "scrooge": "Scrooge",
}

# Traits that had abs MAE > 0.10
FLAGGED_TRAITS = [
    "verbosity", "hot_cold_oscillation", "politeness", "humor_self_enhancing",
    "optimism", "self_mythologizing", "depression", "information_control",
    "fantasy", "attachment_avoidance", "social_dominance", "charm_influence", "trust",
]

trait_dim = {t["name"]: t["dimension"] for t in TRAIT_CATALOG}


def load_quotes(character):
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


def detect(detector, quotes, speaker, sid):
    text = "\n\n".join(f"{speaker}: {q}" for q in quotes)
    conversation = [{"role": "speaker", "text": q} for q in quotes]
    bf = extract_features(conversation, speaker_role="speaker")
    result = detector.analyze(text=text, speaker_id=sid, speaker_label=speaker, behavioral_features=bf)
    bf_adj = compute_adjustments(bf)
    if bf_adj:
        result = apply_adjustments(result, bf_adj)
    return {t.name: t.value for t in result.traits}


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    detector = Detector(api_key=api_key, temperature=0.0)

    print("STR Fix Verification — 3 characters × flagged traits\n")

    all_before = {}  # trait -> [abs_errors] from gt_v41_new.json (before fix)
    all_after = {}   # trait -> [abs_errors] from this run (after fix)

    for char, name in TEST_CHARS.items():
        old_gt = json.loads(Path(f"data/literary/{char}/gt_v41.json").read_text())
        before_gt = json.loads(Path(f"data/literary/{char}/gt_v41_new.json").read_text())

        quotes = load_quotes(char)
        print(f"{char} ({len(quotes)} quotes)...", end=" ", flush=True)
        start = time.time()
        after_gt = detect(detector, quotes, name, f"{char}_str_fix")
        elapsed = time.time() - start
        print(f"done in {elapsed:.0f}s")

        # Save
        with open(Path(f"data/literary/{char}/gt_v41_new2.json"), "w") as f:
            json.dump(after_gt, f, indent=2)

        # Compute per-trait errors
        for trait in FLAGGED_TRAITS:
            if trait in old_gt and trait in before_gt and trait in after_gt:
                before_err = abs(before_gt[trait] - old_gt[trait])
                after_err = abs(after_gt[trait] - old_gt[trait])
                all_before.setdefault(trait, []).append(before_err)
                all_after.setdefault(trait, []).append(after_err)

    # Print comparison
    print(f"\n{'Trait':30s} {'Before':>8} {'After':>8} {'Change':>8}")
    print("-" * 60)

    improvements = 0
    regressions = 0
    for trait in FLAGGED_TRAITS:
        if trait in all_before and trait in all_after:
            b = mean(all_before[trait])
            a = mean(all_after[trait])
            delta = a - b
            marker = "✓" if delta < -0.01 else ("✗" if delta > 0.01 else "=")
            if delta < -0.01:
                improvements += 1
            elif delta > 0.01:
                regressions += 1
            print(f"{trait:30s} {b:8.3f} {a:8.3f} {delta:+8.3f} {marker}")

    # Overall
    all_b = mean(v for vals in all_before.values() for v in vals)
    all_a = mean(v for vals in all_after.values() for v in vals)
    print(f"\n{'OVERALL':30s} {all_b:8.3f} {all_a:8.3f} {all_a-all_b:+8.3f}")
    print(f"\nImproved: {improvements}, Regressed: {regressions}, Stable: {len(FLAGGED_TRAITS)-improvements-regressions}")


if __name__ == "__main__":
    main()
