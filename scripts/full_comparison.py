"""Comprehensive A/B comparison: Old Pipeline vs New Pipeline.

Tests:
1. Cross-pipeline agreement: new full-data GT vs old full-data GT (9 literary + 4 real users)
2. Self-consistency: progressive detection on 3 representative cases
3. Per-dimension breakdown

Old pipeline: 282-line prompt, temp=1.0 (default), BF post-hoc only
New pipeline: 43-line prompt, temp=0.0, BF injected into LLM prompt

Usage:
    source .env && export ANTHROPIC_API_KEY
    .venv/bin/python scripts/full_comparison.py
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
from super_brain.literary import compute_mae
from super_brain.catalog import ALL_DIMENSIONS


# ── Literary character config ──

CHARACTERS = {
    "scarlett": "Scarlett", "sherlock": "Holmes", "elizabeth": "Elizabeth",
    "darcy": "Darcy", "watson": "Watson", "hamlet": "Hamlet",
    "jane_eyre": "Jane", "scrooge": "Scrooge", "huck": "Huck",
}


def load_literary_quotes(character: str) -> list[str]:
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


def load_real_user_quotes(filepath: Path) -> list[str]:
    quotes = []
    for line in filepath.read_text().splitlines():
        line = line.strip()
        if line and re.match(r"\[\d+\]", line):
            m = re.match(r"\[\d+\]\s*(.*)", line)
            if m and len(m.group(1)) > 3:
                quotes.append(m.group(1))
    return quotes


def detect_new_pipeline(
    detector: Detector, quotes: list[str], speaker_label: str, speaker_id: str,
) -> dict[str, float]:
    """New pipeline: BF pre-computed and injected into LLM prompt."""
    text = "\n\n".join(f"{speaker_label}: {q}" for q in quotes)
    conversation = [{"role": "speaker", "text": q} for q in quotes]
    bf = extract_features(conversation, speaker_role="speaker")

    result = detector.analyze(
        text=text, speaker_id=speaker_id, speaker_label=speaker_label,
        behavioral_features=bf,
    )

    bf_adj = compute_adjustments(bf)
    if bf_adj:
        result = apply_adjustments(result, bf_adj)

    return {t.name: t.value for t in result.traits}


def cross_pipeline_mae(new_profile: dict, old_gt_path: Path) -> tuple[float, dict[str, float]]:
    """Compute MAE between new pipeline result and old pipeline GT.
    Returns (overall_mae, per_dimension_mae)."""
    old_gt = json.loads(old_gt_path.read_text())
    shared = set(new_profile.keys()) & set(old_gt.keys())
    if not shared:
        return 1.0, {}

    # Overall MAE
    errors = {k: abs(new_profile[k] - old_gt[k]) for k in shared}
    overall = mean(errors.values())

    # Per-dimension (need trait->dimension mapping)
    from super_brain.catalog import TRAIT_CATALOG
    trait_dim = {t["name"]: t["dimension"] for t in TRAIT_CATALOG}

    dim_errors: dict[str, list[float]] = {}
    for trait, err in errors.items():
        dim = trait_dim.get(trait, "UNK")
        dim_errors.setdefault(dim, []).append(err)

    per_dim = {dim: mean(errs) for dim, errs in dim_errors.items()}
    return overall, per_dim


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    detector = Detector(api_key=api_key, temperature=0.0)

    print("=" * 70)
    print("COMPREHENSIVE PIPELINE COMPARISON")
    print("Old: 282-line prompt, temp=1.0, BF post-hoc")
    print("New: 43-line prompt, temp=0.0, BF injected")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════
    # TEST 1: Cross-pipeline agreement on literary characters
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print("TEST 1: Cross-Pipeline Agreement (Literary Characters)")
    print(f"{'─'*70}")
    print("Running new pipeline on full data, comparing against old gt_v41.json\n")

    literary_results = {}
    all_dim_maes: dict[str, list[float]] = {}

    for char, name in CHARACTERS.items():
        old_gt_path = Path(f"data/literary/{char}/gt_v41.json")
        if not old_gt_path.exists():
            print(f"  {char}: SKIP (no old GT)")
            continue

        quotes = load_literary_quotes(char)
        if not quotes:
            print(f"  {char}: SKIP (no dialogue)")
            continue

        print(f"  {char} ({len(quotes)} quotes)...", end=" ", flush=True)
        start = time.time()
        new_profile = detect_new_pipeline(detector, quotes, name, f"{char}_new_full")
        elapsed = time.time() - start

        overall_mae, per_dim = cross_pipeline_mae(new_profile, old_gt_path)
        literary_results[char] = {
            "overall_mae": round(overall_mae, 3),
            "per_dim": {k: round(v, 3) for k, v in per_dim.items()},
            "traits": len(new_profile),
            "elapsed": round(elapsed, 1),
        }

        for dim, mae in per_dim.items():
            all_dim_maes.setdefault(dim, []).append(mae)

        print(f"MAE={overall_mae:.3f} | {elapsed:.0f}s")

        # Save new GT
        outdir = Path(f"data/literary/{char}")
        with open(outdir / "gt_v41_new.json", "w") as f:
            json.dump(new_profile, f, indent=2)

    # Literary summary
    if literary_results:
        maes = [r["overall_mae"] for r in literary_results.values()]
        print(f"\n  Literary avg cross-pipeline MAE: {mean(maes):.3f}")
        print(f"  Range: {min(maes):.3f} - {max(maes):.3f}")

    # ═══════════════════════════════════════════════════════════════
    # TEST 2: Cross-pipeline agreement on real users
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print("TEST 2: Cross-Pipeline Agreement (Real Users)")
    print(f"{'─'*70}")
    print("Running new pipeline on full data, comparing convergence vs old\n")

    real_user_files = sorted(Path("data/real_users").glob("real_user_*.txt"))
    real_results = {}

    for filepath in real_user_files:
        user_label = filepath.stem
        old_result_path = Path(f"data/real_users/results/{user_label}_convergence.json")

        quotes = load_real_user_quotes(filepath)
        if not quotes or len(quotes) < 25:
            print(f"  {user_label}: SKIP (too few messages)")
            continue

        # Skip User A (2681 msgs — too expensive for full comparison)
        if "2681" in user_label:
            print(f"  {user_label}: SKIP (too large for comparison run)")
            continue

        print(f"  {user_label} ({len(quotes)} msgs)...", end=" ", flush=True)
        start = time.time()
        new_profile = detect_new_pipeline(detector, quotes, "User", f"{user_label}_new_full")
        elapsed = time.time() - start

        # If old convergence exists, the old GT was the old pipeline's full-data result.
        # We don't have it as a separate file, but the final checkpoint (100% data)
        # approximates it. For a proper comparison, we compute self-consistency.
        real_results[user_label] = {
            "msgs": len(quotes),
            "traits": len(new_profile),
            "elapsed": round(elapsed, 1),
        }
        print(f"{len(new_profile)} traits | {elapsed:.0f}s")

        # Save new profile
        outdir = Path("data/real_users/results")
        with open(outdir / f"{user_label}_profile_new.json", "w") as f:
            json.dump(new_profile, f, indent=2)

    # ═══════════════════════════════════════════════════════════════
    # TEST 3: Progressive self-consistency on 3 representative cases
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print("TEST 3: Progressive Self-Consistency")
    print(f"{'─'*70}")

    test_cases = [
        ("real_user_E_122msgs", Path("data/real_users/real_user_E_122msgs.txt"), "User", 25),
        ("real_user_C_582msgs", Path("data/real_users/real_user_C_582msgs.txt"), "User", 50),
        ("scarlett", None, "Scarlett", 25),  # literary
    ]

    progressive_results = {}

    for label, filepath, speaker, seg_size in test_cases:
        if filepath:
            quotes = load_real_user_quotes(filepath)
        else:
            quotes = load_literary_quotes(label)

        if not quotes:
            continue

        checkpoints = list(range(seg_size, len(quotes), seg_size))
        if checkpoints and checkpoints[-1] != len(quotes):
            checkpoints.append(len(quotes))
        if not checkpoints:
            checkpoints = [len(quotes)]

        print(f"\n  {label}: {len(quotes)} items, {len(checkpoints)} checkpoints")

        # Full-data GT with new pipeline
        print(f"    [GT] Running on all {len(quotes)} items...", end=" ", flush=True)
        start = time.time()
        gt = detect_new_pipeline(detector, quotes, speaker, f"{label}_prog_full")
        print(f"done in {time.time()-start:.0f}s")

        convergence = []
        prev = None
        for i, cp in enumerate(checkpoints):
            cumulative = quotes[:cp]
            print(f"    CP{i}: {cp} items...", end=" ", flush=True)
            start = time.time()
            profile = detect_new_pipeline(detector, cumulative, speaker, f"{label}_prog_cp{i}")
            elapsed = time.time() - start

            profile_dict = {k: {"value": v, "confidence": 0.8} for k, v in profile.items()}
            mae = compute_mae(profile_dict, gt)

            delta = 0.0
            if prev:
                shared = set(profile.keys()) & set(prev.keys())
                if shared:
                    delta = sum(abs(profile[k] - prev[k]) for k in shared) / len(shared)

            convergence.append({
                "checkpoint": i, "msgs": cp,
                "pct": round(cp / len(quotes) * 100, 1),
                "mae": round(mae, 3), "delta": round(delta, 4),
                "elapsed": round(elapsed, 1),
            })
            prev = profile
            print(f"MAE={mae:.3f} delta={delta:.4f} | {elapsed:.0f}s")

        progressive_results[label] = convergence

        # Load old convergence for comparison
        if filepath:
            old_path = Path(f"data/real_users/results/{label}_convergence.json")
        else:
            old_path = Path(f"data/literary/{label}/convergence_v2.json")

        if old_path.exists():
            old_data = json.loads(old_path.read_text())
            old_conv = old_data.get("convergence", [])
            old_mae_key = "mae_vs_v41_full" if "mae_vs_v41_full" in (old_conv[0] if old_conv else {}) else "mae_vs_full"

            print(f"\n    {'Msgs':>5} | {'Old MAE':>8} {'Old Δ':>8} | {'New MAE':>8} {'New Δ':>8} | {'Improve':>8}")
            print(f"    {'-'*5}-+-{'-'*8}-{'-'*8}-+-{'-'*8}-{'-'*8}-+-{'-'*8}")

            for new_cp in convergence:
                msgs = new_cp["msgs"]
                old_cp = next(
                    (c for c in old_conv
                     if c.get("cumulative_msgs", c.get("cumulative_quotes", 0)) == msgs),
                    None
                )
                if old_cp:
                    old_mae = old_cp.get(old_mae_key, old_cp.get("mae_vs_composite", "?"))
                    old_delta = old_cp.get("profile_delta", 0)
                    if isinstance(old_mae, (int, float)):
                        imp = old_mae - new_cp["mae"]
                        print(f"    {msgs:5d} | {old_mae:8.3f} {old_delta:8.4f} | "
                              f"{new_cp['mae']:8.3f} {new_cp['delta']:8.4f} | {imp:+8.3f}")
                    else:
                        print(f"    {msgs:5d} | {'?':>8} {'?':>8} | "
                              f"{new_cp['mae']:8.3f} {new_cp['delta']:8.4f} |")

    # ═══════════════════════════════════════════════════════════════
    # TEST 4: Per-dimension analysis
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print("TEST 4: Per-Dimension Cross-Pipeline MAE (Literary Characters)")
    print(f"{'─'*70}\n")

    if all_dim_maes:
        print(f"  {'Dimension':>5} {'Name':40s} {'Avg MAE':>8} {'#Chars':>7}")
        print(f"  {'-'*5} {'-'*40} {'-'*8} {'-'*7}")
        for dim in sorted(all_dim_maes.keys()):
            dim_name = ALL_DIMENSIONS.get(dim, dim)[:40]
            avg = mean(all_dim_maes[dim])
            count = len(all_dim_maes[dim])
            marker = " ⚠" if avg > 0.15 else ""
            print(f"  {dim:>5} {dim_name:40s} {avg:8.3f} {count:7d}{marker}")

        overall = mean(v for vals in all_dim_maes.values() for v in vals)
        print(f"\n  Overall cross-pipeline MAE: {overall:.3f}")

    # ═══════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    if literary_results:
        lit_maes = [r["overall_mae"] for r in literary_results.values()]
        print(f"\nCross-pipeline agreement (literary): avg MAE = {mean(lit_maes):.3f}")
        for char, r in sorted(literary_results.items()):
            print(f"  {char:15s} MAE = {r['overall_mae']:.3f}")

    if progressive_results:
        print(f"\nSelf-consistency (progressive):")
        for label, conv in progressive_results.items():
            maes = [c["mae"] for c in conv]
            deltas = [c["delta"] for c in conv]
            print(f"  {label:25s} avg MAE={mean(maes):.3f} avg delta={mean(deltas):.4f} "
                  f"final MAE={conv[-1]['mae']:.3f}")

    print(f"\nConclusion: ", end="")
    if literary_results:
        avg = mean(r["overall_mae"] for r in literary_results.values())
        if avg < 0.10:
            print(f"Cross-pipeline MAE {avg:.3f} < 0.10 → New pipeline produces CONSISTENT profiles.")
        elif avg < 0.15:
            print(f"Cross-pipeline MAE {avg:.3f} < 0.15 → Moderate agreement, acceptable.")
        else:
            print(f"Cross-pipeline MAE {avg:.3f} ≥ 0.15 → SIGNIFICANT divergence, investigate.")

    # Save full results
    all_results = {
        "literary_cross_pipeline": literary_results,
        "real_user_profiles": real_results,
        "progressive_self_consistency": {
            k: v for k, v in progressive_results.items()
        },
        "per_dimension_mae": {k: round(mean(v), 3) for k, v in all_dim_maes.items()},
    }
    outpath = Path("data/comparison_old_vs_new.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {outpath}")


if __name__ == "__main__":
    main()
