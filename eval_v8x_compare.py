"""V8x prompt compression eval — compare original vs V8x on 3 short texts.

Usage:
    cd /Users/michael/super-brain && .venv/bin/python eval_v8x_compare.py
"""

import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from super_brain.catalog import TRAIT_CATALOG, ALL_DIMENSIONS
from super_brain.detector import (
    Detector,
    DIMENSION_BATCHES,
    _get_traits_for_batch,
    _build_trait_prompt,
    _get_calibration_examples,
    _parse_batch_response,
    _clamp,
    _SYSTEM_PROMPT,
)
from super_brain.v8x_prompt import V8X_SYSTEM_PROMPT, V8X_BATCH_CALIBRATION_EXAMPLES
from super_brain.api_retry import retry_api_call
from super_brain.models import Trait
import anthropic

# ── 3 short personality-rich texts ──────────────────────────────────────────

TEXTS = {
    "dominant_leader": (
        "Alex: Look, I've been leading teams for 15 years. The problem is simple — "
        "nobody wants to make the hard call. I will. We cut the underperforming division, "
        "redirect resources to growth, and stop wasting time on consensus-building. "
        "I told the board exactly that, and they agreed because the data backs me up. "
        "People respect decisiveness, not hand-wringing."
    ),
    "anxious_creative": (
        "Sam: I don't know... maybe this sounds weird but I keep having these vivid "
        "dreams about building entire cities? Like floating ones. And then I wake up "
        "terrified that I'll never actually create anything meaningful. What if I'm "
        "just fooling myself? Sometimes I think maybe I should just get a normal job... "
        "but then another idea hits me and I can't stop thinking about it."
    ),
    "warm_empathetic": (
        "Jamie: Oh gosh, I'm so sorry you're going through that. I remember when my "
        "mom was sick — it felt like the ground was disappearing under me. Please know "
        "you're not alone in this. If you need to talk at 3am, call me. I mean it. "
        "People don't say that enough — that it's okay to not be okay. You're doing "
        "better than you think."
    ),
}

# We'll only run 2 key batches to keep cost/time down
TEST_BATCHES = [
    ["OPN", "CON"],   # Batch 1
    ["EXT", "AGR"],   # Batch 2
    ["NEU", "HON"],   # Batch 3
    ["DRK", "EMO"],   # Batch 4
]


def detect_with_prompt(client, model, system_prompt, calib_source, text, speaker, batches):
    """Run detection with a specific system prompt."""
    all_traits = {}
    for batch_dims in batches:
        batch_traits = _get_traits_for_batch(batch_dims)
        if not batch_traits:
            continue

        dim_labels = ", ".join(f"{d} ({ALL_DIMENSIONS.get(d, d)})" for d in batch_dims)
        trait_prompt = _build_trait_prompt(batch_traits)

        # Get calibration from appropriate source
        key = ",".join(batch_dims)
        if calib_source == "v8x":
            calibration = V8X_BATCH_CALIBRATION_EXAMPLES.get(key, "")
        else:
            calibration = _get_calibration_examples(batch_dims)
        calibration_section = f"\n{calibration}\n" if calibration else ""

        user_message = (
            f"## Text Sample\n\n{text}\n\n"
            f"## Target Speaker\n\nAnalyze speaker labeled '{speaker}'.\n\n"
            f"## Dimensions to Analyze: {dim_labels}\n\n"
            f"{trait_prompt}\n\n"
            f"{calibration_section}"
            f"Return JSON with 'reasoning' and 'scores' arrays as specified. "
            f"Analyze ONLY the {len(batch_traits)} traits listed above. "
            f"You MUST return exactly {len(batch_traits)} scores."
        )

        response = retry_api_call(lambda: client.messages.create(
            model=model,
            max_tokens=8192,
            temperature=0.0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        ))

        raw = response.content[0].text if response and response.content else '{"scores":[]}'
        parsed = _parse_batch_response(raw)

        for item in parsed:
            all_traits[item["name"]] = _clamp(item["value"])

    return all_traits


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    kwargs = {"api_key": api_key}
    if api_key.startswith("sk-or-"):
        kwargs["base_url"] = "https://openrouter.ai/api"
    client = anthropic.Anthropic(**kwargs)
    model = "claude-sonnet-4-20250514"

    print("=" * 70)
    print("SUPER BRAIN V8x PROMPT COMPRESSION EVAL")
    print(f"Batches: {len(TEST_BATCHES)} | Texts: {len(TEXTS)} | Model: {model}")
    print("=" * 70)

    all_diffs = []
    top5_matches = 0
    top5_total = 0

    for text_name, text in TEXTS.items():
        speaker = text.split(":")[0].strip()
        print(f"\n{'─' * 60}")
        print(f"Text: {text_name} (speaker={speaker})")
        print(f"{'─' * 60}")

        # Original prompt
        print("  Running ORIGINAL prompt...", end=" ", flush=True)
        t0 = time.time()
        orig = detect_with_prompt(client, model, _SYSTEM_PROMPT, "original", text, speaker, TEST_BATCHES)
        print(f"done ({time.time()-t0:.1f}s, {len(orig)} traits)")

        # V8x prompt
        print("  Running V8x prompt...", end=" ", flush=True)
        t0 = time.time()
        v8x = detect_with_prompt(client, model, V8X_SYSTEM_PROMPT, "v8x", text, speaker, TEST_BATCHES)
        print(f"done ({time.time()-t0:.1f}s, {len(v8x)} traits)")

        # Compare
        common = set(orig.keys()) & set(v8x.keys())
        diffs = []
        for trait in sorted(common):
            diff = abs(orig[trait] - v8x[trait])
            diffs.append((trait, orig[trait], v8x[trait], diff))
            all_diffs.append(diff)

        # Sort by diff descending
        diffs.sort(key=lambda x: -x[3])

        # Top-5 comparison
        orig_top5 = sorted(common, key=lambda t: orig[t], reverse=True)[:5]
        v8x_top5 = sorted(common, key=lambda t: v8x[t], reverse=True)[:5]
        overlap = len(set(orig_top5) & set(v8x_top5))
        top5_matches += overlap
        top5_total += 5

        print(f"\n  {'Trait':<30} {'Orig':>6} {'V8x':>6} {'Diff':>6}")
        print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*6}")

        # Show worst 10 diffs
        for trait, o, v, d in diffs[:10]:
            flag = " <<<" if d > 0.15 else " <" if d > 0.10 else ""
            print(f"  {trait:<30} {o:>6.2f} {v:>6.2f} {d:>6.3f}{flag}")

        mae = sum(d for _, _, _, d in diffs) / len(diffs) if diffs else 0
        max_diff = max(d for _, _, _, d in diffs) if diffs else 0
        within_01 = sum(1 for _, _, _, d in diffs if d <= 0.10)
        within_015 = sum(1 for _, _, _, d in diffs if d <= 0.15)

        print(f"\n  MAE: {mae:.3f} | Max: {max_diff:.3f} | ≤0.10: {within_01}/{len(diffs)} | ≤0.15: {within_015}/{len(diffs)}")
        print(f"  Top-5 overlap: {overlap}/5 — orig={orig_top5} v8x={v8x_top5}")

    # Overall
    print(f"\n{'=' * 70}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 70}")
    overall_mae = sum(all_diffs) / len(all_diffs) if all_diffs else 0
    overall_max = max(all_diffs) if all_diffs else 0
    within_01 = sum(1 for d in all_diffs if d <= 0.10)
    within_015 = sum(1 for d in all_diffs if d <= 0.15)

    print(f"  Total traits compared: {len(all_diffs)}")
    print(f"  Overall MAE (orig vs v8x): {overall_mae:.3f}")
    print(f"  Max diff: {overall_max:.3f}")
    print(f"  Within ±0.10: {within_01}/{len(all_diffs)} ({100*within_01/len(all_diffs):.1f}%)")
    print(f"  Within ±0.15: {within_015}/{len(all_diffs)} ({100*within_015/len(all_diffs):.1f}%)")
    print(f"  Top-5 trait overlap: {top5_matches}/{top5_total} ({100*top5_matches/top5_total:.1f}%)")

    # PASS/FAIL
    passed = overall_mae <= 0.10 and (within_015 / len(all_diffs)) >= 0.80
    status = "PASS" if passed else "FAIL"
    print(f"\n  VERDICT: {status}")
    print(f"  (Criteria: MAE ≤ 0.10 AND ≥80% within ±0.15)")


if __name__ == "__main__":
    main()
