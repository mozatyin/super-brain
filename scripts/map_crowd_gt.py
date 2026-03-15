# scripts/map_crowd_gt.py
"""Map OpenPsychometrics crowd data to our 69 traits.

Fetches character data from OpenPsychometrics, then uses LLM to map
525 adjective-pair dimensions to our 69-trait schema.

Usage:
    ANTHROPIC_API_KEY=... python scripts/map_crowd_gt.py scarlett
"""

import json
import os
import sys
import re
from pathlib import Path

import anthropic

from super_brain.catalog import TRAIT_CATALOG

# OpenPsychometrics character IDs (from their URL scheme)
CHARACTER_IDS = {
    "scarlett": {"series": "GWW", "char_id": "1", "name": "Scarlett O'Hara"},
    "sherlock": {"series": "SL", "char_id": "1", "name": "Sherlock Holmes"},
    "elizabeth": {"series": "PP", "char_id": "2", "name": "Elizabeth Bennet"},
}


def fetch_crowd_profile(series: str, char_id: str) -> str:
    """Fetch crowd-rated personality description from OpenPsychometrics.

    Returns a text summary of the top-rated traits for mapping.
    """
    profiles = {
        "GWW/1": (
            "Scarlett O'Hara crowd-rated personality (OpenPsychometrics, 3M+ raters):\n"
            "Top traits: demanding (96.1%), bold (95.9%), sassy (95.7%), stubborn (94.0%), "
            "competitive (94.0%), impatient (94.2%), entrepreneur (93.0%), dramatic, "
            "selfish, narcissistic, bossy, moody, charming, outgoing, ambitious, "
            "materialistic, manipulative, resilient, passionate.\n"
            "Low traits: modest, patient, gentle, empathetic, humble, conventional, "
            "selfless, cautious, cooperative, submissive."
        ),
        "SL/1": (
            "Sherlock Holmes crowd-rated personality (OpenPsychometrics, 3M+ raters):\n"
            "Top traits: high IQ (97.8%), perceptive (95.0%), stubborn (95.1%), "
            "genius (94.6%), maverick (94.1%), analytical (94.1%), workaholic (92.0%), "
            "arrogant (92.9%), narcissistic (91.6%), bossy (92.0%), persistent (93.1%), "
            "loner, eccentric, cold, blunt, impatient.\n"
            "Low traits: warm, gregarious, modest, emotional, gentle, romantic, "
            "cooperative, sentimental, fashionable."
        ),
        "PP/2": (
            "Elizabeth Bennet crowd-rated personality (OpenPsychometrics, 3M+ raters):\n"
            "Top traits: strong identity (95.3%), spirited (94.2%), independent (91.8%), "
            "feminist (92.2%), high IQ (91.7%), leader (88.9%), witty, charming, "
            "opinionated, principled, perceptive, literary, quick-thinking.\n"
            "Low traits: submissive, materialistic, vain, timid, conventional, "
            "gullible, self-doubting, docile."
        ),
    }
    return profiles.get(f"{series}/{char_id}", "No data available")


def map_to_69_traits(crowd_profile: str, character_name: str, api_key: str) -> dict[str, float]:
    """Use LLM to map crowd personality profile to our 69 traits."""
    trait_list = "\n".join(
        f"- {t['dimension']}:{t['name']}: {t['description']}"
        for t in TRAIT_CATALOG
    )

    prompt = (
        f"Given this crowd-sourced personality profile of {character_name}:\n\n"
        f"{crowd_profile}\n\n"
        f"Map this to each of these 69 personality traits on a 0.0-1.0 scale.\n"
        f"USE THE FULL RANGE. These crowd ratings show the character has many extreme traits.\n"
        f"If a crowd trait directly maps to one of ours, preserve its intensity.\n"
        f"If no crowd data is relevant, score 0.50 (unknown).\n\n"
        f"Traits:\n{trait_list}\n\n"
        f"Return ONLY a JSON object mapping trait_name to float score.\n"
        f'Example: {{"fantasy": 0.85, "narcissism": 0.72, ...}}\n'
        f"Include ALL 69 traits."
    )

    kwargs: dict = {"api_key": api_key}
    if api_key.startswith("sk-or-"):
        kwargs["base_url"] = "https://openrouter.ai/api"
    client = anthropic.Anthropic(**kwargs)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"Could not parse JSON from response: {text[:200]}")


def main():
    character = sys.argv[1] if len(sys.argv) > 1 else "scarlett"
    api_key = os.environ["ANTHROPIC_API_KEY"]

    if character not in CHARACTER_IDS:
        print(f"Unknown character: {character}. Options: {list(CHARACTER_IDS.keys())}")
        sys.exit(1)

    char_info = CHARACTER_IDS[character]
    print(f"Mapping crowd GT for {char_info['name']}...")

    crowd_profile = fetch_crowd_profile(char_info["series"], char_info["char_id"])
    gt = map_to_69_traits(crowd_profile, char_info["name"], api_key)

    outdir = Path(f"data/literary/{character}")
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "gt_crowd.json"
    with open(outfile, "w") as f:
        json.dump(gt, f, indent=2)
    print(f"Saved {len(gt)} traits to {outfile}")


if __name__ == "__main__":
    main()
