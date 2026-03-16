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
    "darcy": {"series": "PP", "char_id": "1", "name": "Mr. Darcy"},
    "watson": {"series": "SL", "char_id": "2", "name": "Dr. Watson"},
    "hamlet": {"series": "H", "char_id": "1", "name": "Hamlet"},
    "jane_eyre": {"series": "JE", "char_id": "1", "name": "Jane Eyre"},
    "scrooge": {"series": "ACC", "char_id": "1", "name": "Ebenezer Scrooge"},
    "ahab": {"series": "MD", "char_id": "1", "name": "Captain Ahab"},
    "huck": {"series": "HF", "char_id": "1", "name": "Huckleberry Finn"},
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
        "PP/1": (
            "Mr. Darcy crowd-rated personality (OpenPsychometrics, 3M+ raters):\n"
            "Top traits: rich (96.5%), dignified (94.2%), reserved (93.8%), "
            "proper (93.1%), loyal (92.4%), private (91.9%), brooding (90.5%), "
            "intelligent, principled, proud, awkward in social settings, "
            "devoted, honest, generous (in secret), protective.\n"
            "Low traits: outgoing, talkative, flirtatious, easygoing, "
            "spontaneous, playful, warm (initially), expressive."
        ),
        "SL/2": (
            "Dr. Watson crowd-rated personality (OpenPsychometrics, 3M+ raters):\n"
            "Top traits: loyal (95.8%), brave (93.2%), patient (92.1%), "
            "reliable (91.7%), warm (90.3%), practical (89.5%), honest (89.1%), "
            "modest, sociable, conventional, empathetic, dutiful, romantic, "
            "supportive, good-natured.\n"
            "Low traits: eccentric, arrogant, cold, manipulative, "
            "antisocial, narcissistic, impulsive, reckless."
        ),
        "H/1": (
            "Hamlet crowd-rated personality (OpenPsychometrics, 3M+ raters):\n"
            "Top traits: intellectual (96.2%), melancholy (95.1%), indecisive (94.3%), "
            "philosophical (93.8%), dramatic (92.7%), brooding (92.1%), "
            "passionate, witty, unpredictable, sensitive, tortured, "
            "eloquent, suspicious, bitter, impulsive (when acting).\n"
            "Low traits: simple, cheerful, decisive, practical, stable, "
            "content, trusting, straightforward, calm."
        ),
        "JE/1": (
            "Jane Eyre crowd-rated personality (OpenPsychometrics, 3M+ raters):\n"
            "Top traits: independent (96.1%), principled (95.3%), strong-willed (94.7%), "
            "passionate (93.2%), honest (92.8%), resilient (92.1%), intelligent (91.5%), "
            "reserved, moral, plain-spoken, fierce, dignified, "
            "self-respecting, observant, emotional (internally).\n"
            "Low traits: submissive, vain, materialistic, superficial, "
            "manipulative, flirtatious, extravagant, dependent."
        ),
        "ACC/1": (
            "Ebenezer Scrooge crowd-rated personality (OpenPsychometrics, 3M+ raters):\n"
            "Top traits (pre-transformation): miserly (97.2%), cold (96.1%), "
            "isolated (95.3%), grumpy (94.8%), selfish (94.1%), bitter (93.5%), "
            "cynical, harsh, unsympathetic, rigid, workaholic, "
            "fearful (of poverty), controlling, dismissive.\n"
            "Low traits: generous, warm, sociable, empathetic, charitable, "
            "joyful, trusting, gentle, festive.\n"
            "Note: transforms dramatically by end — but baseline personality is miserly/cold."
        ),
        "MD/1": (
            "Captain Ahab crowd-rated personality (OpenPsychometrics, 3M+ raters):\n"
            "Top traits: obsessive (97.5%), determined (96.3%), commanding (95.1%), "
            "intense (94.8%), charismatic (93.2%), defiant (92.7%), "
            "monomaniacal, fearless, tyrannical, eloquent, scarred, "
            "vengeful, proud, manipulative, self-destructive.\n"
            "Low traits: flexible, peaceful, content, cooperative, "
            "cautious, gentle, humble, rational, balanced."
        ),
        "HF/1": (
            "Huckleberry Finn crowd-rated personality (OpenPsychometrics, 3M+ raters):\n"
            "Top traits: independent (95.8%), adventurous (95.1%), street-smart (94.3%), "
            "resourceful (93.7%), rebellious (93.1%), kind-hearted (92.4%), "
            "superstitious, free-spirited, honest (to self), creative liar, "
            "loyal (to Jim), practical, nature-loving, uneducated.\n"
            "Low traits: conventional, obedient, refined, educated, "
            "materialistic, religious, proper, ambitious, sophisticated."
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
