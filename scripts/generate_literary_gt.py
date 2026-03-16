"""Generate LLM omniscient ground truth for literary characters.

Usage:
    ANTHROPIC_API_KEY=... python scripts/generate_literary_gt.py scarlett
    ANTHROPIC_API_KEY=... python scripts/generate_literary_gt.py sherlock
    ANTHROPIC_API_KEY=... python scripts/generate_literary_gt.py elizabeth
"""

import json
import os
import sys
import re
from pathlib import Path

import anthropic

from super_brain.catalog import TRAIT_CATALOG

CHARACTERS = {
    "scarlett": {
        "full_name": "Scarlett O'Hara",
        "book": "Gone with the Wind by Margaret Mitchell",
        "description": (
            "Scarlett O'Hara, the protagonist of Gone with the Wind. "
            "Consider her ENTIRE arc: the spoiled belle, the war survivor, "
            "the ruthless businesswoman, her relationships with Ashley, Rhett, "
            "and Melanie. Use ALL available information: her dialogue, actions, "
            "inner monologue, other characters' perceptions of her."
        ),
    },
    "sherlock": {
        "full_name": "Sherlock Holmes",
        "book": "The Sherlock Holmes stories by Arthur Conan Doyle",
        "description": (
            "Sherlock Holmes as depicted across the original Conan Doyle canon. "
            "Consider: his methods, his relationships with Watson/Mycroft/Moriarty, "
            "his drug use, his attitude toward emotion and sentiment, his violin, "
            "his disguises, his treatment of clients."
        ),
    },
    "elizabeth": {
        "full_name": "Elizabeth Bennet",
        "book": "Pride and Prejudice by Jane Austen",
        "description": (
            "Elizabeth Bennet, the protagonist of Pride and Prejudice. "
            "Consider: her wit, her misjudgment of Darcy and Wickham, "
            "her family dynamics, her independence, her growth arc, "
            "her relationship with Jane, her sharp tongue."
        ),
    },
    "darcy": {
        "full_name": "Mr. Darcy",
        "book": "Pride and Prejudice by Jane Austen",
        "description": (
            "Fitzwilliam Darcy from Pride and Prejudice. "
            "Consider: his pride and social reserve, his letter to Elizabeth, "
            "his transformation, his loyalty to Bingley, his treatment of Wickham, "
            "his generosity in secret, his awkwardness in social situations."
        ),
    },
    "watson": {
        "full_name": "Dr. John Watson",
        "book": "The Sherlock Holmes stories by Arthur Conan Doyle",
        "description": (
            "Dr. Watson as depicted in the original Conan Doyle canon. "
            "Consider: his loyalty to Holmes, his military background, "
            "his warmth and sociability, his practical nature, his role as narrator, "
            "his marriages, his medical practice, his courage."
        ),
    },
    "hamlet": {
        "full_name": "Prince Hamlet",
        "book": "Hamlet by William Shakespeare",
        "description": (
            "Prince Hamlet from Shakespeare's tragedy. "
            "Consider: his indecisiveness, his intellectual nature, his grief, "
            "his relationship with Ophelia/Gertrude/Claudius, his 'antic disposition', "
            "his philosophical soliloquies, his eventual action, his wit and wordplay."
        ),
    },
    "jane_eyre": {
        "full_name": "Jane Eyre",
        "book": "Jane Eyre by Charlotte Brontë",
        "description": (
            "Jane Eyre, the protagonist and narrator. "
            "Consider: her fierce independence, her moral strength, her passion, "
            "her childhood suffering, her relationship with Rochester, "
            "her refusal to compromise her principles, her plain appearance "
            "but rich inner life."
        ),
    },
    "scrooge": {
        "full_name": "Ebenezer Scrooge",
        "book": "A Christmas Carol by Charles Dickens",
        "description": (
            "Ebenezer Scrooge across his FULL arc in A Christmas Carol. "
            "Consider BOTH his cold, miserly nature at the start AND his "
            "transformation after the three spirits. His isolation, his greed, "
            "his fear of death, his eventual generosity and joy. "
            "Weight the pre-transformation personality more heavily (it's his baseline)."
        ),
    },
    "ahab": {
        "full_name": "Captain Ahab",
        "book": "Moby Dick by Herman Melville",
        "description": (
            "Captain Ahab from Moby Dick. "
            "Consider: his monomaniacal obsession with the white whale, "
            "his charisma and ability to command, his defiance of God and nature, "
            "his scarred body and tortured soul, his manipulation of the crew, "
            "his moments of tenderness (Starbuck, Pip), his ultimate self-destruction."
        ),
    },
    "huck": {
        "full_name": "Huckleberry Finn",
        "book": "Adventures of Huckleberry Finn by Mark Twain",
        "description": (
            "Huck Finn as narrator and protagonist. "
            "Consider: his rejection of 'sivilization', his moral growth regarding Jim, "
            "his street-smart survival skills, his superstitions, his kindness, "
            "his ability to lie and improvise, his independence, his dialect speech."
        ),
    },
}


def generate_gt(character_key: str, api_key: str) -> dict[str, float]:
    """Generate ground truth trait scores using LLM omniscient evaluation."""
    char = CHARACTERS[character_key]

    trait_list = "\n".join(
        f"- {t['dimension']}:{t['name']}: {t['description']}"
        for t in TRAIT_CATALOG
    )

    prompt = (
        f"You are a literary psychologist scoring the personality of "
        f"{char['full_name']} from {char['book']}.\n\n"
        f"{char['description']}\n\n"
        f"Score each of these 69 personality traits on a 0.0-1.0 scale.\n"
        f"USE THE FULL RANGE. This is a fictional character with extreme traits — "
        f"do NOT cluster scores in the middle.\n\n"
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
    # Extract JSON from response
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"Could not parse JSON from response: {text[:200]}")


def main():
    character = sys.argv[1] if len(sys.argv) > 1 else "scarlett"
    api_key = os.environ["ANTHROPIC_API_KEY"]

    if character not in CHARACTERS:
        print(f"Unknown character: {character}. Options: {list(CHARACTERS.keys())}")
        sys.exit(1)

    print(f"Generating GT for {CHARACTERS[character]['full_name']}...")
    gt = generate_gt(character, api_key)

    outdir = Path(f"data/literary/{character}")
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "gt_llm.json"
    with open(outfile, "w") as f:
        json.dump(gt, f, indent=2)
    print(f"Saved {len(gt)} traits to {outfile}")


if __name__ == "__main__":
    main()
