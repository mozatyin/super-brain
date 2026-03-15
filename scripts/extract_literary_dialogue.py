"""Extract character dialogue from Project Gutenberg texts.

Downloads public domain texts and extracts explicitly attributed dialogue.

Usage:
    ANTHROPIC_API_KEY=... python scripts/extract_literary_dialogue.py sherlock
    ANTHROPIC_API_KEY=... python scripts/extract_literary_dialogue.py elizabeth
"""

import json
import os
import sys
import re
import subprocess
from pathlib import Path

import anthropic

SOURCES = {
    "sherlock": {
        "urls": [
            "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",  # Adventures
        ],
        "character": "Sherlock Holmes",
        "attribution_patterns": "Holmes said, said Holmes, Holmes remarked, Holmes replied, Holmes cried, Holmes answered, he said (when Holmes is the subject)",
    },
    "elizabeth": {
        "urls": [
            "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",  # Pride & Prejudice
        ],
        "character": "Elizabeth Bennet",
        "attribution_patterns": "Elizabeth said, said Elizabeth, she replied, she cried, Elizabeth cried, Lizzy said, said Lizzy, Miss Bennet said, Elizabeth answered",
    },
}


def download_text(url: str) -> str:
    """Download text from URL using curl."""
    result = subprocess.run(
        ["curl", "-sL", url], capture_output=True, text=True, timeout=60,
    )
    return result.stdout


def extract_dialogue(text: str, character: str, patterns: str, api_key: str) -> list[str]:
    """Use LLM to extract character dialogue from book text.

    Processes in chunks to handle long texts.
    """
    kwargs: dict = {"api_key": api_key}
    if api_key.startswith("sk-or-"):
        kwargs["base_url"] = "https://openrouter.ai/api"
    client = anthropic.Anthropic(**kwargs)

    # Split text into manageable chunks (~30K chars each)
    chunk_size = 30000
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    all_quotes = []
    for i, chunk in enumerate(chunks):
        print(f"  Processing chunk {i+1}/{len(chunks)}...")
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            messages=[{"role": "user", "content": (
                f"Extract ALL dialogue lines spoken by {character} from this text.\n"
                f"Look for attribution patterns like: {patterns}\n\n"
                f"Rules:\n"
                f"- Include ONLY lines explicitly attributed to {character}\n"
                f"- Return one quote per line, numbered [1], [2], etc.\n"
                f"- Include the full quote text, not the attribution\n"
                f"- If no quotes found in this chunk, return 'NONE'\n\n"
                f"TEXT:\n{chunk}"
            )}],
        )
        result = response.content[0].text
        if result.strip() != "NONE":
            # Parse numbered quotes
            for m in re.finditer(r'\[\d+\]\s*"?(.+?)"?\s*$', result, re.MULTILINE):
                quote = m.group(1).strip().strip('"')
                if len(quote) > 5:
                    all_quotes.append(quote)

    return all_quotes


def main():
    character = sys.argv[1] if len(sys.argv) > 1 else "sherlock"
    api_key = os.environ["ANTHROPIC_API_KEY"]

    if character not in SOURCES:
        print(f"Unknown: {character}. Options: {list(SOURCES.keys())}")
        sys.exit(1)

    src = SOURCES[character]
    print(f"Extracting dialogue for {src['character']}...")

    full_text = ""
    for url in src["urls"]:
        print(f"  Downloading {url}...")
        full_text += download_text(url) + "\n\n"

    print(f"  Text length: {len(full_text)} chars")

    quotes = extract_dialogue(
        full_text, src["character"], src["attribution_patterns"], api_key,
    )

    outdir = Path(f"data/literary/{character}")
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "dialogue.txt"
    with open(outfile, "w") as f:
        for i, q in enumerate(quotes, 1):
            f.write(f"[{i}] {q}\n\n")

    print(f"Extracted {len(quotes)} quotes → {outfile}")


if __name__ == "__main__":
    main()
