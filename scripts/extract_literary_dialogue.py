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
    "darcy": {
        "urls": [
            "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",  # Pride & Prejudice
        ],
        "character": "Mr. Darcy",
        "attribution_patterns": "Darcy said, said Darcy, Mr. Darcy said, said Mr. Darcy, Darcy replied, Darcy cried, he said (when Darcy is the subject), he replied (when Darcy is the subject)",
    },
    "watson": {
        "urls": [
            "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",  # Adventures
        ],
        "character": "Dr. Watson",
        "attribution_patterns": "Watson said, said Watson, I said (Watson is the narrator), I cried, I replied, I answered, I remarked, I exclaimed",
    },
    "hamlet": {
        "urls": [
            "https://www.gutenberg.org/cache/epub/1524/pg1524.txt",  # Hamlet
        ],
        "character": "Hamlet",
        "attribution_patterns": "Ham. (speech prefix in play format), HAMLET. — this is a play, so look for lines prefixed with 'Ham.' or 'HAMLET' as the speaker",
    },
    "jane_eyre": {
        "urls": [
            "https://www.gutenberg.org/cache/epub/1260/pg1260.txt",  # Jane Eyre
        ],
        "character": "Jane Eyre",
        "attribution_patterns": "I said, I replied, I answered, I cried, I exclaimed, said I, Jane said, said Jane (Jane is the first-person narrator)",
    },
    "scrooge": {
        "urls": [
            "https://www.gutenberg.org/cache/epub/46/pg46.txt",  # A Christmas Carol
        ],
        "character": "Ebenezer Scrooge",
        "attribution_patterns": "Scrooge said, said Scrooge, Scrooge replied, Scrooge cried, Scrooge exclaimed, he said (when Scrooge is the subject)",
    },
    "ahab": {
        "urls": [
            "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",  # Moby Dick
        ],
        "character": "Captain Ahab",
        "attribution_patterns": "Ahab said, said Ahab, cried Ahab, Ahab cried, Captain Ahab said, he said (when Ahab is the subject)",
    },
    "huck": {
        "urls": [
            "https://www.gutenberg.org/cache/epub/76/pg76.txt",  # Adventures of Huck Finn
        ],
        "character": "Huckleberry Finn",
        "attribution_patterns": "I says, says I, I said, said I, I told (Huck is the first-person narrator using dialect 'I says')",
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
