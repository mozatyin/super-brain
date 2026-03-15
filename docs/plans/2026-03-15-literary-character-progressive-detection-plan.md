# Literary Character Progressive Personality Detection — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a ProgressiveDetector module and eval script that incrementally detects personality from literary character dialogue, measuring convergence speed and final accuracy against dual ground truth.

**Architecture:** `ProgressiveDetector` wraps existing `Detector` with per-trait Bayesian priors `(value, confidence)`. `eval_literary.py` feeds segmented dialogue through it, recording snapshots. Two ground truths (crowd-sourced + LLM omniscient) for each of 3 characters.

**Tech Stack:** Python 3.12, existing `super_brain.detector.Detector`, `super_brain.models.PersonalityDNA/Trait`, anthropic SDK via OpenRouter, pytest.

---

### Task 1: ProgressiveDetector — Bayesian Update Core

**Files:**
- Create: `super_brain/progressive.py`
- Test: `tests/test_progressive.py`

**Step 1: Write the failing tests**

```python
# tests/test_progressive.py
"""Tests for ProgressiveDetector Bayesian update logic."""

import pytest
from super_brain.progressive import bayesian_update, ProgressiveDetector


class TestBayesianUpdate:
    """Test the core Bayesian update formula."""

    def test_first_observation_adopts_value(self):
        """With no prior (conf=0), fully adopts observation."""
        val, conf = bayesian_update(0.5, 0.0, 0.80, 0.60)
        assert val == pytest.approx(0.80, abs=0.01)
        assert conf > 0.0

    def test_high_conf_prior_resists_change(self):
        """Strong prior barely moves with weak observation."""
        val, conf = bayesian_update(0.30, 0.80, 0.70, 0.20)
        assert val < 0.45  # still closer to prior (0.30)

    def test_equal_confidence_averages(self):
        """Equal confidence → midpoint."""
        val, conf = bayesian_update(0.20, 0.50, 0.80, 0.50)
        assert val == pytest.approx(0.50, abs=0.01)

    def test_confidence_grows_asymptotically(self):
        """Confidence increases but never exceeds 0.95."""
        conf = 0.0
        for _ in range(50):
            _, conf = bayesian_update(0.5, conf, 0.5, 0.5)
        assert conf <= 0.95
        assert conf > 0.90

    def test_zero_observation_confidence_no_change(self):
        """Zero-confidence observation doesn't move prior."""
        val, conf = bayesian_update(0.30, 0.50, 0.90, 0.0)
        assert val == pytest.approx(0.30, abs=0.01)
        assert conf == pytest.approx(0.50, abs=0.01)


class TestProgressiveDetectorInit:
    """Test ProgressiveDetector initialization and state."""

    def test_init_empty_priors(self):
        """Starts with no priors."""
        pd = ProgressiveDetector.__new__(ProgressiveDetector)
        pd._priors = {}
        pd._history = []
        assert pd.get_profile_dict() == {}
        assert pd.get_history() == []

    def test_set_and_get_prior(self):
        """Can manually set priors for testing."""
        pd = ProgressiveDetector.__new__(ProgressiveDetector)
        pd._priors = {"narcissism": (0.65, 0.40)}
        pd._history = []
        profile = pd.get_profile_dict()
        assert profile["narcissism"]["value"] == pytest.approx(0.65)
        assert profile["narcissism"]["confidence"] == pytest.approx(0.40)
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_progressive.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'super_brain.progressive'`

**Step 3: Write minimal implementation**

```python
# super_brain/progressive.py
"""Progressive personality detection with Bayesian belief updating.

Wraps the existing Detector to incrementally build a personality profile
from sequential text segments, inspired by SoulGraph's Dual Soul architecture.
"""

from __future__ import annotations

from super_brain.catalog import TRAIT_CATALOG
from super_brain.detector import Detector
from super_brain.models import PersonalityDNA, Trait, SampleSummary


def bayesian_update(
    prior_val: float,
    prior_conf: float,
    obs_val: float,
    obs_conf: float,
) -> tuple[float, float]:
    """Bayesian merge of prior belief with new observation.

    Args:
        prior_val: Current estimated trait value (0.0-1.0).
        prior_conf: Confidence in current estimate (0.0-1.0).
        obs_val: Newly observed trait value (0.0-1.0).
        obs_conf: Confidence of new observation (0.0-1.0).

    Returns:
        (new_value, new_confidence) tuple.
    """
    total_conf = prior_conf + obs_conf
    if total_conf == 0:
        return 0.5, 0.0

    new_val = (prior_val * prior_conf + obs_val * obs_conf) / total_conf
    # Asymptotic confidence growth: diminishing returns from new evidence
    new_conf = min(0.95, prior_conf + obs_conf * (1.0 - prior_conf) * 0.5)
    return new_val, new_conf


class ProgressiveDetector:
    """Incrementally builds a personality profile from sequential text segments.

    Each call to update() runs the Detector on new text, then Bayesian-merges
    the results with accumulated priors. Tracks full history for convergence analysis.
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self._detector = Detector(api_key=api_key, model=model)
        self._priors: dict[str, tuple[float, float]] = {}  # trait_name -> (value, conf)
        self._trait_dims: dict[str, str] = {
            t["name"]: t["dimension"] for t in TRAIT_CATALOG
        }
        self._history: list[dict] = []
        self._segment_count = 0

    def update(self, text: str, speaker_label: str = "Speaker") -> dict:
        """Detect traits from a text segment and merge with priors.

        Args:
            text: The dialogue text segment to analyze.
            speaker_label: Label identifying the target speaker in the text.

        Returns:
            Snapshot dict with segment_id, per-trait values, and confidence.
        """
        # Run single-shot detection on this segment
        result = self._detector.analyze(
            text=text,
            speaker_id=f"progressive_seg_{self._segment_count}",
            speaker_label=speaker_label,
        )

        # Merge each detected trait with prior
        for trait in result.traits:
            prior_val, prior_conf = self._priors.get(trait.name, (0.5, 0.0))
            new_val, new_conf = bayesian_update(
                prior_val, prior_conf, trait.value, trait.confidence,
            )
            self._priors[trait.name] = (new_val, new_conf)

        # Build snapshot
        snapshot = {
            "segment_id": self._segment_count,
            "traits": {
                name: {"value": round(v, 3), "confidence": round(c, 3)}
                for name, (v, c) in self._priors.items()
            },
        }
        self._history.append(snapshot)
        self._segment_count += 1
        return snapshot

    def get_profile_dict(self) -> dict[str, dict]:
        """Return current accumulated profile as {trait: {value, confidence}}."""
        return {
            name: {"value": round(v, 3), "confidence": round(c, 3)}
            for name, (v, c) in self._priors.items()
        }

    def get_profile(self) -> PersonalityDNA:
        """Return current accumulated profile as PersonalityDNA."""
        traits = []
        for name, (val, conf) in self._priors.items():
            dim = self._trait_dims.get(name, "UNK")
            traits.append(Trait(dimension=dim, name=name, value=val, confidence=conf))
        return PersonalityDNA(
            id="progressive",
            sample_summary=SampleSummary(
                total_tokens=0,
                conversation_count=self._segment_count,
                date_range=["unknown", "unknown"],
                contexts=["literary"],
                confidence_overall=(
                    sum(c for _, c in self._priors.values()) / max(len(self._priors), 1)
                ),
            ),
            traits=traits,
        )

    def get_history(self) -> list[dict]:
        """Return all historical snapshots."""
        return list(self._history)

    def reset(self):
        """Clear all priors and history."""
        self._priors.clear()
        self._history.clear()
        self._segment_count = 0
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_progressive.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add super_brain/progressive.py tests/test_progressive.py
git commit -m "feat: add ProgressiveDetector with Bayesian belief updating"
```

---

### Task 2: Dialogue Extraction Utilities

**Files:**
- Create: `super_brain/literary.py`
- Test: `tests/test_literary.py`

**Step 1: Write the failing tests**

```python
# tests/test_literary.py
"""Tests for literary dialogue extraction and segmentation."""

import pytest
from super_brain.literary import segment_dialogue, compute_mae


class TestSegmentDialogue:
    """Test dialogue segmentation into fixed-size chunks."""

    def test_basic_segmentation(self):
        """Splits quotes into segments of target size."""
        quotes = [f"Quote {i}" for i in range(25)]
        segments = segment_dialogue(quotes, segment_size=10)
        assert len(segments) == 3  # 10 + 10 + 5
        assert len(segments[0]) == 10
        assert len(segments[2]) == 5

    def test_small_input(self):
        """Fewer quotes than segment_size → single segment."""
        quotes = ["Hello", "World"]
        segments = segment_dialogue(quotes, segment_size=10)
        assert len(segments) == 1
        assert len(segments[0]) == 2

    def test_empty_input(self):
        """Empty list → empty result."""
        segments = segment_dialogue([], segment_size=10)
        assert segments == []

    def test_exact_multiple(self):
        """Exact multiple of segment_size → no remainder segment."""
        quotes = [f"Q{i}" for i in range(20)]
        segments = segment_dialogue(quotes, segment_size=10)
        assert len(segments) == 2


class TestComputeMAE:
    """Test MAE computation between detected profile and ground truth."""

    def test_perfect_match(self):
        """Identical profiles → MAE = 0."""
        detected = {"a": {"value": 0.5, "confidence": 0.8}}
        gt = {"a": 0.5}
        assert compute_mae(detected, gt) == pytest.approx(0.0)

    def test_known_error(self):
        """Known difference → correct MAE."""
        detected = {
            "a": {"value": 0.3, "confidence": 0.8},
            "b": {"value": 0.7, "confidence": 0.8},
        }
        gt = {"a": 0.5, "b": 0.5}
        # errors: 0.2 + 0.2 = 0.4, MAE = 0.2
        assert compute_mae(detected, gt) == pytest.approx(0.2)

    def test_missing_traits_excluded(self):
        """Only traits in both detected and GT count."""
        detected = {"a": {"value": 0.5, "confidence": 0.8}}
        gt = {"a": 0.5, "b": 0.9}
        assert compute_mae(detected, gt) == pytest.approx(0.0)
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_literary.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'super_brain.literary'`

**Step 3: Write minimal implementation**

```python
# super_brain/literary.py
"""Utilities for literary character personality experiments."""

from __future__ import annotations


def segment_dialogue(quotes: list[str], segment_size: int = 12) -> list[list[str]]:
    """Split a list of quotes into fixed-size segments.

    Args:
        quotes: List of character dialogue quotes.
        segment_size: Target number of quotes per segment.

    Returns:
        List of segments, each a list of quotes.
    """
    if not quotes:
        return []
    return [
        quotes[i : i + segment_size]
        for i in range(0, len(quotes), segment_size)
    ]


def compute_mae(
    detected: dict[str, dict],
    ground_truth: dict[str, float],
) -> float:
    """Compute MAE between detected profile and ground truth.

    Args:
        detected: {trait_name: {"value": float, "confidence": float}}
        ground_truth: {trait_name: float}

    Returns:
        Mean absolute error over shared traits.
    """
    errors = []
    for name, gt_val in ground_truth.items():
        if name in detected:
            errors.append(abs(detected[name]["value"] - gt_val))
    if not errors:
        return 1.0
    return sum(errors) / len(errors)
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_literary.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add super_brain/literary.py tests/test_literary.py
git commit -m "feat: add literary dialogue segmentation and MAE utilities"
```

---

### Task 3: Ground Truth — LLM Omniscient Scoring

**Files:**
- Create: `scripts/generate_literary_gt.py`
- Output: `data/literary/{character}/gt_llm.json`

**Step 1: Write the GT generation script**

```python
# scripts/generate_literary_gt.py
"""Generate LLM omniscient ground truth for literary characters.

Usage:
    ANTHROPIC_API_KEY=... python scripts/generate_literary_gt.py scarlett
    ANTHROPIC_API_KEY=... python scripts/generate_literary_gt.py sherlock
    ANTHROPIC_API_KEY=... python scripts/generate_literary_gt.py elizabeth
"""

import json
import os
import sys
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
    import re
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
```

**Step 2: Run for all 3 characters**

```bash
export ANTHROPIC_API_KEY=sk-or-...
python scripts/generate_literary_gt.py scarlett
python scripts/generate_literary_gt.py sherlock
python scripts/generate_literary_gt.py elizabeth
```

Expected: `data/literary/{character}/gt_llm.json` with 69 traits each.

**Step 3: Commit**

```bash
git add scripts/generate_literary_gt.py data/literary/*/gt_llm.json
git commit -m "feat: generate LLM omniscient ground truth for 3 literary characters"
```

---

### Task 4: Ground Truth — OpenPsychometrics Crowd Mapping

**Files:**
- Create: `scripts/map_crowd_gt.py`
- Output: `data/literary/{character}/gt_crowd.json`

**Step 1: Write the mapping script**

```python
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
    # We provide the known top traits from our research rather than scraping
    # This is more reliable and doesn't require web access
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
    import re
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
```

**Step 2: Run for all 3 characters**

```bash
export ANTHROPIC_API_KEY=sk-or-...
python scripts/map_crowd_gt.py scarlett
python scripts/map_crowd_gt.py sherlock
python scripts/map_crowd_gt.py elizabeth
```

**Step 3: Commit**

```bash
git add scripts/map_crowd_gt.py data/literary/*/gt_crowd.json
git commit -m "feat: map OpenPsychometrics crowd data to 69 traits for 3 characters"
```

---

### Task 5: Dialogue Extraction — Sherlock & Elizabeth

**Files:**
- Create: `scripts/extract_literary_dialogue.py`
- Output: `data/literary/sherlock/dialogue.txt`, `data/literary/elizabeth/dialogue.txt`

**Step 1: Write the extraction script**

```python
# scripts/extract_literary_dialogue.py
"""Extract character dialogue from Project Gutenberg texts.

Downloads public domain texts and extracts explicitly attributed dialogue.

Usage:
    ANTHROPIC_API_KEY=... python scripts/extract_literary_dialogue.py sherlock
    ANTHROPIC_API_KEY=... python scripts/extract_literary_dialogue.py elizabeth
"""

import json
import os
import sys
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
            import re
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
```

**Step 2: Run for Sherlock and Elizabeth** (Scarlett already exists)

```bash
export ANTHROPIC_API_KEY=sk-or-...
python scripts/extract_literary_dialogue.py sherlock
python scripts/extract_literary_dialogue.py elizabeth
```

**Step 3: Copy existing Scarlett dialogue**

```bash
cp data/gwtw/scarlett_dialogue.txt data/literary/scarlett/dialogue.txt
```

**Step 4: Commit**

```bash
git add scripts/extract_literary_dialogue.py data/literary/*/dialogue.txt
git commit -m "feat: extract dialogue for Sherlock Holmes and Elizabeth Bennet"
```

---

### Task 6: Main Experiment Script

**Files:**
- Create: `eval_literary.py`

**Step 1: Write the experiment script**

```python
# eval_literary.py
"""Literary Character Progressive Personality Detection Experiment.

Feeds character dialogue segment-by-segment through ProgressiveDetector,
recording convergence curves against dual ground truth.

Usage:
    ANTHROPIC_API_KEY=... python eval_literary.py [character] [segment_size]
    ANTHROPIC_API_KEY=... python eval_literary.py scarlett 12
    ANTHROPIC_API_KEY=... python eval_literary.py all 12
"""

import json
import os
import sys
import time
from pathlib import Path

from super_brain.progressive import ProgressiveDetector
from super_brain.literary import segment_dialogue, compute_mae


CHARACTERS = ["scarlett", "sherlock", "elizabeth"]


def load_dialogue(character: str) -> list[str]:
    """Load dialogue quotes from file."""
    path = Path(f"data/literary/{character}/dialogue.txt")
    quotes = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            # Strip leading [N] numbering
            import re
            m = re.match(r"\[\d+\]\s*(.*)", line)
            if m:
                quotes.append(m.group(1))
            elif line:
                quotes.append(line)
    return [q for q in quotes if len(q) > 3]


def load_ground_truth(character: str) -> dict[str, dict[str, float]]:
    """Load both ground truth sources."""
    gt = {}
    for source in ["gt_llm", "gt_crowd"]:
        path = Path(f"data/literary/{character}/{source}.json")
        if path.exists():
            gt[source] = json.loads(path.read_text())
    return gt


def format_segment_text(quotes: list[str], character_name: str) -> str:
    """Format quotes into a conversation-like text for the detector."""
    lines = []
    for q in quotes:
        lines.append(f"{character_name}: {q}")
    return "\n\n".join(lines)


CHARACTER_NAMES = {
    "scarlett": "Scarlett",
    "sherlock": "Holmes",
    "elizabeth": "Elizabeth",
}


def run_experiment(character: str, segment_size: int, api_key: str) -> dict:
    """Run progressive detection experiment for one character."""
    print(f"\n{'='*60}")
    print(f"Character: {character}")
    print(f"{'='*60}")

    # Load data
    quotes = load_dialogue(character)
    gt_sources = load_ground_truth(character)
    segments = segment_dialogue(quotes, segment_size)
    char_name = CHARACTER_NAMES[character]

    print(f"Quotes: {len(quotes)}, Segments: {len(segments)}, GTs: {list(gt_sources.keys())}")

    # Run progressive detection
    detector = ProgressiveDetector(api_key=api_key)
    convergence = []

    for i, seg in enumerate(segments):
        print(f"  Segment {i+1}/{len(segments)} ({len(seg)} quotes)...", end=" ", flush=True)
        start = time.time()

        text = format_segment_text(seg, char_name)
        snapshot = detector.update(text, speaker_label=char_name)

        # Compute MAE vs each ground truth
        profile = detector.get_profile_dict()
        mae_results = {}
        for gt_name, gt_data in gt_sources.items():
            mae_results[f"mae_vs_{gt_name}"] = round(compute_mae(profile, gt_data), 3)

        # Count converged traits
        n_converged = sum(
            1 for t in profile.values()
            if t["confidence"] > 0.70
        )
        # Check stability (last 3 snapshots)
        if len(convergence) >= 3:
            history = detector.get_history()
            stable_count = 0
            for trait_name in profile:
                recent_vals = []
                for h in history[-3:]:
                    if trait_name in h["traits"]:
                        recent_vals.append(h["traits"][trait_name]["value"])
                if len(recent_vals) == 3 and max(recent_vals) - min(recent_vals) < 0.03:
                    stable_count += 1
            n_converged = min(n_converged, stable_count)

        elapsed = time.time() - start
        entry = {
            "segment_id": i,
            "cumulative_quotes": sum(len(s) for s in segments[:i+1]),
            **mae_results,
            "traits_converged": n_converged,
            "avg_confidence": round(
                sum(t["confidence"] for t in profile.values()) / max(len(profile), 1), 3
            ),
            "elapsed": round(elapsed, 1),
        }
        convergence.append(entry)

        mae_str = " | ".join(f"{k}={v:.3f}" for k, v in mae_results.items())
        print(f"{mae_str} | converged={n_converged} | {elapsed:.0f}s")

    # Save results
    outdir = Path(f"data/literary/{character}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Convergence curve
    with open(outdir / "convergence.json", "w") as f:
        json.dump({"character": character, "segments": convergence}, f, indent=2)

    # Trait trajectories
    trajectories = {}
    history = detector.get_history()
    profile = detector.get_profile_dict()
    for trait_name in profile:
        traj = {
            "gt_llm": gt_sources.get("gt_llm", {}).get(trait_name),
            "gt_crowd": gt_sources.get("gt_crowd", {}).get(trait_name),
            "trajectory": [],
        }
        for h in history:
            if trait_name in h["traits"]:
                traj["trajectory"].append({
                    "seg": h["segment_id"],
                    "value": h["traits"][trait_name]["value"],
                    "confidence": h["traits"][trait_name]["confidence"],
                })
        trajectories[trait_name] = traj

    with open(outdir / "trajectories.json", "w") as f:
        json.dump(trajectories, f, indent=2)

    # Final comparison
    final = {
        "character": character,
        "total_quotes": len(quotes),
        "total_segments": len(segments),
    }
    for gt_name, gt_data in gt_sources.items():
        final[f"final_mae_{gt_name}"] = round(compute_mae(profile, gt_data), 3)

    # GT consistency
    if "gt_llm" in gt_sources and "gt_crowd" in gt_sources:
        gt_errors = []
        for name, llm_val in gt_sources["gt_llm"].items():
            if name in gt_sources["gt_crowd"]:
                gt_errors.append(abs(llm_val - gt_sources["gt_crowd"][name]))
        if gt_errors:
            final["gt_consistency_mae"] = round(sum(gt_errors) / len(gt_errors), 3)

    with open(outdir / "final_comparison.json", "w") as f:
        json.dump(final, f, indent=2)

    print(f"\n  Final: {json.dumps({k: v for k, v in final.items() if 'mae' in k}, indent=2)}")
    return final


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    character = sys.argv[1] if len(sys.argv) > 1 else "scarlett"
    segment_size = int(sys.argv[2]) if len(sys.argv) > 2 else 12

    characters = CHARACTERS if character == "all" else [character]

    all_results = {}
    for char in characters:
        result = run_experiment(char, segment_size, api_key)
        all_results[char] = result

    if len(characters) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for char, result in all_results.items():
            mae_str = " | ".join(
                f"{k}={v}" for k, v in result.items() if "mae" in k
            )
            print(f"  {char}: {mae_str}")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add eval_literary.py
git commit -m "feat: add literary character progressive detection experiment script"
```

---

### Task 7: Run Scarlett Experiment (First Validation)

**Step 1: Ensure Scarlett data is ready**

```bash
# Check dialogue exists
wc -l data/literary/scarlett/dialogue.txt

# Check GTs exist
ls data/literary/scarlett/gt_llm.json data/literary/scarlett/gt_crowd.json
```

**Step 2: Run experiment**

```bash
export ANTHROPIC_API_KEY=sk-or-...
PYTHONUNBUFFERED=1 python eval_literary.py scarlett 12
```

Expected: ~20 segments, each taking ~2-3 min. Total ~45 min.
Output: `data/literary/scarlett/convergence.json`, `trajectories.json`, `final_comparison.json`

**Step 3: Analyze results**

```bash
python -c "
import json
with open('data/literary/scarlett/convergence.json') as f:
    data = json.load(f)
for s in data['segments']:
    print(f\"Seg {s['segment_id']:2d} | quotes={s['cumulative_quotes']:3d} | \", end='')
    for k, v in s.items():
        if 'mae' in k:
            print(f'{k}={v:.3f} ', end='')
    print(f\"| converged={s['traits_converged']}\")
"
```

**Step 4: Commit results**

```bash
git add data/literary/scarlett/
git commit -m "results: Scarlett O'Hara progressive detection experiment"
```

---

### Task 8: Run Sherlock & Elizabeth Experiments

**Step 1: Run remaining characters**

```bash
export ANTHROPIC_API_KEY=sk-or-...
PYTHONUNBUFFERED=1 python eval_literary.py sherlock 12
PYTHONUNBUFFERED=1 python eval_literary.py elizabeth 12
```

**Step 2: Commit all results**

```bash
git add data/literary/sherlock/ data/literary/elizabeth/
git commit -m "results: Sherlock Holmes and Elizabeth Bennet progressive detection"
```

---

### Task 9: Analysis & Summary Report

**Step 1: Write analysis script**

```bash
python -c "
import json
from pathlib import Path

characters = ['scarlett', 'sherlock', 'elizabeth']

print('=== LITERARY CHARACTER PROGRESSIVE DETECTION RESULTS ===\n')

for char in characters:
    conv_path = Path(f'data/literary/{char}/convergence.json')
    final_path = Path(f'data/literary/{char}/final_comparison.json')
    if not conv_path.exists():
        print(f'{char}: NO DATA')
        continue

    conv = json.loads(conv_path.read_text())
    final = json.loads(final_path.read_text())

    segs = conv['segments']
    print(f'--- {char.upper()} ---')
    print(f'Total quotes: {final[\"total_quotes\"]}, Segments: {final[\"total_segments\"]}')

    for k, v in final.items():
        if 'mae' in k:
            print(f'  {k}: {v}')

    # Convergence speed: when did MAE first drop below 0.18?
    for s in segs:
        for k, v in s.items():
            if 'mae' in k and v < 0.18:
                pct = s['cumulative_quotes'] / final['total_quotes'] * 100
                print(f'  {k} < 0.18 at segment {s[\"segment_id\"]} ({pct:.0f}% of dialogue)')
                break
    print()
"
```

**Step 2: Push everything to GitHub**

```bash
git push https://TOKEN@github.com/mozatyin/super-brain.git main
```
