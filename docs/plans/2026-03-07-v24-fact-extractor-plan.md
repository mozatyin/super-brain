# V2.4 FactExtractor + Soul Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a FactExtractor (separate LLM call extracting facts, reality narrative, secrets, contradictions) with adaptive frequency, and introduce the Soul model that aggregates character + facts + reality.

**Architecture:** New `FactExtractor` class makes a separate LLM call alongside ThinkSlow every N turns. Both extractors independently adjust their frequency (2-5 turns) based on extraction yield. A new `Soul` model accumulates all layers. The eval pipeline reports Soul Coverage metrics alongside MAE.

**Tech Stack:** Python 3.12, Pydantic, Anthropic SDK (with OpenRouter support), pytest

---

### Task 1: Add V2.4 Data Models

**Files:**
- Modify: `super_brain/models.py:1-100`
- Test: `tests/test_models_v24.py`

**Step 1: Write the failing test**

Create `tests/test_models_v24.py`:

```python
"""Tests for V2.4 data models: Fact, Reality, FactExtractionResult, Soul."""

from super_brain.models import (
    Fact, Reality, FactExtractionResult, Soul, PersonalityDNA, SampleSummary,
)


def test_fact_creation():
    f = Fact(category="career", content="software engineer at a startup", confidence=0.8, source_turn=3)
    assert f.category == "career"
    assert f.confidence == 0.8
    assert f.source_turn == 3


def test_fact_confidence_clamped():
    f = Fact(category="hobby", content="plays guitar", confidence=1.0, source_turn=1)
    assert f.confidence == 1.0


def test_reality_creation():
    r = Reality(
        summary="Currently a mid-career engineer exploring entrepreneurship",
        domains={"career": "software engineer", "relationships": "single"},
        constraints=["limited savings"],
        resources=["strong network"],
    )
    assert "engineer" in r.summary
    assert r.domains["career"] == "software engineer"
    assert len(r.constraints) == 1


def test_fact_extraction_result():
    result = FactExtractionResult(
        new_facts=[Fact(category="career", content="engineer", confidence=0.9, source_turn=5)],
        reality=Reality(
            summary="An engineer",
            domains={"career": "engineer"},
            constraints=[],
            resources=[],
        ),
        secrets=["avoids discussing family"],
        contradictions=["said values independence but hates decisions"],
    )
    assert len(result.new_facts) == 1
    assert result.reality is not None
    assert len(result.secrets) == 1
    assert len(result.contradictions) == 1


def test_fact_extraction_result_empty():
    result = FactExtractionResult(new_facts=[], reality=None, secrets=[], contradictions=[])
    assert len(result.new_facts) == 0
    assert result.reality is None


def test_soul_creation_minimal():
    profile = PersonalityDNA(
        id="test",
        sample_summary=SampleSummary(
            total_tokens=0, conversation_count=0,
            date_range=["unknown", "unknown"],
            contexts=["test"], confidence_overall=0.5,
        ),
    )
    soul = Soul(id="test_soul", character=profile)
    assert soul.id == "test_soul"
    assert soul.facts == []
    assert soul.reality is None
    assert soul.secrets == []
    assert soul.contradictions == []


def test_soul_with_facts():
    profile = PersonalityDNA(
        id="test",
        sample_summary=SampleSummary(
            total_tokens=0, conversation_count=0,
            date_range=["unknown", "unknown"],
            contexts=["test"], confidence_overall=0.5,
        ),
    )
    soul = Soul(
        id="test_soul",
        character=profile,
        facts=[
            Fact(category="career", content="engineer", confidence=0.9, source_turn=3),
            Fact(category="hobby", content="guitar", confidence=0.7, source_turn=5),
        ],
        reality=Reality(
            summary="An engineer who plays guitar",
            domains={"career": "engineer", "hobby": "guitar"},
            constraints=[],
            resources=["technical skills"],
        ),
        secrets=["enthusiasm spikes when discussing travel"],
    )
    assert len(soul.facts) == 2
    assert soul.reality is not None
    assert len(soul.secrets) == 1
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/michael/super-brain && .venv/bin/pytest tests/test_models_v24.py -v`
Expected: FAIL with ImportError (Fact, Reality, etc. not defined)

**Step 3: Write minimal implementation**

Add to `super_brain/models.py` (after `ConductorAction` class, at the end of file):

```python
class Fact(BaseModel):
    """A factual piece of information about the person."""
    category: str       # "career", "relationship", "hobby", "education",
                        # "location", "family", "preference", "experience"
    content: str        # "software engineer at a startup"
    confidence: float = Field(ge=0.0, le=1.0)
    source_turn: int    # which turn this was extracted from


class Reality(BaseModel):
    """Current reality snapshot of the person's life situation."""
    summary: str                    # narrative: "Currently a mid-career engineer..."
    domains: dict[str, str]         # {"career": "...", "relationships": "..."}
    constraints: list[str]          # things limiting them
    resources: list[str]            # things they have going for them


class FactExtractionResult(BaseModel):
    """Result of a single FactExtractor cycle."""
    new_facts: list[Fact]           # facts found in this cycle
    reality: Reality | None = None  # updated reality snapshot
    secrets: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)


class Soul(BaseModel):
    """Full Soul model aggregating all layers of understanding about a person."""
    id: str

    # Layer 1: Character (existing 66-trait PersonalityDNA)
    character: PersonalityDNA

    # Layer 2: Facts (accumulated from FactExtractor)
    facts: list[Fact] = Field(default_factory=list)

    # Layer 3: Reality (latest snapshot from FactExtractor)
    reality: Reality | None = None

    # Cross-layer insights
    secrets: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/michael/super-brain && .venv/bin/pytest tests/test_models_v24.py -v`
Expected: ALL PASS (7 tests)

**Step 5: Commit**

```bash
git add super_brain/models.py tests/test_models_v24.py
git commit -m "feat(v2.4): add Fact, Reality, FactExtractionResult, Soul models"
```

---

### Task 2: FactExtractor Module — Core Extraction

**Files:**
- Create: `super_brain/fact_extractor.py`
- Test: `tests/test_fact_extractor.py`

**Context:** The FactExtractor follows the same pattern as `ThinkSlow` in `super_brain/think_slow.py`:
- Takes `api_key` + optional `model` in `__init__`
- OpenRouter auto-detection: `if api_key.startswith("sk-or-"): kwargs["base_url"] = "https://openrouter.ai/api"`
- Single `extract()` method that takes conversation + previous state → returns `FactExtractionResult`
- JSON response parsing with fallback (same `_parse_response()` pattern as `_parse_think_slow_response`)

**Step 1: Write the failing test**

Create `tests/test_fact_extractor.py`:

```python
"""Tests for V2.4 FactExtractor — fact, reality, secrets extraction."""

import json
from unittest.mock import MagicMock, patch

from super_brain.fact_extractor import FactExtractor, _parse_fact_response, _deduplicate_facts
from super_brain.models import Fact, FactExtractionResult


def test_parse_fact_response_valid_json():
    raw = json.dumps({
        "facts": [
            {"category": "career", "content": "software engineer", "confidence": 0.9},
            {"category": "hobby", "content": "plays guitar", "confidence": 0.7},
        ],
        "reality": {
            "summary": "An engineer who plays guitar",
            "domains": {"career": "software engineer"},
            "constraints": [],
            "resources": ["technical skills"],
        },
        "secrets": ["avoids discussing family"],
        "contradictions": [],
    })
    data = _parse_fact_response(raw)
    assert len(data["facts"]) == 2
    assert data["facts"][0]["category"] == "career"
    assert data["reality"]["summary"] == "An engineer who plays guitar"
    assert len(data["secrets"]) == 1


def test_parse_fact_response_code_block():
    raw = "```json\n" + json.dumps({
        "facts": [],
        "reality": None,
        "secrets": [],
        "contradictions": [],
    }) + "\n```"
    data = _parse_fact_response(raw)
    assert data["facts"] == []


def test_parse_fact_response_invalid():
    data = _parse_fact_response("not json at all")
    assert data["facts"] == []
    assert data["reality"] is None
    assert data["secrets"] == []
    assert data["contradictions"] == []


def test_deduplicate_facts_removes_duplicates():
    existing = [
        Fact(category="career", content="software engineer", confidence=0.9, source_turn=3),
    ]
    new_raw = [
        {"category": "career", "content": "software engineer", "confidence": 0.8},  # duplicate
        {"category": "hobby", "content": "plays guitar", "confidence": 0.7},        # new
    ]
    result = _deduplicate_facts(new_raw, existing, current_turn=6)
    assert len(result) == 1
    assert result[0].content == "plays guitar"
    assert result[0].source_turn == 6


def test_deduplicate_facts_case_insensitive():
    existing = [
        Fact(category="career", content="Software Engineer", confidence=0.9, source_turn=3),
    ]
    new_raw = [
        {"category": "career", "content": "software engineer", "confidence": 0.8},
    ]
    result = _deduplicate_facts(new_raw, existing, current_turn=6)
    assert len(result) == 0


def test_deduplicate_facts_empty_existing():
    new_raw = [
        {"category": "career", "content": "engineer", "confidence": 0.9},
    ]
    result = _deduplicate_facts(new_raw, [], current_turn=1)
    assert len(result) == 1
    assert result[0].source_turn == 1


def test_fact_extractor_extract_with_mock():
    """Test FactExtractor.extract() with a mocked LLM response."""
    with patch("super_brain.fact_extractor.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "facts": [
                {"category": "career", "content": "data scientist", "confidence": 0.85},
            ],
            "reality": {
                "summary": "A data scientist exploring new opportunities",
                "domains": {"career": "data science"},
                "constraints": ["limited time"],
                "resources": ["analytical skills"],
            },
            "secrets": ["seems stressed about work"],
            "contradictions": [],
        }))]
        mock_client.messages.create.return_value = mock_response

        extractor = FactExtractor(api_key="test-key")
        conversation = [
            {"role": "chatter", "text": "What do you do for work?"},
            {"role": "speaker", "text": "I'm a data scientist, been doing it for three years."},
        ]
        result = extractor.extract(conversation, existing_facts=[], current_turn=2)

        assert isinstance(result, FactExtractionResult)
        assert len(result.new_facts) == 1
        assert result.new_facts[0].category == "career"
        assert result.new_facts[0].content == "data scientist"
        assert result.new_facts[0].source_turn == 2
        assert result.reality is not None
        assert result.reality.summary == "A data scientist exploring new opportunities"
        assert result.secrets == ["seems stressed about work"]

        # Verify the LLM was called with the right model
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["max_tokens"] == 4096
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/michael/super-brain && .venv/bin/pytest tests/test_fact_extractor.py -v`
Expected: FAIL with ModuleNotFoundError (fact_extractor not found)

**Step 3: Write minimal implementation**

Create `super_brain/fact_extractor.py`:

```python
"""FactExtractor: Periodic extraction of facts, reality, secrets (V2.4).

Runs alongside ThinkSlow every N turns (adaptive frequency). Uses a separate
LLM call focused exclusively on factual information — NOT personality traits.
"""

from __future__ import annotations

import json

import anthropic

from super_brain.models import Fact, Reality, FactExtractionResult


_FACT_EXTRACTOR_SYSTEM = """\
You are analyzing a conversation to extract FACTUAL information about the target \
speaker (labeled "Person B"). Focus on concrete facts, NOT personality traits.

Extract:
1. **Facts**: Specific, concrete information (job, location, family, hobbies, education, etc.)
2. **Reality snapshot**: A brief narrative of their current life situation
3. **Secrets**: Things they avoid discussing, topics that change their energy, \
contradictions between what they say and how they say it
4. **Contradictions**: Statements that conflict with earlier statements

Return ONLY valid JSON:
{
  "facts": [
    {"category": "<category>", "content": "<what you learned>", "confidence": <0.0-1.0>}
  ],
  "reality": {
    "summary": "<1-2 sentence narrative of their current situation>",
    "domains": {"<domain>": "<description>", ...},
    "constraints": ["<limiting factor>", ...],
    "resources": ["<advantage or asset>", ...]
  },
  "secrets": ["<observation about avoidance or hidden pattern>", ...],
  "contradictions": ["<statement X conflicts with statement Y>", ...]
}

Categories for facts: career, relationship, hobby, education, location, family, \
preference, experience, health, financial.

IMPORTANT:
- Only extract facts with CLEAR evidence from the text. Do not infer or guess.
- Reality snapshot should be null if insufficient information.
- Secrets are OBSERVATIONS, not accusations. "Energy drops when family is mentioned" not "hiding something about family".
- Confidence: 0.5-0.7 for mentioned once, 0.8-0.9 for discussed in detail, 1.0 only for explicit statements.
"""


def _format_conversation(conversation: list[dict]) -> str:
    """Format conversation for FactExtractor input."""
    lines = []
    for msg in conversation:
        label = "Person A" if msg["role"] == "chatter" else "Person B"
        lines.append(f"{label}: {msg['text']}")
    return "\n\n".join(lines)


def _parse_fact_response(raw: str) -> dict:
    """Parse FactExtractor JSON response with fallback."""
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {"facts": [], "reality": None, "secrets": [], "contradictions": []}


def _deduplicate_facts(
    new_raw: list[dict],
    existing: list[Fact],
    current_turn: int,
) -> list[Fact]:
    """Deduplicate new facts against existing ones (case-insensitive content match)."""
    existing_contents = {f.content.lower().strip() for f in existing}
    unique: list[Fact] = []
    for raw_fact in new_raw:
        content = raw_fact.get("content", "").strip()
        if content.lower() not in existing_contents:
            unique.append(Fact(
                category=raw_fact.get("category", "unknown"),
                content=content,
                confidence=max(0.0, min(1.0, float(raw_fact.get("confidence", 0.5)))),
                source_turn=current_turn,
            ))
            existing_contents.add(content.lower())
    return unique


class FactExtractor:
    """Extracts facts, reality narrative, and secrets from conversation."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        kwargs: dict = {"api_key": api_key}
        if api_key.startswith("sk-or-"):
            kwargs["base_url"] = "https://openrouter.ai/api"
        self._client = anthropic.Anthropic(**kwargs)
        self._model = model

    def extract(
        self,
        conversation: list[dict],
        existing_facts: list[Fact] | None = None,
        current_turn: int = 0,
    ) -> FactExtractionResult:
        """Extract facts, reality, secrets from conversation.

        Args:
            conversation: Full conversation so far.
            existing_facts: Previously extracted facts (for deduplication).
            current_turn: Current turn number (for fact source tracking).

        Returns:
            FactExtractionResult with new (deduplicated) facts, reality, secrets.
        """
        conv_text = _format_conversation(conversation)
        existing = existing_facts or []

        # Include existing facts context so LLM focuses on NEW information
        existing_section = ""
        if existing:
            known = [f"{f.category}: {f.content}" for f in existing]
            existing_section = (
                "\n\nALREADY KNOWN FACTS (focus on NEW information not listed here):\n"
                + "\n".join(f"- {k}" for k in known)
            )

        user_message = (
            f"## Conversation\n\n{conv_text}\n\n"
            f"## Target Speaker\n\nExtract factual information about Person B."
            f"{existing_section}\n\n"
            f"Return JSON with facts, reality, secrets, contradictions."
        )

        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=_FACT_EXTRACTOR_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
        )

        raw = response.content[0].text
        data = _parse_fact_response(raw)

        # Deduplicate facts
        new_facts = _deduplicate_facts(
            data.get("facts", []), existing, current_turn,
        )

        # Build Reality if provided
        reality = None
        raw_reality = data.get("reality")
        if raw_reality and isinstance(raw_reality, dict) and raw_reality.get("summary"):
            reality = Reality(
                summary=raw_reality["summary"],
                domains=raw_reality.get("domains", {}),
                constraints=raw_reality.get("constraints", []),
                resources=raw_reality.get("resources", []),
            )

        return FactExtractionResult(
            new_facts=new_facts,
            reality=reality,
            secrets=data.get("secrets", []),
            contradictions=data.get("contradictions", []),
        )
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/michael/super-brain && .venv/bin/pytest tests/test_fact_extractor.py -v`
Expected: ALL PASS (7 tests)

**Step 5: Run all existing tests to check no regressions**

Run: `cd /Users/michael/super-brain && .venv/bin/pytest tests/ -v`
Expected: ALL PASS (should include existing 99+ tests + 7 new)

**Step 6: Commit**

```bash
git add super_brain/fact_extractor.py tests/test_fact_extractor.py
git commit -m "feat(v2.4): add FactExtractor module with deduplication"
```

---

### Task 3: Adaptive Frequency Manager

**Files:**
- Create: `super_brain/adaptive_frequency.py`
- Test: `tests/test_adaptive_frequency.py`

**Context:** Each extractor (ThinkSlow, FactExtractor) independently adjusts its run interval based on extraction yield. High yield → shorter interval. No yield → longer interval. Interval range: 2-5 turns.

**Step 1: Write the failing test**

Create `tests/test_adaptive_frequency.py`:

```python
"""Tests for V2.4 adaptive frequency manager."""

from super_brain.adaptive_frequency import AdaptiveFrequency


def test_default_interval():
    af = AdaptiveFrequency(default_interval=3)
    assert af.interval == 3


def test_should_run_at_interval():
    af = AdaptiveFrequency(default_interval=3)
    assert af.should_run(turn=3) is True
    assert af.should_run(turn=4) is False
    assert af.should_run(turn=5) is False
    assert af.should_run(turn=6) is True


def test_high_yield_decreases_interval():
    af = AdaptiveFrequency(default_interval=3)
    af.report_yield(3)  # high yield (>=3 new items)
    assert af.interval == 2  # decreased from 3 to 2


def test_zero_yield_increases_interval():
    af = AdaptiveFrequency(default_interval=3)
    af.report_yield(0)  # nothing new
    assert af.interval == 4  # increased from 3 to 4


def test_normal_yield_keeps_interval():
    af = AdaptiveFrequency(default_interval=3)
    af.report_yield(1)  # normal yield
    assert af.interval == 3  # unchanged


def test_interval_clamped_min():
    af = AdaptiveFrequency(default_interval=2, min_interval=2)
    af.report_yield(5)  # high yield
    assert af.interval == 2  # can't go below min


def test_interval_clamped_max():
    af = AdaptiveFrequency(default_interval=5, max_interval=5)
    af.report_yield(0)  # nothing new
    assert af.interval == 5  # can't go above max


def test_repeated_zero_yield_caps_at_max():
    af = AdaptiveFrequency(default_interval=3, max_interval=5)
    for _ in range(10):
        af.report_yield(0)
    assert af.interval == 5


def test_repeated_high_yield_caps_at_min():
    af = AdaptiveFrequency(default_interval=4, min_interval=2)
    for _ in range(10):
        af.report_yield(5)
    assert af.interval == 2


def test_should_run_adapts_to_new_interval():
    af = AdaptiveFrequency(default_interval=3)
    assert af.should_run(turn=3) is True
    af.report_yield(5)  # high yield → interval becomes 2
    assert af.should_run(turn=4) is False
    assert af.should_run(turn=5) is True  # 3 + 2 = 5
    af.report_yield(0)  # zero yield → interval becomes 3
    assert af.should_run(turn=8) is True  # 5 + 3 = 8
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/michael/super-brain && .venv/bin/pytest tests/test_adaptive_frequency.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

Create `super_brain/adaptive_frequency.py`:

```python
"""Adaptive frequency manager for extraction cycles (V2.4).

Each extractor independently adjusts its interval based on extraction yield:
- High yield (>=3 new items): decrease interval (run more often)
- Normal yield (1-2 items): keep current interval
- Zero yield: increase interval (run less often)
"""

from __future__ import annotations


class AdaptiveFrequency:
    """Manages adaptive run frequency for an extractor.

    Parameters
    ----------
    default_interval : int
        Starting interval in turns (default 3).
    min_interval : int
        Minimum interval — never run more often than this (default 2).
    max_interval : int
        Maximum interval — never run less often than this (default 5).
    high_yield_threshold : int
        Number of new items that counts as "high yield" (default 3).
    """

    def __init__(
        self,
        default_interval: int = 3,
        min_interval: int = 2,
        max_interval: int = 5,
        high_yield_threshold: int = 3,
    ) -> None:
        self._interval = default_interval
        self._min = min_interval
        self._max = max_interval
        self._high_threshold = high_yield_threshold
        self._last_run_turn: int = 0

    @property
    def interval(self) -> int:
        return self._interval

    def should_run(self, turn: int) -> bool:
        """Check if the extractor should run at this turn number."""
        if turn - self._last_run_turn >= self._interval:
            self._last_run_turn = turn
            return True
        return False

    def report_yield(self, new_items_count: int) -> None:
        """Report extraction yield to adjust frequency.

        Args:
            new_items_count: Number of new items extracted in the last cycle.
        """
        if new_items_count >= self._high_threshold:
            self._interval = max(self._min, self._interval - 1)
        elif new_items_count == 0:
            self._interval = min(self._max, self._interval + 1)
        # else: normal yield, keep current interval
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/michael/super-brain && .venv/bin/pytest tests/test_adaptive_frequency.py -v`
Expected: ALL PASS (11 tests)

**Step 5: Commit**

```bash
git add super_brain/adaptive_frequency.py tests/test_adaptive_frequency.py
git commit -m "feat(v2.4): add AdaptiveFrequency manager"
```

---

### Task 4: Integrate FactExtractor + Adaptive Frequency into Simulation Loop

**Files:**
- Modify: `eval_conversation.py:962-1049` (simulate_conversation function)
- Test: `tests/test_simulate_v24.py`

**Context:** The simulation loop in `eval_conversation.py` currently runs ThinkSlow every 3 turns (hardcoded at line 1036: `if think_slow and (turn + 1) % 3 == 0`). We need to:
1. Add FactExtractor as an optional parameter
2. Use AdaptiveFrequency for both ThinkSlow and FactExtractor
3. Accumulate facts into a Soul object
4. Return the Soul alongside conversation and ts_results

**Step 1: Write the failing test**

Create `tests/test_simulate_v24.py`:

```python
"""Tests for V2.4 simulate_conversation with FactExtractor + adaptive frequency."""

import json
from unittest.mock import MagicMock, patch

from super_brain.models import (
    Fact, Reality, FactExtractionResult, Soul, ThinkSlowResult, ThinkFastResult,
    PersonalityDNA, SampleSummary, Trait,
)


def test_simulate_returns_soul_when_fact_extractor_provided():
    """simulate_conversation returns (conversation, ts_results, soul) with fact_extractor."""
    from eval_conversation import simulate_conversation, Chatter, PersonalitySpeaker
    from super_brain.profile_gen import generate_profile
    from super_brain.think_slow import ThinkSlow
    from super_brain.fact_extractor import FactExtractor

    # We need to mock the LLM calls
    with patch.object(Chatter, "next_message", return_value="Tell me more about that."):
        with patch.object(PersonalitySpeaker, "respond", return_value="I work as an engineer."):
            with patch.object(ThinkSlow, "extract") as mock_ts:
                with patch.object(FactExtractor, "extract") as mock_fe:
                    # ThinkSlow mock return
                    mock_profile = PersonalityDNA(
                        id="partial",
                        sample_summary=SampleSummary(
                            total_tokens=0, conversation_count=0,
                            date_range=["u", "u"], contexts=["t"],
                            confidence_overall=0.5,
                        ),
                    )
                    mock_ts.return_value = ThinkSlowResult(
                        partial_profile=mock_profile,
                        confidence_map={},
                        low_confidence_traits=[],
                        observations=[],
                        incisive_questions=[],
                    )

                    # FactExtractor mock return
                    mock_fe.return_value = FactExtractionResult(
                        new_facts=[
                            Fact(category="career", content="engineer", confidence=0.9, source_turn=3),
                        ],
                        reality=Reality(
                            summary="An engineer",
                            domains={"career": "engineer"},
                            constraints=[],
                            resources=[],
                        ),
                        secrets=["avoids personal topics"],
                        contradictions=[],
                    )

                    profile = generate_profile("test", seed=0)
                    chatter = Chatter.__new__(Chatter)
                    speaker = PersonalitySpeaker.__new__(PersonalitySpeaker)
                    think_slow = ThinkSlow.__new__(ThinkSlow)
                    fact_extractor = FactExtractor.__new__(FactExtractor)

                    result = simulate_conversation(
                        chatter, speaker, profile, n_turns=6, seed=0,
                        think_slow=think_slow,
                        fact_extractor=fact_extractor,
                    )

                    # Should return 3-tuple now
                    assert len(result) == 3
                    conversation, ts_results, soul = result
                    assert isinstance(soul, Soul)
                    assert len(soul.facts) >= 1
                    assert soul.facts[0].content == "engineer"


def test_simulate_without_fact_extractor_returns_old_format():
    """Without fact_extractor, simulate_conversation returns (conversation, ts_results)."""
    from eval_conversation import simulate_conversation, Chatter, PersonalitySpeaker
    from super_brain.profile_gen import generate_profile
    from super_brain.think_slow import ThinkSlow

    with patch.object(Chatter, "next_message", return_value="Tell me more."):
        with patch.object(PersonalitySpeaker, "respond", return_value="Sure thing."):
            with patch.object(ThinkSlow, "extract") as mock_ts:
                mock_profile = PersonalityDNA(
                    id="partial",
                    sample_summary=SampleSummary(
                        total_tokens=0, conversation_count=0,
                        date_range=["u", "u"], contexts=["t"],
                        confidence_overall=0.5,
                    ),
                )
                mock_ts.return_value = ThinkSlowResult(
                    partial_profile=mock_profile,
                    confidence_map={},
                    low_confidence_traits=[],
                    observations=[],
                    incisive_questions=[],
                )

                profile = generate_profile("test", seed=0)
                chatter = Chatter.__new__(Chatter)
                speaker = PersonalitySpeaker.__new__(PersonalitySpeaker)
                think_slow = ThinkSlow.__new__(ThinkSlow)

                result = simulate_conversation(
                    chatter, speaker, profile, n_turns=4, seed=0,
                    think_slow=think_slow,
                )

                # Old format: 2-tuple (conversation, ts_results)
                assert len(result) == 2
                conversation, ts_results = result
                assert isinstance(conversation, list)
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/michael/super-brain && .venv/bin/pytest tests/test_simulate_v24.py -v`
Expected: FAIL (simulate_conversation doesn't accept fact_extractor parameter)

**Step 3: Modify simulate_conversation**

In `eval_conversation.py`, modify `simulate_conversation()` (lines 962-1049):

Replace the function signature and body. The key changes are:
1. Add `fact_extractor` optional parameter
2. Use `AdaptiveFrequency` for both ThinkSlow and FactExtractor
3. Accumulate results into a `Soul` object
4. Return 3-tuple when fact_extractor is provided

```python
def simulate_conversation(
    chatter: Chatter,
    speaker: PersonalitySpeaker,
    profile: PersonalityDNA,
    n_turns: int,
    seed: int = 0,
    think_slow: "ThinkSlow | None" = None,
    fact_extractor: "FactExtractor | None" = None,
) -> "list[dict] | tuple[list[dict], list] | tuple[list[dict], list, Soul]":
    """Simulate a natural conversation for n_turns exchanges.

    V2.4: When fact_extractor is provided alongside think_slow, both run on
    adaptive frequency and results accumulate into a Soul model.

    Returns:
        - No extractors: list of messages
        - think_slow only: (conversation, think_slow_results)
        - think_slow + fact_extractor: (conversation, think_slow_results, soul)
    """
    import random as _random
    rng = _random.Random(seed)

    conversation: list[dict] = []
    ts_results: list = []
    previous_ts = None
    current_low_conf: list[str] = []

    # V2.3: Instantiate ThinkFast + Conductor when think_slow is available
    think_fast = ThinkFast() if think_slow else None
    conductor = Conductor() if think_slow else None
    last_tf: ThinkFastResult | None = None

    # V2.4: Adaptive frequency + Soul accumulation
    ts_freq = None
    fe_freq = None
    soul = None
    if fact_extractor is not None and think_slow is not None:
        from super_brain.adaptive_frequency import AdaptiveFrequency
        from super_brain.models import Soul as SoulModel
        ts_freq = AdaptiveFrequency(default_interval=3)
        fe_freq = AdaptiveFrequency(default_interval=3)
        soul = SoulModel(
            id=profile.id + "_soul",
            character=PersonalityDNA(
                id="think_slow_partial",
                sample_summary=SampleSummary(
                    total_tokens=0, conversation_count=0,
                    date_range=["unknown", "unknown"],
                    contexts=["soul"], confidence_overall=0.0,
                ),
            ),
        )

    # Start with a random casual opener
    opener = rng.choice(CASUAL_OPENERS)
    conversation.append({"role": "chatter", "text": opener})

    # Speaker responds
    reply = speaker.respond(profile, conversation, turn_number=0)
    conversation.append({"role": "speaker", "text": reply})

    # V2.3: Analyze the first speaker response with ThinkFast
    if think_fast is not None:
        last_tf = think_fast.analyze(conversation)

    # Continue for n_turns - 1 more exchanges
    for turn in range(1, n_turns):
        # V2.3: Use Conductor to decide chatter action when available
        conductor_action = None
        if conductor is not None and last_tf is not None:
            conductor_action = conductor.decide(
                think_fast=last_tf,
                think_slow=previous_ts,
                turn_number=turn + 1,
            )

        # Chatter follows up naturally (with escalation or Conductor guidance)
        chatter_msg = chatter.next_message(
            conversation, turn_number=turn + 1, total_turns=n_turns,
            low_confidence_traits=current_low_conf if current_low_conf else None,
            conductor_action=conductor_action,
        )
        conversation.append({"role": "chatter", "text": chatter_msg})

        # Speaker responds in character (turn_number drives temporal modulation)
        speaker_reply = speaker.respond(profile, conversation, turn_number=turn)
        conversation.append({"role": "speaker", "text": speaker_reply})

        # V2.3: Analyze each speaker response with ThinkFast
        if think_fast is not None:
            last_tf = think_fast.analyze(conversation)

        # V2.4: Adaptive frequency for ThinkSlow
        if think_slow and ts_freq is not None and ts_freq.should_run(turn + 1):
            focus = previous_ts.low_confidence_traits if previous_ts else None
            ts_result = think_slow.extract(
                conversation=conversation,
                focus_traits=focus,
                previous=previous_ts,
            )
            ts_results.append(ts_result)
            previous_ts = ts_result
            current_low_conf = ts_result.low_confidence_traits
            # Report yield: number of traits estimated
            ts_freq.report_yield(len(ts_result.partial_profile.traits))
            # Update soul character
            if soul is not None:
                soul.character = ts_result.partial_profile

        # V2.4: Adaptive frequency for FactExtractor
        if fact_extractor is not None and fe_freq is not None and fe_freq.should_run(turn + 1):
            fe_result = fact_extractor.extract(
                conversation=conversation,
                existing_facts=soul.facts if soul else [],
                current_turn=turn + 1,
            )
            if soul is not None:
                soul.facts.extend(fe_result.new_facts)
                if fe_result.reality is not None:
                    soul.reality = fe_result.reality
                soul.secrets.extend(fe_result.secrets)
                soul.contradictions.extend(fe_result.contradictions)
            # Report yield: new facts + secrets + contradictions
            fe_freq.report_yield(
                len(fe_result.new_facts) + len(fe_result.secrets) + len(fe_result.contradictions)
            )

        # V2.3 fallback: ThinkSlow on fixed interval when no adaptive frequency
        elif think_slow and ts_freq is None and (turn + 1) % 3 == 0:
            focus = previous_ts.low_confidence_traits if previous_ts else None
            ts_result = think_slow.extract(
                conversation=conversation,
                focus_traits=focus,
                previous=previous_ts,
            )
            ts_results.append(ts_result)
            previous_ts = ts_result
            current_low_conf = ts_result.low_confidence_traits

    # Return appropriate format
    if fact_extractor is not None and soul is not None:
        return conversation, ts_results, soul
    if think_slow is not None:
        return conversation, ts_results
    return conversation
```

Also add the import at the top of `eval_conversation.py` (near the other imports from super_brain):

```python
from super_brain.models import Soul, Fact, Reality, FactExtractionResult
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/michael/super-brain && .venv/bin/pytest tests/test_simulate_v24.py -v`
Expected: ALL PASS

**Step 5: Run all tests for regressions**

Run: `cd /Users/michael/super-brain && .venv/bin/pytest tests/ -v`
Expected: ALL PASS (existing tests should still work because old call signature is preserved)

**Step 6: Commit**

```bash
git add eval_conversation.py tests/test_simulate_v24.py
git commit -m "feat(v2.4): integrate FactExtractor + adaptive frequency into simulation loop"
```

---

### Task 5: Soul Coverage Metrics in Eval

**Files:**
- Modify: `eval_conversation.py:1139-1303` (run_eval function)
- Test: `tests/test_soul_coverage.py`

**Context:** Add Soul Coverage metrics alongside MAE in the eval output. The Soul Coverage score measures how populated the Soul model is after a conversation.

**Step 1: Write the failing test**

Create `tests/test_soul_coverage.py`:

```python
"""Tests for V2.4 Soul Coverage scoring."""

from super_brain.models import (
    Soul, Fact, Reality, PersonalityDNA, SampleSummary,
)
from super_brain.soul_coverage import compute_soul_coverage


def _make_profile():
    return PersonalityDNA(
        id="test",
        sample_summary=SampleSummary(
            total_tokens=0, conversation_count=0,
            date_range=["u", "u"], contexts=["t"], confidence_overall=0.5,
        ),
    )


def test_empty_soul_coverage():
    soul = Soul(id="test", character=_make_profile())
    score = compute_soul_coverage(soul)
    assert score == 0.0


def test_full_soul_coverage():
    soul = Soul(
        id="test",
        character=_make_profile(),
        facts=[
            Fact(category="c", content=f"fact_{i}", confidence=0.9, source_turn=i)
            for i in range(10)
        ],
        reality=Reality(
            summary="Full reality",
            domains={"career": "engineer"},
            constraints=["time"],
            resources=["skills"],
        ),
        secrets=["s1", "s2", "s3"],
    )
    score = compute_soul_coverage(soul)
    assert score == 1.0


def test_partial_soul_coverage():
    soul = Soul(
        id="test",
        character=_make_profile(),
        facts=[
            Fact(category="c", content=f"fact_{i}", confidence=0.9, source_turn=i)
            for i in range(5)  # 5/10 = 0.5
        ],
        reality=Reality(
            summary="Partial",
            domains={},
            constraints=[],
            resources=[],
        ),
        # reality populated = 1.0
        secrets=["s1"],  # 1/3 = 0.333
    )
    score = compute_soul_coverage(soul)
    # (0.5 + 1.0 + 0.333) / 3 = 0.611
    assert abs(score - 0.611) < 0.01


def test_coverage_facts_cap():
    """More than 10 facts still gives 1.0 for facts component."""
    soul = Soul(
        id="test",
        character=_make_profile(),
        facts=[
            Fact(category="c", content=f"fact_{i}", confidence=0.9, source_turn=i)
            for i in range(20)
        ],
    )
    # 20 facts → capped at 1.0 for facts, reality=0, secrets=0
    score = compute_soul_coverage(soul)
    assert abs(score - 1.0 / 3) < 0.01
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/michael/super-brain && .venv/bin/pytest tests/test_soul_coverage.py -v`
Expected: FAIL with ModuleNotFoundError (soul_coverage not found)

**Step 3: Write minimal implementation**

Create `super_brain/soul_coverage.py`:

```python
"""Soul Coverage scoring for V2.4 evaluation.

Measures how populated the Soul model is after a conversation.
Components (V2.4 — facts + reality + secrets):
- facts: min(count / 10, 1.0)
- reality: 1.0 if populated, else 0.0
- secrets: min(count / 3, 1.0)
"""

from __future__ import annotations

from super_brain.models import Soul


def compute_soul_coverage(soul: Soul) -> float:
    """Compute Soul Coverage score (0.0-1.0).

    V2.4 components (3 items, equally weighted):
    - facts: min(len / 10, 1.0) — 10+ facts = full
    - reality: 1.0 if populated, else 0.0
    - secrets: min(len / 3, 1.0) — 3+ secrets = full

    Returns:
        Float between 0.0 and 1.0.
    """
    scores: list[float] = []
    scores.append(min(len(soul.facts) / 10.0, 1.0))
    scores.append(1.0 if soul.reality else 0.0)
    scores.append(min(len(soul.secrets) / 3.0, 1.0))
    return sum(scores) / len(scores) if scores else 0.0
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/michael/super-brain && .venv/bin/pytest tests/test_soul_coverage.py -v`
Expected: ALL PASS (4 tests)

**Step 5: Commit**

```bash
git add super_brain/soul_coverage.py tests/test_soul_coverage.py
git commit -m "feat(v2.4): add Soul Coverage scoring"
```

---

### Task 6: Wire FactExtractor into run_eval + Report Soul Coverage

**Files:**
- Modify: `eval_conversation.py:1139-1303` (run_eval function)

**Context:** The `run_eval()` function needs to:
1. Instantiate a `FactExtractor` alongside ThinkSlow
2. Pass it to `simulate_conversation()`
3. Report Soul Coverage metrics for each profile and in the summary
4. Log FactExtractor progress (facts found, reality populated, secrets)

**Step 1: Modify run_eval**

In `eval_conversation.py`, update `run_eval()` starting at line 1139.

Add import at top of file (near other super_brain imports):
```python
from super_brain.fact_extractor import FactExtractor
from super_brain.soul_coverage import compute_soul_coverage
```

Modify the `run_eval()` function body. Key changes:

After line 1160 (`think_slow = ThinkSlow(api_key=api_key)`), add:
```python
    fact_extractor = FactExtractor(api_key=api_key)
```

Replace lines 1182-1185 (the simulate_conversation call):
```python
        # Simulate full conversation
        print(f"  Simulating {max_turns}-turn conversation...", end=" ", flush=True)
        sim_result = simulate_conversation(
            chatter, speaker, profile, n_turns=max_turns, seed=i,
            think_slow=think_slow,
            fact_extractor=fact_extractor,
        )
        conversation, ts_results, soul = sim_result
        total_words = len(extract_speaker_text(conversation).split())
        print(f"done ({total_words} speaker words)")
```

After the ThinkSlow logging block (after line 1198), add Soul logging:
```python
        # Log Soul state
        if soul is not None:
            coverage = compute_soul_coverage(soul)
            print(f"    Soul: {len(soul.facts)} facts, "
                  f"reality={'yes' if soul.reality else 'no'}, "
                  f"{len(soul.secrets)} secrets, "
                  f"{len(soul.contradictions)} contradictions, "
                  f"coverage={coverage:.2f}")
```

After the per-profile results (around line 1235), store soul coverage:
```python
        if soul is not None:
            profile_results["soul_coverage"] = compute_soul_coverage(soul)
            profile_results["soul_facts_count"] = len(soul.facts)
            profile_results["soul_reality_populated"] = soul.reality is not None
            profile_results["soul_secrets_count"] = len(soul.secrets)
```

In the summary section (after the learning curve, around line 1266), add Soul Coverage summary:
```python
    # ── Soul Coverage summary ─────────────────────────────────────────────
    coverages = [pr.get("soul_coverage", 0.0) for pr in all_results.values()
                 if isinstance(pr.get("soul_coverage"), (int, float))]
    if coverages:
        avg_cov = statistics.mean(coverages)
        fact_counts = [pr.get("soul_facts_count", 0) for pr in all_results.values()]
        secret_counts = [pr.get("soul_secrets_count", 0) for pr in all_results.values()]
        reality_count = sum(1 for pr in all_results.values() if pr.get("soul_reality_populated"))
        print(f"\n{'='*70}")
        print(f"  SOUL COVERAGE (V2.4)")
        print(f"{'='*70}")
        print(f"  Avg coverage score: {avg_cov:.3f}")
        print(f"  Avg facts per profile: {statistics.mean(fact_counts):.1f}")
        print(f"  Reality populated: {reality_count}/{n_profiles}")
        print(f"  Avg secrets per profile: {statistics.mean(secret_counts):.1f}")
```

**Step 2: Run all tests to verify no regressions**

Run: `cd /Users/michael/super-brain && .venv/bin/pytest tests/ -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add eval_conversation.py
git commit -m "feat(v2.4): wire FactExtractor into eval + report Soul Coverage"
```

---

### Task 7: Run V2.4 Eval and Record Results

**Step 1: Run the eval**

```bash
cd /Users/michael/super-brain && ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY .venv/bin/python eval_conversation.py 3 20
```

Expected output includes:
- Standard MAE metrics (target: ≤ 0.185)
- NEW: Soul Coverage section with facts_count, reality_populated, secrets_count
- Target: facts ≥ 5 per profile, reality populated for all profiles

**Step 2: Record results in EVAL_HISTORY.md**

Append V2.4 results to `EVAL_HISTORY.md` following the existing format:
- Version, date, MAE, ≤0.25, ≤0.40, per-dimension table
- NEW: Soul Coverage score, facts count, reality populated, secrets count
- Changelog listing: FactExtractor module, adaptive frequency, Soul model, Soul Coverage metrics

**Step 3: Commit**

```bash
git add EVAL_HISTORY.md eval_conversation_results.json
git commit -m "eval(v2.4): record V2.4 evaluation results"
```

**Step 4: Push to GitHub**

```bash
git push https://$GH_TOKEN@github.com/mozatyin/communication-dna.git main
```

---

## Summary

| Task | What | New Files | Tests |
|------|------|-----------|-------|
| 1 | V2.4 data models | — | test_models_v24.py (7 tests) |
| 2 | FactExtractor module | fact_extractor.py | test_fact_extractor.py (7 tests) |
| 3 | Adaptive frequency | adaptive_frequency.py | test_adaptive_frequency.py (11 tests) |
| 4 | Integration into sim loop | — | test_simulate_v24.py (2 tests) |
| 5 | Soul Coverage scoring | soul_coverage.py | test_soul_coverage.py (4 tests) |
| 6 | Wire into eval + reporting | — | — (manual verification) |
| 7 | Run eval + record results | — | — (eval run) |

**Total new tests: ~31**
**Total new files: 3** (fact_extractor.py, adaptive_frequency.py, soul_coverage.py)
**Total modified files: 2** (models.py, eval_conversation.py)
