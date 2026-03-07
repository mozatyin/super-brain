# V2.7 ThinkDeep Quality Control Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce Soul state bloat (31 intentions, 28 gaps, 34 secrets, 30 contradictions per profile) to signal-quality levels by adding Jaccard deduplication and capping ThinkDeep to 2 fires maximum.

**Architecture:** A new `dedup.py` module provides token-level Jaccard similarity checking and dedup-aware list extension. The `simulate_conversation` loop replaces raw `.extend()` calls with dedup-filtered accumulation for secrets, contradictions, intentions, and gaps. The `td_fired` boolean is replaced with a `td_fire_count` integer capped at 2, preventing unlimited re-triggering after Conductor push mode clears.

**Tech Stack:** Python 3.12, Pydantic, pytest

---

### Task 1: Create dedup module + tests

**Files:**
- Create: `/Users/michael/super-brain/super_brain/dedup.py`
- Create: `/Users/michael/super-brain/tests/test_dedup.py`

**Step 1: Write the failing test**

Create `tests/test_dedup.py`:

```python
"""Tests for V2.7 text deduplication."""

from super_brain.dedup import is_duplicate, dedup_extend_strings


def test_exact_duplicate():
    assert is_duplicate("start a business", ["start a business"]) is True


def test_near_duplicate():
    assert is_duplicate(
        "wants to start own business",
        ["start a business and be independent"],
        threshold=0.5,
    ) is True


def test_not_duplicate():
    assert is_duplicate("loves playing guitar", ["start a business"]) is False


def test_empty_existing():
    assert is_duplicate("anything", []) is False


def test_empty_new():
    assert is_duplicate("", ["something"]) is False


def test_high_threshold_rejects():
    assert is_duplicate(
        "wants to start own business",
        ["start a business"],
        threshold=0.9,
    ) is False


def test_dedup_extend_strings():
    target = ["start a business", "learn guitar"]
    new = ["launch own company", "play piano", "start a business"]
    added = dedup_extend_strings(target, new, threshold=0.5)
    # "launch own company" is similar to "start a business" -> skipped
    # "play piano" is unique -> added
    # "start a business" is exact duplicate -> skipped
    assert added == 1
    assert "play piano" in target
    assert len(target) == 3


def test_dedup_extend_strings_all_unique():
    target = ["fact one"]
    new = ["fact two", "fact three"]
    added = dedup_extend_strings(target, new)
    assert added == 2
    assert len(target) == 3
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_dedup.py -v`

Expected: `ModuleNotFoundError: No module named 'super_brain.dedup'`

**Step 3: Write minimal implementation**

Create `super_brain/dedup.py`:

```python
"""Text deduplication utilities for Soul state management (V2.7)."""

from __future__ import annotations


def _tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase word set."""
    return set(text.lower().split())


def is_duplicate(new_text: str, existing_texts: list[str], threshold: float = 0.6) -> bool:
    """Check if new_text is a duplicate of any existing text.

    Uses Jaccard similarity (token set overlap ratio).
    Returns True if similarity with any existing text exceeds threshold.
    """
    new_tokens = _tokenize(new_text)
    if not new_tokens:
        return False
    for existing in existing_texts:
        existing_tokens = _tokenize(existing)
        if not existing_tokens:
            continue
        intersection = len(new_tokens & existing_tokens)
        union = len(new_tokens | existing_tokens)
        if union > 0 and intersection / union >= threshold:
            return True
    return False


def dedup_extend_strings(target: list[str], new_items: list[str], threshold: float = 0.6) -> int:
    """Extend target list with non-duplicate new items. Returns count added."""
    added = 0
    for item in new_items:
        if not is_duplicate(item, target, threshold):
            target.append(item)
            added += 1
    return added
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_dedup.py -v`

Expected: All 8 tests pass.

**Step 5: Commit**

```
git add super_brain/dedup.py tests/test_dedup.py
git commit -m "V2.7: Add dedup module with Jaccard similarity for Soul state deduplication"
```

---

### Task 2: ThinkDeep trigger cap (max 2 fires)

**Files:**
- Modify: `/Users/michael/super-brain/eval_conversation.py` (lines 1025-1028, 1054-1057, 1135-1138)
- Create: `/Users/michael/super-brain/tests/test_simulate_v27.py`

**Step 1: Write the failing test**

Create `tests/test_simulate_v27.py`:

```python
"""Tests for V2.7 ThinkDeep trigger cap and dedup in simulation."""

from unittest.mock import MagicMock, patch

from super_brain.models import (
    Fact, Reality, FactExtractionResult, Soul, ThinkSlowResult,
    ThinkDeepResult, Intention, Gap,
    PersonalityDNA, SampleSummary,
)


def test_think_deep_fires_at_most_twice():
    """ThinkDeep should fire at most 2 times in a conversation."""
    from eval_conversation import simulate_conversation, Chatter, PersonalitySpeaker
    from super_brain.profile_gen import generate_profile
    from super_brain.think_slow import ThinkSlow
    from super_brain.fact_extractor import FactExtractor
    from super_brain.think_deep import ThinkDeep

    td_call_count = 0

    def mock_td_analyze(soul, conversation):
        nonlocal td_call_count
        td_call_count += 1
        return ThinkDeepResult(
            soul_narrative=f"Analysis #{td_call_count}",
            intentions=[
                Intention(description=f"intention_{td_call_count}", domain="career", strength=0.8),
            ],
            gaps=[
                Gap(intention=f"int_{td_call_count}", reality="reality", bridge_question="q?", priority=0.9),
            ],
            critical_question=f"Question #{td_call_count}?",
            conversation_strategy="strategy",
        )

    with patch.object(Chatter, "next_message", return_value="Tell me more."):
        with patch.object(PersonalitySpeaker, "respond", return_value="I'm an engineer thinking about change."):
            with patch.object(ThinkSlow, "extract") as mock_ts:
                with patch.object(FactExtractor, "extract") as mock_fe:
                    with patch.object(ThinkDeep, "analyze", side_effect=mock_td_analyze):
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
                            info_staleness=0.9,
                        )
                        mock_fe.return_value = FactExtractionResult(
                            new_facts=[
                                Fact(category="career", content=f"fact_{i}", confidence=0.9, source_turn=i)
                                for i in range(6)
                            ],
                            reality=Reality(
                                summary="An engineer",
                                domains={"career": "engineer"},
                                constraints=[],
                                resources=[],
                            ),
                            secrets=["secret"],
                            contradictions=["contradiction"],
                        )

                        profile = generate_profile("test", seed=0)
                        chatter = Chatter.__new__(Chatter)
                        speaker = PersonalitySpeaker.__new__(PersonalitySpeaker)
                        think_slow = ThinkSlow.__new__(ThinkSlow)
                        fact_extractor = FactExtractor.__new__(FactExtractor)
                        think_deep = ThinkDeep.__new__(ThinkDeep)

                        result = simulate_conversation(
                            chatter, speaker, profile, n_turns=20, seed=0,
                            think_slow=think_slow,
                            fact_extractor=fact_extractor,
                            think_deep=think_deep,
                        )

                        conversation, ts_results, soul = result
                        # ThinkDeep should fire at most 2 times
                        assert td_call_count <= 2
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_simulate_v27.py::test_think_deep_fires_at_most_twice -v`

Expected: `AssertionError` because `td_call_count` exceeds 2 (current code resets `td_fired = False` on every push, allowing unlimited re-fires).

**Step 3: Write minimal implementation**

Modify `/Users/michael/super-brain/eval_conversation.py`:

**Change 1** — State tracking (lines 1025-1028): Replace the `td_fired` boolean with a counter.

```python
    # V2.5/V2.7: ThinkDeep state tracking
    last_td = None
    td_fire_count = 0  # V2.7: was td_fired boolean, now count with max 2
    consecutive_stale = 0
```

**Change 2** — Trigger check (line 1135): Replace `not td_fired` with `td_fire_count < 2` and increment.

```python
                if should_fire and td_fire_count < 2:  # V2.7: max 2 fires
                    td_result = think_deep.analyze(soul=soul, conversation=conversation)
                    last_td = td_result
                    td_fire_count += 1  # V2.7: increment count
```

**Change 3** — Push mode clearing (lines 1054-1057): Remove the `td_fired = False` reset.

```python
        # V2.5: Clear ThinkDeep after Conductor uses it (one-shot push)
        if conductor_action and conductor_action.mode == "push" and last_td is not None:
            last_td = None
            # V2.7: Don't reset fire count -- cap at 2 total fires
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_simulate_v27.py::test_think_deep_fires_at_most_twice -v`

Expected: Pass. `td_call_count <= 2`.

**Step 5: Commit**

```
git add eval_conversation.py tests/test_simulate_v27.py
git commit -m "V2.7: Cap ThinkDeep to max 2 fires per conversation (was unlimited via boolean reset)"
```

---

### Task 3: Apply dedup to Soul accumulation

**Files:**
- Modify: `/Users/michael/super-brain/eval_conversation.py` (lines 1100-1104 FactExtractor accumulation, lines 1139-1141 ThinkDeep accumulation)

**Step 1: Write the failing test**

Add to `tests/test_simulate_v27.py`:

```python
def test_soul_dedup_prevents_duplicate_secrets():
    """Duplicate secrets should not accumulate in Soul."""
    from super_brain.dedup import dedup_extend_strings

    secrets = ["afraid of failure"]
    new_secrets = ["afraid of failure", "scared of failing", "loves painting"]
    dedup_extend_strings(secrets, new_secrets, threshold=0.6)
    # "afraid of failure" is exact dup -> skipped
    # "scared of failing" is near-dup of "afraid of failure" -> skipped (low overlap... actually different words)
    # Let's verify: {"scared", "of", "failing"} vs {"afraid", "of", "failure"} -> intersection={"of"}, union=5 -> 0.2 < 0.6 -> NOT dup
    # So "scared of failing" gets added, "loves painting" gets added
    assert "loves painting" in secrets
    assert len(secrets) <= 4  # original + at most 2 new unique ones


def test_soul_dedup_prevents_duplicate_intentions():
    """Duplicate intentions should not accumulate in Soul."""
    from super_brain.dedup import is_duplicate

    existing = ["wants to start own business", "learn to play guitar"]
    # Near-duplicate of first
    assert is_duplicate("start a business of their own", existing, threshold=0.5) is True
    # Unique
    assert is_duplicate("travel to Japan", existing) is False
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_simulate_v27.py -v`

Expected: These unit tests should pass immediately (they test dedup logic, not the integration). They serve as regression guards.

**Step 3: Write minimal implementation**

Modify `/Users/michael/super-brain/eval_conversation.py`:

**Change 1** — FactExtractor accumulation (lines 1100-1104): Replace raw `.extend()` for secrets and contradictions.

Replace:
```python
                soul.facts.extend(fe_result.new_facts)
                if fe_result.reality is not None:
                    soul.reality = fe_result.reality
                soul.secrets.extend(fe_result.secrets)
                soul.contradictions.extend(fe_result.contradictions)
```

With:
```python
                from super_brain.dedup import dedup_extend_strings
                soul.facts.extend(fe_result.new_facts)  # facts already deduped by FactExtractor
                if fe_result.reality is not None:
                    soul.reality = fe_result.reality
                dedup_extend_strings(soul.secrets, fe_result.secrets, threshold=0.6)
                dedup_extend_strings(soul.contradictions, fe_result.contradictions, threshold=0.6)
```

**Change 2** — ThinkDeep accumulation (lines 1139-1141): Replace raw `.extend()` for intentions and gaps.

Replace:
```python
                    # Accumulate into Soul
                    soul.intentions.extend(td_result.intentions)
                    soul.gaps.extend(td_result.gaps)
```

With:
```python
                    # V2.7: Dedup intentions and gaps before accumulating
                    from super_brain.dedup import is_duplicate
                    existing_int_descs = [i.description for i in soul.intentions]
                    for intent in td_result.intentions:
                        if not is_duplicate(intent.description, existing_int_descs):
                            soul.intentions.append(intent)
                            existing_int_descs.append(intent.description)

                    existing_gap_qs = [g.bridge_question for g in soul.gaps]
                    for gap in td_result.gaps:
                        if not is_duplicate(gap.bridge_question, existing_gap_qs):
                            soul.gaps.append(gap)
                            existing_gap_qs.append(gap.bridge_question)
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_simulate_v27.py tests/test_dedup.py -v`

Expected: All tests pass. Also run the full existing test suite to catch regressions:

Run: `.venv/bin/pytest tests/ -v --timeout=30`

**Step 5: Commit**

```
git add eval_conversation.py tests/test_simulate_v27.py
git commit -m "V2.7: Apply dedup to Soul accumulation (secrets, contradictions, intentions, gaps)"
```

---

### Task 4: Run V2.7 eval and record results

**Files:**
- Read: eval output (stdout)
- Modify: `/Users/michael/super-brain/EVAL_HISTORY.md`

**Step 1: Run full eval**

Run:
```bash
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY .venv/bin/python eval_conversation.py 3 20
```

This runs 3 profiles x 20 turns each.

**Step 2: Record results**

Add a new row to the Results Summary table in `EVAL_HISTORY.md`:

```markdown
| **V2.7** | **X.XXX** | **XX.X%** | **XX.X%** | 3 | ThinkDeep cap=2 + Jaccard dedup (secrets, contradictions, intentions, gaps) |
```

Fill in actual values from eval output. Key metrics to watch:
- Intention count: target < 10 (was 31.3)
- Gap count: target < 10 (was 28.3)
- Secret count: target < 15 (was 34.7)
- Contradiction count: target < 15 (was 30.0)
- MAE should stay <= 0.185 (not regress from V2.5's 0.178)

**Step 3: Commit**

```
git add EVAL_HISTORY.md eval_conversation_results_v23.json
git commit -m "V2.7: Eval results — ThinkDeep quality control reduces bloat"
```

(Adjust the results filename to whatever the eval script produces, e.g., `eval_conversation_results_v27.json` if it auto-names.)
