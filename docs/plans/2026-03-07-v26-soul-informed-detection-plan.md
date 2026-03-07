# V2.6 Soul-Informed Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Pass accumulated Soul context (facts, reality, intentions) to the Detector so it can better calibrate personality trait scores against what is already known about the person.

**Architecture:** A new helper function `_build_detector_soul_context()` in `eval_conversation.py` serializes the most relevant Soul layers (top 10 facts by confidence, reality summary/constraints/resources, top 3 intentions by strength) into a concise markdown block. This block is injected into every Detector batch prompt between the "Target Speaker" section and the "Dimensions to Analyze" section. The Detector's `analyze()` method gains an optional `soul_context` parameter, and `detect_and_compare()` gains an optional `soul` parameter that wires everything together. The prompt instructs the LLM to use Soul context for calibration but to base scores on observed behavior.

**Tech Stack:** Python 3.12, Pydantic, Anthropic SDK, pytest

---

### Task 1: Add `_build_detector_soul_context()` helper + tests

**Files:**
- Modify: `/Users/michael/super-brain/eval_conversation.py` (add helper near line 1162, after `extract_speaker_text`)
- Create: `/Users/michael/super-brain/tests/test_detector_soul_context.py`

**Step 1: Write the failing test**

Create `tests/test_detector_soul_context.py`:

```python
"""Tests for V2.6 Soul-informed detection context builder."""

from eval_conversation import _build_detector_soul_context
from super_brain.models import (
    Soul, Fact, Reality, Intention, Gap,
    PersonalityDNA, SampleSummary,
)


def _make_profile():
    return PersonalityDNA(
        id="test",
        sample_summary=SampleSummary(
            total_tokens=0, conversation_count=0,
            date_range=["u", "u"], contexts=["t"], confidence_overall=0.5,
        ),
    )


def test_empty_soul_returns_empty():
    soul = Soul(id="test", character=_make_profile())
    result = _build_detector_soul_context(soul)
    assert result == ""


def test_soul_with_facts_only():
    soul = Soul(
        id="test",
        character=_make_profile(),
        facts=[
            Fact(category="career", content="software engineer", confidence=0.9, source_turn=3),
            Fact(category="hobby", content="plays guitar", confidence=0.7, source_turn=5),
        ],
    )
    result = _build_detector_soul_context(soul)
    assert "software engineer" in result
    assert "guitar" in result
    assert "Background Information" in result
    assert "OBSERVED BEHAVIOR" in result


def test_soul_with_reality():
    soul = Soul(
        id="test",
        character=_make_profile(),
        facts=[Fact(category="career", content="engineer", confidence=0.9, source_turn=1)],
        reality=Reality(
            summary="Mid-career engineer feeling stuck",
            domains={"career": "engineering"},
            constraints=["mortgage", "limited savings"],
            resources=["technical skills", "network"],
        ),
    )
    result = _build_detector_soul_context(soul)
    assert "feeling stuck" in result
    assert "mortgage" in result
    assert "technical skills" in result


def test_soul_with_intentions():
    soul = Soul(
        id="test",
        character=_make_profile(),
        facts=[Fact(category="career", content="engineer", confidence=0.9, source_turn=1)],
        intentions=[
            Intention(description="start own business", domain="career", strength=0.85),
            Intention(description="travel more", domain="personal_growth", strength=0.6),
            Intention(description="learn cooking", domain="creative", strength=0.4),
            Intention(description="exercise daily", domain="health", strength=0.3),
        ],
    )
    result = _build_detector_soul_context(soul)
    # Should include top 3 by strength
    assert "start own business" in result
    assert "travel more" in result
    assert "learn cooking" in result
    # 4th intention (lowest strength) should NOT be included
    assert "exercise daily" not in result


def test_facts_limited_to_10():
    soul = Soul(
        id="test",
        character=_make_profile(),
        facts=[
            Fact(category="misc", content=f"fact_{i}", confidence=0.5 + i * 0.01, source_turn=i)
            for i in range(20)
        ],
    )
    result = _build_detector_soul_context(soul)
    # Should have at most 10 fact lines
    fact_lines = [line for line in result.split("\n") if line.startswith("- [")]
    assert len(fact_lines) <= 10
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_detector_soul_context.py -v`

Expected: `ImportError` or `AttributeError` because `_build_detector_soul_context` does not exist yet.

**Step 3: Write minimal implementation**

Add the following function to `/Users/michael/super-brain/eval_conversation.py`, immediately after `extract_speaker_text()` (after line 1177):

```python
def _build_detector_soul_context(soul: "Soul") -> str:
    """Build concise Soul context for Detector calibration."""
    sections = []

    # Top facts by confidence (max 10)
    if soul.facts:
        top_facts = sorted(soul.facts, key=lambda f: -f.confidence)[:10]
        fact_lines = [f"- [{f.category}] {f.content}" for f in top_facts]
        sections.append("Known facts:\n" + "\n".join(fact_lines))

    # Reality summary
    if soul.reality:
        sections.append(f"Reality: {soul.reality.summary}")
        if soul.reality.constraints:
            sections.append(f"Constraints: {', '.join(soul.reality.constraints)}")
        if soul.reality.resources:
            sections.append(f"Resources: {', '.join(soul.reality.resources)}")

    # Top intentions (max 3)
    if soul.intentions:
        top_int = sorted(soul.intentions, key=lambda i: -i.strength)[:3]
        int_lines = [f"- {i.description} ({i.domain}, strength={i.strength:.1f})" for i in top_int]
        sections.append("Key intentions:\n" + "\n".join(int_lines))

    if not sections:
        return ""

    return (
        "## Background Information About This Person\n"
        "The following facts were gathered from earlier in the conversation:\n"
        + "\n".join(sections)
        + "\n\nUse this as CONTEXT to calibrate your scoring, but base scores on "
        "OBSERVED BEHAVIOR in the text, not on these facts alone."
    )
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_detector_soul_context.py -v`

Expected: All 5 tests pass.

**Step 5: Commit**

```
git add tests/test_detector_soul_context.py eval_conversation.py
git commit -m "V2.6: add _build_detector_soul_context() helper + tests

Serializes top Soul facts, reality, and intentions into a concise
markdown block for Detector prompt injection.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Modify `Detector.analyze()` to accept `soul_context` parameter

**Files:**
- Modify: `/Users/michael/super-brain/super_brain/detector.py` (add parameter + inject into prompt)
- Modify: `/Users/michael/super-brain/tests/test_detector_soul_context.py` (add signature test)

**Step 1: Write the failing test**

Append to `tests/test_detector_soul_context.py`:

```python
def test_detector_analyze_accepts_soul_context():
    """Verify Detector.analyze() signature accepts soul_context parameter."""
    import inspect
    from super_brain.detector import Detector
    sig = inspect.signature(Detector.analyze)
    assert "soul_context" in sig.parameters
    assert sig.parameters["soul_context"].default is None
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_detector_soul_context.py::test_detector_analyze_accepts_soul_context -v`

Expected: `AssertionError` because `soul_context` is not in the signature yet.

**Step 3: Write minimal implementation**

In `/Users/michael/super-brain/super_brain/detector.py`, modify the `analyze()` method signature (line 343-349) to add the `soul_context` parameter:

```python
    def analyze(
        self,
        text: str,
        speaker_id: str,
        speaker_label: str = "Speaker",
        context: str = "general",
        soul_context: str | None = None,
    ) -> PersonalityDNA:
```

Then modify the user_message construction (line 371-380) to inject the soul context between the "Target Speaker" section and the "Dimensions to Analyze" section:

```python
            soul_section = f"\n\n{soul_context}\n" if soul_context else ""

            user_message = (
                f"## Text Sample\n\n{text}\n\n"
                f"## Target Speaker\n\nAnalyze speaker labeled '{speaker_label}'.\n\n"
                f"{soul_section}"
                f"## Dimensions to Analyze: {dim_labels}\n\n"
                f"{trait_prompt}\n\n"
                f"{calibration_section}"
                f"Return JSON with 'reasoning' and 'scores' arrays as specified. "
                f"Analyze ONLY the {len(batch_traits)} traits listed above. "
                f"You MUST return exactly {len(batch_traits)} scores."
            )
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_detector_soul_context.py -v`

Expected: All 6 tests pass (5 from Task 1 + 1 new).

**Step 5: Commit**

```
git add super_brain/detector.py tests/test_detector_soul_context.py
git commit -m "V2.6: add soul_context parameter to Detector.analyze()

Injects Soul context block into each batch prompt between the Target
Speaker section and Dimensions to Analyze section.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Wire Soul into `detect_and_compare()` and `run_eval()`

**Files:**
- Modify: `/Users/michael/super-brain/eval_conversation.py` (`detect_and_compare` at line 1180, `run_eval` call at line 1348)

**Step 1: Write the failing test**

Append to `tests/test_detector_soul_context.py`:

```python
def test_detect_and_compare_accepts_soul_parameter():
    """Verify detect_and_compare() signature accepts soul parameter."""
    import inspect
    from eval_conversation import detect_and_compare
    sig = inspect.signature(detect_and_compare)
    assert "soul" in sig.parameters
    assert sig.parameters["soul"].default is None
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_detector_soul_context.py::test_detect_and_compare_accepts_soul_parameter -v`

Expected: `AssertionError` because `soul` is not in the signature yet.

**Step 3: Write minimal implementation**

Modify `detect_and_compare()` in `/Users/michael/super-brain/eval_conversation.py` (line 1180-1196):

```python
def detect_and_compare(
    detector: Detector,
    conversation: list[dict],
    profile: PersonalityDNA,
    profile_name: str,
    soul: "Soul | None" = None,
) -> dict:
    """Run detection on full conversation and compare with ground truth."""
    # V0.2: Feed FULL conversation to detector, not just speaker text
    full_text = format_full_conversation(conversation)
    speaker_text = extract_speaker_text(conversation)
    word_count = len(speaker_text.split())

    # V2.6: Build Soul context for Detector
    soul_ctx = _build_detector_soul_context(soul) if soul else None

    detected = detector.analyze(
        text=full_text,
        speaker_id=f"eval_{profile_name}",
        speaker_label="Person B",
        soul_context=soul_ctx,
    )
```

Modify the `run_eval()` call to `detect_and_compare` (line 1348):

```python
            result = detect_and_compare(detector, conv_slice, profile, profile_name, soul=soul)
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_detector_soul_context.py -v`

Expected: All 7 tests pass.

**Step 5: Commit**

```
git add eval_conversation.py tests/test_detector_soul_context.py
git commit -m "V2.6: wire Soul context into detect_and_compare() and run_eval()

detect_and_compare() now accepts an optional Soul, builds context via
_build_detector_soul_context(), and passes it to Detector.analyze().
run_eval() passes the Soul from simulate_conversation() through.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Run V2.6 eval and record results

**Files:**
- Run: `/Users/michael/super-brain/eval_conversation.py`
- Modify: `/Users/michael/super-brain/EVAL_HISTORY.md`

**Step 1: Run full test suite to confirm nothing is broken**

Run: `.venv/bin/pytest tests/ -v`

Expected: All tests pass.

**Step 2: Run the V2.6 eval**

Run:

```bash
cd /Users/michael/super-brain && ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY .venv/bin/python eval_conversation.py 3 20
```

Expected: 3 profiles, 20-turn conversations, detection at checkpoints. Capture output for recording.

**Step 3: Record results in EVAL_HISTORY.md**

Add a new section to `/Users/michael/super-brain/EVAL_HISTORY.md` with the V2.6 results following the existing format. Include:
- Version label: V2.6 Soul-Informed Detection
- Date: 2026-03-07
- MAE, within-0.25, within-0.40 counts at each checkpoint
- Per-dimension MAE breakdown
- Brief comparison note vs V2.5 results

**Step 4: Commit and push**

```
git add EVAL_HISTORY.md
git commit -m "V2.6: record eval results — Soul-informed detection

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
git push https://$GH_TOKEN@github.com/mozatyin/super-brain.git HEAD
```
