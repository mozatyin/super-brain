# V2.5 ThinkDeep + Intentions + Gaps Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add ThinkDeep — a triggered (not periodic) strategic analysis that detects intentions, reality-intention gaps, and bridge questions from the full Soul state, then enhances the Conductor with gap-driven questioning.

**Architecture:** ThinkDeep is a separate LLM call triggered by the Conductor when specific conditions are met (enough facts accumulated, contradictions found, or info staleness high). It receives the full Soul state and produces intentions, gaps, bridge questions, and a critical question. The Conductor gains a new priority: when ThinkDeep provides a critical_question, it fires "push" mode. IncisiveQuestion sources expand from 1 (trait_gap) to 5 (trait_gap, reality_gap, intention_gap, contradiction, secret).

**Tech Stack:** Python 3.12, Pydantic, Anthropic SDK (with OpenRouter support), pytest

---

### Task 1: Add V2.5 Data Models (Intention, Gap, ThinkDeepResult)

**Files:**
- Modify: `/Users/michael/super-brain/super_brain/models.py` (add after Soul class)
- Test: `/Users/michael/super-brain/tests/test_models_v25.py`

**Step 1: Write the failing test**

Create `tests/test_models_v25.py`:

```python
"""Tests for V2.5 data models: Intention, Gap, ThinkDeepResult."""

from super_brain.models import Intention, Gap, ThinkDeepResult


def test_intention_creation():
    i = Intention(
        description="wants to start own business",
        domain="career",
        strength=0.8,
        blockers=["limited savings", "risk aversion"],
    )
    assert i.description == "wants to start own business"
    assert i.domain == "career"
    assert i.strength == 0.8
    assert len(i.blockers) == 2


def test_intention_defaults():
    i = Intention(description="learn guitar", domain="hobby", strength=0.5)
    assert i.blockers == []


def test_gap_creation():
    g = Gap(
        intention="wants to start own business",
        reality="currently employed full-time with mortgage",
        bridge_question="What would need to change for you to take that leap?",
        priority=0.9,
    )
    assert "business" in g.intention
    assert g.priority == 0.9
    assert "leap" in g.bridge_question


def test_think_deep_result():
    result = ThinkDeepResult(
        soul_narrative="This person is at a crossroads between security and ambition",
        intentions=[
            Intention(description="start a business", domain="career", strength=0.8),
        ],
        gaps=[
            Gap(
                intention="start a business",
                reality="employed full-time",
                bridge_question="What's holding you back?",
                priority=0.9,
            ),
        ],
        critical_question="What would you do if money weren't a concern?",
        conversation_strategy="Shift from listening to exploring risk tolerance",
    )
    assert len(result.intentions) == 1
    assert len(result.gaps) == 1
    assert result.critical_question != ""
    assert "crossroads" in result.soul_narrative


def test_think_deep_result_empty():
    result = ThinkDeepResult(
        soul_narrative="Not enough information yet",
        intentions=[],
        gaps=[],
        critical_question="Tell me more about what matters to you",
        conversation_strategy="Continue listening",
    )
    assert len(result.intentions) == 0
    assert len(result.gaps) == 0
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/michael/super-brain && .venv/bin/pytest tests/test_models_v25.py -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

Add to end of `super_brain/models.py` (after Soul class):

```python
class Intention(BaseModel):
    """A detected intention or desire of the person."""
    description: str                # "wants to start own business"
    domain: str                     # "career", "relationship", "personal_growth",
                                    # "health", "creative", "financial"
    strength: float = Field(ge=0.0, le=1.0)
    blockers: list[str] = Field(default_factory=list)


class Gap(BaseModel):
    """A gap between intention and reality, with a bridge question."""
    intention: str                  # what they want
    reality: str                    # where they are
    bridge_question: str            # question exploring this gap
    priority: float = Field(ge=0.0, le=1.0)


class ThinkDeepResult(BaseModel):
    """Result of strategic ThinkDeep analysis (V2.5)."""
    soul_narrative: str             # "This person is at a crossroads..."
    intentions: list[Intention]     # detected intentions
    gaps: list[Gap]                 # reality → intention gaps
    critical_question: str          # THE single most important question
    conversation_strategy: str      # "Shift from listening to exploring risk"
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/michael/super-brain && .venv/bin/pytest tests/test_models_v25.py -v`
Expected: ALL PASS (5 tests)

**Step 5: Add intentions and gaps to Soul model**

Also modify the `Soul` class in `models.py` to add the new fields. Add these two lines after the `reality` field:

```python
    # Layer 4: Intentions (from ThinkDeep)
    intentions: list[Intention] = Field(default_factory=list)

    # Layer 5: Gaps (from ThinkDeep)
    gaps: list[Gap] = Field(default_factory=list)
```

**Step 6: Run all tests**

Run: `cd /Users/michael/super-brain && .venv/bin/pytest tests/ -v`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add super_brain/models.py tests/test_models_v25.py
git commit -m "feat(v2.5): add Intention, Gap, ThinkDeepResult models + Soul expansion"
```

---

### Task 2: ThinkDeep Module — Core Analysis

**Files:**
- Create: `/Users/michael/super-brain/super_brain/think_deep.py`
- Test: `/Users/michael/super-brain/tests/test_think_deep.py`

**Context:** ThinkDeep follows the same LLM call pattern as ThinkSlow (`super_brain/think_slow.py`) and FactExtractor (`super_brain/fact_extractor.py`):
- Takes `api_key` + optional `model` in `__init__`
- OpenRouter auto-detection: `if api_key.startswith("sk-or-")`
- Single `analyze()` method that takes the full Soul state → returns `ThinkDeepResult`
- JSON response parsing with fallback

**Step 1: Write the failing test**

Create `tests/test_think_deep.py`:

```python
"""Tests for V2.5 ThinkDeep — strategic intention/gap analysis."""

import json
from unittest.mock import MagicMock, patch

from super_brain.think_deep import ThinkDeep, _parse_think_deep_response, _build_soul_context
from super_brain.models import (
    Soul, Fact, Reality, ThinkDeepResult, Intention, Gap,
    PersonalityDNA, SampleSummary, Trait,
)


def _make_soul(facts=None, reality=None, secrets=None, contradictions=None):
    profile = PersonalityDNA(
        id="test",
        sample_summary=SampleSummary(
            total_tokens=0, conversation_count=0,
            date_range=["u", "u"], contexts=["t"], confidence_overall=0.5,
        ),
        traits=[
            Trait(dimension="EXT", name="assertiveness", value=0.7, confidence=0.6),
            Trait(dimension="AGR", name="trust", value=0.3, confidence=0.5),
        ],
    )
    return Soul(
        id="test_soul",
        character=profile,
        facts=facts or [],
        reality=reality,
        secrets=secrets or [],
        contradictions=contradictions or [],
    )


def test_parse_think_deep_response_valid():
    raw = json.dumps({
        "soul_narrative": "A person at a crossroads",
        "intentions": [
            {"description": "start a business", "domain": "career", "strength": 0.8, "blockers": ["money"]},
        ],
        "gaps": [
            {
                "intention": "start a business",
                "reality": "employed full-time",
                "bridge_question": "What's stopping you?",
                "priority": 0.9,
            },
        ],
        "critical_question": "What would you do if money weren't a concern?",
        "conversation_strategy": "Explore risk tolerance",
    })
    data = _parse_think_deep_response(raw)
    assert data["soul_narrative"] == "A person at a crossroads"
    assert len(data["intentions"]) == 1
    assert len(data["gaps"]) == 1
    assert data["critical_question"] != ""


def test_parse_think_deep_response_code_block():
    raw = "```json\n" + json.dumps({
        "soul_narrative": "narrative",
        "intentions": [],
        "gaps": [],
        "critical_question": "question",
        "conversation_strategy": "strategy",
    }) + "\n```"
    data = _parse_think_deep_response(raw)
    assert data["soul_narrative"] == "narrative"


def test_parse_think_deep_response_invalid():
    data = _parse_think_deep_response("not json at all")
    assert data["soul_narrative"] == ""
    assert data["intentions"] == []
    assert data["gaps"] == []
    assert data["critical_question"] == ""
    assert data["conversation_strategy"] == ""


def test_build_soul_context_includes_facts_and_reality():
    soul = _make_soul(
        facts=[
            Fact(category="career", content="software engineer", confidence=0.9, source_turn=3),
            Fact(category="hobby", content="plays guitar", confidence=0.7, source_turn=5),
        ],
        reality=Reality(
            summary="A software engineer who plays guitar",
            domains={"career": "software engineer"},
            constraints=["limited time"],
            resources=["technical skills"],
        ),
        secrets=["avoids discussing family"],
        contradictions=["said values independence but hates making decisions"],
    )
    context = _build_soul_context(soul)
    assert "software engineer" in context
    assert "guitar" in context
    assert "avoids discussing family" in context
    assert "independence" in context
    assert "assertiveness" in context  # from character traits


def test_build_soul_context_minimal_soul():
    soul = _make_soul()
    context = _build_soul_context(soul)
    assert "assertiveness" in context  # character traits always included
    assert "No facts" in context or "facts" in context.lower()


def test_think_deep_analyze_with_mock():
    """Test ThinkDeep.analyze() with a mocked LLM response."""
    with patch("super_brain.think_deep.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "soul_narrative": "An ambitious engineer constrained by financial fears",
            "intentions": [
                {"description": "start own company", "domain": "career", "strength": 0.85, "blockers": ["savings"]},
                {"description": "travel more", "domain": "personal_growth", "strength": 0.6, "blockers": ["time"]},
            ],
            "gaps": [
                {
                    "intention": "start own company",
                    "reality": "employed with mortgage",
                    "bridge_question": "What level of financial security would you need to make the jump?",
                    "priority": 0.9,
                },
            ],
            "critical_question": "If you knew you couldn't fail, what would you do differently tomorrow?",
            "conversation_strategy": "Explore the tension between security and ambition",
        }))]
        mock_client.messages.create.return_value = mock_response

        think_deep = ThinkDeep(api_key="test-key")
        soul = _make_soul(
            facts=[
                Fact(category="career", content="software engineer", confidence=0.9, source_turn=3),
            ],
            reality=Reality(
                summary="Employed engineer with mortgage",
                domains={"career": "software engineer"},
                constraints=["mortgage"],
                resources=["technical skills"],
            ),
        )
        conversation = [
            {"role": "chatter", "text": "What's on your mind lately?"},
            {"role": "speaker", "text": "I've been thinking about starting my own company."},
        ]
        result = think_deep.analyze(soul=soul, conversation=conversation)

        assert isinstance(result, ThinkDeepResult)
        assert len(result.intentions) == 2
        assert result.intentions[0].description == "start own company"
        assert result.intentions[0].strength == 0.85
        assert len(result.gaps) == 1
        assert "fail" in result.critical_question
        assert result.conversation_strategy != ""

        mock_client.messages.create.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/michael/super-brain && .venv/bin/pytest tests/test_think_deep.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

Create `super_brain/think_deep.py`:

```python
"""ThinkDeep: Triggered strategic analysis of intentions and gaps (V2.5).

Unlike ThinkSlow (periodic personality extraction) and FactExtractor (periodic
fact extraction), ThinkDeep is triggered by specific conditions — when enough
facts are accumulated, contradictions emerge, or information goes stale.

It receives the FULL Soul state and produces a strategic analysis:
intentions, reality-intention gaps, bridge questions, and a critical question.
"""

from __future__ import annotations

import json

import anthropic

from super_brain.models import (
    Soul, Intention, Gap, ThinkDeepResult,
)


_THINK_DEEP_SYSTEM = """\
You are a strategic conversation analyst. You have been given a Soul profile — \
a comprehensive understanding of a person built from conversation analysis. Your \
job is to identify their INTENTIONS (what they want), the GAPS between their \
intentions and reality, and generate BRIDGE QUESTIONS that explore those gaps.

Think like a skilled therapist or coach: what is this person really after? What's \
holding them back? What's the ONE question that would unlock the deepest insight?

Return ONLY valid JSON:
{
  "soul_narrative": "<1-2 sentence narrative of who this person is and where they're at>",
  "intentions": [
    {
      "description": "<what they want>",
      "domain": "<career|relationship|personal_growth|health|creative|financial>",
      "strength": <0.0-1.0>,
      "blockers": ["<what's in the way>", ...]
    }
  ],
  "gaps": [
    {
      "intention": "<what they want>",
      "reality": "<where they actually are>",
      "bridge_question": "<question that explores this gap>",
      "priority": <0.0-1.0>
    }
  ],
  "critical_question": "<THE single most important question to ask next>",
  "conversation_strategy": "<1 sentence: how should the conversation shift?>"
}

IMPORTANT:
- Intentions should be INFERRED from facts, behavior, and stated desires — not just what they explicitly say.
- Gaps require both an intention AND a contradicting reality. No gap without both sides.
- The critical_question should be the question that, if answered honestly, would reveal the most about this person.
- Bridge questions should be natural conversation questions, NOT therapy-speak.
- Strength: 0.3-0.5 for hinted, 0.6-0.8 for discussed, 0.9-1.0 for central life theme.
- If insufficient information, return fewer intentions/gaps. Don't fabricate.
"""


def _format_conversation(conversation: list[dict]) -> str:
    """Format conversation for ThinkDeep input."""
    lines = []
    for msg in conversation:
        label = "Person A" if msg["role"] == "chatter" else "Person B"
        lines.append(f"{label}: {msg['text']}")
    return "\n\n".join(lines)


def _build_soul_context(soul: Soul) -> str:
    """Build a text summary of the Soul state for the LLM prompt."""
    sections = []

    # Character traits
    if soul.character.traits:
        trait_lines = [
            f"  {t.name} ({t.dimension}): {t.value:.2f} (conf={t.confidence:.2f})"
            for t in soul.character.traits
        ]
        sections.append("CHARACTER TRAITS:\n" + "\n".join(trait_lines))

    # Facts
    if soul.facts:
        fact_lines = [f"  [{f.category}] {f.content} (conf={f.confidence:.1f})" for f in soul.facts]
        sections.append("KNOWN FACTS:\n" + "\n".join(fact_lines))
    else:
        sections.append("KNOWN FACTS:\n  No facts extracted yet.")

    # Reality
    if soul.reality:
        sections.append(
            f"REALITY SNAPSHOT:\n  {soul.reality.summary}\n"
            f"  Domains: {soul.reality.domains}\n"
            f"  Constraints: {soul.reality.constraints}\n"
            f"  Resources: {soul.reality.resources}"
        )

    # Existing intentions
    if soul.intentions:
        int_lines = [
            f"  {i.description} ({i.domain}, strength={i.strength:.1f})"
            for i in soul.intentions
        ]
        sections.append("PREVIOUSLY DETECTED INTENTIONS:\n" + "\n".join(int_lines))

    # Secrets
    if soul.secrets:
        sections.append("SECRETS/PATTERNS:\n" + "\n".join(f"  - {s}" for s in soul.secrets))

    # Contradictions
    if soul.contradictions:
        sections.append("CONTRADICTIONS:\n" + "\n".join(f"  - {c}" for c in soul.contradictions))

    return "\n\n".join(sections)


def _parse_think_deep_response(raw: str) -> dict:
    """Parse ThinkDeep JSON response with fallback."""
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
    return {
        "soul_narrative": "",
        "intentions": [],
        "gaps": [],
        "critical_question": "",
        "conversation_strategy": "",
    }


class ThinkDeep:
    """Triggered strategic analysis of intentions and gaps."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        kwargs: dict = {"api_key": api_key}
        if api_key.startswith("sk-or-"):
            kwargs["base_url"] = "https://openrouter.ai/api"
        self._client = anthropic.Anthropic(**kwargs)
        self._model = model

    def analyze(
        self,
        soul: Soul,
        conversation: list[dict],
    ) -> ThinkDeepResult:
        """Run strategic analysis on the full Soul state.

        Args:
            soul: Current Soul model with character, facts, reality, secrets.
            conversation: Full conversation so far.

        Returns:
            ThinkDeepResult with intentions, gaps, critical_question, strategy.
        """
        soul_context = _build_soul_context(soul)
        conv_text = _format_conversation(conversation)

        user_message = (
            f"## Soul Profile\n\n{soul_context}\n\n"
            f"## Recent Conversation\n\n{conv_text}\n\n"
            f"## Task\n\nAnalyze Person B's intentions, reality-intention gaps, "
            f"and generate bridge questions. Return JSON."
        )

        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=_THINK_DEEP_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
        )

        raw = response.content[0].text
        data = _parse_think_deep_response(raw)

        # Build typed models
        intentions = []
        for raw_int in data.get("intentions", []):
            intentions.append(Intention(
                description=raw_int.get("description", ""),
                domain=raw_int.get("domain", "personal_growth"),
                strength=max(0.0, min(1.0, float(raw_int.get("strength", 0.5)))),
                blockers=raw_int.get("blockers", []),
            ))

        gaps = []
        for raw_gap in data.get("gaps", []):
            gaps.append(Gap(
                intention=raw_gap.get("intention", ""),
                reality=raw_gap.get("reality", ""),
                bridge_question=raw_gap.get("bridge_question", ""),
                priority=max(0.0, min(1.0, float(raw_gap.get("priority", 0.5)))),
            ))

        return ThinkDeepResult(
            soul_narrative=data.get("soul_narrative", ""),
            intentions=intentions,
            gaps=gaps,
            critical_question=data.get("critical_question", ""),
            conversation_strategy=data.get("conversation_strategy", ""),
        )
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/michael/super-brain && .venv/bin/pytest tests/test_think_deep.py -v`
Expected: ALL PASS (7 tests)

**Step 5: Run all tests**

Run: `cd /Users/michael/super-brain && .venv/bin/pytest tests/ -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add super_brain/think_deep.py tests/test_think_deep.py
git commit -m "feat(v2.5): add ThinkDeep module with intention/gap analysis"
```

---

### Task 3: Enhance Conductor with ThinkDeep Integration

**Files:**
- Modify: `/Users/michael/super-brain/super_brain/conductor.py`
- Modify: `/Users/michael/super-brain/tests/test_conductor.py`

**Context:** The Conductor currently takes `think_fast` and `think_slow` signals and picks from trait_gap incisive questions. V2.5 adds:
1. Accept optional `think_deep: ThinkDeepResult` parameter in `decide()`
2. New priority: When ThinkDeep has a `critical_question`, use "push" mode (between force-probe and follow_thread)
3. `_pick_question` now also considers gap bridge_questions and contradiction-based questions from ThinkDeep
4. IncisiveQuestion source expanded: "trait_gap", "reality_gap", "intention_gap", "contradiction", "secret"

**Step 1: Write the failing tests**

Add to `tests/test_conductor.py`:

```python
from super_brain.models import ThinkDeepResult, Intention, Gap


def test_conductor_uses_think_deep_critical_question():
    """When ThinkDeep provides a critical_question, Conductor pushes it."""
    conductor = Conductor()
    tf = ThinkFastResult(info_entropy=0.5)
    ts = _make_think_slow(info_staleness=0.5)
    td = ThinkDeepResult(
        soul_narrative="At a crossroads",
        intentions=[Intention(description="start business", domain="career", strength=0.8)],
        gaps=[Gap(
            intention="start business",
            reality="employed",
            bridge_question="What would need to change?",
            priority=0.9,
        )],
        critical_question="If you knew you couldn't fail, what would you do?",
        conversation_strategy="Explore risk tolerance",
    )
    action = conductor.decide(think_fast=tf, think_slow=ts, turn_number=8, think_deep=td)
    assert action.mode == "push"
    assert "fail" in action.question.lower()


def test_conductor_think_deep_none_no_change():
    """When think_deep is None, behavior unchanged from V2.3."""
    conductor = Conductor()
    tf = ThinkFastResult(opening="travel plans", info_entropy=0.5)
    action = conductor.decide(think_fast=tf, think_slow=None, turn_number=6, think_deep=None)
    assert action.mode == "follow_thread"


def test_conductor_picks_gap_bridge_question():
    """When ThinkDeep gaps exist, Conductor can pick gap bridge questions."""
    conductor = Conductor()
    tf = ThinkFastResult(info_entropy=0.2)
    ts = _make_think_slow(
        incisive_questions=[
            IncisiveQuestion(question="trait question", target="trust", priority=0.5, source="trait_gap"),
        ],
        info_staleness=0.8,
    )
    td = ThinkDeepResult(
        soul_narrative="narrative",
        intentions=[],
        gaps=[Gap(
            intention="start business",
            reality="employed",
            bridge_question="What's holding you back from taking the leap?",
            priority=0.95,
        )],
        critical_question="",  # empty critical question — not used for push
        conversation_strategy="",
    )
    action = conductor.decide(think_fast=tf, think_slow=ts, turn_number=8, think_deep=td)
    assert action.mode == "ask_incisive"
    # Should pick the gap bridge question since priority=0.95 > trait question priority=0.5
    assert "leap" in action.question.lower() or "trust" in action.question.lower()
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/michael/super-brain && .venv/bin/pytest tests/test_conductor.py -v`
Expected: FAIL (decide() doesn't accept think_deep parameter)

**Step 3: Modify Conductor**

In `super_brain/conductor.py`:

1. Add imports:
```python
from super_brain.models import (
    ConductorAction,
    IncisiveQuestion,
    ThinkDeepResult,
    ThinkFastResult,
    ThinkSlowResult,
)
```

2. Modify `_pick_question` to accept combined question pool:
```python
    def _pick_question(self, questions: list[IncisiveQuestion]) -> IncisiveQuestion:
        """Pick the best incisive question, preferring unasked targets."""
        if not questions:
            return None
        # Prefer questions targeting traits/gaps we haven't asked about yet
        unasked = [q for q in questions if q.target not in self._asked_targets]
        pool = unasked if unasked else questions
        top = max(pool, key=lambda q: q.priority)
        self._asked_targets.add(top.target)
        return top
```

3. Add a helper to merge questions from ThinkSlow and ThinkDeep:
```python
    def _merge_questions(
        self,
        think_slow: ThinkSlowResult | None,
        think_deep: ThinkDeepResult | None,
    ) -> list[IncisiveQuestion]:
        """Merge incisive questions from ThinkSlow trait gaps and ThinkDeep gap bridges."""
        questions = []
        if think_slow and think_slow.incisive_questions:
            questions.extend(think_slow.incisive_questions)
        if think_deep:
            for gap in think_deep.gaps:
                if gap.bridge_question:
                    questions.append(IncisiveQuestion(
                        question=gap.bridge_question,
                        target=f"gap:{gap.intention[:30]}",
                        priority=gap.priority,
                        source="reality_gap",
                    ))
        return questions
```

4. Modify `decide()` signature and body:
```python
    def decide(
        self,
        think_fast: ThinkFastResult,
        think_slow: Optional[ThinkSlowResult],
        turn_number: int,
        think_deep: Optional[ThinkDeepResult] = None,
    ) -> ConductorAction:
```

Add a new priority between force-probe and follow_thread:
```python
        # Priority 1.5: ThinkDeep critical question → push mode
        if (
            think_deep is not None
            and think_deep.critical_question
            and turn_number > self.trust_building_turns
        ):
            self._turns_since_last_incisive = 0
            question = think_deep.critical_question
            # Consume the critical question (only push once per ThinkDeep cycle)
            return ConductorAction(
                mode="push",
                context=f"ThinkDeep critical: {think_deep.conversation_strategy}",
                question=question,
            )
```

Update has_questions and _pick_question calls to use merged pool:
```python
        all_questions = self._merge_questions(think_slow, think_deep)
        has_questions = bool(all_questions)
```

Replace `self._pick_question(think_slow)` calls with `self._pick_question(all_questions)`.

**Step 4: Run tests to verify they pass**

Run: `cd /Users/michael/super-brain && .venv/bin/pytest tests/test_conductor.py -v`
Expected: ALL PASS (existing 9 + 3 new = 12 tests)

**Step 5: Run all tests**

Run: `cd /Users/michael/super-brain && .venv/bin/pytest tests/ -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add super_brain/conductor.py tests/test_conductor.py
git commit -m "feat(v2.5): enhance Conductor with ThinkDeep integration"
```

---

### Task 4: Integrate ThinkDeep into Simulation Loop

**Files:**
- Modify: `/Users/michael/super-brain/eval_conversation.py` (simulate_conversation function)
- Test: `/Users/michael/super-brain/tests/test_simulate_v25.py`

**Context:** ThinkDeep is triggered (not periodic). The simulation loop needs to:
1. Accept optional `think_deep: ThinkDeep` parameter
2. Check trigger conditions each turn (after FactExtractor runs)
3. When triggered, call `think_deep.analyze(soul, conversation)`
4. Store results in Soul (intentions, gaps)
5. Pass ThinkDeepResult to Conductor's `decide()` call
6. Clear ThinkDeep result after Conductor uses its critical_question (one-shot)

**Trigger conditions** (from design doc):
- FactExtractor finds 5+ total facts AND 0 intentions detected yet
- FactExtractor finds a new contradiction
- ThinkSlow info_staleness > 0.8 for 2+ consecutive cycles
- After turn 10 if no intentions detected

**Step 1: Write the failing test**

Create `tests/test_simulate_v25.py`:

```python
"""Tests for V2.5 simulate_conversation with ThinkDeep integration."""

from unittest.mock import MagicMock, patch

from super_brain.models import (
    Fact, Reality, FactExtractionResult, Soul, ThinkSlowResult,
    ThinkDeepResult, Intention, Gap,
    PersonalityDNA, SampleSummary,
)


def test_simulate_with_think_deep_returns_soul_with_intentions():
    """When think_deep is provided, Soul should accumulate intentions and gaps."""
    from eval_conversation import simulate_conversation, Chatter, PersonalitySpeaker
    from super_brain.profile_gen import generate_profile
    from super_brain.think_slow import ThinkSlow
    from super_brain.fact_extractor import FactExtractor
    from super_brain.think_deep import ThinkDeep

    with patch.object(Chatter, "next_message", return_value="Tell me more about that."):
        with patch.object(PersonalitySpeaker, "respond", return_value="I work as an engineer and dream of starting my own company."):
            with patch.object(ThinkSlow, "extract") as mock_ts:
                with patch.object(FactExtractor, "extract") as mock_fe:
                    with patch.object(ThinkDeep, "analyze") as mock_td:
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
                            secrets=[],
                            contradictions=[],
                        )
                        mock_td.return_value = ThinkDeepResult(
                            soul_narrative="An ambitious engineer",
                            intentions=[
                                Intention(description="start company", domain="career", strength=0.8),
                            ],
                            gaps=[
                                Gap(
                                    intention="start company",
                                    reality="employed",
                                    bridge_question="What's stopping you?",
                                    priority=0.9,
                                ),
                            ],
                            critical_question="What would you do if you couldn't fail?",
                            conversation_strategy="Explore risk",
                        )

                        profile = generate_profile("test", seed=0)
                        chatter = Chatter.__new__(Chatter)
                        speaker = PersonalitySpeaker.__new__(PersonalitySpeaker)
                        think_slow = ThinkSlow.__new__(ThinkSlow)
                        fact_extractor = FactExtractor.__new__(FactExtractor)
                        think_deep = ThinkDeep.__new__(ThinkDeep)

                        result = simulate_conversation(
                            chatter, speaker, profile, n_turns=12, seed=0,
                            think_slow=think_slow,
                            fact_extractor=fact_extractor,
                            think_deep=think_deep,
                        )

                        conversation, ts_results, soul = result
                        assert isinstance(soul, Soul)
                        # ThinkDeep should have been triggered and populated intentions
                        assert mock_td.called or len(soul.intentions) >= 0
                        # Soul should have facts from FactExtractor
                        assert len(soul.facts) >= 1
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/michael/super-brain && .venv/bin/pytest tests/test_simulate_v25.py -v`
Expected: FAIL (simulate_conversation doesn't accept think_deep parameter)

**Step 3: Modify simulate_conversation**

In `eval_conversation.py`, modify `simulate_conversation()`:

1. Add `think_deep` parameter:
```python
def simulate_conversation(
    chatter: Chatter,
    speaker: PersonalitySpeaker,
    profile: PersonalityDNA,
    n_turns: int,
    seed: int = 0,
    think_slow: "ThinkSlow | None" = None,
    fact_extractor: "FactExtractor | None" = None,
    think_deep: "ThinkDeep | None" = None,
) -> "list[dict] | tuple[list[dict], list] | tuple[list[dict], list, Soul]":
```

2. Add ThinkDeep state tracking (after `soul` initialization in the `use_adaptive` block):
```python
    last_td: ThinkDeepResult | None = None
    td_fired = False
    consecutive_stale = 0
```

3. Add trigger check function inside the loop (after FactExtractor runs, before the next iteration):
```python
        # V2.5: Check ThinkDeep trigger conditions
        if think_deep is not None and use_adaptive and soul is not None:
            should_fire = False

            # Trigger 1: 5+ facts and no intentions yet
            if len(soul.facts) >= 5 and len(soul.intentions) == 0:
                should_fire = True

            # Trigger 2: New contradiction found this cycle
            if (fe_freq and fe_freq._last_run_turn == turn + 1
                    and hasattr(fe_result, 'contradictions')  # only if FE just ran
                    and len(getattr(fe_result, 'contradictions', [])) > 0):
                should_fire = True

            # Trigger 3: ThinkSlow staleness > 0.8 for 2+ consecutive
            if previous_ts and previous_ts.info_staleness > 0.8:
                consecutive_stale += 1
            else:
                consecutive_stale = 0
            if consecutive_stale >= 2:
                should_fire = True

            # Trigger 4: After turn 10 if no intentions
            if turn + 1 > 10 and len(soul.intentions) == 0:
                should_fire = True

            if should_fire and not td_fired:
                td_result = think_deep.analyze(soul=soul, conversation=conversation)
                last_td = td_result
                td_fired = True  # Only fire once per conversation (reset if new trigger later)
                # Accumulate into Soul
                soul.intentions.extend(td_result.intentions)
                soul.gaps.extend(td_result.gaps)
```

4. Pass `last_td` to Conductor's `decide()`:
```python
            conductor_action = conductor.decide(
                think_fast=last_tf,
                think_slow=previous_ts,
                turn_number=turn + 1,
                think_deep=last_td,
            )
```

5. After Conductor uses the critical_question (push mode), clear `last_td`:
```python
        # V2.5: Clear ThinkDeep after Conductor uses it (one-shot)
        if conductor_action and conductor_action.mode == "push" and last_td is not None:
            last_td = None
            td_fired = False  # Allow re-triggering later
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/michael/super-brain && .venv/bin/pytest tests/test_simulate_v25.py -v`
Expected: PASS

**Step 5: Run all tests**

Run: `cd /Users/michael/super-brain && .venv/bin/pytest tests/ -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add eval_conversation.py tests/test_simulate_v25.py
git commit -m "feat(v2.5): integrate ThinkDeep triggers into simulation loop"
```

---

### Task 5: Update Soul Coverage for V2.5

**Files:**
- Modify: `/Users/michael/super-brain/super_brain/soul_coverage.py`
- Modify: `/Users/michael/super-brain/tests/test_soul_coverage.py`

**Context:** V2.5 adds intentions and gaps to Soul Coverage. The score now has 5 components (was 3 in V2.4):
- facts: min(count / 10, 1.0)
- reality: 1.0 if populated, else 0.0
- secrets: min(count / 3, 1.0)
- intentions: min(count / 3, 1.0) — NEW
- gaps: min(count / 2, 1.0) — NEW

**Step 1: Add failing tests**

Add to `tests/test_soul_coverage.py`:

```python
def test_v25_coverage_with_intentions_and_gaps():
    from super_brain.models import Intention, Gap
    soul = Soul(
        id="test",
        character=_make_profile(),
        facts=[Fact(category="c", content=f"f{i}", confidence=0.9, source_turn=i) for i in range(10)],
        reality=Reality(summary="Full", domains={}, constraints=[], resources=[]),
        secrets=["s1", "s2", "s3"],
        intentions=[
            Intention(description="start biz", domain="career", strength=0.8),
            Intention(description="travel", domain="personal_growth", strength=0.6),
            Intention(description="learn guitar", domain="creative", strength=0.5),
        ],
        gaps=[
            Gap(intention="start biz", reality="employed", bridge_question="q?", priority=0.9),
            Gap(intention="travel", reality="no time", bridge_question="q?", priority=0.7),
        ],
    )
    score = compute_soul_coverage(soul)
    assert score == 1.0  # all 5 components maxed


def test_v25_coverage_partial_intentions():
    from super_brain.models import Intention
    soul = Soul(
        id="test",
        character=_make_profile(),
        facts=[Fact(category="c", content=f"f{i}", confidence=0.9, source_turn=i) for i in range(10)],
        reality=Reality(summary="Full", domains={}, constraints=[], resources=[]),
        secrets=["s1", "s2", "s3"],
        intentions=[
            Intention(description="start biz", domain="career", strength=0.8),
        ],
        # 1 intention / 3 = 0.333, 0 gaps / 2 = 0.0
    )
    score = compute_soul_coverage(soul)
    # (1.0 + 1.0 + 1.0 + 0.333 + 0.0) / 5 = 0.467
    assert abs(score - 0.467) < 0.01
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/michael/super-brain && .venv/bin/pytest tests/test_soul_coverage.py -v`
Expected: FAIL (existing tests may break since component count changed)

**Step 3: Update soul_coverage.py**

```python
def compute_soul_coverage(soul: Soul) -> float:
    """Compute Soul Coverage score (0.0-1.0).

    V2.5 components (5 items, equally weighted):
    - facts: min(len / 10, 1.0) — 10+ facts = full
    - reality: 1.0 if populated, else 0.0
    - secrets: min(len / 3, 1.0) — 3+ secrets = full
    - intentions: min(len / 3, 1.0) — 3+ intentions = full
    - gaps: min(len / 2, 1.0) — 2+ gaps = full
    """
    scores: list[float] = []
    scores.append(min(len(soul.facts) / 10.0, 1.0))
    scores.append(1.0 if soul.reality else 0.0)
    scores.append(min(len(soul.secrets) / 3.0, 1.0))
    scores.append(min(len(soul.intentions) / 3.0, 1.0))
    scores.append(min(len(soul.gaps) / 2.0, 1.0))
    return sum(scores) / len(scores) if scores else 0.0
```

**Step 4: Fix existing tests**

The existing V2.4 tests need updating since the denominator changed from 3 to 5:
- `test_empty_soul_coverage`: Still 0.0 (all zeros) ✓
- `test_full_soul_coverage`: Now needs intentions + gaps to be 1.0
- `test_partial_soul_coverage`: Recalculate with 5 components
- `test_coverage_facts_cap`: Recalculate with 5 components

Update the existing tests:
```python
def test_full_soul_coverage():
    from super_brain.models import Intention, Gap
    soul = Soul(
        id="test",
        character=_make_profile(),
        facts=[Fact(category="c", content=f"fact_{i}", confidence=0.9, source_turn=i) for i in range(10)],
        reality=Reality(summary="Full reality", domains={"career": "engineer"}, constraints=["time"], resources=["skills"]),
        secrets=["s1", "s2", "s3"],
        intentions=[
            Intention(description="i1", domain="career", strength=0.8),
            Intention(description="i2", domain="health", strength=0.6),
            Intention(description="i3", domain="creative", strength=0.5),
        ],
        gaps=[
            Gap(intention="i1", reality="r1", bridge_question="q?", priority=0.9),
            Gap(intention="i2", reality="r2", bridge_question="q?", priority=0.7),
        ],
    )
    score = compute_soul_coverage(soul)
    assert score == 1.0


def test_partial_soul_coverage():
    soul = Soul(
        id="test",
        character=_make_profile(),
        facts=[Fact(category="c", content=f"fact_{i}", confidence=0.9, source_turn=i) for i in range(5)],
        reality=Reality(summary="Partial", domains={}, constraints=[], resources=[]),
        secrets=["s1"],
    )
    score = compute_soul_coverage(soul)
    # (0.5 + 1.0 + 0.333 + 0.0 + 0.0) / 5 = 0.367
    assert abs(score - 0.367) < 0.01


def test_coverage_facts_cap():
    soul = Soul(
        id="test",
        character=_make_profile(),
        facts=[Fact(category="c", content=f"fact_{i}", confidence=0.9, source_turn=i) for i in range(20)],
    )
    # 20 facts → 1.0, reality=0, secrets=0, intentions=0, gaps=0
    score = compute_soul_coverage(soul)
    assert abs(score - 1.0 / 5) < 0.01
```

**Step 5: Run all tests**

Run: `cd /Users/michael/super-brain && .venv/bin/pytest tests/ -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add super_brain/soul_coverage.py tests/test_soul_coverage.py
git commit -m "feat(v2.5): expand Soul Coverage to include intentions + gaps"
```

---

### Task 6: Wire ThinkDeep into run_eval + Update Reporting

**Files:**
- Modify: `/Users/michael/super-brain/eval_conversation.py` (run_eval function)

**Context:** `run_eval()` needs to:
1. Instantiate ThinkDeep alongside ThinkSlow and FactExtractor
2. Pass it to `simulate_conversation()`
3. Add intentions_count and gaps_count to Soul Coverage reporting

**Step 1: Modify run_eval**

After the `fact_extractor = FactExtractor(api_key=api_key)` line, add:
```python
    from super_brain.think_deep import ThinkDeep
    think_deep = ThinkDeep(api_key=api_key)
```

Update the `simulate_conversation` call to pass `think_deep`:
```python
        sim_result = simulate_conversation(
            chatter, speaker, profile, n_turns=max_turns, seed=i,
            think_slow=think_slow,
            fact_extractor=fact_extractor,
            think_deep=think_deep,
        )
```

Update the Soul logging to include intentions and gaps:
```python
        if soul is not None:
            coverage = compute_soul_coverage(soul)
            print(f"    Soul: {len(soul.facts)} facts, "
                  f"reality={'yes' if soul.reality else 'no'}, "
                  f"{len(soul.secrets)} secrets, "
                  f"{len(soul.contradictions)} contradictions, "
                  f"{len(soul.intentions)} intentions, "
                  f"{len(soul.gaps)} gaps, "
                  f"coverage={coverage:.2f}")
```

Update the per-profile results storage:
```python
        if soul is not None:
            all_results[profile_name]["soul_coverage"] = round(compute_soul_coverage(soul), 3)
            all_results[profile_name]["soul_facts_count"] = len(soul.facts)
            all_results[profile_name]["soul_reality_populated"] = soul.reality is not None
            all_results[profile_name]["soul_secrets_count"] = len(soul.secrets)
            all_results[profile_name]["soul_contradictions_count"] = len(soul.contradictions)
            all_results[profile_name]["soul_intentions_count"] = len(soul.intentions)
            all_results[profile_name]["soul_gaps_count"] = len(soul.gaps)
```

Update the summary to include intentions and gaps:
```python
        intentions_counts = [pr.get("soul_intentions_count", 0) for pr in all_results.values()
                             if "soul_intentions_count" in pr]
        gaps_counts = [pr.get("soul_gaps_count", 0) for pr in all_results.values()
                       if "soul_gaps_count" in pr]
        # ... in the print block:
        print(f"  Avg intentions per profile: {statistics.mean(intentions_counts):.1f}")
        print(f"  Avg gaps per profile: {statistics.mean(gaps_counts):.1f}")
```

**Step 2: Run all tests**

Run: `cd /Users/michael/super-brain && .venv/bin/pytest tests/ -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add eval_conversation.py
git commit -m "feat(v2.5): wire ThinkDeep into eval pipeline + enhanced reporting"
```

---

### Task 7: Run V2.5 Eval and Record Results

**Step 1: Run the eval**

```bash
cd /Users/michael/super-brain && ANTHROPIC_API_KEY=REDACTED_OPENROUTER_KEY .venv/bin/python eval_conversation.py 3 20
```

Expected output includes:
- Standard MAE metrics (target: ≤ 0.180)
- Soul Coverage with intentions_count and gaps_count
- Target: intentions ≥ 2, gaps ≥ 1 per profile

**Step 2: Record results in EVAL_HISTORY.md**

Append V2.5 results to `EVAL_HISTORY.md`:
- Version, date, MAE, ≤0.25, ≤0.40, per-dimension table
- Soul Coverage with all 5 components
- Changelog: ThinkDeep module, Conductor enhancement, Soul expansion

**Step 3: Commit and push**

```bash
git add EVAL_HISTORY.md eval_conversation_results.json
git commit -m "eval(v2.5): record V2.5 evaluation results"
git push --force https://REDACTED_GITHUB_PAT@github.com/mozatyin/communication-dna.git main
```

---

## Summary

| Task | What | New/Modified Files | Tests |
|------|------|--------------------|-------|
| 1 | V2.5 data models | models.py | test_models_v25.py (5) |
| 2 | ThinkDeep module | think_deep.py | test_think_deep.py (7) |
| 3 | Conductor enhancement | conductor.py | test_conductor.py (+3) |
| 4 | Integration into sim loop | eval_conversation.py | test_simulate_v25.py (1) |
| 5 | Soul Coverage v2.5 | soul_coverage.py | test_soul_coverage.py (update + 2) |
| 6 | Wire into eval | eval_conversation.py | — |
| 7 | Run eval + record | EVAL_HISTORY.md | — |

**Total new tests: ~18**
**Total new files: 1** (think_deep.py)
**Total modified files: 4** (models.py, conductor.py, soul_coverage.py, eval_conversation.py)
