# Dynamic Probabilistic Soul Architecture — Design Doc

## Context

V2.0-2.2 proved that Deep Listening improves signal quality (10-turn MAE 0.176 beats V1.8's 0.184). But two problems remain:

1. **Fixed phases hurt**: Turns 1-7 rapport → 8-14 deepening → 15+ incisive questions is too rigid. The "incisive questions" phase at turn 15+ fights the detector's "casual conversation" calibration, causing 20-turn regression (MAE 0.196).
2. **Soul is more than personality**: The current system only detects 66 personality traits. SoulMap defines Soul as personality + facts + reality + intention graph + the gap between reality and intentions.

## User's Architectural Vision

> "变成一个动态的概率的think fast, think slow, think deep, 一定的概率是deep listening。然后第二件事情呢，去问问题，然后呢，推动这件事情"

Key principles:
- **ThinkSlow every 3-5 exchanges**: Collect new facts AND accumulate incisive questions
- **Incisive questions target unexposed information**: Use accumulated "secrets" to craft the most important next question when new material runs dry
- **Soul = personality + intention graph + reality + gaps**: Not just personality detection
- **Dynamic probabilistic modes**: Some probability of deep listening, asking questions, pushing forward
- **Questions bridge reality→intention gap**: Know where the user is AND where they want to go

## Architecture: Three Thinking Agents + Conductor

```
                    ┌──────────────────┐
                    │    Soul Model    │  character + facts + reality
                    │   (accumulated)  │  + intentions + gaps + secrets
                    └────────┬─────────┘
                             │ updates
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼─────┐  ┌────▼─────┐  ┌────▼─────┐
        │Think Fast │  │Think Slow│  │Think Deep│
        │(every turn)│  │(every 3-5)│  │(triggered)│
        │ rule-based │  │ full LLM  │  │ large LLM │
        └─────┬─────┘  └────┬─────┘  └────┬─────┘
              │              │              │
              └──────────────┼──────────────┘
                             │ signals
                    ┌────────▼─────────┐
                    │    Conductor     │  probabilistic action selection
                    │  P(listen|ask|   │  based on trust, gaps, staleness
                    │   push|follow)   │
                    └────────┬─────────┘
                             │ action
                    ┌────────▼─────────┐
                    │     Chatter      │  executes chosen action with
                    │  (Deep Listener) │  context from Soul Model
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │     Speaker      │  responds in character
                    │  (method actor)  │  (eval only)
                    └──────────────────┘
```

### Key differences from V2.0-2.2:
- **No fixed phases** — Conductor decides dynamically each turn
- **ThinkFast runs every turn** — catches openings, shifts, contradictions in real-time
- **ThinkSlow accumulates incisive questions** — not just confidence maps
- **Soul is multi-layered** — character + facts + reality + intentions + gaps + secrets
- **Questions bridge reality→intention gap** — not just trait-specific topics

---

## Component 1: Think Fast

**Runs**: After every speaker response
**Cost**: Near-free (rule-based + optional small LLM)
**Purpose**: Real-time signal detection — what just happened in this exchange?

```python
class ThinkFastResult(BaseModel):
    new_facts: list[str]           # "mentioned working at a startup"
    emotional_shift: str | None    # "became guarded when discussing family"
    contradiction: str | None      # "said values independence but hates decisions"
    opening: str | None            # "mentioned unfulfilled dream — follow up?"
    info_entropy: float            # 0-1: how much NEW info this exchange contained
```

**What it detects**:
1. **New facts**: Job, relationships, hobbies, location, education — any concrete information
2. **Emotional shifts**: Tone changes, avoidance, sudden enthusiasm, guardedness
3. **Contradictions**: Inconsistencies between current and previous statements
4. **Openings**: Topics hinted at but not fully explored — these are gold
5. **Information entropy**: Is the conversation producing new information or going in circles?

**Implementation**: Start rule-based (keyword detection, sentiment analysis). Upgrade to small LLM (Haiku) if rules prove insufficient.

---

## Component 2: Think Slow

**Runs**: Every 3-5 exchanges (adaptive frequency)
**Cost**: 1 full LLM call per cycle
**Purpose**: Deep analysis — update the Soul model, identify gaps, accumulate incisive questions

```python
class ThinkSlowResult(BaseModel):
    # Character layer (existing, enhanced)
    trait_updates: dict[str, TraitEstimate]
    confidence_map: dict[str, float]

    # Facts layer
    facts: list[Fact]

    # Reality layer
    reality_snapshot: str    # narrative: "Currently a mid-career engineer,
                              # unhappy with work-life balance..."

    # Intention layer
    intentions: list[Intention]

    # Gap analysis
    reality_intention_gaps: list[Gap]

    # Accumulated incisive questions (KEY NEW FEATURE)
    incisive_questions: list[IncisiveQuestion]

    # Secrets (accumulated insights)
    secrets: list[str]       # "user avoids discussing father"
                              # "enthusiasm spikes when discussing travel"

    # Meta-signals
    info_staleness: float    # how much new info since last ThinkSlow
    suggested_mode: str      # "listen" | "ask" | "push"
```

### Incisive Questions Accumulation

This is the core innovation. ThinkSlow doesn't just track confidence — it generates ranked questions from multiple sources:

```python
class IncisiveQuestion(BaseModel):
    question: str       # the actual question to ask
    target: str         # what it aims to reveal
    priority: float     # 0-1, based on gap importance
    source: str         # where it came from:
                        #   "trait_gap" — low-confidence personality trait
                        #   "reality_gap" — unknown aspect of current situation
                        #   "intention_gap" — unexplored goal/desire
                        #   "contradiction" — conflicting signals
                        #   "secret" — pattern noticed worth exploring
```

**Question generation sources**:
1. **Trait gaps**: Low-confidence traits → natural conversation topics (from trait_topic_map.py)
2. **Reality gaps**: Missing information about situation → factual questions
3. **Intention gaps**: Unexplored domains → "what would your ideal X look like?"
4. **Contradictions**: Conflicting signals → "you mentioned X but also Y — help me understand?"
5. **Secrets**: Observed patterns → "you seem to light up when discussing X — what draws you?"

### Adaptive Frequency

```python
# Default: every 3 exchanges
# More frequent (every 2) when info_entropy is high (lots of new info)
# Less frequent (every 5) when conversation is stable/repetitive
think_slow_interval = 3
if last_think_fast.info_entropy > 0.7:
    think_slow_interval = 2
elif last_think_fast.info_entropy < 0.2:
    think_slow_interval = 5
```

---

## Component 3: Think Deep

**Runs**: Triggered by specific conditions (rare — maybe 1-2 times per conversation)
**Cost**: 1 large LLM call
**Purpose**: Strategic meta-analysis — step back and see the big picture

```python
class ThinkDeepResult(BaseModel):
    soul_narrative: str              # "This person is at a crossroads..."
    reality_intention_bridge: str    # "The key question connecting their
                                     #  reality to intention is..."
    conversation_strategy: str       # "Shift from listening to exploring
                                     #  their relationship with risk"
    critical_question: str           # THE single most important question
```

**Trigger conditions**:
- Reality-intention gap score > 0.7 (major disconnect detected)
- ThinkFast detects a major contradiction
- ThinkSlow info_staleness > 0.8 (conversation going in circles for 2+ cycles)
- After 10+ turns with no new intention detected

Think Deep produces the **bridge question** — the question that connects where the user IS to where they WANT TO BE. This is the deepest level of SoulMap's methodology.

---

## Component 4: The Conductor

The Conductor is the dynamic probabilistic decision engine.

```python
class ConductorAction(BaseModel):
    mode: str           # "listen" | "follow_thread" | "ask_incisive" | "push"
    context: str        # guidance for the Chatter
    question: str|None  # specific question (from accumulated queue)

def conduct(
    think_fast: ThinkFastResult,
    think_slow: ThinkSlowResult | None,
    think_deep: ThinkDeepResult | None,
    turn_number: int,
    soul: Soul,
) -> ConductorAction:

    # Early turns: mostly listen (build trust)
    if turn_number <= 3:
        return ConductorAction(mode="listen", context="Build rapport")

    # Priority 1: ThinkFast detected an opening → follow it immediately
    if think_fast.opening:
        return ConductorAction(
            mode="follow_thread",
            context=f"Follow up on: {think_fast.opening}"
        )

    # Priority 2: ThinkDeep has a critical question → push
    if think_deep and think_deep.critical_question:
        return ConductorAction(
            mode="push",
            question=think_deep.critical_question
        )

    # Priority 3: Info is stale + accumulated questions exist → ask
    if (think_fast.info_entropy < 0.3
            and think_slow
            and think_slow.incisive_questions):
        top_q = think_slow.incisive_questions[0]
        return ConductorAction(
            mode="ask_incisive",
            question=top_q.question,
            context=f"Target: {top_q.target}"
        )

    # Priority 4: Contradiction detected → explore it
    if think_fast.contradiction:
        return ConductorAction(
            mode="ask_incisive",
            context=f"Explore contradiction: {think_fast.contradiction}"
        )

    # Default: keep listening
    return ConductorAction(mode="listen", context="Continue deep listening")
```

### Mode Behaviors

| Mode | Chatter Behavior |
|------|-----------------|
| **listen** | Pure deep listening. Short responses. Reflect back. One follow-up question. |
| **follow_thread** | Follow an opening the speaker just hinted at. "You mentioned X — tell me more?" |
| **ask_incisive** | Ask a specific accumulated question, woven naturally into conversation flow. |
| **push** | Challenge or probe deeper. "You said you want X but you also mentioned Y — what would it look like to reconcile those?" |

---

## Component 5: Soul Model

```python
class Soul(BaseModel):
    """Complete Soul: Character + Facts + Reality + Intentions + Gaps."""
    id: str

    # Layer 1: Character (existing 66 traits)
    character: PersonalityDNA

    # Layer 2: Facts (observed factual info)
    facts: list[Fact] = []

    # Layer 3: Reality (current state/situation)
    reality: Reality | None = None

    # Layer 4: Intentions (goals, desires)
    intentions: list[Intention] = []

    # Cross-layer analysis
    gaps: list[Gap] = []              # reality → intention bridges
    secrets: list[str] = []           # accumulated insights
    contradictions: list[str] = []    # unresolved contradictions

class Fact(BaseModel):
    category: str       # "career", "relationship", "hobby", "education", "location", "family"
    content: str        # "software engineer at a startup"
    confidence: float
    source_turn: int

class Reality(BaseModel):
    summary: str                     # narrative of current state
    domains: dict[str, str]          # {"career": "...", "relationships": "..."}
    constraints: list[str]           # things limiting them
    resources: list[str]             # things they have going for them

class Intention(BaseModel):
    description: str                 # "wants to start own business"
    domain: str                      # "career", "relationship", "personal_growth"
    strength: float                  # 0-1
    blockers: list[str]              # what's in the way

class Gap(BaseModel):
    intention: str                   # what they want
    reality: str                     # where they are
    bridge_question: str             # question exploring this gap
    priority: float                  # 0-1
```

---

## Implementation Plan (Phased)

### V2.3: Think Fast + Conductor (replace fixed phases)
- Implement ThinkFast (rule-based)
- Implement Conductor (replaces `_build_chatter_system()` phase logic)
- Chatter now receives ConductorAction instead of phase instructions
- ThinkSlow remains as-is but passes accumulated questions to Conductor
- **Eval**: Character MAE should improve by removing the 20-turn regression

### V2.4: Enhanced ThinkSlow (accumulate questions + secrets)
- ThinkSlow prompt expanded to generate incisive questions from gaps
- ThinkSlow detects facts and reality alongside personality traits
- Soul model expanded with Fact, Reality types
- **Eval**: Character MAE + Soul Coverage metrics

### V2.5: Think Deep + Intention Detection
- Implement ThinkDeep (triggered by Conductor)
- ThinkSlow begins detecting intentions
- Gap analysis: reality → intention bridge questions
- **Eval**: Character MAE + Intention detection quality

### V2.6: Full Soul Profile + Adaptive ThinkSlow Frequency
- All four Soul layers populated
- ThinkSlow frequency adapts based on info_entropy
- Conductor uses full Soul context for action selection
- **Eval**: Full Soul eval framework

---

## Eval Strategy

### Core metric: Character MAE (continuity with V0.1-V2.2)
Same eval framework: generate profile → simulate conversation → detect → compare.

### New metrics (introduced progressively):

| Metric | Introduced | Description |
|--------|-----------|-------------|
| Character MAE | V0.1+ | Mean absolute error on 66 traits |
| 10t/20t gap | V2.3 | MAE difference between 10-turn and 20-turn detection |
| Soul Coverage | V2.4 | % of Soul model populated (facts, reality, intentions) |
| Question Quality | V2.4 | Manual review: did accumulated questions target real gaps? |
| Intention F1 | V2.5 | Precision/recall on detected intentions |
| Gap Quality | V2.6 | Did reality→intention gaps generate useful bridge questions? |

### Expected results:

| Version | Target MAE (20t) | Key improvement |
|---------|-------------------|-----------------|
| V2.2 (current) | 0.196 | Deep Listening + ThinkSlow + Gap-Aware |
| V2.3 | < 0.180 | Dynamic Conductor eliminates 20t regression |
| V2.4 | < 0.170 | Better incisive questions from accumulated gaps |
| V2.5 | < 0.160 | Intention detection cross-validates personality |
| V2.6 | < 0.150 | Full Soul context improves all detection |

---

## Key Design Principles

1. **Dynamic over fixed**: No predetermined phases — the Conductor adapts to what's happening
2. **Accumulate, don't discard**: Every exchange adds to the Soul model. Nothing is thrown away.
3. **Questions from gaps**: Incisive questions are generated from what we DON'T know, not from a fixed list
4. **Secrets are data**: Patterns like avoidance, enthusiasm spikes, and contradictions are first-class signals
5. **Bridge reality→intention**: The deepest insight comes from understanding where someone IS vs. where they WANT TO BE
6. **Soft corrections only**: Never hard-code overrides (lesson from V1.0). All adjustments are probabilistic.
7. **Measurable at every step**: Each version has a clear eval metric before moving to the next.
